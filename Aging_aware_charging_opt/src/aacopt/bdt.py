"""Inference-only Battery Digital Twin.

Architecture is a verbatim copy of ``rw_transfer.models.digital_twin.BatteryDigitalTwin``
(v9 Transformer decoder). Training code is not included. Checkpoints are the
inherited ``.pt`` files listed in ``configs/paths.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class BatteryDigitalTwin(nn.Module):
    """v9-style twin: 3-scalar initial state (age, V0, T0), no ambient."""

    def __init__(
        self,
        seq_len: int = 150,
        d_model: int = 150,
        nhead: int = 20,
        num_layers: int = 1,
        dropout: float = 0.1,
        temp_delta_scale: float = 0.1,
        author_style: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.temp_delta_scale = float(temp_delta_scale)
        self.author_style = bool(author_style)
        full_dim = 2 * d_model
        if full_dim % nhead != 0:
            raise ValueError(f"Decoder d_model*2 ({full_dim}) must be divisible by nhead ({nhead})")

        self.linear_in = nn.Linear(5, d_model)
        self.linear_in_1 = nn.Linear(d_model, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(seq_len, d_model))
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=full_dim, nhead=nhead, dropout=dropout, batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.linear_out1 = nn.Linear(full_dim, 5 * 2)
        self.linear_out2 = nn.Linear(5 * 2, 2 * 2)
        self.linear_out3 = nn.Linear(2 * 2, 2)
        self.gelu = nn.GELU()
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(size, device=device)

    def forward(
        self,
        age: torch.Tensor,
        v0: torch.Tensor,
        t0: torch.Tensor,
        current_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps = current_seq.shape
        initial_state = torch.stack([age, v0, t0], dim=-1)
        scaled_state = initial_state.clone()
        scaled_state[:, 1] = scaled_state[:, 1] / 3.0
        scaled_state[:, 2] = scaled_state[:, 2] / 30.0

        actions = (current_seq / 5.0).unsqueeze(-1)
        actions_delta = actions.clone()
        actions_delta[:, :-1, :] = actions[:, :-1, :] - actions[:, 1:, :]
        actions_delta[:, -1, :] = 0.0

        state_repeated = scaled_state.unsqueeze(1).expand(batch_size, steps, 3)
        transformer_input = torch.cat([state_repeated, actions, actions_delta], dim=-1)
        if self.author_style:
            transformer_input = self.linear_in_1(self.linear_in(transformer_input))
        else:
            transformer_input = self.gelu(
                self.linear_in_1(self.gelu(self.linear_in(transformer_input)))
            )

        pos_encoding = self.positional_encoding[:steps].unsqueeze(0).expand(batch_size, -1, -1)
        transformer_input = torch.cat([transformer_input, pos_encoding], dim=-1)
        tgt_mask = self._causal_mask(steps, transformer_input.device)
        transformer_output = self.transformer_decoder(
            transformer_input, transformer_input, tgt_mask=tgt_mask,
        )
        residual = self.gelu(self.linear_out1(transformer_output))
        residual = self.gelu(self.linear_out2(residual))
        residual = self.linear_out3(residual)

        base_voltage = initial_state[:, 1].unsqueeze(1).expand(batch_size, steps)
        base_temperature = initial_state[:, 2].unsqueeze(1).expand(batch_size, steps)
        voltage = base_voltage + residual[:, :, 0]
        temperature = base_temperature + residual[:, :, 1] * self.temp_delta_scale
        return voltage, temperature

    def _predict_chunk(
        self,
        relative_age: float,
        v0: float,
        t0: float,
        current_chunk: np.ndarray,
        v_stats: Tuple[float, float],
        t_stats: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        device = next(self.parameters()).device
        chunk_len = len(current_chunk)
        padded = np.zeros(self.seq_len, dtype=np.float32)
        padded[:chunk_len] = current_chunk
        age_t = torch.tensor([relative_age], dtype=torch.float32, device=device)
        v0_t = torch.tensor([v0], dtype=torch.float32, device=device)
        t0_t = torch.tensor([t0], dtype=torch.float32, device=device)
        curr_t = torch.from_numpy(padded[np.newaxis]).to(dtype=torch.float32, device=device)
        voltage, temperature = self.forward(age_t, v0_t, t0_t, curr_t)
        volt = voltage[0, :chunk_len].detach().cpu().numpy()
        temp = temperature[0, :chunk_len].detach().cpu().numpy()
        volt_scale = float(v_stats[1]) if abs(v_stats[1] - 1.0) > 1e-8 else 1.0
        temp_scale = float(t_stats[1]) if abs(t_stats[1] - 1.0) > 1e-8 else 1.0
        if volt_scale != 1.0:
            volt = v0 + (volt - v0) * volt_scale
        if temp_scale != 1.0:
            temp = t0 + (temp - t0) * temp_scale
        return volt, temp

    @torch.no_grad()
    def predict(
        self,
        relative_age: float,
        v0: float,
        t0: float,
        current_profile: np.ndarray,
        v_stats: Tuple[float, float] = (0.0, 1.0),
        t_stats: Tuple[float, float] = (0.0, 1.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        profile = np.asarray(current_profile, dtype=np.float32)
        if profile.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
        volt_parts, temp_parts = [], []
        cursor_v, cursor_t = float(v0), float(t0)
        for start in range(0, profile.shape[0], self.seq_len):
            chunk = profile[start: start + self.seq_len]
            volt_chunk, temp_chunk = self._predict_chunk(
                relative_age=relative_age, v0=cursor_v, t0=cursor_t,
                current_chunk=chunk, v_stats=v_stats, t_stats=t_stats,
            )
            volt_parts.append(volt_chunk)
            temp_parts.append(temp_chunk)
            cursor_v = float(volt_chunk[-1])
            cursor_t = float(temp_chunk[-1])
        return np.concatenate(volt_parts), np.concatenate(temp_parts)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_twin(checkpoint: Path, device: str = "auto") -> BatteryDigitalTwin:
    """Load an inherited checkpoint without the training trainer class."""
    dev = resolve_device(device)
    ckpt = torch.load(Path(checkpoint), map_location=dev, weights_only=False)
    model = BatteryDigitalTwin(
        seq_len=int(ckpt.get("seq_len", 150)),
        d_model=int(ckpt.get("twin_d_model", 150)),
        nhead=int(ckpt.get("twin_nhead", 20)),
        num_layers=int(ckpt.get("twin_num_layers", 1)),
        dropout=float(ckpt.get("twin_dropout", 0.1)),
        temp_delta_scale=float(ckpt.get("temp_delta_scale", 0.1)),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class FrozenBDT:
    """Closed-loop wrapper used by the charging simulator."""

    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        self.checkpoint = Path(checkpoint)
        self.model = load_twin(self.checkpoint, device=device)
        self.device = next(self.model.parameters()).device
        self.seq_len = self.model.seq_len

    def predict_traj(
        self, age: float, v0: float, t0: float, current_profile: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict(
            relative_age=float(age), v0=float(v0), t0=float(t0),
            current_profile=np.asarray(current_profile, dtype=np.float32),
        )

    def single_step(
        self,
        state: Dict[str, float],
        action_a: float,
        n_steps: int,
        v_ceiling: float = 4.2,
        switch_pad: int = 5,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, bool]:
        prev_i = float(state.get("prev_i", 0.0))
        pad = int(switch_pad) if abs(prev_i - float(action_a)) > 1e-9 else 0
        profile = np.concatenate([
            np.full(pad, prev_i, dtype=np.float32),
            np.full(int(n_steps), float(action_a), dtype=np.float32),
        ])
        v_pred, t_pred = self.predict_traj(state["age"], state["v0"], state["t0"], profile)
        v_traj, t_traj = v_pred[pad:], t_pred[pad:]
        over = np.flatnonzero(v_traj >= v_ceiling)
        terminated = bool(over.size)
        if terminated:
            cut = int(over[0]) + 1
            v_traj, t_traj = v_traj[:cut], t_traj[:cut]
        next_state = {
            "v0": float(v_traj[-1]) if v_traj.size else state["v0"],
            "t0": float(t_traj[-1]) if t_traj.size else state["t0"],
            "age": state["age"],
            "prev_i": float(action_a) if v_traj.size else prev_i,
        }
        return next_state, v_traj, t_traj, terminated

    @torch.no_grad()
    def predict_windows(self, X: np.ndarray, batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Batched window inference. ``X`` is ``[N, 3+seq_len]`` = age, V0, T0, I[0:T]."""
        self.model.eval()
        x = np.asarray(X, dtype=np.float32)
        if x.size == 0:
            return np.empty((0, self.seq_len)), np.empty((0, self.seq_len))
        v_parts, t_parts = [], []
        for i in range(0, x.shape[0], int(batch_size)):
            xb = x[i : i + int(batch_size)]
            age = torch.tensor(xb[:, 0], device=self.device, dtype=torch.float32)
            v0 = torch.tensor(xb[:, 1], device=self.device, dtype=torch.float32)
            t0 = torch.tensor(xb[:, 2], device=self.device, dtype=torch.float32)
            curr = torch.tensor(xb[:, 3:], device=self.device, dtype=torch.float32)
            v_hat, t_hat = self.model(age, v0, t0, curr)
            v_parts.append(v_hat.detach().cpu().numpy())
            t_parts.append(t_hat.detach().cpu().numpy())
        return np.concatenate(v_parts, axis=0), np.concatenate(t_parts, axis=0)
