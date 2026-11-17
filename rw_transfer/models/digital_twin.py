"""
Transformer-decoder digital twin for battery voltage/temperature prediction.

Matches v9_rescaling_adaptive_TransformerModel3Decoder (Old_Codes):
  [age, V₀/3, T₀/30] per step + current/5 + Δ(current/5) → decoder → ΔV, ΔT/10.

No ambient input (author trajectory).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class BatteryDigitalTwin(nn.Module):
  """
  v9-style digital twin: 3-scalar initial state (age, V₀, T₀), no ambient.

  Temperature head: ``T = T₀ + residual_T * temp_delta_scale`` with default scale
  0.1 (equivalent to patent ``/10`` on the raw residual logit).
  """

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

    # Per-step: [age, v₀/3, t₀/30, I/5, Δ(I/5)] → 5 features
    self.linear_in = nn.Linear(5, d_model)
    self.linear_in_1 = nn.Linear(d_model, d_model)
    self.positional_encoding = nn.Parameter(torch.zeros(seq_len, d_model))
    nn.init.trunc_normal_(self.positional_encoding, std=0.02)

    decoder_layer = nn.TransformerDecoderLayer(
      d_model=full_dim,
      nhead=nhead,
      dropout=dropout,
      batch_first=True,
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
      transformer_input,
      transformer_input,
      tgt_mask=tgt_mask,
    )

    residual = self.gelu(self.linear_out1(transformer_output))
    residual = self.gelu(self.linear_out2(residual))
    residual = self.linear_out3(residual)

    base_voltage = initial_state[:, 1].unsqueeze(1).expand(batch_size, steps)
    base_temperature = initial_state[:, 2].unsqueeze(1).expand(batch_size, steps)
    voltage = base_voltage + residual[:, :, 0]
    temperature = base_temperature + residual[:, :, 1] * self.temp_delta_scale
    return voltage, temperature

  def forward_author(
    self,
    starting_state: torch.Tensor,
    actions: torch.Tensor,
  ) -> torch.Tensor:
    """
    Author API: ``starting_state`` (B, 3), ``actions`` (B, T, 1) → ``(B, T, 2)``.
    """
    age = starting_state[:, 0]
    v0 = starting_state[:, 1]
    t0 = starting_state[:, 2]
    current_seq = actions.squeeze(-1)
    v_hat, t_hat = self.forward(age, v0, t0, current_seq)
    return torch.stack([v_hat, t_hat], dim=-1)

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

  # ── Freeze / unfreeze helpers for staged fine-tuning ──────────────────────

  def freeze_backbone(self) -> None:
    """Freeze everything except the output projection head (linear_out1/2/3).

    Use before stage-1 temperature warmup so the transformer's learned voltage
    physics is preserved while the output head re-calibrates to the target cell.
    """
    head_prefixes = ("linear_out1", "linear_out2", "linear_out3")
    for name, param in self.named_parameters():
      is_head = any(name.startswith(p) for p in head_prefixes)
      param.requires_grad_(is_head)

  def unfreeze_all(self) -> None:
    """Re-enable gradients for all parameters (call after stage-1 warmup)."""
    for param in self.parameters():
      param.requires_grad_(True)

  @property
  def n_trainable_params(self) -> int:
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
    """Predict V/T for profiles longer than ``seq_len`` via chained chunks."""
    self.eval()
    profile = np.asarray(current_profile, dtype=np.float32)
    if profile.size == 0:
      return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    volt_parts, temp_parts = [], []
    cursor_v, cursor_t = float(v0), float(t0)
    for start in range(0, profile.shape[0], self.seq_len):
      chunk = profile[start : start + self.seq_len]
      volt_chunk, temp_chunk = self._predict_chunk(
        relative_age=relative_age,
        v0=cursor_v,
        t0=cursor_t,
        current_chunk=chunk,
        v_stats=v_stats,
        t_stats=t_stats,
      )
      volt_parts.append(volt_chunk)
      temp_parts.append(temp_chunk)
      cursor_v = float(volt_chunk[-1])
      cursor_t = float(temp_chunk[-1])

    return np.concatenate(volt_parts), np.concatenate(temp_parts)
