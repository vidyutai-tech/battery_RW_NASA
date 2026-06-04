"""
MLP for State-of-Charge (SOC) estimation (Stage 2).

Supported input variants (``soc_variant``):
  v_only  — [voltage]                         author-style baseline
  vta     — [voltage, temperature, age]       default multi-feature model
  vta_i   — [voltage, temperature, age, |I|]  discharge-rate-aware model

Training in this repo uses **measured** V/T only (``soc_input: measured``).
At optimization inference, twin-predicted V/T may be used when explicitly wired.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

SOC_VARIANTS = {
    "v_only": 1,
    "vta": 3,
    "vta_i": 4,
}

SOC_INPUT_DIM_TWIN = SOC_VARIANTS["vta"]


def soc_variant_input_dim(variant: str) -> int:
    if variant not in SOC_VARIANTS:
        raise ValueError(
            f"Unknown soc_variant {variant!r}; expected one of {list(SOC_VARIANTS)}"
        )
    return SOC_VARIANTS[variant]


def soc_variant_ckpt_name(variant: str) -> str:
    return f"soc_model_{variant}.pt"


def soc_variant_registry_key(variant: str) -> str:
    return f"soc_model_{variant}"


class SOCModel(nn.Module):
    """3-layer MLP with skip connection for SOC regression."""

    def __init__(self, input_dim: int = SOC_INPUT_DIM_TWIN, hidden: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),   # enforce [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, input_dim) → SOC : (B, 1)"""
        h = self.stem(x)
        h = h + self.hidden1(h)   # residual
        h = self.hidden2(h)
        return self.head(h)

    @staticmethod
    def stack_features(
        voltage: np.ndarray,
        temperature: Optional[np.ndarray] = None,
        relative_age: float = 0.0,
        current_abs: Optional[np.ndarray] = None,
        variant: str = "vta",
        input_dim: Optional[int] = None,
    ) -> np.ndarray:
        """
        Build (T, D) feature matrix for batch or sequence inference.

        ``input_dim`` is kept for backward compatibility with older checkpoints
        that only stored dimensionality (3 → ``vta``, 1 → ``v_only``, 4 → ``vta_i``).
        """
        if input_dim is not None and variant == "vta" and input_dim != SOC_VARIANTS["vta"]:
            if input_dim == 1:
                variant = "v_only"
            elif input_dim == 4:
                variant = "vta_i"

        n = len(voltage)
        v = voltage.astype(np.float32)

        if variant == "v_only":
            return v.reshape(-1, 1)

        t = (
            temperature.astype(np.float32)
            if temperature is not None
            else np.full(n, 25.0, dtype=np.float32)
        )
        age_col = np.full(n, relative_age, dtype=np.float32)

        if variant == "vta":
            return np.stack([v, t, age_col], axis=1)

        if variant == "vta_i":
            if current_abs is None:
                raise ValueError("soc_variant='vta_i' requires current_abs")
            i_abs = np.abs(current_abs.astype(np.float32))
            return np.stack([v, t, age_col, i_abs], axis=1)

        raise ValueError(f"Unknown soc_variant: {variant!r}")

    @torch.no_grad()
    def predict(
        self,
        voltage: float,
        temperature: float = 25.0,
        relative_age: float = 0.0,
        current_abs: float = 0.0,
        variant: str = "vta",
    ) -> float:
        self.eval()
        features = SOCModel.stack_features(
            np.array([voltage], dtype=np.float32),
            np.array([temperature], dtype=np.float32),
            relative_age=relative_age,
            current_abs=np.array([current_abs], dtype=np.float32),
            variant=variant,
            input_dim=self.input_dim,
        )
        x = torch.tensor(features, dtype=torch.float32)
        return float(self.forward(x).item())

    # ------------------------------------------------------------------
    # Physics-based fallback (no model needed)
    # ------------------------------------------------------------------

    @staticmethod
    def voltage_to_soc(
        voltage: float, v_min: float = 3.0, v_max: float = 4.2
    ) -> float:
        """Linear SOC estimate from voltage (works without a trained model)."""
        return float(np.clip((voltage - v_min) / (v_max - v_min), 0.0, 1.0))

    @staticmethod
    def sequence_soc(
        voltage_sequence: np.ndarray, v_min: float = 3.0, v_max: float = 4.2
    ) -> np.ndarray:
        return np.clip((voltage_sequence - v_min) / (v_max - v_min), 0.0, 1.0)
