"""Voltage→SoC MLP from Old_Codes/SOC_modelling (rest / low-current OCV proxy)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class VoltageSOCMLP(nn.Module):
    """Matches ``MyModel`` in ``Old_Codes/SOC_modelling/training.ipynb``."""

    def __init__(self) -> None:
        super().__init__()
        self.input_layer = nn.Linear(1, 10)
        self.hidden_layer = nn.Linear(10, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        return torch.sigmoid(self.hidden_layer(x))


def load_voltage_soc_mlp(
    ckpt_path: str | Path,
    *,
    device: str = "cpu",
) -> VoltageSOCMLP:
    model = VoltageSOCMLP()
    state = torch.load(Path(ckpt_path), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def mlp_ocv_soc_grid(
    model: VoltageSOCMLP,
    v_min: float = 3.0,
    v_max: float = 4.3,
    n_points: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dense (V, SoC) grid with monotone non-decreasing SoC."""
    v = np.linspace(v_min, v_max, n_points, dtype=np.float64)
    vt = torch.tensor(v, dtype=torch.float32).unsqueeze(-1)
    soc = model(vt).numpy().ravel().astype(np.float64)
    soc = np.maximum.accumulate(soc)
    soc = np.clip(soc, 0.0, 1.0)
    return v, soc
