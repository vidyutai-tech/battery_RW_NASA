"""
Stitch NASA RW ``.mat`` data the same way as the original author's ``data_loading.py``.

* Every step in the file is included (no RW comment filter).
* ``age`` at each sample = ``step_index / n_steps`` (author: ``i / len(list_)``).
* Arrays are concatenated in step order (voltage, current, temperature, times).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from rw_transfer.data.mat_loader import load_cell_steps, mat_path_for_cell


@dataclass
class AuthorStitchedSeries:
    """Stitched 1-D arrays matching the author's ``MyDataset`` inputs."""

    cell_id: str
    non_relative_time_s: np.ndarray
    relative_time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray
    age: np.ndarray
    n_steps: int

    @property
    def n_samples(self) -> int:
        return int(self.voltage_v.size)

    @property
    def duration_hours(self) -> float:
        if self.non_relative_time_s.size < 2:
            return 0.0
        return float(
            (self.non_relative_time_s[-1] - self.non_relative_time_s[0]) / 3600.0
        )


def load_author_stitched_series(
    matlab_dir: str,
    cell_id: str,
    decimation: int = 1,
) -> AuthorStitchedSeries:
    """
    Load and stitch all steps from ``RW*.mat`` (author-style, no step filter).
    """
    path = mat_path_for_cell(matlab_dir, cell_id)
    steps = load_cell_steps(path, step_mode="all")
    if not steps:
        raise ValueError(f"No steps in {path}")

    n_steps = len(steps)
    parts: Dict[str, list] = {
        "non_relative_time_s": [],
        "relative_time_s": [],
        "voltage_v": [],
        "current_a": [],
        "temperature_c": [],
        "age": [],
    }

    for i, step in enumerate(steps):
        n = step.voltage_v.size
        if n == 0:
            continue
        age_val = float(i) / float(n_steps)
        parts["non_relative_time_s"].append(step.time_s)
        parts["relative_time_s"].append(step.relative_time_s)
        parts["voltage_v"].append(step.voltage_v)
        parts["current_a"].append(step.current_a)
        parts["temperature_c"].append(step.temperature_c)
        parts["age"].append(np.full(n, age_val, dtype=np.float64))

    dec = max(int(decimation), 1)
    sl = slice(None, None, dec)

    cid = cell_id.upper()
    if not cid.startswith("RW"):
        cid = f"RW{cid}"

    return AuthorStitchedSeries(
        cell_id=cid,
        non_relative_time_s=np.concatenate(parts["non_relative_time_s"])[sl].astype(np.float64),
        relative_time_s=np.concatenate(parts["relative_time_s"])[sl].astype(np.float64),
        voltage_v=np.concatenate(parts["voltage_v"])[sl].astype(np.float32),
        current_a=np.concatenate(parts["current_a"])[sl].astype(np.float32),
        temperature_c=np.concatenate(parts["temperature_c"])[sl].astype(np.float32),
        age=np.concatenate(parts["age"])[sl].astype(np.float32),
        n_steps=n_steps,
    )
