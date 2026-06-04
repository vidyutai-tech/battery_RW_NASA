"""Stitch filtered steps into a continuous time series."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from rw_transfer.data.mat_loader import load_cell_steps, mat_path_for_cell


@dataclass
class BatteryTimeSeries:
    cell_id: str
    time_s: np.ndarray
    relative_time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray
    age: np.ndarray
    comment: np.ndarray
    step_index: np.ndarray

    @property
    def duration_hours(self) -> float:
        if self.time_s.size < 2:
            return 0.0
        return float((self.time_s[-1] - self.time_s[0]) / 3600.0)

    def slice_by_time(self, t_end_s: float) -> "BatteryTimeSeries":
        """First segment with ``time_s <= t_end_s`` (chronological prefix)."""
        mask = self.time_s <= t_end_s
        if not np.any(mask):
            mask = np.zeros_like(self.time_s, dtype=bool)
            mask[0] = True
        return BatteryTimeSeries(
            cell_id=self.cell_id,
            time_s=self.time_s[mask],
            relative_time_s=self.relative_time_s[mask],
            voltage_v=self.voltage_v[mask],
            current_a=self.current_a[mask],
            temperature_c=self.temperature_c[mask],
            age=self.age[mask],
            comment=self.comment[mask],
            step_index=self.step_index[mask],
        )

    def slice_by_fraction(self, frac: float) -> "BatteryTimeSeries":
        frac = float(np.clip(frac, 0.0, 1.0))
        if frac <= 0.0 or self.time_s.size == 0:
            return self.slice_by_time(self.time_s[0])
        t0, t1 = self.time_s[0], self.time_s[-1]
        t_end = t0 + frac * (t1 - t0)
        return self.slice_by_time(t_end)

    def remainder_after(self, prefix: "BatteryTimeSeries") -> "BatteryTimeSeries":
        if prefix.time_s.size == 0:
            return self
        t_cut = prefix.time_s[-1]
        mask = self.time_s > t_cut
        return BatteryTimeSeries(
            cell_id=self.cell_id,
            time_s=self.time_s[mask],
            relative_time_s=self.relative_time_s[mask],
            voltage_v=self.voltage_v[mask],
            current_a=self.current_a[mask],
            temperature_c=self.temperature_c[mask],
            age=self.age[mask],
            comment=self.comment[mask],
            step_index=self.step_index[mask],
        )


def slice_battery_series(series: BatteryTimeSeries, sl: slice) -> BatteryTimeSeries:
    """Chronological index slice (used for SOC train/val/test splits)."""
    return BatteryTimeSeries(
        cell_id=series.cell_id,
        time_s=series.time_s[sl],
        relative_time_s=series.relative_time_s[sl],
        voltage_v=series.voltage_v[sl],
        current_a=series.current_a[sl],
        temperature_c=series.temperature_c[sl],
        age=series.age[sl],
        comment=series.comment[sl],
        step_index=series.step_index[sl],
    )


def load_battery_series(
    matlab_dir: str | Path,
    cell_id: str,
    step_mode: str = "rw_operational",
    decimation: int = 1,
) -> BatteryTimeSeries:
    path = mat_path_for_cell(matlab_dir, cell_id)
    steps = load_cell_steps(path, step_mode=step_mode)
    if not steps:
        raise ValueError(f"No steps after filter for {cell_id} ({step_mode})")

    n_steps = len(steps)
    parts: Dict[str, list] = {
        k: []
        for k in (
            "time_s",
            "relative_time_s",
            "voltage_v",
            "current_a",
            "temperature_c",
            "comment",
            "step_index",
        )
    }

    for i, step in enumerate(steps):
        n = step.voltage_v.size
        if n == 0:
            continue
        parts["time_s"].append(step.time_s)
        parts["relative_time_s"].append(step.relative_time_s)
        parts["voltage_v"].append(step.voltage_v)
        parts["current_a"].append(step.current_a)
        parts["temperature_c"].append(step.temperature_c)
        parts["comment"].append(np.full(n, step.comment, dtype=object))
        parts["step_index"].append(np.full(n, i, dtype=np.int32))

    time_s = np.concatenate(parts["time_s"])
    step_idx = np.concatenate(parts["step_index"])
    age = (step_idx / max(n_steps - 1, 1)).astype(np.float32)

    cid = cell_id.upper()
    if not cid.startswith("RW"):
        cid = f"RW{cid}"
    dec = max(int(decimation), 1)
    sl = slice(None, None, dec)
    return BatteryTimeSeries(
        cell_id=cid,
        time_s=np.concatenate(parts["time_s"])[sl],
        relative_time_s=np.concatenate(parts["relative_time_s"])[sl],
        voltage_v=np.concatenate(parts["voltage_v"]).astype(np.float32)[sl],
        current_a=np.concatenate(parts["current_a"]).astype(np.float32)[sl],
        temperature_c=np.concatenate(parts["temperature_c"]).astype(np.float32)[sl],
        age=age[sl],
        comment=np.concatenate(parts["comment"])[sl],
        step_index=step_idx[sl],
    )
