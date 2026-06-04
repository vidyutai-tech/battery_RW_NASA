"""Sliding-window dataset for digital-twin training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from rw_transfer.data.series import BatteryTimeSeries


@dataclass
class WindowBatch:
    X: np.ndarray
    Y_voltage: np.ndarray
    Y_temperature: np.ndarray
    # each row of X: [age, v0, t0, I_1..I_T]
    window_start_idx: np.ndarray
    Y_temperature_raw: Optional[np.ndarray] = None


def build_twin_windows(
    series: BatteryTimeSeries,
    seq_len: int = 150,
    stride: int = 50,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    temperature_c: Optional[np.ndarray] = None,
) -> WindowBatch:
    """
  Build sliding windows on a (possibly sliced) time series.

  ``start_idx`` / ``end_idx`` index into ``series`` arrays (chronological).
    """
    n = series.voltage_v.size
    temp_series = (
        temperature_c if temperature_c is not None else series.temperature_c
    )
    temp_raw = series.temperature_c
    if end_idx is None:
        end_idx = n
    end_idx = min(end_idx, n)
    if end_idx - start_idx < seq_len + 1:
        return WindowBatch(
            X=np.empty((0, 3 + seq_len), dtype=np.float32),
            Y_voltage=np.empty((0, seq_len), dtype=np.float32),
            Y_temperature=np.empty((0, seq_len), dtype=np.float32),
            window_start_idx=np.empty((0,), dtype=np.int64),
        )

    X_list, Yv, Yt, Yt_raw, starts = [], [], [], [], []
    store_raw = temperature_c is not None
    for start in range(start_idx, end_idx - seq_len, stride):
        end = start + seq_len
        v0 = float(series.voltage_v[start])
        t0 = float(temp_series[start])
        age = float(series.age[start])
        curr = series.current_a[start:end].astype(np.float32)
        volt = series.voltage_v[start:end].astype(np.float32)
        temp = temp_series[start:end].astype(np.float32)
        x_row = np.concatenate([[age, v0, t0], curr])
        if np.isnan(x_row).any() or np.isnan(volt).any() or np.isnan(temp).any():
            continue
        X_list.append(x_row)
        Yv.append(volt)
        Yt.append(temp)
        if store_raw:
            Yt_raw.append(temp_raw[start:end].astype(np.float32))
        starts.append(start)

    if not X_list:
        return WindowBatch(
            X=np.empty((0, 3 + seq_len), dtype=np.float32),
            Y_voltage=np.empty((0, seq_len), dtype=np.float32),
            Y_temperature=np.empty((0, seq_len), dtype=np.float32),
            window_start_idx=np.empty((0,), dtype=np.int64),
        )
    return WindowBatch(
        X=np.stack(X_list, axis=0),
        Y_voltage=np.stack(Yv, axis=0),
        Y_temperature=np.stack(Yt, axis=0),
        window_start_idx=np.array(starts, dtype=np.int64),
        Y_temperature_raw=np.stack(Yt_raw, axis=0) if store_raw else None,
    )


def chronological_split_indices(
    n_samples: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Split window indices in time order (windows already chronological)."""
    n_train = int(n_samples * train_frac)
    n_val = int(n_samples * val_frac)
    idx = np.arange(n_samples)
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def split_windows_by_series_fraction(
    batch: WindowBatch,
    train_frac: float,
    val_frac: float,
) -> Dict[str, WindowBatch]:
    splits = chronological_split_indices(len(batch.X), train_frac, val_frac)
    out = {}
    for name, inds in splits.items():
        if inds.size == 0:
            out[name] = WindowBatch(
                X=np.empty((0, batch.X.shape[1]), dtype=np.float32),
                Y_voltage=np.empty((0, batch.Y_voltage.shape[1]), dtype=np.float32),
                Y_temperature=np.empty((0, batch.Y_temperature.shape[1]), dtype=np.float32),
                window_start_idx=np.empty((0,), dtype=np.int64),
                Y_temperature_raw=None,
            )
        else:
            raw = batch.Y_temperature_raw
            out[name] = WindowBatch(
                X=batch.X[inds],
                Y_voltage=batch.Y_voltage[inds],
                Y_temperature=batch.Y_temperature[inds],
                window_start_idx=batch.window_start_idx[inds],
                Y_temperature_raw=raw[inds] if raw is not None else None,
            )
    return out


def index_range_for_series_prefix(
    series: BatteryTimeSeries,
    prefix: BatteryTimeSeries,
    seq_len: int,
    stride: int,
) -> Tuple[int, int]:
    """Sample index range in ``build_twin_windows(series)`` lying in chronological prefix."""
    if prefix.time_s.size == 0:
        return 0, 0
    t_max = float(prefix.time_s[-1])
    n = series.voltage_v.size
    # last sample index included in prefix
    sample_end = int(np.searchsorted(series.time_s, t_max, side="right"))
    start_w = 0
    end_w = max(0, (sample_end - seq_len) // stride)
    return start_w, end_w


class TwinWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        Y_voltage: np.ndarray,
        Y_temperature: np.ndarray,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y_voltage = torch.tensor(Y_voltage, dtype=torch.float32)
        self.Y_temperature = torch.tensor(Y_temperature, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        age, v0, t0 = x[0], x[1], x[2]
        curr = x[3:]
        return age, v0, t0, curr, self.Y_voltage[idx], self.Y_temperature[idx]
