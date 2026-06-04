"""
Author-style chunk dataset and random train/val/test split (seed 42).

Mirrors ``MyDataset`` in the original ``data_loading.py``:
  * ``chunk_size`` contiguous samples, stride = ``chunk_size`` (non-overlapping)
  * ``__getitem__`` returns ``(starting_state, actions, next_states)``
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from rw_transfer.data.author_loader import AuthorStitchedSeries
from rw_transfer.data.windows import WindowBatch


class AuthorChunkDataset(Dataset):
    """Non-overlapping chunks of length ``chunk_size`` (author default 150)."""

    def __init__(
        self,
        series: AuthorStitchedSeries,
        chunk_size: int = 150,
    ):
        self.chunk_size = int(chunk_size)
        self.voltage = torch.tensor(series.voltage_v, dtype=torch.float32).unsqueeze(1)
        self.current = torch.tensor(series.current_a, dtype=torch.float32).unsqueeze(1)
        self.temperature = torch.tensor(series.temperature_c, dtype=torch.float32).unsqueeze(1)
        self.age = torch.tensor(series.age, dtype=torch.float32).unsqueeze(1)
        self.length = int(series.n_samples)
        self.number_chunks = self.length // self.chunk_size

    def __len__(self) -> int:
        # Author: return self.number_chunks - 1
        return max(0, self.number_chunks - 1)

    def __getitem__(self, index: int):
        cs = self.chunk_size
        start = cs * index
        end = start + cs + 1

        starting_state = torch.cat(
            [self.age[start], self.voltage[start], self.temperature[start]],
            dim=0,
        )
        actions = self.current[start + 1 : end]
        next_voltages = self.voltage[start + 1 : end]
        next_temperatures = self.temperature[start + 1 : end]
        next_states = torch.cat([next_voltages, next_temperatures], dim=1)
        return starting_state, actions, next_states


def subset_author_train_by_fraction(train_set: Subset, frac: float) -> Subset:
    """Deterministic prefix of the train split — used for % adaptation sweeps."""
    frac = float(np.clip(frac, 0.0, 1.0))
    indices = sorted(int(i) for i in train_set.indices)
    if not indices or frac <= 0.0:
        return Subset(train_set.dataset, [])
    n = max(1, int(round(frac * len(indices))))
    n = min(n, len(indices))
    return Subset(train_set.dataset, indices[:n])


def random_split_author_dataset(
    dataset: AuthorChunkDataset,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """Random index split — matches ``torch.randperm`` in author code."""
    total = len(dataset)
    train_size = int(train_frac * total)
    val_size = int(val_frac * total)
    test_size = total - train_size - val_size

    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(total, generator=gen)
    train_idx = perm[:train_size].tolist()
    val_idx = perm[train_size : train_size + val_size].tolist()
    test_idx = perm[train_size + val_size :].tolist()

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def author_subset_to_window_batch(
    subset: Subset,
    max_windows: Optional[int] = 512,
) -> WindowBatch:
    """
    Materialise a ``WindowBatch`` from author chunks for plotting / RMSE metrics.
    """
    base: AuthorChunkDataset = subset.dataset
    indices = list(subset.indices)
    if max_windows is not None:
        indices = indices[:max_windows]
    if not indices:
        cs = base.chunk_size
        return WindowBatch(
            X=np.empty((0, 3 + cs), dtype=np.float32),
            Y_voltage=np.empty((0, cs), dtype=np.float32),
            Y_temperature=np.empty((0, cs), dtype=np.float32),
            window_start_idx=np.empty((0,), dtype=np.int64),
        )

    xs, yv, yt, starts = [], [], [], []
    for idx in indices:
        state, actions, next_states = base[idx]
        age, v0, t0 = state[0].item(), state[1].item(), state[2].item()
        curr = actions.squeeze(-1).numpy()
        x_row = np.concatenate([[age, v0, t0], curr.astype(np.float32)])
        xs.append(x_row)
        yv.append(next_states[:, 0].numpy())
        yt.append(next_states[:, 1].numpy())
        starts.append(idx * base.chunk_size)

    cs = base.chunk_size
    return WindowBatch(
        X=np.stack(xs, axis=0),
        Y_voltage=np.stack(yv, axis=0),
        Y_temperature=np.stack(yt, axis=0),
        window_start_idx=np.array(starts, dtype=np.int64),
    )
