"""
Load ``lfp_processed.mat`` into the same ``AuthorStitchedSeries`` format used by NASA RW.

The MAT file stores per-step cell arrays:
  Time_all, Voltage_all, Current_all, Temperature_all, Comments_all

Each step is stitched in order with ``age = step_index / n_steps`` (author recipe).
Timestamps in the file are datetime strings (~1 Hz); we use cumulative sample indices
as seconds so parsing locale-specific strings is not required for training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
from scipy.io import loadmat

from lfp_transfer.constants import DEFAULT_CELL_ID, LFP_STEP_MODE_COMMENTS
from rw_transfer.data.author_loader import AuthorStitchedSeries


def _normalize_comment(raw: object) -> str:
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return ""
        raw = raw.flat[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def _step_comment(comments_arr: np.ndarray) -> str:
    if comments_arr.size == 0:
        return ""
    return _normalize_comment(comments_arr.flat[0])


def _as_1d_float(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).ravel()


def _step_times_seconds(n_samples: int, step_offset_s: float) -> tuple[np.ndarray, np.ndarray]:
    """1 Hz relative / cumulative absolute time (seconds)."""
    rel = np.arange(n_samples, dtype=np.float64)
    non_rel = step_offset_s + rel
    return non_rel, rel


def load_lfp_stitched_series(
    mat_path: str | Path,
    cell_id: str = DEFAULT_CELL_ID,
    *,
    comment_mode: str = "rw_only",
    allowed_comments: Optional[Set[str]] = None,
    decimation: int = 1,
) -> AuthorStitchedSeries:
    """
    Parse ``lfp_processed.mat`` and return an ``AuthorStitchedSeries``.

    Parameters
    ----------
    comment_mode
        ``rw_only`` | ``rw_plus_reference`` | ``all``
    allowed_comments
        Override the comment whitelist.
    decimation
        Keep every N-th sample within each step (use e.g. 10 for ~680k samples).
    """
    mat_path = Path(mat_path)
    if not mat_path.is_file():
        raise FileNotFoundError(mat_path)

    if allowed_comments is None:
        allowed = LFP_STEP_MODE_COMMENTS.get(comment_mode)
        if allowed is None and comment_mode != "all":
            raise ValueError(f"Unknown comment_mode: {comment_mode!r}")
    else:
        allowed = allowed_comments

    raw = loadmat(str(mat_path), squeeze_me=True)
    time_all = raw["Time_all"]
    voltage_all = raw["Voltage_all"]
    current_all = raw["Current_all"]
    temperature_all = raw["Temperature_all"]
    comments_all = raw["Comments_all"]

    n_file_steps = len(voltage_all)
    parts: Dict[str, List[np.ndarray]] = {
        "non_relative_time_s": [],
        "relative_time_s": [],
        "voltage_v": [],
        "current_a": [],
        "temperature_c": [],
        "age": [],
    }

    kept_steps: List[int] = []
    step_offset_s = 0.0
    dec = max(int(decimation), 1)

    for i in range(n_file_steps):
        comment = _step_comment(comments_all[i])
        if allowed is not None and comment not in allowed:
            continue

        voltage = _as_1d_float(voltage_all[i])
        n = voltage.size
        if n == 0:
            continue

        sl = slice(None, None, dec)
        voltage = voltage[sl]
        current = _as_1d_float(current_all[i])[sl]
        temperature = _as_1d_float(temperature_all[i])[sl]
        n_kept = voltage.size

        non_rel, rel = _step_times_seconds(n_kept, step_offset_s)
        step_offset_s += float(n_kept)

        age_val = float(i) / float(n_file_steps)
        kept_steps.append(i)

        parts["non_relative_time_s"].append(non_rel)
        parts["relative_time_s"].append(rel)
        parts["voltage_v"].append(voltage)
        parts["current_a"].append(current)
        parts["temperature_c"].append(temperature)
        parts["age"].append(np.full(n_kept, age_val, dtype=np.float64))

    if not kept_steps:
        raise ValueError(f"No steps kept from {mat_path} (comment_mode={comment_mode!r})")

    n_steps = len(kept_steps)
    cid = cell_id.upper()

    return AuthorStitchedSeries(
        cell_id=cid,
        non_relative_time_s=np.concatenate(parts["non_relative_time_s"]).astype(np.float64),
        relative_time_s=np.concatenate(parts["relative_time_s"]).astype(np.float64),
        voltage_v=np.concatenate(parts["voltage_v"]).astype(np.float32),
        current_a=np.concatenate(parts["current_a"]).astype(np.float32),
        temperature_c=np.concatenate(parts["temperature_c"]).astype(np.float32),
        age=np.concatenate(parts["age"]).astype(np.float32),
        n_steps=n_steps,
    )
