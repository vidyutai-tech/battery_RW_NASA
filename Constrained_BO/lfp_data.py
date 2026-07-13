"""Load protocol steps from ``lfp_processed.mat`` for OCV / capacity calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.io import loadmat

from rw_transfer.data.mat_loader import BatteryStep

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LFP_MAT = REPO_ROOT / "lfp_processed.mat"

REF_1C_COMMENT = "1C Reference"


@dataclass(frozen=True)
class LfpStepRecord:
    step_index: int
    comment: str
    time_s: np.ndarray
    relative_time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray

    def to_battery_step(self) -> BatteryStep:
        return BatteryStep(
            comment=self.comment,
            step_type="",
            time_s=self.time_s,
            relative_time_s=self.relative_time_s,
            voltage_v=self.voltage_v,
            current_a=self.current_a,
            temperature_c=self.temperature_c,
        )


def _step_comment(comments_arr: np.ndarray) -> str:
    if comments_arr.size == 0:
        return ""
    raw = comments_arr.flat[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def load_lfp_steps(
    mat_path: str | Path = DEFAULT_LFP_MAT,
    *,
    comment: Optional[str] = None,
) -> List[LfpStepRecord]:
    mat_path = Path(mat_path)
    raw = loadmat(str(mat_path), squeeze_me=True)
    out: List[LfpStepRecord] = []
    n = len(raw["Voltage_all"])
    for i in range(n):
        c = _step_comment(raw["Comments_all"][i])
        if comment is not None and c != comment:
            continue
        n_samp = len(raw["Voltage_all"][i])
        rel = np.arange(n_samp, dtype=np.float64)
        out.append(
            LfpStepRecord(
                step_index=i,
                comment=c,
                time_s=rel.copy(),
                relative_time_s=rel,
                voltage_v=np.asarray(raw["Voltage_all"][i], dtype=np.float64).ravel(),
                current_a=np.asarray(raw["Current_all"][i], dtype=np.float64).ravel(),
                temperature_c=np.asarray(raw["Temperature_all"][i], dtype=np.float64).ravel(),
            )
        )
    return out


def extract_discharge_segment(
    step: LfpStepRecord | BatteryStep,
    *,
    discharge_negative: bool = True,
    threshold_a: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (voltage, current, temperature, relative_time) for the main discharge leg.

    LFP raw data: negative current = discharge (opposite of NASA charge sign).
    """
    i = np.asarray(step.current_a, dtype=np.float64)
    if discharge_negative:
        mask = i < -threshold_a
    else:
        mask = i > threshold_a
    if not np.any(mask):
        raise ValueError("No discharge samples found in step")

    idx = np.flatnonzero(mask)
    # take longest contiguous block
    breaks = np.where(np.diff(idx) > 1)[0]
    spans: List[Tuple[int, int]] = []
    start = 0
    for b in breaks:
        spans.append((idx[start], idx[b] + 1))
        start = b + 1
    spans.append((idx[start], idx[-1] + 1))
    a, b = max(spans, key=lambda ab: ab[1] - ab[0])

    sl = slice(a, b)
    v = np.asarray(step.voltage_v, dtype=np.float64)[sl]
    cur = i[sl]
    t = np.asarray(step.temperature_c, dtype=np.float64)[sl]
    rel = np.arange(v.size, dtype=np.float64)
    return v, cur, t, rel


def estimate_lfp_capacity_ah(
    mat_path: str | Path = DEFAULT_LFP_MAT,
    *,
    use_first_reference: bool = True,
) -> float:
    """Rated capacity (Ah) from the first 1C Reference discharge leg."""
    steps = load_lfp_steps(mat_path, comment=REF_1C_COMMENT)
    if not steps:
        raise ValueError(f"No '{REF_1C_COMMENT}' steps in {mat_path}")
    step = steps[0]
    _, cur, _, _ = extract_discharge_segment(step)
    q_as = float(np.sum(np.abs(cur)))  # 1 Hz samples
    return q_as / 3600.0
