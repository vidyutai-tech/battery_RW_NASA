"""NASA RW ``.mat`` parsing.

Clean reimplementation (see docs/old_vs_new.md). Sign convention as recorded in
the dataset: negative current = charge, positive current = discharge.

The whole chronological step list is kept, including rests and reference
characterization steps, because calendar aging accrues during rest and the
reference discharges are the measurement events the degradation model is
calibrated against.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.io import loadmat

REF_DISCHARGE = "reference discharge"
REF_CHARGE = "reference charge"
LOW_CURRENT_DISCHARGE = "low current discharge at 0.04a"


@dataclass(frozen=True)
class Step:
    """One protocol step of the NASA RW record."""

    index: int
    comment: str
    step_type: str
    time_s: np.ndarray          # absolute time since test start [s]
    relative_time_s: np.ndarray  # time since step start [s]
    voltage_v: np.ndarray
    current_a: np.ndarray        # negative = charge
    temperature_c: np.ndarray

    @property
    def n(self) -> int:
        return int(self.voltage_v.size)

    @property
    def duration_s(self) -> float:
        if self.time_s.size >= 2:
            return float(self.time_s[-1] - self.time_s[0])
        if self.relative_time_s.size >= 2:
            return float(self.relative_time_s[-1] - self.relative_time_s[0])
        return 0.0

    def dt_s(self) -> np.ndarray:
        """Per-sample interval, forward-differenced with the last value held."""
        t = self.time_s if self.time_s.size == self.n else self.relative_time_s
        if t.size < 2:
            return np.full(self.n, 1.0)
        d = np.diff(t)
        return np.concatenate([d, d[-1:]])

    def charge_ah(self) -> float:
        """Charge moved during the step [Ah], unsigned."""
        return float(np.sum(np.abs(self.current_a) * self.dt_s()) / 3600.0)


def _text(raw: object) -> str:
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return ""
        raw = raw.flat[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip()


def _vec(raw: object, n_hint: int = 0) -> np.ndarray:
    a = np.asarray(raw, dtype=np.float64).ravel()
    if a.size == 0 and n_hint:
        return np.zeros(n_hint)
    return a


def load_steps(mat_path: str | Path) -> List[Step]:
    """Parse a whole ``RW*.mat`` into chronological :class:`Step` records."""
    raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["data"]
    steps: List[Step] = []
    for i, s in enumerate(raw.step):
        v = _vec(s.voltage)
        n = int(v.size)
        if n == 0:
            continue
        steps.append(
            Step(
                index=i,
                comment=_text(s.comment).lower(),
                step_type=_text(s.type),
                time_s=_vec(s.time, n),
                relative_time_s=_vec(s.relativeTime, n),
                voltage_v=v,
                current_a=_vec(s.current, n),
                temperature_c=_vec(s.temperature, n),
            )
        )
    return steps


def step_comment_counts(steps: List[Step]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for s in steps:
        out[s.comment] = out.get(s.comment, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def reference_discharge_indices(steps: List[Step], *, min_samples: int = 10) -> List[int]:
    return [
        k for k, s in enumerate(steps)
        if s.comment == REF_DISCHARGE and s.n >= min_samples
    ]


def low_current_discharge_steps(steps: List[Step]) -> List[Step]:
    """Near-equilibrium 0.04 A sweeps used to fit the OCV-SOC curve."""
    return [s for s in steps if s.comment == LOW_CURRENT_DISCHARGE]


def test_span_hours(steps: List[Step]) -> float:
    if not steps:
        return 0.0
    t0 = steps[0].time_s
    t1 = steps[-1].time_s
    if t0.size == 0 or t1.size == 0:
        return 0.0
    return float((t1[-1] - t0[0]) / 3600.0)


def summarize_cell(steps: List[Step], *, q_nominal_ah: float = 2.2) -> Dict[str, object]:
    """Descriptive stats used by the dataset report (no modelling)."""
    dis_mag: List[float] = []
    chg_mag: List[float] = []
    for s in steps:
        if s.comment == "discharge (random walk)":
            if s.n:
                dis_mag.append(float(np.abs(s.current_a).mean()))
        elif s.comment == "charge (random walk)":
            if s.n:
                chg_mag.append(float(np.abs(s.current_a).mean()))
    d = np.asarray(dis_mag) if dis_mag else np.zeros(1)
    c = np.asarray(chg_mag) if chg_mag else np.zeros(1)
    return {
        "n_steps": len(steps),
        "test_span_h": test_span_hours(steps),
        "n_reference_discharge": len(reference_discharge_indices(steps)),
        "n_low_current_discharge": len(low_current_discharge_steps(steps)),
        "rw_discharge_abs_i_a": {
            "min": float(d.min()), "median": float(np.median(d)), "max": float(d.max()),
        },
        "rw_charge_abs_i_a": {
            "min": float(c.min()), "median": float(np.median(c)), "max": float(c.max()),
        },
        "observed_c_rate_range": [
            float(min(d.min(), c.min()) / q_nominal_ah),
            float(max(d.max(), c.max()) / q_nominal_ah),
        ],
        "step_comment_counts": step_comment_counts(steps),
    }
