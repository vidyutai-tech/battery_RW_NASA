"""
OCV-SoC curve fitting, coulomb counting, and capacity-fade utilities.

Design decisions (verified against RW9.mat):

* The OCV-SoC curve is fitted on the **low-current discharge at 0.04A** steps.
  At 0.04 A the IR drop is ~2 mV, so terminal voltage ~ OCV. RW9 contains two
  such steps (~44 h each, dt = 10 s, spanning 3.20-4.196 V): the first (fresh
  cell) is used for fitting, the second (aged cell) for held-out validation.

* SoC *within a session* is computed by coulomb counting, NOT by inverting the
  loaded terminal voltage through the OCV curve. Under charge at up to -4.5 A
  the terminal voltage sits well above OCV (IR drop), which would
  systematically over-credit aggressive currents. During a simulated rollout
  the current is the action and is known exactly, so coulomb counting is
  exact up to capacity error.

* Capacity Q(age) is extracted from the 80 reference discharges (1 A CC down
  to 3.2 V). RW9 fades from ~2.1 Ah to ~0.75 Ah, so an age-independent Q
  would mis-scale Delta-SoC by up to ~3x late in life. Each measured Q is
  corrected for the not-quite-full starting voltage via the OCV curve:
      Q_full = Q_measured / (soc_ocv(V_start) - soc_ocv(V_end)).

The OCV-SoC relationship is treated as age-invariant when SoC is defined
relative to the *current* full capacity (standard assumption; the held-out
validation on the aged low-current step quantifies how well it holds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from rw_transfer.data.mat_loader import BatteryStep, load_cell_steps, mat_path_for_cell

LOW_CURRENT_COMMENT = "low current discharge at 0.04a"
REF_DISCHARGE_COMMENT = "reference discharge"

DEFAULT_V_MIN = 3.2
DEFAULT_V_MAX = 4.2


# ---------------------------------------------------------------------------
# Step loading with the same age convention as rw_transfer.data.series
# ---------------------------------------------------------------------------

def load_steps_with_age(
    matlab_dir: str | Path,
    cell_id: str,
) -> Tuple[List[BatteryStep], np.ndarray]:
    """All steps of one cell plus per-step normalized age (i / (n_steps-1))."""
    path = mat_path_for_cell(matlab_dir, cell_id)
    steps = load_cell_steps(path, step_mode="all")
    n = len(steps)
    age = np.arange(n, dtype=np.float64) / max(n - 1, 1)
    return steps, age


def _dt_seconds(time_s: np.ndarray) -> np.ndarray:
    tt = np.asarray(time_s, dtype=np.float64)
    dt = np.diff(tt, prepend=tt[0] if tt.size else 0.0)
    if dt.size:
        dt[0] = 0.0
    return dt


# ---------------------------------------------------------------------------
# OCV-SoC pairs from low-current discharge
# ---------------------------------------------------------------------------

def find_low_current_steps(steps: List[BatteryStep]) -> List[int]:
    return [
        i for i, s in enumerate(steps)
        if s.comment.strip().lower() == LOW_CURRENT_COMMENT
    ]


def extract_ocv_soc_pairs(step: BatteryStep) -> Tuple[np.ndarray, np.ndarray]:
    """
    (OCV, SoC) pairs from one low-current discharge step.

    SoC by coulomb counting within the step: the step covers the full usable
    window (~4.2 V down to 3.2 V), so SoC(t) = 1 - delivered(t) / Q_step.
    """
    v = np.asarray(step.voltage_v, dtype=np.float64)
    i = np.abs(np.asarray(step.current_a, dtype=np.float64))
    dt = _dt_seconds(step.relative_time_s)
    delivered = np.cumsum(i * dt)
    q_step = float(delivered[-1])
    if q_step <= 0:
        raise ValueError("Low-current discharge step has zero throughput")
    soc = 1.0 - delivered / q_step
    return v, soc


# ---------------------------------------------------------------------------
# Monotone OCV -> SoC spline
# ---------------------------------------------------------------------------

def fit_ocv_soc_curve(
    ocv_values: np.ndarray,
    soc_values: np.ndarray,
    v_bin_width: float = 0.005,
) -> PchipInterpolator:
    """
    Fit a strictly monotone OCV -> SoC spline.

    Voltage is binned (default 5 mV), SoC bin-averaged, monotonicity enforced
    via a cumulative max in ascending-voltage order, then PCHIP interpolated
    (PCHIP preserves monotonicity of the knots).
    """
    v = np.asarray(ocv_values, dtype=np.float64)
    s = np.asarray(soc_values, dtype=np.float64)
    order = np.argsort(v)
    v, s = v[order], s[order]

    edges = np.arange(v[0], v[-1] + v_bin_width, v_bin_width)
    which = np.digitize(v, edges)
    v_knots, s_knots = [], []
    for b in np.unique(which):
        m = which == b
        v_knots.append(float(v[m].mean()))
        s_knots.append(float(s[m].mean()))
    v_knots = np.asarray(v_knots)
    s_knots = np.asarray(s_knots)

    # Enforce monotone non-decreasing SoC, then drop duplicate plateaus so the
    # spline is strictly increasing wherever the data allows.
    s_knots = np.maximum.accumulate(s_knots)
    keep = np.concatenate([[True], np.diff(v_knots) > 1e-9])
    return PchipInterpolator(v_knots[keep], s_knots[keep], extrapolate=True)


def validate_ocv_curve(
    spline: PchipInterpolator,
    v_min: float = DEFAULT_V_MIN,
    v_max: float = DEFAULT_V_MAX,
) -> Dict[str, float]:
    """Boundary / monotonicity checks. Prints warnings, returns the numbers."""
    test_v = np.linspace(v_min, v_max, 200)
    test_soc = spline(test_v)
    mono = bool(np.all(np.diff(test_soc) >= -1e-9))
    out = {
        "soc_at_v_min": float(test_soc[0]),
        "soc_at_v_max": float(test_soc[-1]),
        "monotone": mono,
    }
    print("OCV curve validation:")
    print(f"  V={v_min:.2f}V -> SoC={out['soc_at_v_min']:.3f}  (expected ~0)")
    print(f"  V={v_max:.2f}V -> SoC={out['soc_at_v_max']:.3f}  (expected ~1)")
    print(f"  Monotone increasing: {mono}")
    if not mono:
        print("  WARNING: curve is not monotone; inversion will be unreliable.")
    if abs(out["soc_at_v_min"]) > 0.1:
        print(f"  WARNING: SoC at V_min is {out['soc_at_v_min']:.3f}, expected ~0.")
    if abs(out["soc_at_v_max"] - 1.0) > 0.1:
        print(f"  WARNING: SoC at V_max is {out['soc_at_v_max']:.3f}, expected ~1.")
    return out


def save_ocv_curve(spline: PchipInterpolator, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(3.0, 4.3, 600)
    y = np.clip(spline(x), 0.0, None)
    np.savez(path, ocv=x, soc=y)
    print(f"OCV-SoC curve saved -> {path}")


def load_ocv_curve(path: str | Path) -> PchipInterpolator:
    data = np.load(path)
    soc = np.maximum.accumulate(data["soc"])  # guard against fp wiggle
    keep = np.concatenate([[True], np.diff(data["ocv"]) > 1e-9])
    return PchipInterpolator(data["ocv"][keep], soc[keep], extrapolate=True)


def soc_from_ocv(spline: Callable, voltage: float | np.ndarray) -> np.ndarray:
    """Rest-voltage -> SoC, clipped to [0, 1]. Only valid near zero current."""
    return np.clip(spline(voltage), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Capacity fade Q(age) from reference discharges
# ---------------------------------------------------------------------------

def capacity_fade_table(
    steps: List[BatteryStep],
    step_age: np.ndarray,
    ocv_spline: Optional[PchipInterpolator] = None,
    v_end_max: float = 3.25,
) -> Dict[str, np.ndarray]:
    """
    Measured capacity per reference discharge.

    Returns dict of arrays: age, q_measured_as, q_full_as, v_start, v_end.
    ``q_full_as`` rescales the measured throughput to a full 0-100% window via
    the OCV curve (reference discharges start at ~4.05-4.11 V after a rest,
    i.e. slightly below 100% SoC).
    """
    ages, q_meas, q_full, v0s, v1s = [], [], [], [], []
    for i, s in enumerate(steps):
        if s.comment.strip().lower() != REF_DISCHARGE_COMMENT:
            continue
        v = np.asarray(s.voltage_v, dtype=np.float64)
        if v.size < 10 or v[-1] > v_end_max:
            continue  # aborted / partial discharge
        dt = _dt_seconds(s.relative_time_s)
        q = float(np.sum(np.abs(s.current_a) * dt))
        if ocv_spline is not None:
            span = float(
                np.clip(ocv_spline(v[0]), 0, 1) - np.clip(ocv_spline(v[-1]), 0, 1)
            )
            qf = q / span if span > 0.05 else np.nan
        else:
            qf = q
        ages.append(float(step_age[i]))
        q_meas.append(q)
        q_full.append(qf)
        v0s.append(float(v[0]))
        v1s.append(float(v[-1]))
    return {
        "age": np.asarray(ages),
        "q_measured_as": np.asarray(q_meas),
        "q_full_as": np.asarray(q_full),
        "v_start": np.asarray(v0s),
        "v_end": np.asarray(v1s),
    }


def fit_capacity_curve(
    age: np.ndarray,
    q_as: np.ndarray,
    smooth_window: int = 5,
) -> Callable[[float | np.ndarray], np.ndarray]:
    """
    Smoothed, interpolatable Q(age) in Ampere-seconds.

    Rolling-median smoothing absorbs capacity-regeneration wiggles; linear
    interpolation between smoothed points, constant extrapolation at the ends.
    """
    m = np.isfinite(q_as)
    a, q = np.asarray(age)[m], np.asarray(q_as)[m]
    order = np.argsort(a)
    a, q = a[order], q[order]
    half = max(smooth_window // 2, 1)
    q_s = np.array([
        np.median(q[max(0, j - half): j + half + 1]) for j in range(len(q))
    ])

    def q_of_age(x):
        return np.interp(np.asarray(x, dtype=np.float64), a, q_s)

    return q_of_age


def save_capacity_curve(
    table: Dict[str, np.ndarray], path: str | Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **table)
    print(f"Capacity-fade table saved -> {path}")


def load_capacity_curve(path: str | Path) -> Callable[[float | np.ndarray], np.ndarray]:
    data = np.load(path)
    return fit_capacity_curve(data["age"], data["q_full_as"])


# ---------------------------------------------------------------------------
# Coulomb counting within a charging session
# ---------------------------------------------------------------------------

def coulomb_delta_soc(
    current_a: np.ndarray,
    dt_seconds: float | np.ndarray,
    q_as: float,
) -> float:
    """
    Delta-SoC of a charging segment (NASA RW convention: I < 0 is charge).

    Exact given the applied current (the action) and the cell capacity at the
    session's age. Discharge portions subtract.
    """
    i = np.asarray(current_a, dtype=np.float64)
    dt = np.broadcast_to(np.asarray(dt_seconds, dtype=np.float64), i.shape)
    net_charge = float(np.sum(-i * dt))  # positive when charging
    return net_charge / float(q_as)
