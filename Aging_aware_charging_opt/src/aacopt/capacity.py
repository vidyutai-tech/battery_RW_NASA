"""OCV-SOC curve fitting and measured reference-discharge capacity.

This module produces the *measurement* the degradation model is calibrated
against. It contains no degradation modelling of any kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator

from aacopt.nasa_data import (
    LOW_CURRENT_DISCHARGE,
    REF_CHARGE,
    REF_DISCHARGE,
    Step,
    low_current_discharge_steps,
)

VOLTAGE_BIN_V = 0.005      # 5 mV bins for the OCV sweep
MIN_SOC_WINDOW = 0.05      # reject references traversing less than this
MIN_REF_SAMPLES = 10


@dataclass
class OcvCurve:
    """Monotone OCV-SOC relation fitted from a near-equilibrium sweep."""

    soc_grid: np.ndarray       # increasing in [0, 1]
    ocv_grid: np.ndarray       # increasing voltage [V]
    source_step_index: int
    n_raw_points: int
    v_min: float
    v_max: float

    def soc_from_voltage(self, v: float | np.ndarray) -> np.ndarray:
        """Invert the curve; clamped outside the fitted voltage range."""
        vv = np.clip(np.asarray(v, dtype=np.float64), self.v_min, self.v_max)
        return np.clip(np.interp(vv, self.ocv_grid, self.soc_grid), 0.0, 1.0)

    def voltage_from_soc(self, soc: float | np.ndarray) -> np.ndarray:
        ss = np.clip(np.asarray(soc, dtype=np.float64), 0.0, 1.0)
        return np.interp(ss, self.soc_grid, self.ocv_grid)

    def nominal_voltage(self) -> float:
        """Mean OCV over the 10-90 % SOC working window."""
        s = np.linspace(0.1, 0.9, 81)
        return float(np.mean(self.voltage_from_soc(s)))

    @classmethod
    def from_npz(cls, path: Path, cell: str) -> "OcvCurve":
        blob = np.load(Path(path))
        soc = np.asarray(blob[f"{cell.upper()}_soc"], dtype=np.float64)
        ocv = np.asarray(blob[f"{cell.upper()}_ocv"], dtype=np.float64)
        return cls(
            soc_grid=soc, ocv_grid=ocv,
            source_step_index=-1, n_raw_points=int(soc.size),
            v_min=float(ocv[0]), v_max=float(ocv[-1]),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_step_index": self.source_step_index,
            "n_raw_points": self.n_raw_points,
            "n_grid_points": int(self.soc_grid.size),
            "v_min": self.v_min,
            "v_max": self.v_max,
            "nominal_voltage_v": self.nominal_voltage(),
        }


def fit_ocv_curve(steps: List[Step]) -> OcvCurve:
    """Fit OCV-SOC from the *first* 0.04 A discharge (freshest cell state).

    The 0.04 A sweep is ~1/55 C, so ohmic and diffusion overpotentials are
    small and terminal voltage approximates the equilibrium potential.
    """
    sweeps = low_current_discharge_steps(steps)
    if not sweeps:
        raise ValueError(f"no {LOW_CURRENT_DISCHARGE!r} step found; cannot fit OCV")
    s = sweeps[0]

    dt = s.dt_s()
    q_removed = np.cumsum(np.abs(s.current_a) * dt) / 3600.0
    q_total = float(q_removed[-1])
    if q_total <= 0:
        raise ValueError("low-current discharge moved no charge")

    # Discharge sweep: SOC falls from 1 to 0 as charge is removed.
    soc = 1.0 - q_removed / q_total
    v = s.voltage_v

    order = np.argsort(v)
    v_sorted, soc_sorted = v[order], soc[order]

    # Average SOC within 5 mV voltage bins, then enforce monotonicity so the
    # relation is invertible in both directions.
    edges = np.arange(v_sorted[0], v_sorted[-1] + VOLTAGE_BIN_V, VOLTAGE_BIN_V)
    idx = np.clip(np.digitize(v_sorted, edges) - 1, 0, len(edges) - 2)
    v_b, s_b = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            v_b.append(float(v_sorted[m].mean()))
            s_b.append(float(soc_sorted[m].mean()))
    v_b = np.asarray(v_b)
    s_b = np.asarray(s_b)
    s_b = np.maximum.accumulate(s_b)

    keep = np.concatenate([[True], np.diff(s_b) > 1e-9])
    v_b, s_b = v_b[keep], s_b[keep]
    if v_b.size < 8:
        raise ValueError(f"OCV fit degenerate: only {v_b.size} usable bins")

    # Resample onto a dense uniform SOC grid via a monotone (PCHIP) spline so
    # both directions of the mapping are smooth and non-oscillatory.
    spline = PchipInterpolator(s_b, v_b, extrapolate=True)
    soc_grid = np.linspace(float(s_b[0]), float(s_b[-1]), 600)
    ocv_grid = np.asarray(spline(soc_grid), dtype=np.float64)
    ocv_grid = np.maximum.accumulate(ocv_grid)

    return OcvCurve(
        soc_grid=soc_grid,
        ocv_grid=ocv_grid,
        source_step_index=s.index,
        n_raw_points=s.n,
        v_min=float(ocv_grid[0]),
        v_max=float(ocv_grid[-1]),
    )


def validate_ocv_curve(steps: List[Step], ocv: OcvCurve) -> Optional[Dict[str, float]]:
    """Check the fitted curve against a second 0.04 A sweep if one exists.

    The second sweep happens on an aged cell, so a residual is expected; this
    reports it rather than using it to adjust the fit.
    """
    sweeps = low_current_discharge_steps(steps)
    if len(sweeps) < 2:
        return None
    s = sweeps[1]
    q = np.cumsum(np.abs(s.current_a) * s.dt_s()) / 3600.0
    q_total = float(q[-1])
    if q_total <= 0:
        return None
    soc_meas = 1.0 - q / q_total
    soc_pred = ocv.soc_from_voltage(s.voltage_v)
    err = soc_pred - soc_meas
    return {
        "n": int(err.size),
        "soc_rmse": float(np.sqrt(np.mean(err ** 2))),
        "soc_mae": float(np.mean(np.abs(err))),
        "soc_bias": float(np.mean(err)),
        "note": "second 0.04 A sweep is on an aged cell; residual is expected",
    }


@dataclass
class ReferenceMeasurement:
    """One accepted reference-discharge capacity measurement."""

    ref_number: int
    step_index: int
    t_start_h: float
    t_end_h: float
    q_measured_ah: float
    q_full_ah: float
    v_start: float
    v_end: float
    soc_start: float
    soc_end: float
    soc_window: float
    mean_temperature_c: float


def reference_capacity_table(
    steps: List[Step],
    ocv: OcvCurve,
    *,
    min_soc_window: float = MIN_SOC_WINDOW,
    min_samples: int = MIN_REF_SAMPLES,
) -> Tuple[List[ReferenceMeasurement], Dict[str, object]]:
    """Measured capacity at every usable reference discharge.

    ``q_measured_ah`` is the raw coulomb count of the step. ``q_full_ah``
    corrects it to a full 0-100 % SOC window using the cell's own OCV curve,
    since a reference discharge does not always start full or end empty:

        q_full = q_measured / (soc(V_start) - soc(V_end))
    """
    out: List[ReferenceMeasurement] = []
    rejected: List[Dict[str, object]] = []
    n_seen = 0

    for k, s in enumerate(steps):
        if s.comment != REF_DISCHARGE:
            continue
        n_seen += 1
        if s.n < min_samples:
            rejected.append({"step_index": s.index, "reason": "too few samples", "n": s.n})
            continue

        q_meas = s.charge_ah()
        v0, v1 = float(s.voltage_v[0]), float(s.voltage_v[-1])
        soc0 = float(ocv.soc_from_voltage(v0))
        soc1 = float(ocv.soc_from_voltage(v1))
        window = soc0 - soc1
        if window < min_soc_window:
            rejected.append({
                "step_index": s.index, "reason": "soc window too narrow",
                "soc_window": window,
            })
            continue

        out.append(ReferenceMeasurement(
            ref_number=len(out) + 1,
            step_index=k,
            t_start_h=float(s.time_s[0] / 3600.0),
            t_end_h=float(s.time_s[-1] / 3600.0),
            q_measured_ah=q_meas,
            q_full_ah=q_meas / window,
            v_start=v0,
            v_end=v1,
            soc_start=soc0,
            soc_end=soc1,
            soc_window=window,
            mean_temperature_c=float(s.temperature_c.mean()),
        ))

    info = {
        "n_reference_steps_seen": n_seen,
        "n_accepted": len(out),
        "n_rejected": len(rejected),
        "rejected": rejected[:50],
        "min_soc_window": min_soc_window,
        "min_samples": min_samples,
    }
    return out, info


def fractional_capacity_loss(refs: List[ReferenceMeasurement]) -> np.ndarray:
    """y = 1 - Q_k / Q_0, the regression target (a real capacity fraction)."""
    if not refs:
        return np.zeros(0)
    q = np.asarray([r.q_full_ah for r in refs], dtype=np.float64)
    return 1.0 - q / q[0]
