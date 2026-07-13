"""OCV–SoC calibration for LFP cells from ``lfp_processed.mat``."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from charging_opt.soc_utils import fit_ocv_soc_curve, save_ocv_curve, validate_ocv_curve

from Constrained_BO.config import REPO_ROOT, SOC_START
from Constrained_BO.lfp_data import (
    DEFAULT_LFP_MAT,
    LfpStepRecord,
    extract_discharge_segment,
    load_lfp_steps,
)
from Constrained_BO.ocv import ocv_data_dir, ocv_curve_path, ocv_plot_path

LFP_V_MIN = 2.50
LFP_V_MAX = 3.65
CELL_ID = "LFP"
LFP_OCV_FALLBACK = REPO_ROOT / "outputs" / "lfp_ocv"


def resolve_lfp_ocv_dir() -> Path:
    """Prefer ``Constrained_BO/data/LFP``; fall back to ``outputs/lfp_ocv`` if not writable."""
    import os

    primary = ocv_data_dir(CELL_ID)
    probe = primary / ".write_probe"
    try:
        primary.mkdir(parents=True, exist_ok=True)
        probe.touch()
        probe.unlink(missing_ok=True)
        return primary
    except OSError:
        pass

    fallback = LFP_OCV_FALLBACK
    fallback.mkdir(parents=True, exist_ok=True)
    if os.access(fallback, os.W_OK):
        print(f"  Note: using writable OCV dir {fallback} (primary not writable)")
        return fallback
    return primary


def lfp_ocv_curve_path() -> Path:
    primary = ocv_curve_path(CELL_ID)
    if primary.is_file():
        return primary
    fallback = LFP_OCV_FALLBACK / "ocv_soc_curve.npz"
    return fallback if fallback.is_file() else primary


def _ocv_soc_from_discharge(
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    relative_time_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """SoC from coulomb counting on a discharge leg (SoC=1 at start)."""
    v = np.asarray(voltage_v, dtype=np.float64)
    i = np.asarray(current_a, dtype=np.float64)
    dt = np.diff(relative_time_s, prepend=0.0)
    if dt.size:
        dt[0] = 1.0
    delivered = np.cumsum(np.abs(i) * dt)
    q = float(delivered[-1])
    if q <= 0:
        raise ValueError("Zero discharge throughput")
    soc = 1.0 - delivered / q
    return v, soc


def fit_lfp_ocv_curve(
    *,
    mat_path: str | Path = DEFAULT_LFP_MAT,
    out_dir: Optional[Path] = None,
    plot: bool = True,
    step_index: int = 0,
) -> Tuple[PchipInterpolator, Path]:
    """
    Fit OCV→SoC from the discharge leg of a 1C Reference step.

    Uses the freshest reference (first in file by default).
    """
    steps = load_lfp_steps(mat_path, comment="1C Reference")
    if not steps:
        raise ValueError(f"No 1C Reference steps in {mat_path}")
    rec = steps[min(step_index, len(steps) - 1)]
    v_seg, i_seg, _, rel = extract_discharge_segment(rec)
    ocv_v, soc = _ocv_soc_from_discharge(v_seg, i_seg, rel)
    # PCHIP requires strictly increasing voltage
    order = np.argsort(ocv_v)
    ocv_v = ocv_v[order]
    soc = soc[order]
    # collapse duplicate voltages (keep lowest SoC = most discharged)
    uniq_v, uniq_idx = np.unique(ocv_v, return_index=True)
    if uniq_v.size < ocv_v.size:
        ocv_v = uniq_v
        soc = soc[uniq_idx]
    spline = fit_ocv_soc_curve(ocv_v, soc)
    validate_ocv_curve(spline, v_min=LFP_V_MIN, v_max=LFP_V_MAX)

    out = out_dir or resolve_lfp_ocv_dir()
    out.mkdir(parents=True, exist_ok=True)
    npz_path = out / "ocv_soc_curve.npz"
    save_ocv_curve(spline, npz_path, v_min=LFP_V_MIN, v_max=LFP_V_MAX)

    if plot:
        _plot_lfp_ocv(ocv_v, soc, spline, out / "ocv_soc_curve.png")

    return spline, npz_path


def _plot_lfp_ocv(
    ocv_fit: np.ndarray,
    soc_fit: np.ndarray,
    spline: PchipInterpolator,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = np.linspace(0, ocv_fit.size - 1, min(4000, ocv_fit.size)).astype(int)
    ax.plot(ocv_fit[sub], soc_fit[sub], ".", ms=1.5, alpha=0.25, label="1C ref discharge")
    v_line = np.linspace(float(ocv_fit.min()), float(ocv_fit.max()), 300)
    ax.plot(v_line, spline(v_line), "k-", lw=1.5, label="PCHIP OCV curve")
    ax.set_xlabel("OCV (V)")
    ax.set_ylabel("SoC")
    ax.set_title(f"{CELL_ID} OCV–SoC (1C Reference discharge)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def load_or_fit_lfp_ocv(
    *,
    mat_path: str | Path = DEFAULT_LFP_MAT,
    refit: bool = False,
) -> PchipInterpolator:
    path = lfp_ocv_curve_path()
    if path.exists() and not refit:
        from charging_opt.soc_utils import load_ocv_curve
        return load_ocv_curve(path)
    spline, _ = fit_lfp_ocv_curve(mat_path=mat_path, plot=True)
    return spline


def ocv_from_soc_lfp(
    spline: PchipInterpolator,
    soc: float,
    *,
    v_min: float = LFP_V_MIN,
    v_max: float = LFP_V_MAX,
) -> float:
    soc = float(np.clip(soc, 0.0, 1.0))
    v_grid = np.linspace(v_min, v_max, 800)
    s_grid = np.clip(spline(v_grid), 0.0, 1.0)
    return float(np.interp(soc, s_grid, v_grid))


def build_lfp_start_state(
    *,
    soc: float = SOC_START,
    age: float = 0.0,
    mat_path: str | Path = DEFAULT_LFP_MAT,
    ocv_spline: Optional[PchipInterpolator] = None,
    refit_ocv: bool = False,
    ambient_t_c: float = 24.0,
) -> Dict[str, float]:
    spline = ocv_spline or load_or_fit_lfp_ocv(mat_path=mat_path, refit=refit_ocv)
    v0 = ocv_from_soc_lfp(spline, soc)
    return {
        "soc": float(soc),
        "v0": float(v0),
        "t0": float(ambient_t_c),
        "age": float(age),
        "prev_i": 0.0,
    }


def nominal_voltage_lfp(
    spline: PchipInterpolator,
    soc: float = 0.5,
) -> float:
    return ocv_from_soc_lfp(spline, soc)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Fit LFP OCV–SoC from 1C Reference discharge")
    p.add_argument("--mat", default=str(DEFAULT_LFP_MAT))
    p.add_argument("--refit", action="store_true")
    args = p.parse_args()

    if args.refit or not lfp_ocv_curve_path().exists():
        spline, npz = fit_lfp_ocv_curve(mat_path=args.mat)
        print(f"Saved {npz}")
        print(f"Saved {npz.with_suffix('.png')}")
    else:
        from charging_opt.soc_utils import load_ocv_curve
        path = lfp_ocv_curve_path()
        spline = load_ocv_curve(path)
        print(f"Loaded {path}")

    q_ah = __import__(
        "Constrained_BO.lfp_data", fromlist=["estimate_lfp_capacity_ah"]
    ).estimate_lfp_capacity_ah(args.mat)
    state = build_lfp_start_state(mat_path=args.mat, ocv_spline=spline)
    print(f"  Estimated Q_rated ≈ {q_ah:.3f} Ah")
    print(f"  OCV @ {SOC_START:.0%} SoC -> {state['v0']:.4f} V")
    print(f"  Start state: {state}")
