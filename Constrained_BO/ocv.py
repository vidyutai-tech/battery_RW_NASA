"""OCV–SoC curve fit, inverse lookup, and aligned start-state construction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from charging_opt.soc_utils import (
    extract_ocv_soc_pairs,
    find_low_current_steps,
    fit_ocv_soc_curve,
    load_ocv_curve,
    load_steps_with_age,
    save_ocv_curve,
    soc_from_ocv,
    validate_ocv_curve,
)

from Constrained_BO.config import REPO_ROOT, SOC_START

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MATLAB_DIR = (
    REPO_ROOT
    / "NASA_RW/dataset/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post"
    / "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab"
)


def ocv_data_dir(cell_id: str) -> Path:
    return DATA_DIR / cell_id.upper()


def ocv_curve_path(cell_id: str) -> Path:
    return ocv_data_dir(cell_id) / "ocv_soc_curve.npz"


def ocv_plot_path(cell_id: str) -> Path:
    return ocv_data_dir(cell_id) / "ocv_soc_curve.png"


def fit_ocv_curve(
    cell_id: str,
    *,
    matlab_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    plot: bool = True,
) -> Tuple[PchipInterpolator, Path]:
    """Fit OCV→SoC on the first low-current (0.04 A) discharge step."""
    mat_dir = Path(matlab_dir or DEFAULT_MATLAB_DIR)
    out = out_dir or ocv_data_dir(cell_id)
    out.mkdir(parents=True, exist_ok=True)

    steps, _ = load_steps_with_age(mat_dir, cell_id)
    lc_idx = find_low_current_steps(steps)
    if not lc_idx:
        raise ValueError(f"{cell_id}: no low-current discharge step for OCV fit")

    fit_step = steps[lc_idx[0]]
    ocv_v, soc = extract_ocv_soc_pairs(fit_step)
    spline = fit_ocv_soc_curve(ocv_v, soc)
    validate_ocv_curve(spline)

    npz_path = out / "ocv_soc_curve.npz"
    save_ocv_curve(spline, npz_path)

    if plot:
        _plot_ocv_curve(
            cell_id,
            ocv_v,
            soc,
            spline,
            out / "ocv_soc_curve.png",
            val_step=steps[lc_idx[1]] if len(lc_idx) > 1 else None,
        )

    return spline, npz_path


def _plot_ocv_curve(
    cell_id: str,
    ocv_fit: np.ndarray,
    soc_fit: np.ndarray,
    spline: PchipInterpolator,
    out_path: Path,
    *,
    val_step=None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = np.linspace(0, ocv_fit.size - 1, min(4000, ocv_fit.size)).astype(int)
    ax.plot(ocv_fit[sub], soc_fit[sub], ".", ms=1.5, alpha=0.25, label="fit step (0.04 A)")

    if val_step is not None:
        ocv_val, soc_val = extract_ocv_soc_pairs(val_step)
        subv = np.linspace(0, ocv_val.size - 1, min(4000, ocv_val.size)).astype(int)
        ax.plot(ocv_val[subv], soc_val[subv], ".", ms=1.5, alpha=0.25, label="held-out step")

    v_line = np.linspace(float(ocv_fit.min()), float(ocv_fit.max()), 300)
    ax.plot(v_line, spline(v_line), "k-", lw=1.5, label="PCHIP OCV curve")

    ax.set_xlabel("OCV (V)")
    ax.set_ylabel("SoC")
    ax.set_title(f"{cell_id} OCV–SoC curve (low-current discharge)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def load_or_fit_ocv(
    cell_id: str,
    *,
    matlab_dir: Optional[Path] = None,
    refit: bool = False,
) -> PchipInterpolator:
    path = ocv_curve_path(cell_id)
    if path.exists() and not refit:
        return load_ocv_curve(path)
    spline, _ = fit_ocv_curve(cell_id, matlab_dir=matlab_dir, plot=True)
    return spline


def ocv_from_soc(spline: PchipInterpolator, soc: float) -> float:
    """Invert monotone OCV→SoC: rest voltage at a given SoC."""
    soc = float(np.clip(soc, 0.0, 1.0))
    v_grid = np.linspace(3.0, 4.3, 800)
    s_grid = np.clip(spline(v_grid), 0.0, 1.0)
    return float(np.interp(soc, s_grid, v_grid))


NOMINAL_SOC = 0.5


def nominal_voltage_from_ocv(
    cell_id: str,
    *,
    soc: float = NOMINAL_SOC,
    matlab_dir: Optional[Path] = None,
    ocv_spline: Optional[PchipInterpolator] = None,
    refit_ocv: bool = False,
) -> float:
    """Nominal pack voltage from NASA OCV data: rest OCV at ``soc`` (default 50%)."""
    spline = ocv_spline or load_or_fit_ocv(
        cell_id, matlab_dir=matlab_dir, refit=refit_ocv,
    )
    return ocv_from_soc(spline, soc)


def extract_rest_temperature(
    cell_id: str,
    target_soc: float,
    ocv_spline: PchipInterpolator,
    *,
    matlab_dir: Optional[Path] = None,
    default_t0: float = 24.7,
) -> float:
    """Pick rest temperature from data near target SoC, else default."""
    from rw_transfer.data.series import load_battery_series

    try:
        series = load_battery_series(
            matlab_dir or DEFAULT_MATLAB_DIR,
            cell_id,
            step_mode="rw_operational",
        )
    except Exception:
        return default_t0

    mask = (
        np.array(["rest" in str(c).lower() for c in series.comment])
        & (np.abs(series.current_a) <= 0.02)
        & np.isfinite(series.temperature_c)
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return default_t0

    soc_est = soc_from_ocv(ocv_spline, series.voltage_v[idx])
    j = int(idx[int(np.argmin(np.abs(soc_est - target_soc)))])
    return float(series.temperature_c[j])


def build_start_state(
    cell_id: str,
    *,
    soc: float = SOC_START,
    age: float = 0.0,
    matlab_dir: Optional[Path] = None,
    ocv_spline: Optional[PchipInterpolator] = None,
    refit_ocv: bool = False,
) -> Dict[str, float]:
    """
    Rest start state with SoC and OCV-aligned voltage.

    At zero current before charge: terminal voltage ≈ OCV(SoC).
    """
    spline = ocv_spline or load_or_fit_ocv(cell_id, matlab_dir=matlab_dir, refit=refit_ocv)
    v0 = ocv_from_soc(spline, soc)
    t0 = extract_rest_temperature(cell_id, soc, spline, matlab_dir=matlab_dir)
    return {
        "soc": float(soc),
        "v0": float(v0),
        "t0": float(t0),
        "age": float(age),
        "prev_i": 0.0,
    }


def _fit_ocv_cli() -> None:
    import argparse

    import matplotlib
    matplotlib.use("Agg")

    from Constrained_BO.config import SOC_START

    p = argparse.ArgumentParser(description="Fit OCV–SoC curve from low-current discharge")
    p.add_argument("--cell", default="RW9")
    p.add_argument("--cells", nargs="+", default=None)
    p.add_argument("--refit", action="store_true", help="Overwrite existing curve")
    args = p.parse_args()

    cells = args.cells or [args.cell.upper()]
    for cell_id in cells:
        cell_id = cell_id.upper()
        print(f"\n=== {cell_id} ===")
        if args.refit or not ocv_curve_path(cell_id).exists():
            spline, npz = fit_ocv_curve(cell_id)
            print(f"Saved {npz}")
            print(f"Saved {ocv_plot_path(cell_id)}")
        else:
            spline = load_or_fit_ocv(cell_id)
            print(f"Loaded {ocv_curve_path(cell_id)}")

        v20 = ocv_from_soc(spline, SOC_START)
        state = build_start_state(cell_id, soc=SOC_START, age=0.0)
        print(f"  OCV @ {SOC_START:.0%} SoC -> {v20:.4f} V")
        print(f"  Start state: {state}")


if __name__ == "__main__":
    _fit_ocv_cli()
