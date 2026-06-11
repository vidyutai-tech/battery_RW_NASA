#!/usr/bin/env python3
"""
Export the notebook SOC MLP to ``ocv_soc_curve.npz`` for charging BO.

The MLP replaces the PCHIP OCV curve for **rest voltage → SoC** and capacity-fade
Q correction. Charging simulation still uses coulomb counting (unchanged).

Outputs
-------
    outputs/charging_opt/models/stage1_state_estimation/ocv_soc_curve.npz
    outputs/charging_opt/models/stage1_state_estimation/capacity_fade.npz  (if missing)
    outputs/charging_opt/plots/stage1_state_estimation/ocv_soc_curve.png

Usage
-----
    venv/bin/python scripts/export_soc_mlp_to_ocv.py
    venv/bin/python scripts/export_soc_mlp_to_ocv.py --ckpt outputs/soc_model/soc_model.pt --cell RW9
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rw_transfer.config import load_config
from charging_opt import paths as P
from charging_opt.artifacts import CANONICAL, OPTIONAL, update_master_registry, write_stage_registry
from charging_opt.soc_mlp import load_voltage_soc_mlp, mlp_ocv_soc_grid
from charging_opt.soc_utils import (
    capacity_fade_table,
    extract_ocv_soc_pairs,
    find_low_current_steps,
    fit_capacity_curve,
    load_ocv_curve,
    load_steps_with_age,
    save_capacity_curve,
    soc_from_ocv,
    validate_ocv_curve,
)


def _export_ocv_npz(
    ckpt: Path,
    out_npz: Path,
    *,
    v_min: float = 3.0,
    v_max: float = 4.3,
    n_points: int = 600,
) -> dict:
    model = load_voltage_soc_mlp(ckpt)
    v, soc = mlp_ocv_soc_grid(model, v_min=v_min, v_max=v_max, n_points=n_points)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, ocv=v, soc=soc)
    spline = load_ocv_curve(out_npz)
    checks = validate_ocv_curve(spline, v_min=v_min, v_max=v_max)
    return {"v_grid": v, "soc_grid": soc, "checks": checks, "spline": spline}


def _heldout_validation(spline, matlab_dir: str, cell: str) -> dict | None:
    steps, step_age = load_steps_with_age(matlab_dir, cell)
    lc_idx = find_low_current_steps(steps)
    if len(lc_idx) < 2:
        return None
    val_step = steps[lc_idx[-1]]
    ocv_val, soc_val = extract_ocv_soc_pairs(val_step)
    soc_pred = soc_from_ocv(spline, ocv_val)
    err = soc_pred - soc_val
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((soc_val - soc_val.mean()) ** 2))
    return {
        "n": int(soc_val.size),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "step_age": float(step_age[lc_idx[-1]]),
    }


def _ensure_capacity_fade(
    spline,
    matlab_dir: str,
    cell: str,
    out_npz: Path,
    *,
    force: bool = False,
) -> dict:
    if out_npz.is_file() and not force:
        print(f"  Keeping existing capacity fade -> {out_npz}")
        data = np.load(out_npz)
        q_of_age = fit_capacity_curve(data["age"], data["q_full_as"])
        return {
            "rebuilt": False,
            "n_reference_discharges": int(data["age"].size),
            "q_age0_ah": float(q_of_age(0.0) / 3600),
            "q_age1_ah": float(q_of_age(1.0) / 3600),
        }

    steps, step_age = load_steps_with_age(matlab_dir, cell)
    table = capacity_fade_table(steps, step_age, ocv_spline=spline)
    q_of_age = fit_capacity_curve(table["age"], table["q_full_as"])
    save_capacity_curve(table, out_npz)
    q0, q1 = float(q_of_age(0.0)), float(q_of_age(1.0))
    print(f"  Capacity fade: {len(table['age'])} reference discharges")
    print(f"    Q(age=0) = {q0/3600:.3f} Ah   Q(age=1) = {q1/3600:.3f} Ah")
    return {
        "rebuilt": True,
        "n_reference_discharges": int(len(table["age"])),
        "q_age0_ah": q0 / 3600,
        "q_age1_ah": q1 / 3600,
    }


def _plot_ocv(
    v: np.ndarray,
    soc: np.ndarray,
    out_png: Path,
    *,
    cell: str,
    val_metrics: dict | None,
    fit_v: np.ndarray | None = None,
    fit_soc: np.ndarray | None = None,
) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if fit_v is not None and fit_soc is not None:
        sub = slice(None, None, 5)
        ax.plot(fit_v[sub], fit_soc[sub], ".", ms=2, alpha=0.25,
                color="tab:gray", label="train low-current discharge")
    ax.plot(v, soc, "-", lw=2.2, color="tab:blue", label="SOC MLP (monotone)")
    ax.set_xlabel("OCV (V)")
    ax.set_ylabel("SoC")
    title = f"{cell} OCV–SoC from SOC MLP (low-current 0.04 A discharge)"
    if val_metrics is not None:
        title += (
            f"\nheld-out inversion: RMSE={val_metrics['rmse']:.3f}, "
            f"R²={val_metrics['r2']:.3f}"
        )
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  Plot -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Export SOC MLP to charging-opt OCV npz.")
    p.add_argument("--ckpt", default="outputs/soc_model/soc_model.pt")
    p.add_argument("--cell", default="RW9")
    p.add_argument("--config", default=None)
    p.add_argument("--v-min", type=float, default=3.0)
    p.add_argument("--v-max", type=float, default=4.3)
    p.add_argument("--n-points", type=int, default=600)
    p.add_argument(
        "--rebuild-capacity-fade",
        action="store_true",
        help="Recompute capacity_fade.npz using the MLP OCV curve",
    )
    args = p.parse_args()

    ckpt = ROOT / args.ckpt
    if not ckpt.is_file():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    P.ensure_layout(ROOT)
    models_dir = ROOT / P.STAGE1_MODELS
    plots_dir = ROOT / P.STAGE1_PLOTS
    ocv_path = ROOT / CANONICAL["ocv_curve"]
    cap_path = ROOT / CANONICAL["capacity_fade"]

    cfg = load_config(args.config)
    matlab_dir = cfg["data"]["matlab_dir"]

    print(f"\n{'='*60}")
    print("  Export SOC MLP → charging_opt OCV curve")
    print(f"{'='*60}")
    print(f"  Checkpoint : {ckpt}")
    print(f"  Cell       : {args.cell}")
    print(f"  OCV out    : {ocv_path}\n")

    exported = _export_ocv_npz(
        ckpt, ocv_path, v_min=args.v_min, v_max=args.v_max, n_points=args.n_points,
    )
    print(f"  Saved OCV curve -> {ocv_path}")

    val_metrics = _heldout_validation(exported["spline"], matlab_dir, args.cell)
    if val_metrics:
        print("\nHeld-out SoC validation (aged low-current discharge, MLP inversion):")
        print(f"  RMSE={val_metrics['rmse']:.4f}  MAE={val_metrics['mae']:.4f}  "
              f"R²={val_metrics['r2']:.4f}  (n={val_metrics['n']})")

    cap_info = _ensure_capacity_fade(
        exported["spline"],
        matlab_dir,
        args.cell,
        cap_path,
        force=args.rebuild_capacity_fade or not cap_path.is_file(),
    )

    steps, _ = load_steps_with_age(matlab_dir, args.cell)
    lc_idx = find_low_current_steps(steps)
    fit_v = fit_soc = None
    if lc_idx:
        fit_v, fit_soc = extract_ocv_soc_pairs(steps[lc_idx[0]])

    _plot_ocv(
        exported["v_grid"],
        exported["soc_grid"],
        ROOT / OPTIONAL["plot_ocv_soc"],
        cell=args.cell,
        val_metrics=val_metrics,
        fit_v=fit_v,
        fit_soc=fit_soc,
    )

    summary = {
        "cell": args.cell,
        "source": "soc_mlp",
        "checkpoint": str(ckpt.relative_to(ROOT)) if ckpt.is_relative_to(ROOT) else str(ckpt),
        "ocv_checks": exported["checks"],
        "heldout_soc_inversion": val_metrics,
        "capacity_fade": cap_info,
        "note": (
            "OCV from notebook VoltageSOCMLP; charging rollout still uses coulomb counting."
        ),
        "artifacts": {
            "ocv_curve": CANONICAL["ocv_curve"],
            "capacity_fade": CANONICAL["capacity_fade"],
        },
    }
    write_stage_registry(P.STAGE1_MODELS, summary, root=ROOT)
    update_master_registry(root=ROOT)

    print(f"\n{'='*60}")
    print("  Done")
    print(f"  OCV curve      : {ocv_path}")
    print(f"  Capacity fade  : {cap_path}")
    print(f"  Registry       : {CANONICAL['stage1_registry']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
