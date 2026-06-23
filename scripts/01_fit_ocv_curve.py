#!/usr/bin/env python3
"""
Step 0 of the charging-profile system: fit and validate the OCV-SoC curve and
the capacity-fade curve Q(age) for one cell.

What it does
------------
1. Fits a monotone OCV->SoC spline on the FIRST low-current (0.04 A) discharge
   step (fresh cell).
2. Validates on the SECOND low-current discharge step (aged cell, held out):
   inverts measured voltage through the fitted curve and compares against the
   coulomb-counted SoC reference -> RMSE / R^2. This is the proper replacement
   for the previously broken SoC metric.
3. Extracts Q(age) from all full reference discharges (1 A CC to 3.2 V),
   corrected for the not-quite-full start voltage via the OCV curve.
4. Saves curve + table + diagnostic plots.

Outputs
-------
    models/stage1_state_estimation/
        ocv_soc_curve.npz, capacity_fade.npz, registry.json
    plots/stage1_state_estimation/
        ocv_soc_curve.png, capacity_fade.png

Usage
-----
    venv/bin/python scripts/01_fit_ocv_curve.py [--cell RW9]
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
from charging_opt.artifacts import CANONICAL, update_master_registry, write_stage_registry
from charging_opt.soc_utils import (
    capacity_fade_table,
    extract_ocv_soc_pairs,
    find_low_current_steps,
    fit_capacity_curve,
    fit_ocv_soc_curve,
    load_steps_with_age,
    save_capacity_curve,
    save_ocv_curve,
    soc_from_ocv,
    validate_ocv_curve,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Fit OCV-SoC + Q(age) curves.")
    p.add_argument("--cell", default="RW9")
    p.add_argument("--config", default=None)
    p.add_argument(
        "--artifact_root",
        default=None,
        help="Per-cell output root (e.g. outputs/charging_opt/cells/RW10). "
             "Writes stage1_state_estimation/ and plots/stage1_state_estimation/. "
             "Default: canonical outputs/charging_opt/models|plots/stage1_state_estimation.",
    )
    args = p.parse_args()

    P.ensure_layout(ROOT)
    if args.artifact_root:
        art = Path(args.artifact_root)
        if not art.is_absolute():
            art = ROOT / art
        models_dir = art / "stage1_state_estimation"
        plots_dir = art / "plots" / "stage1_state_estimation"
    else:
        models_dir = ROOT / P.STAGE1_MODELS
        plots_dir = ROOT / P.STAGE1_PLOTS
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    matlab_dir = cfg["data"]["matlab_dir"]

    print(f"Loading {args.cell} steps ...")
    steps, step_age = load_steps_with_age(matlab_dir, args.cell)
    lc_idx = find_low_current_steps(steps)
    print(f"  {len(steps):,} steps | low-current discharge steps at {lc_idx} "
          f"(ages {[f'{step_age[i]:.3f}' for i in lc_idx]})")
    if not lc_idx:
        raise SystemExit("No low-current discharge steps found; cannot fit OCV curve.")

    # ── 1. Fit on first (fresh) low-current discharge ─────────────────────────
    fit_step = steps[lc_idx[0]]
    ocv_fit, soc_fit = extract_ocv_soc_pairs(fit_step)
    spline = fit_ocv_soc_curve(ocv_fit, soc_fit)
    checks = validate_ocv_curve(spline, v_min=float(ocv_fit.min()), v_max=float(ocv_fit.max()))
    save_ocv_curve(spline, models_dir / "ocv_soc_curve.npz")

    # ── 2. Held-out validation on second (aged) low-current discharge ─────────
    val_metrics = None
    if len(lc_idx) > 1:
        val_step = steps[lc_idx[-1]]
        ocv_val, soc_val = extract_ocv_soc_pairs(val_step)
        soc_pred = soc_from_ocv(spline, ocv_val)
        err = soc_pred - soc_val
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((soc_val - soc_val.mean()) ** 2))
        val_metrics = {
            "n": int(soc_val.size),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - ss_res / ss_tot,
            "step_age": float(step_age[lc_idx[-1]]),
        }
        print("\nHeld-out SoC validation (aged low-current discharge, OCV inversion):")
        print(f"  RMSE={val_metrics['rmse']:.4f}  MAE={val_metrics['mae']:.4f}  "
              f"R2={val_metrics['r2']:.4f}  (n={val_metrics['n']})")

    # ── 3. Capacity fade from reference discharges ─────────────────────────────
    table = capacity_fade_table(steps, step_age, ocv_spline=spline)
    q_of_age = fit_capacity_curve(table["age"], table["q_full_as"])
    save_capacity_curve(table, models_dir / "capacity_fade.npz")
    q0, q1 = float(q_of_age(0.0)), float(q_of_age(1.0))
    print(f"\nCapacity fade: {len(table['age'])} full reference discharges")
    print(f"  Q(age=0) = {q0/3600:.3f} Ah   Q(age=1) = {q1/3600:.3f} Ah   "
          f"({100*(1-q1/q0):.1f}% fade)")

    # ── 4. Plots ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sub = slice(None, None, 5)
    ax.plot(ocv_fit[sub], soc_fit[sub], ".", ms=2, alpha=0.25,
            color="tab:gray", label=f"fit data (age={step_age[lc_idx[0]]:.2f})")
    if val_metrics is not None:
        ax.plot(ocv_val[sub], soc_val[sub], ".", ms=2, alpha=0.25,
                color="tab:orange",
                label=f"held-out (age={val_metrics['step_age']:.2f})")
    vg = np.linspace(3.1, 4.25, 400)
    ax.plot(vg, np.clip(spline(vg), 0, 1.05), "-", lw=2.2, color="tab:blue",
            label="fitted monotone PCHIP")
    ax.set_xlabel("OCV (V)")
    ax.set_ylabel("SoC")
    title = f"{args.cell} OCV-SoC curve (low-current 0.04A discharge)"
    if val_metrics is not None:
        title += f"\nheld-out inversion: RMSE={val_metrics['rmse']:.3f}, R2={val_metrics['r2']:.3f}"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "ocv_soc_curve.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table["age"], table["q_measured_as"] / 3600, "o", ms=4, alpha=0.5,
            color="tab:gray", label="measured (partial window)")
    ax.plot(table["age"], table["q_full_as"] / 3600, "o", ms=4, alpha=0.7,
            color="tab:blue", label="OCV-corrected full capacity")
    ag = np.linspace(0, 1, 200)
    ax.plot(ag, q_of_age(ag) / 3600, "-", lw=2.2, color="tab:red", label="Q(age) fit")
    ax.set_xlabel("Normalized age")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title(f"{args.cell} capacity fade from reference discharges")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "capacity_fade.png", dpi=140)
    plt.close(fig)
    print(f"Plots saved -> {plots_dir}/")

    summary = {
        "cell": args.cell,
        "low_current_step_indices": [int(i) for i in lc_idx],
        "ocv_checks": checks,
        "heldout_soc_inversion": val_metrics,
        "n_reference_discharges": int(len(table["age"])),
        "q_age0_ah": q0 / 3600,
        "q_age1_ah": q1 / 3600,
        "artifacts": {
            "ocv_curve": str(models_dir / "ocv_soc_curve.npz"),
            "capacity_fade": str(models_dir / "capacity_fade.npz"),
        },
    }
    if args.artifact_root:
        from datetime import datetime, timezone
        reg_path = models_dir / "registry.json"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        with reg_path.open("w") as f:
            json.dump({
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics": summary,
            }, f, indent=2, default=float)
        print(f"Cell registry -> {reg_path}")
    else:
        write_stage_registry(P.STAGE1_MODELS, summary, root=ROOT)
        update_master_registry(root=ROOT)
        print(f"Registry -> {CANONICAL['stage1_registry']}")
        print(f"Master   -> {CANONICAL['master_registry']}")


if __name__ == "__main__":
    main()
