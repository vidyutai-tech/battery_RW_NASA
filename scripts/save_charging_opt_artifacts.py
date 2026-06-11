#!/usr/bin/env python3
"""
Verify, snapshot, and regenerate plots for the charging-profile pipeline.

Layout: ``outputs/charging_opt/{models,plots,registry}/`` — see ``charging_opt/paths.py``.

Actions (combine flags as needed):
    --verify              print manifest; exit 1 if any required file missing
    --snapshot            copy all artifacts -> outputs/charging_opt/snapshots/snapshot_<ts>/
    --regenerate-plots    rebuild PNGs from saved .npz / registry JSON (no BDT inference)

Examples
--------
    venv/bin/python scripts/save_charging_opt_artifacts.py --verify
    venv/bin/python scripts/save_charging_opt_artifacts.py --verify --snapshot
    venv/bin/python scripts/migrate_charging_opt_layout.py   # one-time legacy move
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

from charging_opt import paths as P
from charging_opt.artifacts import (
    CANONICAL,
    OPTIONAL,
    snapshot_artifacts,
    update_master_registry,
    verify_artifacts,
    write_manifest,
)


def _stage_metrics(registry_path: Path) -> dict:
    if not registry_path.is_file():
        return {}
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return data.get("metrics", data)


def _regenerate_ocv_plots(root: Path) -> None:
    ocv_path = root / CANONICAL["ocv_curve"]
    cap_path = root / CANONICAL["capacity_fade"]
    plots = root / P.STAGE1_PLOTS
    plots.mkdir(parents=True, exist_ok=True)
    if not ocv_path.is_file():
        print(f"  SKIP OCV plots: {ocv_path} missing")
        return

    data = np.load(ocv_path)
    ocv, soc = data["ocv"], data["soc"]
    title_extra = ""
    reg = _stage_metrics(root / CANONICAL["stage1_registry"])
    v = reg.get("heldout_soc_inversion")
    if v:
        title_extra = f"\nheld-out inversion: RMSE={v['rmse']:.3f}, R2={v['r2']:.3f}"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ocv, soc, lw=2.2, color="tab:blue", label="fitted monotone curve")
    ax.set_xlabel("OCV (V)")
    ax.set_ylabel("SoC")
    ax.set_title(f"OCV-SoC curve (from saved npz){title_extra}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "ocv_soc_curve.png", dpi=140)
    plt.close(fig)

    if cap_path.is_file():
        cap = np.load(cap_path)
        age, q_full = cap["age"], cap["q_full_as"]
        from charging_opt.soc_utils import fit_capacity_curve

        q_fn = fit_capacity_curve(age, q_full)
        ag = np.linspace(0, 1, 200)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(age, cap["q_measured_as"] / 3600, "o", ms=4, alpha=0.5,
                color="tab:gray", label="measured")
        ax.plot(age, q_full / 3600, "o", ms=4, alpha=0.7,
                color="tab:blue", label="OCV-corrected")
        ax.plot(ag, q_fn(ag) / 3600, "-", lw=2.2, color="tab:red", label="Q(age) fit")
        ax.set_xlabel("Normalized age")
        ax.set_ylabel("Capacity (Ah)")
        ax.set_title("Capacity fade (from saved npz)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots / "capacity_fade.png", dpi=140)
        plt.close(fig)
    print(f"  Stage1 plots -> {plots}/")


def _regenerate_drift_plots(root: Path) -> None:
    margins = root / CANONICAL["conformal_margins"]
    if not margins.is_file():
        print(f"  SKIP drift plots: {margins} missing")
        return
    m = np.load(margins)
    horizon_s, v_q50, v_q95 = m["horizon_s"], m["v_q50"], m["v_q95"]
    t_q50, t_q95 = m["t_q50"], m["t_q95"]
    plots = root / P.STAGE1_DRIFT_PLOTS
    plots.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, q50, q95, label, unit in (
        (axes[0], v_q50, v_q95, "Voltage", "V"),
        (axes[1], t_q50, t_q95, "Temperature", "degC"),
    ):
        ax.plot(horizon_s, q50, lw=1.8, label="median |err|")
        ax.plot(horizon_s, q95, lw=1.8, color="tab:red", label="p95 |err| (margin)")
        for w in (150, 300, 600):
            ax.axvline(w, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Prediction horizon (s)")
        ax.set_ylabel(f"|{label} error| ({unit})")
        ax.set_title(f"{label}: open-loop chained drift")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "drift_vs_horizon.png", dpi=140)
    plt.close(fig)

    summary = _stage_metrics(root / CANONICAL["drift_registry"])
    per_action = summary.get("per_action", {})
    if per_action:
        acts = sorted(per_action.keys(), key=lambda x: float(x))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        v_rmse = [per_action[a]["v_rmse"] for a in acts if per_action[a]["v_rmse"]]
        t_rmse = [per_action[a]["t_rmse"] for a in acts if per_action[a]["t_rmse"]]
        acts_ok = [a for a in acts if per_action[a]["v_rmse"] is not None]
        axes[0].bar([f"{float(a):+.2f}" for a in acts_ok], v_rmse, color="tab:blue")
        axes[0].set_ylabel("V RMSE (V)")
        axes[1].bar([f"{float(a):+.2f}" for a in acts_ok], t_rmse, color="tab:orange")
        axes[1].set_ylabel("T RMSE (degC)")
        for ax in axes:
            ax.set_xlabel("Charge setpoint (A)")
            ax.grid(alpha=0.3, axis="y")
        fig.suptitle("Single-chunk error by action (from stage registry)")
        fig.tight_layout()
        fig.savefig(plots / "per_action_rmse.png", dpi=140)
        plt.close(fig)
    print(f"  Stage1 drift plots -> {plots}/")


def main() -> None:
    p = argparse.ArgumentParser(description="Verify / snapshot / replot charging_opt artifacts.")
    p.add_argument("--verify", action="store_true", help="check required files exist")
    p.add_argument("--snapshot", action="store_true", help="copy all artifacts to dated snapshot")
    p.add_argument("--regenerate-plots", action="store_true",
                   help="rebuild PNGs from saved data (no BDT inference)")
    p.add_argument("--manifest", default=f"{P.REGISTRY}/artifacts_manifest.json",
                   help="where to write the verification manifest")
    args = p.parse_args()

    if not any([args.verify, args.snapshot, args.regenerate_plots]):
        args.verify = True

    P.ensure_layout(ROOT)
    ok, manifest = verify_artifacts(root=ROOT)
    manifest_path = ROOT / args.manifest
    write_manifest(manifest_path, manifest)
    update_master_registry(root=ROOT)

    print(f"\nArtifact verification ({manifest_path})")
    print(f"  Layout: models/ | plots/ | registry/")
    print(f"  Required: {'ALL OK' if ok else 'MISSING FILES'}")
    for name, info in manifest["required"].items():
        status = "OK" if info["exists"] else "MISSING"
        size = f"{info['size_bytes']/1024/1024:.2f} MB" if info["exists"] else ""
        print(f"    [{status:7s}] {name:<22} {CANONICAL[name]}  {size}")
    opt_present = sum(1 for v in manifest["optional"].values() if v["exists"])
    print(f"  Optional: {opt_present}/{len(OPTIONAL)} present")

    if args.regenerate_plots:
        print("\nRegenerating plots (no retraining) ...")
        _regenerate_ocv_plots(ROOT)
        _regenerate_drift_plots(ROOT)

    if args.snapshot:
        if not ok:
            print("\nERROR: cannot snapshot — fix missing required artifacts first.")
            sys.exit(1)
        dest = snapshot_artifacts(root=ROOT)
        print(f"\nSnapshot written -> {dest}/")
        print(f"  pointer  -> {P.CHARGING_OPT}/LATEST_SNAPSHOT.txt")

    if args.verify and not ok:
        sys.exit(1)
    if ok:
        print("\nAll required artifacts present under models/ + registry/.")
        print("Master registry:", CANONICAL["master_registry"])


if __name__ == "__main__":
    main()
