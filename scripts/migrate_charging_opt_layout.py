#!/usr/bin/env python3
"""
One-time migration: move legacy flat ``outputs/charging_opt/`` files into
``models/``, ``plots/``, and ``registry/`` stage subfolders.

Usage
-----
    venv/bin/python scripts/migrate_charging_opt_layout.py
    venv/bin/python scripts/migrate_charging_opt_layout.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from charging_opt import paths as P
from charging_opt.artifacts import CANONICAL, OPTIONAL, update_master_registry

# legacy relative path -> new relative path
MOVES = [
    ("outputs/charging_opt/ocv_soc_curve.npz", CANONICAL["ocv_curve"]),
    ("outputs/charging_opt/capacity_fade.npz", CANONICAL["capacity_fade"]),
    ("outputs/charging_opt/ocv_fit_summary.json", CANONICAL["stage1_registry"]),
    ("outputs/charging_opt/drift/conformal_margins.npz", CANONICAL["conformal_margins"]),
    ("outputs/charging_opt/drift/drift_summary.json", CANONICAL["drift_registry"]),
    ("outputs/charging_opt/plots/ocv_soc_curve.png", OPTIONAL["plot_ocv_soc"]),
    ("outputs/charging_opt/plots/capacity_fade.png", OPTIONAL["plot_capacity_fade"]),
    ("outputs/charging_opt/drift/plots/drift_vs_horizon.png", OPTIONAL["plot_drift"]),
    ("outputs/charging_opt/drift/plots/per_action_rmse.png", OPTIONAL["plot_per_action"]),
    ("outputs/charging_opt/lifetime_bo/optimization_result.json", OPTIONAL["lifetime_bo_result"]),
    ("outputs/charging_opt/lifetime_bo/best_session.json",
     f"{P.STAGE3_MODELS}/best_session.json"),
    ("outputs/charging_opt/lifetime_bo/best_profile.png", OPTIONAL["plot_best_profile"]),
    ("outputs/charging_opt/lifetime_bo/bo_convergence.png", OPTIONAL["plot_bo_convergence"]),
    ("outputs/charging_opt/lifetime_bo/sweep/cc_sweep.json", OPTIONAL["cc_sweep_json"]),
    ("outputs/charging_opt/lifetime_bo/sweep/cc_sweep_sei_and_time.png", OPTIONAL["plot_cc_sweep"]),
    ("outputs/charging_opt/lifetime_bo/sweep/cc_sweep_tradeoff.png", OPTIONAL["plot_cc_tradeoff"]),
    ("outputs/charging_opt/artifacts_manifest.json",
     f"{P.REGISTRY}/artifacts_manifest.json"),
]


def _wrap_legacy_registry(src: Path, dst: Path) -> None:
    """If migrating ocv_fit_summary.json, wrap as stage registry."""
    import json
    if dst.name != "registry.json" or not src.is_file():
        return
    if "ocv_fit" in src.name or "drift_summary" in src.name:
        data = json.loads(src.read_text())
        payload = {
            "migrated_from": str(src.name),
            "metrics": data,
        }
        dst.parent.mkdir(parents=True, exist_ok=True)
        with dst.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Migrate charging_opt output layout.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    P.ensure_layout(ROOT)
    moved = 0
    for old_rel, new_rel in MOVES:
        src = ROOT / old_rel
        dst = ROOT / new_rel
        if not src.is_file():
            continue
        if dst.is_file():
            print(f"  skip (exists): {new_rel}")
            continue
        print(f"  {'would move' if args.dry_run else 'move'}: {old_rel} -> {new_rel}")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if old_rel.endswith("ocv_fit_summary.json") or old_rel.endswith("drift_summary.json"):
                _wrap_legacy_registry(src, dst)
            else:
                shutil.copy2(src, dst)
        moved += 1

    if not args.dry_run and moved:
        update_master_registry(root=ROOT)
        print(f"\nMoved/copied {moved} files. Master registry updated.")
    elif args.dry_run:
        print(f"\nDry run: {moved} files would be migrated.")
    else:
        print("\nNothing to migrate (already on new layout or no legacy files).")


if __name__ == "__main__":
    main()
