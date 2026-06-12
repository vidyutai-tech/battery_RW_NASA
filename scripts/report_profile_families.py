#!/usr/bin/env python3
"""
Regenerate Priority-1 family comparison artifacts from saved JSON (no BO re-run).

Usage
-----
    venv/bin/python scripts/report_profile_families.py

    venv/bin/python scripts/report_profile_families.py \\
        --result outputs/charging_opt_user/hima/stage3_optimization/models/family_optimization_results.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from charging_opt import paths as P
from charging_opt.artifacts import CANONICAL, resolve_bdt_ckpt
from charging_opt.family_reporting import (
    export_family_artifacts,
    load_results_payload,
    rehydrate_results_from_json,
)
from charging_opt.io_utils import resolve_stage3_family_dirs
from charging_opt.profile_simulator import ProfileSimulator


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate multi-family comparison plots/CSV.")
    p.add_argument(
        "--result",
        default=None,
        help="family_optimization_results.json (default: auto-detect)",
    )
    p.add_argument("--out_dir", default=None)
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument("--max_minutes", type=int, default=150)
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--max_duration_min", type=float, default=105.0)
    args = p.parse_args()

    out_base = Path(args.out_dir).resolve() if args.out_dir else None
    models_dir, plots_dir = resolve_stage3_family_dirs(ROOT, out_dir=out_base)

    result_path = Path(args.result) if args.result else models_dir / "family_optimization_results.json"
    if not result_path.is_file():
        raise SystemExit(f"Missing {result_path} — run scripts/03_optimize_profile_families.py first.")

    data = load_results_payload(result_path)
    start = data["initial_state"]
    constraints = data.get("constraints", {})
    soc_target = float(constraints.get("soc_target", args.soc_target))
    max_dur = constraints.get("max_duration_min", args.max_duration_min)
    if max_dur is not None:
        max_dur = float(max_dur)

    sim = ProfileSimulator(
        bdt_path=resolve_bdt_ckpt(data.get("bdt_checkpoint", args.bdt_ckpt), root=ROOT),
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=args.max_minutes,
        soc_target=soc_target,
    )

    results = rehydrate_results_from_json(
        data, sim, start, soc_target=soc_target, max_duration_min=max_dur,
    )
    written = export_family_artifacts(
        results,
        plots_dir,
        csv_path=models_dir / "comparison_table.csv",
    )

    print(f"Regenerated {len(results)} families from {result_path}")
    for key, path in sorted(written.items()):
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
