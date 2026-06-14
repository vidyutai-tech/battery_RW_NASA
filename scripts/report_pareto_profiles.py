#!/usr/bin/env python3
"""
Priority 3 — Pareto analysis from saved family optimization JSON (no BO re-run).

Builds non-dominated fronts over charge duration, SEI/ΔSoC, voltage stress,
and temperature penalty; tags Fastest / Balanced / Lifetime profiles.

Usage
-----
    venv/bin/python scripts/report_pareto_profiles.py

    venv/bin/python scripts/report_pareto_profiles.py \\
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

from charging_opt.family_reporting import load_results_payload
from charging_opt.io_utils import resolve_stage3_pareto_dirs
from charging_opt.pareto_reporting import export_pareto_artifacts


def main() -> None:
    p = argparse.ArgumentParser(description="Pareto front analysis (Priority 3).")
    p.add_argument(
        "--result",
        default=None,
        help="family_optimization_results.json (default: auto-detect)",
    )
    p.add_argument("--out_dir", default=None, help="stage3 output base directory")
    args = p.parse_args()

    out_base = Path(args.out_dir).resolve() if args.out_dir else None
    models_dir, pareto_plots_dir = resolve_stage3_pareto_dirs(ROOT, out_dir=out_base)

    result_path = (
        Path(args.result)
        if args.result
        else models_dir / "family_optimization_results.json"
    )
    if not result_path.is_file():
        raise SystemExit(
            f"Missing {result_path} — run scripts/03_optimize_profile_families.py first."
        )

    data = load_results_payload(result_path)
    written = export_pareto_artifacts(
        data, pareto_plots_dir, models_dir=models_dir,
    )

    from charging_opt.pareto_analysis import analyze_results_payload

    analysis = analyze_results_payload(data)
    tg = analysis.tagged_global
    deg_key = analysis.degradation_key
    deg_label = analysis.degradation_label

    print(f"Pareto analysis from {result_path}")
    print(f"  Degradation axis: {deg_label} ({deg_key})")
    print(f"  Feasible evals : {analysis.n_feasible_total}")
    print(f"  Pareto front   : {analysis.n_pareto_global} non-dominated")
    print()
    for tag in ("fastest", "lifetime", "balanced"):
        cand = getattr(tg, tag)
        if cand is None:
            print(f"  {tag:10s}: —")
            continue
        deg_val = getattr(cand, deg_key, None)
        if deg_val is None:
            deg_val = cand.sei_per_pct_soc
        print(
            f"  {tag:10s}: {cand.family_label:28s}  "
            f"dur={cand.duration_min:.1f} min  {deg_key}={deg_val:.4f}  "
            f"V²·min={cand.voltage_stress_v2_min:.2f}  loss={cand.loss:.2f}"
        )
    print()
    for key, path in sorted(written.items()):
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
