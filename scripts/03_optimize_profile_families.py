#!/usr/bin/env python3
"""
Multi-family charging profile optimization (Priority 1).

Runs Bayesian optimization independently for each profile family:
  - CCCV, Reduced-CV CCCV
  - Adaptive 2-step / 3-step (SoC-triggered)
  - Exponential taper
  - CC-taper, Multi-step taper, Pulsed (legacy voltage/pulse families)

Outputs (under models/ and plots/profile_families/):
  - family_optimization_results.json
  - comparison_table.csv
  - best_<family>.png
  - profile_family_comparison.png
  - plots/pareto/ — Pareto trade-off figures (Priority 3)
  - models/pareto_analysis.json, pareto_profiles.csv

Usage
-----
    venv/bin/python scripts/03_optimize_profile_families.py \\
        --out_dir outputs/charging_opt_user/hima \\
        --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \\
        --n_calls 40 --n_initial 10

    venv/bin/python scripts/report_profile_families.py \\
        --out_dir outputs/charging_opt_user/hima
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from rw_transfer.config import load_config
from rw_transfer.data.series import load_battery_series
from charging_opt import paths as P
from charging_opt.artifacts import CANONICAL, resolve_bdt_ckpt
from charging_opt.charging_profile_family import DEFAULT_FAMILY_IDS, FAMILY_LABELS
from charging_opt.family_optimizer import optimize_families, save_family_results
from charging_opt.family_reporting import export_family_artifacts
from charging_opt.io_utils import current_user, resolve_stage3_family_dirs, resolve_stage3_pareto_dirs, user_stage3_root
from charging_opt.pareto_analysis import analyze_family_results
from charging_opt.pareto_reporting import export_pareto_artifacts
from charging_opt.profile_simulator import ProfileSimulator
from charging_opt.soc_utils import load_ocv_curve
from charging_opt.objective_cli import add_objective_args, objective_from_args
from charging_opt.state_utils import extract_rest_states, pick_start_state


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-family charging profile BO (Priority 1).")
    p.add_argument("--config", default=None)
    p.add_argument("--cell", default="RW9")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--ocv_curve", default=CANONICAL["ocv_curve"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILY_IDS),
        help=f"comma-separated family ids (default: all 8). Options: {DEFAULT_FAMILY_IDS}",
    )
    p.add_argument("--n_calls", type=int, default=40, help="BO evaluations per family")
    p.add_argument("--n_initial", type=int, default=10)
    p.add_argument("--max_minutes", type=int, default=150)
    p.add_argument("--max_duration_min", type=float, default=105.0)
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--no_time_limit", action="store_true")
    p.add_argument("--soc", type=float, default=None)
    p.add_argument("--v0", type=float, default=None)
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--age", type=float, default=None)
    p.add_argument(
        "--out_dir",
        default=None,
        help="output base (default: outputs/charging_opt_user/<USER>/stage3_optimization)",
    )
    add_objective_args(p)
    args = p.parse_args()

    weights, objective_mode, refs = objective_from_args(args)
    objective_config = {
        "objective_mode": objective_mode,
        "v_ref_stress": refs["v_ref_stress"],
        "t_comfort_c": refs["t_comfort_c"],
        "weights": weights.to_dict(),
    }

    P.ensure_layout(ROOT)
    out_base = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else user_stage3_root(ROOT, current_user())
    )
    models_dir, plots_dir = resolve_stage3_family_dirs(ROOT, out_dir=out_base)

    if args.soc is not None:
        start = {
            "soc": args.soc,
            "v0": args.v0 if args.v0 is not None else 3.7,
            "t0": args.t0 if args.t0 is not None else 25.0,
            "age": args.age if args.age is not None else 0.0,
            "prev_i": 0.0,
        }
    else:
        cfg = load_config(args.config)
        series = load_battery_series(cfg["data"]["matlab_dir"], args.cell, step_mode="all")
        ocv = load_ocv_curve(ROOT / args.ocv_curve)
        states = extract_rest_states(series, ocv, max_states=3000)
        start = pick_start_state(states)

    max_duration = None if args.no_time_limit else args.max_duration_min
    family_ids = [f.strip() for f in args.families.split(",") if f.strip()]

    bdt_path = resolve_bdt_ckpt(args.bdt_ckpt, root=ROOT)
    try:
        bdt_display = str(bdt_path.relative_to(ROOT))
    except ValueError:
        bdt_display = str(bdt_path)

    print(f"Output base: {out_base}")
    print(f"Start: SoC={start['soc']:.2f}  V={start['v0']:.3f}  "
          f"T={start['t0']:.1f}°C  age={start['age']:.3f}")
    print(f"BDT: {bdt_display}")
    print(f"Families ({len(family_ids)}): {family_ids}")
    print(f"Objective: {objective_mode}  weights={weights.to_dict()}")
    print(f"BO: {args.n_calls} evals/family ({args.n_initial} random initial)\n")

    sim = ProfileSimulator(
        bdt_path=bdt_path,
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=args.max_minutes,
        soc_target=args.soc_target,
    )

    def _save_partial(partial: dict) -> None:
        save_family_results(
            partial,
            models_dir,
            initial_state=start,
            bdt_path=bdt_display,
            soc_target=args.soc_target,
            max_duration_min=max_duration,
            objective_config=objective_config,
            repo_root=ROOT,
        )

    results = optimize_families(
        sim,
        start,
        family_ids,
        n_calls=args.n_calls,
        n_initial_points=args.n_initial,
        soc_target=args.soc_target,
        max_duration_min=max_duration,
        weights=weights,
        objective_mode=objective_mode,
        v_ref_stress=refs["v_ref_stress"],
        t_comfort_c=refs["t_comfort_c"],
        on_family_done=_save_partial,
    )

    json_path = save_family_results(
        results,
        models_dir,
        initial_state=start,
        bdt_path=bdt_display,
        soc_target=args.soc_target,
        max_duration_min=max_duration,
        objective_config=objective_config,
        repo_root=ROOT,
    )

    artifacts = export_family_artifacts(
        results,
        plots_dir,
        csv_path=models_dir / "comparison_table.csv",
    )

    _, pareto_plots_dir = resolve_stage3_pareto_dirs(ROOT, out_dir=out_base)
    families_payload = {fid: r.to_dict() for fid, r in results.items()}
    pareto_payload = {
        "families": families_payload,
        "constraints": {
            "soc_target": args.soc_target,
            "max_duration_min": max_duration,
            **objective_config,
        },
    }
    pareto_artifacts = export_pareto_artifacts(
        pareto_payload, pareto_plots_dir, models_dir=models_dir,
    )

    print(f"\n{'=' * 72}")
    print(f"  MULTI-FAMILY RESULTS ({objective_mode} objective)")
    print(f"{'=' * 72}")
    feasible = [r for r in results.values() if r.best_metrics.get("feasible")]
    best_fid = min(feasible, key=lambda r: r.best_loss).family_id if feasible else None
    for fid, r in sorted(results.items(), key=lambda kv: kv[1].best_loss):
        m = r.best_metrics
        mark = "  ← best" if fid == best_fid else ""
        print(
            f"  {FAMILY_LABELS.get(fid, fid):28s}  loss={r.best_loss:.1f}  "
            f"dur={m.get('duration_min', float('nan')):.1f} min  "
            f"SEI/%SoC={m.get('sei_per_pct_soc', float('nan')):.1f}  "
            f"V²·min={m.get('voltage_stress_v2_min', float('nan')):.2f}  "
            f"feasible={m.get('feasible')}{mark}"
        )
    if feasible:
        best = min(feasible, key=lambda r: r.best_loss)
        print(f"\n  Best overall: {best.family_label} (loss={best.best_loss:.2f})")
    pareto = analyze_family_results(families_payload)
    pt = pareto.tagged_global
    print(f"\n  Pareto reference profiles ({pareto.n_pareto_global} on front):")
    for tag in ("fastest", "lifetime", "balanced"):
        c = getattr(pt, tag)
        if c:
            print(
                f"    {tag:10s} {c.family_label:26s}  "
                f"dur={c.duration_min:.1f} min  SEI={c.sei_per_pct_soc:.1f}"
            )
    print(f"\n  JSON -> {json_path}")
    print(f"  CSV  -> {artifacts['comparison_csv']}")
    print(f"  plots-> {plots_dir}/")
    print(f"  pareto-> {pareto_plots_dir}/")
    print(f"  pareto JSON -> {pareto_artifacts['pareto_json']}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
