#!/usr/bin/env python3
"""
Multi-family charging profile optimization (Priority 1).

Runs Bayesian optimization independently for each profile family:
  - CCCV, Reduced-CV CCCV
  - Adaptive 2-step / 3-step (SoC-triggered)
  - Exponential taper
  - CC-taper, Multi-step taper, Pulsed (legacy voltage/pulse families)
  - Polynomial I(t) [NEW — superset of all families, Enhancement 1]

ENHANCEMENTS vs original:
  --acq_func        EI | PI | LCB  (PI recommended, Paper 3)
  --age_conditioning               evaluate each candidate at multiple ages
  --age_points      0.0,0.25,0.5,0.75   age checkpoints (comma-separated)
  --age_weights     0.15,0.30,0.35,0.20 corresponding weights
  --chebyshev_omega 0.0–1.0         Chebyshev scalarization weight
                                   (0=lifetime, 1=fastest, 0.5=balanced)
  --families        polynomial_current   new joint-search family

Outputs (under models/ and plots/profile_families/):
  - family_optimization_results.json
  - comparison_table.csv
  - best_<family>.png
  - profile_family_comparison.png
  - plots/pareto/ — Pareto trade-off figures (Priority 3)
  - models/pareto_analysis.json, pareto_profiles.csv

Usage
-----
    # Original run (now uses PI instead of EI)
    venv/bin/python scripts/03_optimize_profile_families.py \\
        --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0

    # With age conditioning (Enhancement 2)
    venv/bin/python scripts/03_optimize_profile_families.py \\
        --age_conditioning --soc 0.15 --v0 3.711 --t0 24.7

    # Chebyshev sweep (Enhancement 4) — run for each omega
    for omega in 0.0 0.2 0.4 0.6 0.8 1.0; do
        venv/bin/python scripts/03_optimize_profile_families.py \\
            --chebyshev_omega $omega \\
            --out_dir outputs/charging_opt_user/hima/pareto_sweep/omega_$omega
    done

    # Polynomial family (Enhancement 1) — joint search across all shapes
    venv/bin/python scripts/03_optimize_profile_families.py \\
        --families polynomial_current --n_calls 80 \\
        --age_conditioning
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
from charging_opt.family_optimizer import (
    DEFAULT_AGE_POINTS,
    DEFAULT_AGE_WEIGHTS,
    optimize_families,
    save_family_results,
)
from charging_opt.family_reporting import export_family_artifacts
from charging_opt.io_utils import (
    current_user,
    resolve_stage3_family_dirs,
    resolve_stage3_pareto_dirs,
    user_stage3_root,
)
from charging_opt.pareto_analysis import (
    analyze_family_results,
    degradation_summary,
    resolve_pareto_config,
)
from charging_opt.pareto_reporting import export_pareto_artifacts
from charging_opt.profile_simulator import ProfileSimulator
from charging_opt.soc_utils import load_ocv_curve
from charging_opt.objective_cli import add_objective_args, objective_from_args
from charging_opt.state_utils import extract_rest_states, pick_start_state


def _parse_float_list(s: str) -> list[float]:
    """Parse '0.0,0.25,0.5,0.75' -> [0.0, 0.25, 0.5, 0.75]."""
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def main() -> None:
    p = argparse.ArgumentParser(
        description="Multi-family charging profile BO.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Existing args (unchanged) ──────────────────────────────────────────
    p.add_argument("--config", default=None)
    p.add_argument("--cell", default="RW9")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--ocv_curve", default=CANONICAL["ocv_curve"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILY_IDS),
        help=(
            f"Comma-separated family ids. Default: all {len(DEFAULT_FAMILY_IDS)} families. "
            f"New: 'polynomial_current' for joint shape+parameter search (Enhancement 1)."
        ),
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

    # ── NEW: Enhancement 3 — acquisition function ──────────────────────────
    p.add_argument(
        "--acq_func",
        default="PI",
        choices=["EI", "PI", "LCB"],
        help=(
            "GP acquisition function. PI recommended (Paper 3: Jiang et al. 2022). "
            "EI = original. LCB = lower confidence bound (uses kappa=4.0)."
        ),
    )

    # ── NEW: Enhancement 2 — age conditioning ─────────────────────────────
    p.add_argument(
        "--age_conditioning",
        action="store_true",
        help=(
            "Evaluate each candidate at multiple battery ages (Enhancement 2). "
            "Optimizes for lifetime-robust profiles, not just fresh-cell performance. "
            "Requires ~4x more BDT calls per eval. Recommended for final results."
        ),
    )
    p.add_argument(
        "--age_points",
        default=",".join(str(a) for a in DEFAULT_AGE_POINTS),
        help=(
            f"Comma-separated age checkpoints for age conditioning. "
            f"Default: {DEFAULT_AGE_POINTS}. "
            f"Based on RW9-RW12 lifespan (age 0=fresh, 1=end-of-life)."
        ),
    )
    p.add_argument(
        "--age_weights",
        default=",".join(str(w) for w in DEFAULT_AGE_WEIGHTS),
        help=(
            f"Weights for each age point (must sum to ~1). "
            f"Default: {DEFAULT_AGE_WEIGHTS} — weights later life more."
        ),
    )

    # ── NEW: Enhancement 4 — Chebyshev scalarization ──────────────────────
    p.add_argument(
        "--chebyshev_omega",
        type=float,
        default=None,
        help=(
            "Chebyshev scalarization weight omega ∈ [0,1] (Enhancement 4, Paper 2). "
            "omega=0 → minimize SEI only (Lifetime). "
            "omega=1 → minimize time only (Fastest). "
            "omega=0.5 → balanced (knee of Pareto front). "
            "If not set, uses linear scalarization (composite objective, original behavior). "
            "Run with multiple omega values to construct the full Pareto front."
        ),
    )

    # ── Thermal management (Level 1) ───────────────────────────────────────
    p.add_argument(
        "--thermal_derating",
        action="store_true",
        help="Apply BDT temperature-aware current derating in simulator",
    )
    p.add_argument(
        "--thermal_derate_comfort_c",
        type=float,
        default=33.0,
        help="Start derating current above this BDT-predicted T (°C)",
    )
    p.add_argument(
        "--thermal_loss",
        action="store_true",
        help="Add extra temperature_loss on top of composite objective",
    )
    p.add_argument("--thermal_w_comfort", type=float, default=0.5)
    p.add_argument("--thermal_w_hard", type=float, default=5.0)

    args = p.parse_args()

    weights, objective_mode, refs = objective_from_args(args)

    # Chebyshev overrides the objective mode for BO, but we still report composite metrics
    if args.chebyshev_omega is not None:
        objective_mode = "chebyshev"

    objective_config = {
        "objective_mode": objective_mode,
        "v_ref_stress": refs["v_ref_stress"],
        "t_comfort_c": refs["t_comfort_c"],
        "weights": weights.to_dict(),
    }
    if args.chebyshev_omega is not None:
        objective_config["chebyshev_omega"] = args.chebyshev_omega
    if args.thermal_derating or args.thermal_loss:
        objective_config["thermal_derating"] = args.thermal_derating
        objective_config["thermal_loss"] = args.thermal_loss
        objective_config["thermal_derate_comfort_c"] = args.thermal_derate_comfort_c

    age_points = _parse_float_list(args.age_points)
    age_weights = _parse_float_list(args.age_weights)
    if len(age_points) != len(age_weights):
        p.error("--age_points and --age_weights must have the same number of values")

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

    # ── Summary header ─────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  MULTI-FAMILY CHARGING PROFILE OPTIMIZATION")
    print(f"{'=' * 72}")
    print(f"  Output base   : {out_base}")
    print(f"  BDT           : {bdt_display}")
    print(f"  Start state   : SoC={start['soc']:.2f}  V={start['v0']:.3f}  "
          f"T={start['t0']:.1f}°C  age={start['age']:.3f}")
    print(f"  Families ({len(family_ids)}): {family_ids}")
    print(f"  BO calls/family: {args.n_calls}  initial: {args.n_initial}")
    print(f"  Acquisition   : {args.acq_func}")
    print(f"  Objective     : {objective_mode}  weights={weights.to_dict()}")
    if args.age_conditioning:
        print(f"  Age conditioning: ON")
        print(f"    ages   = {age_points}")
        print(f"    weights = {age_weights}")
        print(f"    (Each eval runs BDT at {len(age_points)} ages — "
              f"~{len(age_points)}x more calls)")
    else:
        print(f"  Age conditioning: OFF (single age={start['age']:.3f})")
    if args.chebyshev_omega is not None:
        print(f"  Chebyshev omega : {args.chebyshev_omega:.2f}  "
              f"({'lifetime' if args.chebyshev_omega < 0.2 else 'fastest' if args.chebyshev_omega > 0.8 else 'balanced'})")
    if args.thermal_derating or args.thermal_loss:
        print(f"  Thermal derating: {args.thermal_derating}  "
              f"comfort={args.thermal_derate_comfort_c:.1f}°C  extra_loss={args.thermal_loss}")
    print(f"{'=' * 72}\n")

    from charging_opt.thermal_management import ThermalDeratingController

    thermal_controller = None
    if args.thermal_derating or args.thermal_loss:
        thermal_controller = ThermalDeratingController(
            t_comfort_c=args.thermal_derate_comfort_c,
        )

    sim = ProfileSimulator(
        bdt_path=bdt_path,
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=args.max_minutes,
        soc_target=args.soc_target,
        thermal_controller=thermal_controller if args.thermal_derating else None,
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
        # New parameters
        acq_func=args.acq_func,
        use_age_conditioning=args.age_conditioning,
        age_points=age_points,
        age_weights=age_weights,
        chebyshev_omega=args.chebyshev_omega,
        thermal_controller=thermal_controller if args.thermal_loss else None,
        thermal_w_comfort=args.thermal_w_comfort,
        thermal_w_hard=args.thermal_w_hard,
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
        constraints={
            "soc_target": args.soc_target,
            "max_duration_min": max_duration,
            **objective_config,
        },
        simulator=sim,
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

    # ── Results summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  RESULTS  (objective={objective_mode}, acq={args.acq_func}, "
          f"age_cond={args.age_conditioning})")
    print(f"{'=' * 72}")
    summary_constraints = {
        "soc_target": args.soc_target,
        "max_duration_min": max_duration,
        **objective_config,
    }
    feasible = [r for r in results.values() if r.best_metrics.get("feasible")]
    best_fid = min(feasible, key=lambda r: r.best_loss).family_id if feasible else None
    for fid, r in sorted(results.items(), key=lambda kv: kv[1].best_loss):
        m = r.best_metrics
        mark = "  ← best" if fid == best_fid else ""
        print(
            f"  {FAMILY_LABELS.get(fid, fid):28s}  loss={r.best_loss:.1f}  "
            f"dur={m.get('duration_min', float('nan')):.1f} min  "
            f"{degradation_summary(m, summary_constraints)}  "
            f"V²·min={m.get('voltage_stress_v2_min', float('nan')):.2f}  "
            f"feasible={m.get('feasible')}{mark}"
        )
    if feasible:
        best = min(feasible, key=lambda r: r.best_loss)
        print(f"\n  Best overall: {best.family_label} (loss={best.best_loss:.2f})")

    pareto = analyze_family_results(families_payload, constraints=summary_constraints)
    pt = pareto.tagged_global
    _, deg_key, deg_label = resolve_pareto_config(summary_constraints)
    print(f"\n  Pareto reference profiles ({pareto.n_pareto_global} on front):")
    for tag in ("fastest", "lifetime", "balanced"):
        c = getattr(pt, tag)
        if c:
            deg_val = getattr(c, deg_key, c.sei_per_pct_soc)
            if deg_key == "capacity_fade_pct":
                deg_str = f"{deg_val:.3f}"
            else:
                deg_str = f"{deg_val:.1f}"
            print(
                f"    {tag:10s} {c.family_label:26s}  "
                f"dur={c.duration_min:.1f} min  {deg_label}={deg_str}"
            )

    print(f"\n  JSON   -> {json_path}")
    print(f"  CSV    -> {artifacts.get('comparison_csv', '—')}")
    print(f"  plots  -> {plots_dir}/")
    print(f"  pareto -> {pareto_plots_dir}/")
    print(f"  pareto JSON -> {pareto_artifacts['pareto_json']}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
