#!/usr/bin/env python3
"""
Directed Pareto front construction via Chebyshev scalarization sweep.

WHAT THIS DOES:
  Runs one BO optimization per omega value, where omega controls the
  SEI vs charging time tradeoff. Collecting results across all omegas gives
  a uniformly sampled Pareto front — superior to mining a single run's history.

  From Paper 2 (Wang & Jiang 2023): "By selecting different value of omega,
  the Pareto front solutions for the multi-objective fast-charging optimization
  problem using the proposed cTS-BO method can be obtained."

WHY THIS IS BETTER THAN YOUR CURRENT APPROACH:
  Your current Pareto front (37 non-dominated points) comes from post-hoc
  mining of 8 independent BO runs, each focused on a single scalar loss.
  This approach is biased toward the linear-weighted optimum, missing
  non-convex parts of the frontier.

  This script instead places BO budget DIRECTLY on each part of the frontier
  by sweeping omega ∈ {0, 0.1, ..., 1.0} — 11 points = 11 × n_calls evaluations
  total, each targeting a different balance between speed and longevity.

USAGE:
    # Quick sweep (11 omegas × 30 evals = 330 BDT calls)
    venv/bin/python scripts/run_chebyshev_pareto_sweep.py \\
        --families cccv pulsed adaptive_two_step polynomial_current \\
        --n_calls 30 \\
        --soc 0.15 --v0 3.711 --t0 24.7 \\
        --out_dir outputs/charging_opt_user/hima/chebyshev_sweep

    # With age conditioning for lifetime-robust Pareto front
    venv/bin/python scripts/run_chebyshev_pareto_sweep.py \\
        --age_conditioning \\
        --families polynomial_current \\
        --n_calls 40 \\
        --out_dir outputs/charging_opt_user/hima/pareto_age_conditioned
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
from rw_transfer.data.series import load_battery_series
from charging_opt.artifacts import CANONICAL, resolve_bdt_ckpt
from charging_opt.charging_profile_family import DEFAULT_FAMILY_IDS, get_family
from charging_opt.family_optimizer import (
    DEFAULT_AGE_POINTS,
    DEFAULT_AGE_WEIGHTS,
    FamilyBayesianOptimizer,
)
from charging_opt.lifetime_reward import LifetimeWeights
from charging_opt.profile_simulator import ProfileSimulator
from charging_opt.soc_utils import load_ocv_curve
from charging_opt.state_utils import extract_rest_states, pick_start_state


def _parse_float_list(s: str) -> list[float]:
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def plot_chebyshev_pareto(results_by_omega: dict, out_path: Path) -> None:
    """Plot the Pareto front obtained by Chebyshev sweep."""
    fig, ax = plt.subplots(figsize=(10, 6))

    omega_values = sorted(results_by_omega.keys())
    cmap = plt.get_cmap("RdYlGn")

    for omega in omega_values:
        runs = results_by_omega[omega]
        for run in runs:
            if run.get("feasible"):
                color = cmap(1.0 - omega)  # green=low omega (lifetime), red=high (fast)
                ax.scatter(
                    run["duration_min"],
                    run["sei_per_pct_soc"],
                    c=[color],
                    s=80,
                    alpha=0.8,
                    edgecolors="white",
                    linewidths=0.5,
                )

    # Connect Pareto points
    pareto_points = []
    for omega in omega_values:
        runs = results_by_omega[omega]
        best = min(
            (r for r in runs if r.get("feasible")),
            key=lambda r: r["duration_min"],
            default=None,
        )
        if best:
            pareto_points.append((best["duration_min"], best["sei_per_pct_soc"], omega))

    if pareto_points:
        pareto_points.sort(key=lambda x: x[0])
        xs = [p[0] for p in pareto_points]
        ys = [p[1] for p in pareto_points]
        ax.plot(xs, ys, "k--", lw=1.5, alpha=0.6, label="Chebyshev Pareto front")
        for x, y, omega in pareto_points:
            ax.annotate(
                f"ω={omega:.1f}",
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("RdYlGn_r"), norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="omega (0=lifetime, 1=fastest)")

    ax.set_xlabel("Charge duration (min)")
    ax.set_ylabel("SEI / ΔSoC (lower = better lifetime)")
    ax.set_title(
        f"Directed Pareto Front — Chebyshev Scalarization Sweep\n"
        f"({sum(len(v) for v in results_by_omega.values())} total evaluations)"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Pareto plot -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Directed Pareto front via Chebyshev omega sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default=None)
    p.add_argument("--cell", default="RW9")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument("--ocv_curve", default=CANONICAL["ocv_curve"])
    p.add_argument(
        "--families",
        nargs="+",
        default=["adaptive_two_step", "pulsed", "cccv"],
        help="Profile families to optimize for each omega value",
    )
    p.add_argument(
        "--omegas",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated omega values to sweep (default: 0 to 1 in 0.1 steps)",
    )
    p.add_argument("--n_calls", type=int, default=30, help="BO evaluations per omega per family")
    p.add_argument("--n_initial", type=int, default=8)
    p.add_argument("--max_minutes", type=int, default=150)
    p.add_argument("--max_duration_min", type=float, default=105.0)
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--soc", type=float, default=None)
    p.add_argument("--v0", type=float, default=None)
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--age", type=float, default=0.0)
    p.add_argument("--out_dir", required=True, help="Output directory for sweep results")
    p.add_argument(
        "--acq_func", default="PI", choices=["EI", "PI", "LCB"],
        help="Acquisition function (PI recommended, Paper 3)",
    )
    p.add_argument(
        "--age_conditioning", action="store_true",
        help="Evaluate each candidate at multiple ages (Enhancement 2)",
    )
    p.add_argument(
        "--age_points",
        default=",".join(str(a) for a in DEFAULT_AGE_POINTS),
    )
    p.add_argument(
        "--age_weights",
        default=",".join(str(w) for w in DEFAULT_AGE_WEIGHTS),
    )
    args = p.parse_args()

    omegas = _parse_float_list(args.omegas)
    age_points = _parse_float_list(args.age_points)
    age_weights = _parse_float_list(args.age_weights)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.soc is not None:
        start = {
            "soc": args.soc,
            "v0": args.v0 if args.v0 is not None else 3.7,
            "t0": args.t0 if args.t0 is not None else 25.0,
            "age": args.age,
            "prev_i": 0.0,
        }
    else:
        cfg = load_config(args.config)
        series = load_battery_series(cfg["data"]["matlab_dir"], args.cell, step_mode="all")
        ocv = load_ocv_curve(ROOT / args.ocv_curve)
        states = extract_rest_states(series, ocv, max_states=3000)
        start = pick_start_state(states)

    bdt_path = resolve_bdt_ckpt(args.bdt_ckpt, root=ROOT)

    sim = ProfileSimulator(
        bdt_path=bdt_path,
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=args.max_minutes,
        soc_target=args.soc_target,
    )

    print(f"\nChebyshev Pareto sweep: {len(omegas)} omegas × {len(args.families)} families")
    print(f"Total BO runs: {len(omegas) * len(args.families)}")
    print(f"Total BDT calls: ~{len(omegas) * len(args.families) * args.n_calls}")
    if args.age_conditioning:
        print(f"  (×{len(age_points)} for age conditioning)")

    # results_by_omega[omega] = list of {duration_min, sei_per_pct_soc, feasible, family_id, params}
    results_by_omega: dict[float, list[dict]] = {omega: [] for omega in omegas}
    all_runs = []

    for omega in omegas:
        print(f"\n{'─' * 60}")
        print(f"  omega = {omega:.1f}  ({'lifetime' if omega < 0.2 else 'fastest' if omega > 0.8 else 'balanced'})")
        print(f"{'─' * 60}")

        for fid in args.families:
            family = get_family(fid)
            opt = FamilyBayesianOptimizer(
                sim,
                family,
                start,
                soc_target=args.soc_target,
                max_duration_min=args.max_duration_min,
                weights=LifetimeWeights(),
                objective_mode="composite",
                acq_func=args.acq_func,
                use_age_conditioning=args.age_conditioning,
                age_points=age_points,
                age_weights=age_weights,
                chebyshev_omega=omega,
                random_state=42,
            )
            result = opt.optimize(n_calls=args.n_calls, n_initial_points=args.n_initial)

            m = result.best_metrics
            run_record = {
                "omega": omega,
                "family_id": fid,
                "family_label": family.label,
                "feasible": bool(m.get("feasible", False)),
                "duration_min": m.get("duration_min"),
                "sei_per_pct_soc": m.get("sei_per_pct_soc"),
                "loss": result.best_loss,
                "params": result.best_params.to_dict(),
            }
            results_by_omega[omega].append(run_record)
            all_runs.append(run_record)

            status = "✓ FEASIBLE" if run_record["feasible"] else "✗ infeasible"
            print(
                f"    {family.label:26s} {status}  "
                f"dur={run_record.get('duration_min', float('nan')):.1f} min  "
                f"SEI={run_record.get('sei_per_pct_soc', float('nan')):.1f}"
            )

    # Save results
    payload = {
        "config": {
            "omegas": omegas,
            "families": args.families,
            "n_calls": args.n_calls,
            "acq_func": args.acq_func,
            "age_conditioning": args.age_conditioning,
            "age_points": age_points,
            "age_weights": age_weights,
            "start_state": start,
        },
        "results_by_omega": {str(k): v for k, v in results_by_omega.items()},
        "all_runs": all_runs,
    }
    json_path = out_dir / "chebyshev_sweep_results.json"
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    plot_chebyshev_pareto(results_by_omega, out_dir / "chebyshev_pareto_front.png")

    # Print Pareto summary
    feasible_runs = [r for r in all_runs if r["feasible"]]
    print(f"\n{'=' * 60}")
    print(f"  CHEBYSHEV PARETO SWEEP COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total feasible runs: {len(feasible_runs)} / {len(all_runs)}")
    if feasible_runs:
        fastest = min(feasible_runs, key=lambda r: r["duration_min"])
        lifetime = min(feasible_runs, key=lambda r: r["sei_per_pct_soc"])
        print(f"  Fastest  : {fastest['family_label']:26s} "
              f"dur={fastest['duration_min']:.1f} min  SEI={fastest['sei_per_pct_soc']:.1f}  "
              f"(omega={fastest['omega']:.1f})")
        print(f"  Lifetime : {lifetime['family_label']:26s} "
              f"dur={lifetime['duration_min']:.1f} min  SEI={lifetime['sei_per_pct_soc']:.1f}  "
              f"(omega={lifetime['omega']:.1f})")
    print(f"\n  Results -> {json_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
