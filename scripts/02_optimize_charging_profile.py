#!/usr/bin/env python3
"""
Stage 3 — Bayesian optimization for lifetime-optimal charging profiles.

Three-stage pipeline (patent-aligned):
    1. Frozen BDT simulates V/T for each candidate current profile.
    2. Lifetime reward aggregates SEI proxy, SoC target, temperature comfort.
    3. Gaussian-process BO searches the best parametric profile.

Prerequisites (run once per cell / chemistry):
    scripts/01_fit_ocv_curve.py
    scripts/00_diagnose_drift.py

Adapt to another battery: pass a fine-tuned BDT checkpoint via --bdt_ckpt.

Usage
-----
    venv/bin/python scripts/02_optimize_charging_profile.py

    # Fine-tuned RW10 twin, custom start state
    venv/bin/python scripts/02_optimize_charging_profile.py \\
        --bdt_ckpt outputs/finetune_two_stage_v2/registry/finetune_RW10_frac0.40.pt \\
        --soc 0.15 --v0 3.71 --t0 24.7 --age 0.0
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
from charging_opt.artifacts import (
    CANONICAL,
    OPTIONAL,
    resolve_bdt_ckpt,
    update_master_registry,
    write_stage_registry,
)
from charging_opt.bayesian_optimizer import (
    LifetimeBayesianOptimizer,
    save_optimization_result,
)
from charging_opt import paths as P
from charging_opt.profile_simulator import ProfileSimulator
from charging_opt.soc_utils import load_ocv_curve
from charging_opt.objective_cli import add_objective_args, objective_from_args
from charging_opt.state_utils import extract_rest_states, pick_start_state


def plot_best_profile(session: dict, metrics: dict, out_path: Path) -> None:
    t_min = session["time_s"] / 60.0
    i_a = -session["current_a"]
    v = session["voltage_v"]
    soc_pct = session["soc"] * 100.0
    s0 = session["initial_state"]
    spec = session["profile_spec"]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()
    ax1.plot(t_min, i_a, color="tab:blue", lw=2.0, label="Charge current")
    ax1.plot(t_min, v, color="tab:red", lw=1.8, ls="--", label="Cell voltage")
    ax2.plot(t_min, soc_pct, color="black", lw=2.2, ls="-.", label="SoC")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Charge current (A)  /  Voltage (V)")
    ax2.set_ylabel("State of charge (%)")
    ax2.set_ylim(0, 105)
    ax1.grid(alpha=0.3)
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=8)

    feasible = metrics.get("feasible", False)
    loss = metrics.get("loss", float("nan"))
    tapered = any(d.get("ceiling_hit") for d in session.get("decisions", []))
    if spec["pulse_rest_min"] >= 5.0:
        mode_line = (
            f"pulse {spec['pulse_on_min']:.1f} min ON / {spec['pulse_rest_min']:.1f} min rest  |  "
        )
    elif tapered:
        mode_line = "CC-taper (ceiling hit)  |  "
    elif abs(spec["i_charge"] - spec["i_floor"]) < 1e-6:
        mode_line = "constant CC (no taper)  |  "
    else:
        mode_line = "continuous CC  |  "
    subtitle = (
        f"I_cc={spec['i_charge']:.2f} A  |  {mode_line}"
        f"I_floor={spec['i_floor']:.2f} A  |  "
        f"{'FEASIBLE' if feasible else 'INFEASIBLE'}  loss={loss:.2f}\n"
        f"Start SoC={s0['soc']:.0%}, V={s0['v0']:.2f} V, T={s0['t0']:.1f} °C, age={s0['age']:.2f}\n"
        f"ΔSoC={metrics['delta_soc_pct_total']:.1f}%  |  "
        f"Duration={metrics['duration_min']:.1f} min"
        + (
            f" (limit {metrics.get('components', {}).get('max_duration_min', '—')} min)  |  "
            if metrics.get("components", {}).get("max_duration_min") is not None else "  |  "
        )
        + f"SEI/%SoC={metrics.get('sei_per_pct_soc', float('nan')):.1f}  |  "
        f"End: {metrics['end_reason']}"
    )
    fig.suptitle(f"Lifetime-optimal profile (Bayesian optimization)\n{subtitle}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(history: list, out_path: Path) -> None:
    losses = [h["loss"] for h in history]
    feas = [h.get("feasible", False) for h in history]
    best = np.minimum.accumulate(losses)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:green" if f else "tab:gray" for f in feas]
    ax.scatter(range(1, len(losses) + 1), losses, c=colors, s=28, alpha=0.7, label="evaluations")
    ax.plot(range(1, len(best) + 1), best, lw=2, color="tab:red", label="best loss so far")
    ax.axhline(200, color="orange", ls=":", lw=1, label="infeasible threshold (~200)")
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("Loss (lower is better; feasible ≈ SEI/%SoC)")
    ax.set_title("Bayesian optimization — green = feasible (SoC + time limit)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Lifetime charging profile — Bayesian optimization.")
    p.add_argument("--config", default=None)
    p.add_argument("--cell", default="RW9")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--ocv_curve", default=CANONICAL["ocv_curve"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument("--n_calls", type=int, default=40)
    p.add_argument("--n_initial", type=int, default=8)
    p.add_argument("--max_minutes", type=int, default=150,
                   help="simulation horizon (must be >= max_duration_min)")
    p.add_argument("--max_duration_min", type=float, default=105.0,
                   help="hard deadline: profile must reach soc_target within this many minutes")
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--no_time_limit", action="store_true",
                   help="disable max_duration_min (lifetime-only, may pick 0.75 A / 141 min)")
    p.add_argument("--allow-pulsed", action="store_true",
                   help="search pulsed rest (default: CC-taper only)")
    # Manual start state (overrides dataset pick when any is set)
    p.add_argument("--soc", type=float, default=None)
    p.add_argument("--v0", type=float, default=None)
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--age", type=float, default=None)
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
    models_dir = (ROOT / P.STAGE3_MODELS).resolve()
    plots_dir = (ROOT / P.STAGE3_PLOTS).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    if max_duration is not None and args.max_minutes < max_duration:
        print(f"NOTE: raising simulation horizon to {max_duration:.0f} min "
              f"(matches max_duration_min)")
        args.max_minutes = int(max_duration)

    bdt_path = resolve_bdt_ckpt(args.bdt_ckpt, root=ROOT)
    try:
        bdt_display = bdt_path.relative_to(ROOT)
    except ValueError:
        bdt_display = bdt_path

    print(f"Start state: SoC={start['soc']:.2f}  V={start['v0']:.3f}  "
          f"T={start['t0']:.1f}°C  age={start['age']:.3f}")
    print(f"BDT: {bdt_display}")
    if max_duration is not None:
        print(f"Constraints: SoC>={args.soc_target:.0%}, duration<={max_duration:.0f} min")
    else:
        print(f"Constraints: SoC>={args.soc_target:.0%} (no time limit)")
    print(f"Simulation horizon: {args.max_minutes} min")
    print(f"Objective: {objective_mode}  weights={weights.to_dict()}")
    print(f"BO: {args.n_calls} evaluations ({args.n_initial} random initial)\n")

    sim = ProfileSimulator(
        bdt_path=bdt_path,
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=args.max_minutes,
        soc_target=args.soc_target,
    )
    optimizer = LifetimeBayesianOptimizer(
        sim, start, soc_target=args.soc_target,
        max_duration_min=max_duration,
        weights=weights,
        objective_mode=objective_mode,
        v_ref_stress=refs["v_ref_stress"],
        t_comfort_c=refs["t_comfort_c"],
        allow_pulsed=args.allow_pulsed,
        random_state=42,
    )
    result = optimizer.optimize(n_calls=args.n_calls, n_initial_points=args.n_initial)

    result.best_session["profile_spec"] = result.best_spec.to_dict()
    save_optimization_result(
        result, models_dir,
        initial_state=start,
        bdt_path=str(bdt_display),
        soc_target=args.soc_target,
        max_duration_min=max_duration,
        objective_config=objective_config,
    )
    plot_best_profile(
        result.best_session, result.best_metrics,
        plots_dir / "best_profile.png",
    )
    plot_convergence(result.history, plots_dir / "bo_convergence.png")

    stage3_metrics = {
        "best_spec": result.best_spec.to_dict(),
        "best_metrics": result.best_metrics,
        "constraints": {
            "soc_target": args.soc_target,
            "max_duration_min": max_duration,
            **objective_config,
        },
        "artifacts": {
            "optimization_result": OPTIONAL["lifetime_bo_result"],
            "best_profile_plot": OPTIONAL["plot_best_profile"],
        },
    }
    write_stage_registry(P.STAGE3, stage3_metrics, root=ROOT)
    update_master_registry(root=ROOT)

    print("\n" + "=" * 72)
    print("BEST LIFETIME PROFILE")
    print("=" * 72)
    for k, v in result.best_spec.to_dict().items():
        print(f"  {k}: {v}")
    m = result.best_metrics
    print(f"\n  ΔSoC: {m['delta_soc_pct_total']:.1f}%")
    print(f"  SEI/%SoC: {m.get('sei_per_pct_soc', float('nan')):.2f}")
    print(f"  Duration: {m['duration_min']:.1f} min")
    print(f"  Feasible: {m.get('feasible', False)}")
    print(f"  Loss: {m.get('loss', float('nan')):.3f}")
    comp = m.get("components", {})
    if comp.get("objective_mode") == "composite":
        print(
            f"    SEI term={comp.get('sei_term', float('nan')):.2f}  "
            f"time={comp.get('time_term', float('nan')):.2f}  "
            f"temp={comp.get('temperature_term', float('nan')):.2f}  "
            f"V-stress={comp.get('voltage_stress_term', float('nan')):.2f}"
        )
    print(f"  End: {m['end_reason']}")
    print(f"\nOutputs:")
    print(f"  models  -> {models_dir}/")
    print(f"  plots   -> {plots_dir}/")
    print(f"  registry-> {ROOT / P.REGISTRY}/")


if __name__ == "__main__":
    main()
