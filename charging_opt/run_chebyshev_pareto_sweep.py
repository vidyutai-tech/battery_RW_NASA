#!/usr/bin/env python3
"""
Directed Pareto front construction via Chebyshev scalarization sweep.

Sweep omega ∈ {0, 0.1, ..., 1.0} with BO per (omega, family). Supports:
  - composite / SEI Chebyshev (default)
  - physics Wang ΔQ/Q₀ Chebyshev (--objective physics)
  - thermal derating + loss (--thermal_derating --thermal_loss)

USAGE:
    # Physics + thermal Chebyshev sweep
    venv/bin/python scripts/run_chebyshev_pareto_sweep.py \\
        --objective physics --thermal_derating --thermal_loss \\
        --families pulsed cccv adaptive_two_step \\
        --n_calls 30 --soc 0.15 --v0 3.711 --t0 24.7 \\
        --out_dir outputs/charging_opt_user/hima/chebyshev_sweep_physics

    # Replot existing JSON with physics metrics (BDT re-score + thermal)
    venv/bin/python scripts/run_chebyshev_pareto_sweep.py --plots_only \\
        --objective physics --thermal_derating --thermal_loss \\
        --out_dir outputs/charging_opt_user/hima/chebyshev_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from charging_opt.charging_profile_family import ProfileParams, get_family
from charging_opt.family_optimizer import (
    DEFAULT_AGE_POINTS,
    DEFAULT_AGE_WEIGHTS,
    FamilyBayesianOptimizer,
)
from charging_opt.lifetime_reward import LifetimeWeights
from charging_opt.objective_cli import add_objective_args, objective_from_args
from charging_opt.pareto_analysis import resolve_pareto_config
from charging_opt.profile_simulator import ProfileSimulator
from charging_opt.soc_utils import load_ocv_curve
from charging_opt.state_utils import extract_rest_states, pick_start_state
from charging_opt.thermal_management import ThermalDeratingController


def _parse_float_list(s: str) -> list[float]:
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def _fmt_deg(y: Optional[float], degradation_key: str) -> str:
    if y is None or not np.isfinite(y):
        return "nan"
    if degradation_key == "capacity_fade_pct":
        return f"{y:.3f}"
    return f"{y:.1f}"


def _run_y_value(run: Dict[str, Any], degradation_key: str) -> Optional[float]:
    if degradation_key == "capacity_fade_pct":
        val = run.get("capacity_fade_pct")
        if val is not None:
            return float(val)
    val = run.get("sei_per_pct_soc")
    return float(val) if val is not None else None


def plot_chebyshev_pareto(
    results_by_omega: dict,
    out_path: Path,
    *,
    degradation_key: str = "sei_per_pct_soc",
    degradation_label: str = "SEI / ΔSoC",
    objective_mode: str = "composite",
    thermal: bool = False,
) -> None:
    """Plot directed Pareto front from Chebyshev sweep."""
    fig, ax = plt.subplots(figsize=(11, 7))
    omega_values = sorted(results_by_omega.keys())
    cmap = plt.get_cmap("RdYlGn")

    for omega in omega_values:
        for run in results_by_omega[omega]:
            if not run.get("feasible"):
                continue
            y = _run_y_value(run, degradation_key)
            if y is None or not np.isfinite(y):
                continue
            color = cmap(1.0 - omega)
            ax.scatter(
                run["duration_min"],
                y,
                c=[color],
                s=90,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.6,
            )

    pareto_points = []
    for omega in omega_values:
        feasible = [
            r for r in results_by_omega[omega]
            if r.get("feasible") and _run_y_value(r, degradation_key) is not None
        ]
        if not feasible:
            continue
        best = min(feasible, key=lambda r: r["duration_min"])
        y = _run_y_value(best, degradation_key)
        if y is not None:
            pareto_points.append((best["duration_min"], y, omega))

    if pareto_points:
        pareto_points.sort(key=lambda x: x[0])
        xs = [p[0] for p in pareto_points]
        ys = [p[1] for p in pareto_points]
        ax.plot(xs, ys, "k--", lw=1.8, alpha=0.65, label="Chebyshev front (best/family)")
        for x, y, omega in pareto_points:
            ax.annotate(
                f"ω={omega:.1f}",
                (x, y),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=9,
                alpha=0.8,
            )

    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("RdYlGn_r"), norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="ω  (0=lifetime → 1=fastest)")
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("Charge duration (min)", fontsize=12)
    ylabel = f"{degradation_label}  (lower = better)"
    ax.set_ylabel(ylabel, fontsize=12)
    obj_note = "Wang ΔQ/Q₀" if objective_mode == "physics" else "SEI proxy"
    thermal_note = " · thermal derating + loss" if thermal else ""
    ax.set_title(
        f"Directed Pareto front — Chebyshev sweep ({obj_note}{thermal_note})\n"
        f"{sum(len(v) for v in results_by_omega.values())} BO runs",
        fontsize=13,
    )
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"  Pareto plot -> {out_path}")


def _metrics_to_run_record(
    omega: float,
    fid: str,
    family_label: str,
    params: dict,
    metrics: dict,
    loss: float,
) -> dict:
    return {
        "omega": omega,
        "family_id": fid,
        "family_label": family_label,
        "feasible": bool(metrics.get("feasible", False)),
        "duration_min": metrics.get("duration_min"),
        "sei_per_pct_soc": metrics.get("sei_per_pct_soc"),
        "capacity_fade_pct": metrics.get("capacity_fade_pct"),
        "peak_temperature": metrics.get("peak_temperature"),
        "loss": loss,
        "params": params,
    }


def rescore_chebyshev_json(
    json_path: Path,
    sim: ProfileSimulator,
    start: dict,
    *,
    soc_target: float,
    max_duration_min: float,
    weights: LifetimeWeights,
    objective_mode: str,
    v_ref_stress: float,
    t_comfort_c: float,
) -> dict[float, list[dict]]:
    """Re-simulate saved params; attach physics metrics for plotting."""
    from charging_opt.lifetime_reward import aggregate_lifetime_reward

    payload = json.loads(json_path.read_text())
    results_by_omega: dict[float, list[dict]] = {}
    for omega_str, entries in payload["results_by_omega"].items():
        omega = float(omega_str)
        results_by_omega[omega] = []
        for entry in entries:
            params = ProfileParams.from_dict(entry["params"])
            family = get_family(params.family_id)
            session = sim.simulate_params(start, params, family=family)
            _, metrics = aggregate_lifetime_reward(
                session,
                soc_target=soc_target,
                max_duration_min=max_duration_min,
                weights=weights,
                objective_mode=objective_mode,
                v_ref_stress=v_ref_stress,
                t_comfort_c=t_comfort_c,
            )
            results_by_omega[omega].append(
                _metrics_to_run_record(
                    omega,
                    entry["family_id"],
                    entry.get("family_label", family.label),
                    entry["params"],
                    metrics,
                    float(entry.get("loss", metrics.get("loss", 0))),
                )
            )
    return results_by_omega


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
    )
    p.add_argument(
        "--omegas",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    p.add_argument("--n_calls", type=int, default=30)
    p.add_argument("--n_initial", type=int, default=8)
    p.add_argument("--max_minutes", type=int, default=150)
    p.add_argument("--max_duration_min", type=float, default=105.0)
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--soc", type=float, default=None)
    p.add_argument("--v0", type=float, default=None)
    p.add_argument("--t0", type=float, default=None)
    p.add_argument("--age", type=float, default=0.0)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--acq_func", default="PI", choices=["EI", "PI", "LCB"])
    p.add_argument("--age_conditioning", action="store_true")
    p.add_argument("--age_points", default=",".join(str(a) for a in DEFAULT_AGE_POINTS))
    p.add_argument("--age_weights", default=",".join(str(w) for w in DEFAULT_AGE_WEIGHTS))
    p.add_argument(
        "--plots_only",
        action="store_true",
        help="Re-score saved JSON and regenerate chebyshev_pareto_front.png only",
    )
    p.add_argument("--thermal_derating", action="store_true")
    p.add_argument("--thermal_loss", action="store_true")
    p.add_argument("--thermal_derate_comfort_c", type=float, default=33.0)
    add_objective_args(p)
    args = p.parse_args()

    weights, objective_mode, refs = objective_from_args(args)
    if objective_mode == "chebyshev":
        objective_mode = "composite"
    constraints = {
        "objective_mode": objective_mode,
        "thermal_derating": args.thermal_derating,
        "thermal_loss": args.thermal_loss,
    }
    _, deg_key, deg_label = resolve_pareto_config(constraints)

    omegas = _parse_float_list(args.omegas)
    age_points = _parse_float_list(args.age_points)
    age_weights = _parse_float_list(args.age_weights)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "chebyshev_sweep_results.json"

    if args.soc is not None:
        start = {
            "soc": args.soc,
            "v0": args.v0 if args.v0 is not None else 3.7,
            "t0": args.t0 if args.t0 is not None else 25.0,
            "age": args.age,
            "prev_i": 0.0,
        }
    elif json_path.is_file():
        start = json.loads(json_path.read_text())["config"]["start_state"]
    else:
        cfg = load_config(args.config)
        series = load_battery_series(cfg["data"]["matlab_dir"], args.cell, step_mode="all")
        ocv = load_ocv_curve(ROOT / args.ocv_curve)
        states = extract_rest_states(series, ocv, max_states=3000)
        start = pick_start_state(states)

    bdt_path = resolve_bdt_ckpt(args.bdt_ckpt, root=ROOT)
    thermal_controller = None
    if args.thermal_derating or args.thermal_loss:
        thermal_controller = ThermalDeratingController(t_comfort_c=args.thermal_derate_comfort_c)

    sim = ProfileSimulator(
        bdt_path=bdt_path,
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=args.max_minutes,
        soc_target=args.soc_target,
        thermal_controller=thermal_controller if args.thermal_derating else None,
    )

    if args.plots_only:
        if not json_path.is_file():
            raise SystemExit(f"Missing {json_path} — run sweep first or omit --plots_only")
        print(f"Re-scoring {json_path} with objective={objective_mode}, thermal={args.thermal_derating}")
        results_by_omega = rescore_chebyshev_json(
            json_path, sim, start,
            soc_target=args.soc_target,
            max_duration_min=args.max_duration_min,
            weights=weights,
            objective_mode=objective_mode,
            v_ref_stress=refs["v_ref_stress"],
            t_comfort_c=refs["t_comfort_c"],
        )
        plot_chebyshev_pareto(
            results_by_omega,
            out_dir / "chebyshev_pareto_front.png",
            degradation_key=deg_key,
            degradation_label=deg_label,
            objective_mode=objective_mode,
            thermal=args.thermal_derating or args.thermal_loss,
        )
        return

    bo_objective: str = objective_mode
    print(f"\nChebyshev sweep: {len(omegas)} omegas × {len(args.families)} families")
    print(f"  objective={bo_objective}  degradation axis={deg_label}")
    if args.thermal_derating or args.thermal_loss:
        print(f"  thermal: derating={args.thermal_derating}  loss={args.thermal_loss}")

    results_by_omega: dict[float, list[dict]] = {omega: [] for omega in omegas}
    all_runs: List[dict] = []

    for omega in omegas:
        print(f"\n{'─' * 60}\n  omega = {omega:.1f}\n{'─' * 60}")
        for fid in args.families:
            family = get_family(fid)
            opt = FamilyBayesianOptimizer(
                sim,
                family,
                start,
                soc_target=args.soc_target,
                max_duration_min=args.max_duration_min,
                weights=weights,
                objective_mode=bo_objective,
                v_ref_stress=refs["v_ref_stress"],
                t_comfort_c=refs["t_comfort_c"],
                acq_func=args.acq_func,
                use_age_conditioning=args.age_conditioning,
                age_points=age_points,
                age_weights=age_weights,
                chebyshev_omega=omega,
                thermal_controller=thermal_controller if args.thermal_loss else None,
                random_state=42,
            )
            result = opt.optimize(n_calls=args.n_calls, n_initial_points=args.n_initial)
            m = result.best_metrics
            run_record = _metrics_to_run_record(
                omega, fid, family.label,
                result.best_params.to_dict(), m, result.best_loss,
            )
            results_by_omega[omega].append(run_record)
            all_runs.append(run_record)
            y = _run_y_value(run_record, deg_key)
            status = "✓" if run_record["feasible"] else "✗"
            print(
                f"    {family.label:26s} {status}  "
                f"dur={run_record.get('duration_min', float('nan')):.1f} min  "
                f"{deg_label}={_fmt_deg(y, deg_key)}"
            )

    payload = {
        "config": {
            "omegas": omegas,
            "families": args.families,
            "n_calls": args.n_calls,
            "acq_func": args.acq_func,
            "objective_mode": objective_mode,
            "thermal_derating": args.thermal_derating,
            "thermal_loss": args.thermal_loss,
            "thermal_derate_comfort_c": args.thermal_derate_comfort_c,
            "degradation_key": deg_key,
            "degradation_label": deg_label,
            "age_conditioning": args.age_conditioning,
            "age_points": age_points,
            "age_weights": age_weights,
            "start_state": start,
        },
        "results_by_omega": {str(k): v for k, v in results_by_omega.items()},
        "all_runs": all_runs,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    plot_chebyshev_pareto(
        results_by_omega,
        out_dir / "chebyshev_pareto_front.png",
        degradation_key=deg_key,
        degradation_label=deg_label,
        objective_mode=objective_mode,
        thermal=args.thermal_derating or args.thermal_loss,
    )

    feasible_runs = [r for r in all_runs if r["feasible"]]
    print(f"\n{'=' * 60}\n  CHEBYSHEV SWEEP COMPLETE\n{'=' * 60}")
    print(f"  Feasible: {len(feasible_runs)} / {len(all_runs)}")
    if feasible_runs:
        fastest = min(feasible_runs, key=lambda r: r["duration_min"])
        lifetime = min(
            feasible_runs,
            key=lambda r: _run_y_value(r, deg_key) or float("inf"),
        )
        yf = _run_y_value(fastest, deg_key)
        yl = _run_y_value(lifetime, deg_key)
        print(
            f"  Fastest  : {fastest['family_label']:26s} "
            f"dur={fastest['duration_min']:.1f} min  {deg_label}={_fmt_deg(yf, deg_key)}"
        )
        print(
            f"  Lifetime : {lifetime['family_label']:26s} "
            f"dur={lifetime['duration_min']:.1f} min  {deg_label}={_fmt_deg(yl, deg_key)}"
        )
    print(f"  Results -> {json_path}\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
