#!/usr/bin/env python3
"""Ambient temperature sensitivity sweep (thermal Level 2)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig_{os.getenv('USER', 'default')}")

from charging_opt.artifacts import CANONICAL, resolve_bdt_ckpt
from charging_opt.charging_profile_family import DEFAULT_FAMILY_IDS
from charging_opt.family_optimizer import optimize_families, save_family_results
from charging_opt.family_reporting import export_family_artifacts, rehydrate_results_from_json
from charging_opt.io_utils import current_user, resolve_stage3_family_dirs, user_stage3_root
from charging_opt.objective_cli import add_objective_args, objective_from_args
from charging_opt.profile_simulator import ProfileSimulator
from charging_opt.thermal_management import ambient_sensitivity_states, compare_ambient_results


def _plot_ambient_summary(summary: dict, out_png: Path) -> None:
    """Cross-temperature comparison of best feasible profile per T0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temps = sorted(float(t) for t in summary)
    if not temps:
        return
    losses = [summary[str(t)]["best_loss"] for t in temps]
    durs = [summary[str(t)]["duration_min"] for t in temps]
    seis = [summary[str(t)]["sei_per_pct_soc"] for t in temps]
    peaks = [summary[str(t)]["peak_temperature"] for t in temps]
    labels = [summary[str(t)]["best_family"] for t in temps]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(temps, losses, "o-", color="#2166ac")
    axes[0].set_xlabel("Initial temperature T0 (°C)")
    axes[0].set_ylabel("Best physics loss")
    axes[0].set_title("Objective vs ambient T0")
    axes[0].grid(True, alpha=0.35)

    axes[1].plot(temps, durs, "s-", color="#d6604d", label="Duration (min)")
    ax1b = axes[1].twinx()
    ax1b.plot(temps, seis, "^--", color="#1b7837", label="SEI/ΔSoC")
    axes[1].set_xlabel("Initial temperature T0 (°C)")
    axes[1].set_ylabel("Duration (min)", color="#d6604d")
    ax1b.set_ylabel("SEI / ΔSoC", color="#1b7837")
    axes[1].set_title("Speed & degradation vs T0")
    axes[1].grid(True, alpha=0.35)

    axes[2].bar([str(int(t)) for t in temps], peaks, color="#762a83", alpha=0.85)
    axes[2].set_xlabel("T0 (°C)")
    axes[2].set_ylabel("Peak temperature (°C)")
    axes[2].set_title("Peak T during charge")
    for i, (t, fam) in enumerate(zip(temps, labels)):
        axes[0].annotate(fam, (t, losses[i]), fontsize=8, xytext=(4, 4), textcoords="offset points")

    fig.suptitle("Ambient Temperature Sensitivity — Best Profile per T0", fontsize=12)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument("--families", default="cccv,pulsed,adaptive_two_step")
    p.add_argument("--ambient_temps", default="15,25,35")
    p.add_argument("--n_calls", type=int, default=20)
    p.add_argument("--n_initial", type=int, default=6)
    p.add_argument("--max_duration_min", type=float, default=105.0)
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--soc", type=float, default=0.15)
    p.add_argument("--v0", type=float, default=3.711)
    p.add_argument("--t0", type=float, default=24.7)
    p.add_argument("--age", type=float, default=0.0)
    p.add_argument("--out_dir", default=None)
    add_objective_args(p)
    p.add_argument("--thermal_derating", action="store_true")
    p.add_argument("--thermal_derate_comfort_c", type=float, default=33.0)
    p.add_argument("--thermal_loss", action="store_true")
    p.add_argument("--thermal_w_comfort", type=float, default=0.5)
    p.add_argument("--thermal_w_hard", type=float, default=5.0)
    p.add_argument("--acq_func", default="PI", choices=["EI", "PI", "LCB"])
    p.add_argument(
        "--plots_only",
        action="store_true",
        help="Regenerate PNGs/CSV from saved JSON (no BO re-run)",
    )
    args = p.parse_args()

    weights, objective_mode, refs = objective_from_args(args)
    ambient_temps = [float(x.strip()) for x in args.ambient_temps.split(",") if x.strip()]
    family_ids = [f.strip() for f in args.families.split(",") if f.strip()]
    base_state = {"soc": args.soc, "v0": args.v0, "t0": args.t0, "age": args.age, "prev_i": 0.0}

    out_base = Path(args.out_dir).resolve() if args.out_dir else user_stage3_root(ROOT, current_user()) / "ambient_sensitivity"
    bdt_path = resolve_bdt_ckpt(args.bdt_ckpt, root=ROOT)

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
        max_minutes=150,
        thermal_controller=thermal_controller if args.thermal_derating else None,
    )

    results_by_temp = {}
    if args.plots_only:
        from charging_opt.family_reporting import load_results_payload

        for t_amb in ambient_temps:
            temp_dir = out_base / f"T{t_amb:.0f}"
            json_path = temp_dir / "models" / "family_optimization_results.json"
            if not json_path.is_file():
                print(f"SKIP T{t_amb:.0f}: missing {json_path}")
                continue
            state = ambient_sensitivity_states(base_state, [t_amb])[0]
            data = load_results_payload(json_path)
            models_dir, plots_dir = resolve_stage3_family_dirs(ROOT, out_dir=temp_dir)
            results = rehydrate_results_from_json(
                data, sim, state,
                soc_target=args.soc_target,
                max_duration_min=args.max_duration_min,
                family_ids=family_ids,
            )
            export_family_artifacts(
                results, plots_dir, csv_path=models_dir / "comparison_table.csv",
            )
            print(f"T{t_amb:.0f}: plots -> {plots_dir}/")
            results_by_temp[t_amb] = results
    else:
        for state, t_amb in zip(ambient_sensitivity_states(base_state, ambient_temps), ambient_temps):
            print(f"\n=== Ambient T0 = {t_amb:.1f}°C ===")
            temp_dir = out_base / f"T{t_amb:.0f}"
            models_dir, plots_dir = resolve_stage3_family_dirs(ROOT, out_dir=temp_dir)
            results = optimize_families(
                sim, state, family_ids,
                n_calls=args.n_calls,
                n_initial_points=args.n_initial,
                max_duration_min=args.max_duration_min,
                weights=weights,
                objective_mode=objective_mode,
                v_ref_stress=refs["v_ref_stress"],
                t_comfort_c=refs["t_comfort_c"],
                acq_func=args.acq_func,
                thermal_controller=thermal_controller if args.thermal_loss else None,
                thermal_w_comfort=args.thermal_w_comfort,
                thermal_w_hard=args.thermal_w_hard,
            )
            save_family_results(
                results, models_dir,
                initial_state=state,
                bdt_path=str(bdt_path),
                max_duration_min=args.max_duration_min,
                repo_root=ROOT,
            )
            export_family_artifacts(
                results,
                plots_dir,
                csv_path=models_dir / "comparison_table.csv",
            )
            print(f"  Plots -> {plots_dir}/")
            results_by_temp[t_amb] = results

    if not results_by_temp:
        print("No ambient results to summarize.")
        return

    summary = compare_ambient_results(results_by_temp)
    # JSON keys are floats; normalize for file I/O
    summary_serializable = {str(k): v for k, v in summary.items()}
    out_base.mkdir(parents=True, exist_ok=True)
    summary_path = out_base / "ambient_sensitivity_summary.json"
    summary_path.write_text(json.dumps(summary_serializable, indent=2, default=float) + "\n")
    _plot_ambient_summary(summary_serializable, out_base / "ambient_sensitivity_comparison.png")
    print(f"\nSummary -> {summary_path}")
    print(f"Comparison plot -> {out_base / 'ambient_sensitivity_comparison.png'}")


if __name__ == "__main__":
    main()
