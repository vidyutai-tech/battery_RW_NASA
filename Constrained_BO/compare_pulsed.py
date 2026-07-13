#!/usr/bin/env python3
"""Compare random vs optimized pulsed charging profile rewards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Constrained_BO.config import CellConfig, get_cell_config
from Constrained_BO.objective import evaluate_session
from Constrained_BO.profile_catalog import ProfileBounds
from Constrained_BO.profiles import PulsedFamily, ProfileParams, set_profile_bounds
from Constrained_BO.simulator import ChargingSimulator

_COMPARE_OUTPUTS = ("pulsed_comparison.json", "pulsed_comparison.png")


def _is_writable_out_dir(d: Path) -> bool:
    """True if comparison outputs can be created or overwritten in ``d``."""
    import os

    d = Path(d)
    if d.exists():
        if not os.access(d, os.W_OK):
            return False
        for name in _COMPARE_OUTPUTS:
            f = d / name
            if f.exists() and not os.access(f, os.W_OK):
                return False
        return True
    parent = d.parent
    return parent.exists() and os.access(parent, os.W_OK)


def _resolve_compare_out_dir(
    requested: Path | None,
    default: Path,
    cell_id: str,
) -> Path:
    """Pick a writable directory for comparison artifacts."""
    import getpass

    if requested is not None:
        return Path(requested)

    default = Path(default)
    if _is_writable_out_dir(default):
        return default

    user = getpass.getuser()
    fallback = Path(__file__).resolve().parent / "results" / "_compare" / user / cell_id
    print(
        f"Warning: {default} has root-owned or read-only comparison outputs; "
        f"writing to {fallback} instead.\n"
        f"  To reuse {default}, run: sudo chown -R $USER {default}"
    )
    return fallback


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    import os
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(
            f"Cannot write {path} (permission denied). "
            f"Re-run without --out-dir to auto-fallback, or run: "
            f"sudo chown -R $USER {path.parent}"
        )
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _charge_current_plot(i_a: np.ndarray) -> np.ndarray:
    return -np.asarray(i_a, dtype=np.float64)


def _cell_from_meta(meta: Dict[str, Any]) -> CellConfig:
    """Rebuild CellConfig from saved results metadata."""
    cell_id = meta["cell"]
    cell = get_cell_config(cell_id)
    overrides: Dict[str, Any] = {}
    if meta.get("constraint_mode") == "energy":
        overrides["energy_fraction"] = meta.get("energy_fraction")
        overrides["soc_target"] = meta.get("soc_target")
    else:
        overrides["soc_target"] = meta.get("soc_target")
    if meta.get("max_duration_min") is not None:
        overrides["max_duration_min"] = meta["max_duration_min"]
    if meta.get("v_nom") is not None:
        overrides["v_nom"] = meta["v_nom"]
    if meta.get("decision_interval_s") is not None:
        overrides["decision_interval_s"] = meta["decision_interval_s"]
        overrides["auto_decision_interval"] = False
    cell = cell.with_run_overrides(**overrides)
    cell.start_state = dict(meta["start_state"])
    if meta.get("profile_bounds"):
        cell.profile_bounds = ProfileBounds(**meta["profile_bounds"])
    return cell


def _params_from_dict(d: Dict[str, Any]) -> ProfileParams:
    vals = {k: v for k, v in d.items() if k != "family_id"}
    return PulsedFamily.from_dict(vals)


def _evaluate(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    params: ProfileParams,
    *,
    w_time: float,
    w_temperature: float,
) -> Tuple[Dict, Dict]:
    family = PulsedFamily()
    session = simulator.simulate(initial_state, params, family=family)
    _, metrics = evaluate_session(
        session, w_time=w_time, w_temperature=w_temperature,
    )
    return session, metrics


def _plot_comparison(
    *,
    cell_id: str,
    optimized: Dict[str, Any],
    random_pick: Dict[str, Any],
    random_stats: Dict[str, float],
    baseline_label: str,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.35, wspace=0.25)

    # Reward bar chart
    ax_bar = fig.add_subplot(gs[0, :])
    labels = ["Optimized pulsed", f"Random pulsed ({baseline_label})"]
    totals = [optimized["metrics"]["total_reward"], random_pick["metrics"]["total_reward"]]
    times = [optimized["metrics"]["time_reward"], random_pick["metrics"]["time_reward"]]
    temps = [optimized["metrics"]["temperature_reward"], random_pick["metrics"]["temperature_reward"]]
    x = np.arange(len(labels))
    w = 0.25
    ax_bar.bar(x - w, totals, w, label="Total reward", color="#2563eb")
    ax_bar.bar(x, times, w, label="Time reward", color="#f59e0b")
    ax_bar.bar(x + w, temps, w, label="Temperature reward", color="#22c55e")
    ax_bar.axhline(0, color="gray", lw=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylabel("Reward (higher is better)")
    ax_bar.set_title(
        f"Pulsed profile reward comparison — {cell_id}\n"
        f"Random baseline: n={int(random_stats['n'])} "
        f"({int(random_stats.get('n_feasible', 0))} feasible)  "
        f"feasible mean={random_stats.get('feasible_mean_total_reward', random_stats['mean_total_reward']):.2f}  "
        f"feasible median={random_stats.get('feasible_median_total_reward', random_stats['median_total_reward']):.2f}",
        fontweight="bold",
    )
    ax_bar.legend(loc="lower right", ncol=3, fontsize=9)
    ax_bar.grid(True, axis="y", alpha=0.3)

    cases = [
        ("Optimized pulsed", optimized),
        ("Random pulsed (baseline)", random_pick),
    ]
    row_labels = ["Current (A)", "Voltage (V)", "SoC (%)", "Temperature (°C)"]

    for col, (title, case) in enumerate(cases):
        session = case["session"]
        metrics = case["metrics"]
        t_min = session["time_s"] / 60.0

        for row, (ylabel, data, color) in enumerate([
            (row_labels[0], _charge_current_plot(session["current_a"]), "C0"),
            (row_labels[1], session["voltage_v"], "C1"),
            (row_labels[2], session["soc"] * 100.0, "C2"),
            (row_labels[3], session["temperature_c"], "C3"),
        ]):
            ax = fig.add_subplot(gs[row + 1, col])
            ax.plot(t_min, data, color=color, lw=1.2)
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(
                    f"{title}\n"
                    f"reward={metrics['total_reward']:.2f}  "
                    f"time={metrics['duration_min']:.1f} min  "
                    f"peak T={metrics['peak_temperature']:.1f} °C",
                    fontsize=9,
                )
            if row == 3:
                ax.set_xlabel("Time (min)")
                ax.axhspan(15, 35, color="green", alpha=0.08)

    fig.suptitle("Random vs optimized pulsed charging", fontsize=12, y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare random vs optimized pulsed profiles")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "results/hima/RW9/constrained_bo_results.json",
        help="Existing optimization results JSON (for optimized params + meta)",
    )
    parser.add_argument("--n-random", type=int, default=80, help="Number of random pulsed samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: same folder as --results)",
    )
    args = parser.parse_args()

    with open(args.results) as f:
        payload = json.load(f)
    meta = payload["meta"]
    pulsed = payload["families"]["pulsed"]
    optimized_params = _params_from_dict(pulsed["best_params"])

    cell = _cell_from_meta(meta)
    if cell.profile_bounds is not None:
        set_profile_bounds(cell.profile_bounds)

    w_time = float(meta["reward_weights"]["w_time"])
    w_temperature = float(meta["reward_weights"]["w_temperature"])
    simulator = ChargingSimulator.from_cell(cell, device=args.device)
    simulator.decision_interval_info = meta.get("decision_interval_selection", {})

    rng = np.random.default_rng(args.seed)
    family = PulsedFamily()
    random_results: List[Dict[str, Any]] = []
    for _ in range(args.n_random):
        params = family.sample_random(rng)
        session, metrics = _evaluate(
            simulator,
            cell.start_state,
            params,
            w_time=w_time,
            w_temperature=w_temperature,
        )
        random_results.append({
            "params": params.to_dict(),
            "session": session,
            "metrics": metrics,
        })

    opt_session, opt_metrics = _evaluate(
        simulator,
        cell.start_state,
        optimized_params,
        w_time=w_time,
        w_temperature=w_temperature,
    )
    optimized = {
        "params": optimized_params.to_dict(),
        "session": opt_session,
        "metrics": opt_metrics,
    }

    rewards = [r["metrics"]["total_reward"] for r in random_results]
    feasible_results = [r for r in random_results if r["metrics"]["feasible"]]
    feasible_rewards = [r["metrics"]["total_reward"] for r in feasible_results]

    def _median_case(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            raise ValueError("No random profiles to compare")
        rs = [r["metrics"]["total_reward"] for r in results]
        idx = int(np.argsort(rs)[len(rs) // 2])
        return results[idx]

    random_pick = _median_case(feasible_results if feasible_results else random_results)
    # For temperature comparison, typical "random profile" = median of all random draws.
    random_temp_baseline = _median_case(random_results)
    random_stats = {
        "n": float(args.n_random),
        "n_feasible": float(len(feasible_results)),
        "mean_total_reward": float(np.mean(rewards)),
        "median_total_reward": float(np.median(rewards)),
        "std_total_reward": float(np.std(rewards)),
        "best_total_reward": float(np.max(rewards)),
        "worst_total_reward": float(np.min(rewards)),
        "mean_duration_min": float(np.mean([r["metrics"]["duration_min"] for r in random_results])),
        "mean_peak_temperature": float(np.mean([r["metrics"]["peak_temperature"] for r in random_results])),
    }
    if feasible_rewards:
        random_stats.update({
            "feasible_mean_total_reward": float(np.mean(feasible_rewards)),
            "feasible_median_total_reward": float(np.median(feasible_rewards)),
            "feasible_best_total_reward": float(np.max(feasible_rewards)),
            "feasible_worst_total_reward": float(np.min(feasible_rewards)),
            "feasible_mean_duration_min": float(np.mean([
                r["metrics"]["duration_min"] for r in feasible_results
            ])),
        })

    baseline_label = "feasible median" if feasible_results else "median (incl. infeasible)"
    improvement_vs_median = opt_metrics["total_reward"] - random_pick["metrics"]["total_reward"]
    improvement_vs_mean = opt_metrics["total_reward"] - random_stats.get(
        "feasible_mean_total_reward", random_stats["mean_total_reward"],
    )
    temp_improvement_vs_random = (
        opt_metrics["temperature_reward"] - random_temp_baseline["metrics"]["temperature_reward"]
    )

    out_dir = _resolve_compare_out_dir(args.out_dir, args.results.parent, meta["cell"])
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "cell": meta["cell"],
        "constraint_mode": meta["constraint_mode"],
        "energy_fraction": meta.get("energy_fraction"),
        "soc_target": meta.get("soc_target"),
        "n_random": args.n_random,
        "seed": args.seed,
        "optimized": {
            "params": optimized["params"],
            "metrics": {k: v for k, v in opt_metrics.items() if k != "reward_weights"},
        },
        "random_baseline_label": baseline_label,
        "random_baseline_case": {
            "params": random_pick["params"],
            "metrics": {k: v for k, v in random_pick["metrics"].items() if k != "reward_weights"},
        },
        "random_temp_baseline_case": {
            "params": random_temp_baseline["params"],
            "metrics": {
                k: v for k, v in random_temp_baseline["metrics"].items() if k != "reward_weights"
            },
        },
        "random_stats": random_stats,
        "improvement": {
            "total_reward_vs_random_baseline": float(improvement_vs_median),
            "total_reward_vs_random_feasible_mean": float(improvement_vs_mean),
            "temperature_reward_vs_random_median": float(temp_improvement_vs_random),
            "duration_min_saved_vs_baseline": float(
                random_pick["metrics"]["duration_min"] - opt_metrics["duration_min"]
            ),
        },
    }

    json_path = out_dir / "pulsed_comparison.json"
    _write_json(json_path, summary)

    png_path = out_dir / "pulsed_comparison.png"
    _plot_comparison(
        cell_id=meta["cell"],
        optimized=optimized,
        random_pick=random_pick,
        random_stats=random_stats,
        baseline_label=baseline_label,
        out_path=png_path,
    )

    print(f"\n=== Pulsed profile reward comparison ({meta['cell']}) ===")
    print(f"Optimized:  total_reward={opt_metrics['total_reward']:7.2f}  "
          f"time={opt_metrics['time_reward']:6.2f}  "
          f"temp={opt_metrics['temperature_reward']:.3f}  "
          f"dT_pk_ch={opt_metrics.get('dT_peak_charge', 0):.2f}°C  "
          f"dT/dt_max={opt_metrics.get('dT_dt_max_charge', 0):.3f}°C/s  "
          f"duration={opt_metrics['duration_min']:.1f} min")
    print(f"Random ({baseline_label} of {args.n_random}, {len(feasible_results)} feasible):  "
          f"total_reward={random_pick['metrics']['total_reward']:7.2f}  "
          f"time={random_pick['metrics']['time_reward']:6.2f}  "
          f"temp={random_pick['metrics']['temperature_reward']:.3f}  "
          f"dT_pk_ch={random_pick['metrics'].get('dT_peak_charge', 0):.2f}°C  "
          f"dT/dt_max={random_pick['metrics'].get('dT_dt_max_charge', 0):.3f}°C/s  "
          f"duration={random_pick['metrics']['duration_min']:.1f} min  "
          f"feasible={random_pick['metrics']['feasible']}")
    if feasible_rewards:
        print(f"Feasible random mean:  total_reward={random_stats['feasible_mean_total_reward']:.2f}  "
              f"range=[{random_stats['feasible_worst_total_reward']:.2f}, "
              f"{random_stats['feasible_best_total_reward']:.2f}]")
    print(f"Random temp baseline (median of all {args.n_random}):  "
          f"temp_reward={random_temp_baseline['metrics']['temperature_reward']:.3f}  "
          f"dT_pk_ch={random_temp_baseline['metrics'].get('dT_peak_charge', 0):.2f}°C  "
          f"dT/dt_max={random_temp_baseline['metrics'].get('dT_dt_max_charge', 0):.3f}°C/s")
    print(f"\nImprovement (optimized − random baseline): {improvement_vs_median:+.2f} total reward")
    print(f"Temperature improvement (optimized − random median): {temp_improvement_vs_random:+.3f}")
    print(f"Wrote {json_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
