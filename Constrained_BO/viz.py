"""Plot reward curves and best charging profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Constrained_BO.objective import (
    TEMP_HIGH_C,
    TEMP_LOW_C,
    TEMP_MAX_C,
    TIME_ZERO_AT_S,
    temperature_reward,
    time_reward,
)
from Constrained_BO.profiles import DEFAULT_FAMILIES, get_family

REWARDS_OUT_DIR = Path(__file__).resolve().parent / "data" / "rewards"


def _charge_current_plot(i_a: np.ndarray) -> np.ndarray:
    """Display magnitude (positive = charge)."""
    return -np.asarray(i_a, dtype=np.float64)


def plot_temperature_reward(out_path: Path | None = None) -> plt.Figure:
    t = np.linspace(0, TEMP_MAX_C, 500)
    r = np.array([temperature_reward(x) for x in t])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, r, color="#2563eb", lw=2.2)
    band_label = f"Optimal band ({TEMP_LOW_C:.0f}–{TEMP_HIGH_C:.0f} °C)"
    ax.axvspan(TEMP_LOW_C, TEMP_HIGH_C, color="#22c55e", alpha=0.12, label=band_label)
    ax.axhline(1.5, color="gray", ls="--", lw=0.9, alpha=0.6)
    ax.axvline(TEMP_LOW_C, color="#22c55e", ls=":", lw=1, alpha=0.7)
    ax.axvline(TEMP_HIGH_C, color="#22c55e", ls=":", lw=1, alpha=0.7)
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Transformed reward", fontsize=11)
    ax.set_title("Temperature reward", fontsize=12, fontweight="bold")
    ax.set_xlim(0, TEMP_MAX_C)
    ax.set_ylim(-2.4, 1.7)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", fontsize=9)
    note = (
        f"Reward = {1.5:.1f} for {TEMP_LOW_C:.0f} °C ≤ T ≤ {TEMP_HIGH_C:.0f} °C.\n"
        "Linear penalty outside this range (continues below 0)."
    )
    ax.text(
        0.03, 0.05, note, transform=ax.transAxes, fontsize=8.5,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8dc", edgecolor="#d4a84b", alpha=0.95),
    )
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_time_reward(out_path: Path | None = None) -> plt.Figure:
    t_max = 2400.0
    t = np.linspace(0, t_max, 400)
    r = np.array([time_reward(x) for x in t])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, r, color="#2563eb", lw=2.2)
    ax.axhline(0.0, color="gray", ls="--", lw=0.9, alpha=0.6)
    ax.scatter([0, TIME_ZERO_AT_S], [1.5, 0.0], color="#2563eb", s=40, zorder=5)
    ax.annotate("(0 s, 1.5)", (0, 1.5), textcoords="offset points", xytext=(8, -12), fontsize=9)
    ax.annotate(f"({TIME_ZERO_AT_S:.0f} s, 0)", (TIME_ZERO_AT_S, 0), textcoords="offset points", xytext=(-55, 10), fontsize=9)
    ax.set_xlabel("Time elapsed since charging started (s)", fontsize=11)
    ax.set_ylabel("Transformed reward", fontsize=11)
    ax.set_title("Time reward", fontsize=12, fontweight="bold")
    ax.set_xlim(0, t_max)
    ax.set_ylim(-10.0, 1.65)
    ax.grid(True, alpha=0.35)
    note = "R(t) = 1.5 − 0.01·t  (continues negative after 150 s)"
    ax.text(0.55, 0.92, note, transform=ax.transAxes, fontsize=9, ha="center",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef4ff", edgecolor="#93b4e8", alpha=0.95))
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_rewards_combined(out_path: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    t_temp = np.linspace(0, TEMP_MAX_C, 500)
    r_temp = np.array([temperature_reward(x) for x in t_temp])
    axes[0].plot(t_temp, r_temp, color="#2563eb", lw=2.2)
    axes[0].axvspan(TEMP_LOW_C, TEMP_HIGH_C, color="#22c55e", alpha=0.12)
    axes[0].axhline(1.5, color="gray", ls="--", lw=0.9, alpha=0.6)
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Transformed reward")
    axes[0].set_title("Temperature reward", fontweight="bold")
    axes[0].set_xlim(0, TEMP_MAX_C)
    axes[0].set_ylim(-2.4, 1.7)
    axes[0].grid(True, alpha=0.35)

    t_time = np.linspace(0, 2400.0, 400)
    r_time = np.array([time_reward(x) for x in t_time])
    axes[1].plot(t_time, r_time, color="#2563eb", lw=2.2)
    axes[1].axhline(0.0, color="gray", ls="--", lw=0.9, alpha=0.6)
    axes[1].scatter([0, TIME_ZERO_AT_S], [1.5, 0.0], color="#2563eb", s=35)
    axes[1].set_xlabel("Time elapsed since charging started (s)")
    axes[1].set_ylabel("Transformed reward")
    axes[1].set_title("Time reward", fontweight="bold")
    axes[1].set_xlim(0, 2400.0)
    axes[1].set_ylim(-10.0, 1.65)
    axes[1].grid(True, alpha=0.35)

    fig.suptitle("Constrained_BO reward functions (temp + time)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_hybrid_reward_components(
    out_path: Path | None = None,
    metrics: Optional[Dict] = None,
) -> plt.Figure:
    """Bar chart of hybrid reward components (SoC, capacity-loss index parts, time, total)."""
    if metrics is None:
        # Illustrative synthetic breakdown for --rewards-only
        components = {
            "SoC reward": 0.60,
            "Calendar loss index": -0.02,
            "Cyclic loss index": -0.15,
            "Time penalty": -0.08,
            "Total reward": 0.35,
        }
    else:
        w_q = float(metrics.get("reward_weights", {}).get("w_qloss", 1.0))
        components = {
            "SoC reward": float(metrics.get("soc_reward", 0.0)),
            "Calendar loss index": -float(metrics.get("qloss_calendar", 0.0)) * w_q,
            "Cyclic loss index": -float(metrics.get("qloss_cyclic", 0.0)) * w_q,
            "Time penalty": -float(metrics.get("time_penalty", 0.0)),
            "Total reward": float(metrics.get("total_reward", 0.0)),
        }

    labels = list(components.keys())
    values = list(components.values())
    colors = ["#2563eb", "#f59e0b", "#ef4444", "#8b5cf6", "#16a34a"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="#1f2937", lw=0.6)
    ax.axhline(0.0, color="gray", lw=0.9)
    ax.set_ylabel("Reward contribution")
    ax.set_title("Hybrid degradation reward components", fontweight="bold")
    ax.text(
        0.5, -0.22,
        "Loss index = Relative Capacity-Loss Index (dimensionless ranking signal, not calibrated % capacity fade)",
        transform=ax.transAxes, fontsize=7.5, ha="center", color="#5b6573", style="italic",
    )
    ax.grid(True, axis="y", alpha=0.35)
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, val),
            xytext=(0, 4 if val >= 0 else -12),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reward_curves(out_dir: Path | None = None) -> None:
    out_dir = out_dir or REWARDS_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("temperature_reward.png", plot_temperature_reward),
        ("time_reward.png", plot_time_reward),
        ("rewards_combined.png", plot_rewards_combined),
        ("hybrid_reward_components.png", plot_hybrid_reward_components),
    ]:
        fig = fn(out_dir / name)
        plt.close(fig)
        print(f"Wrote {out_dir / name}")


def plot_best_profiles(
    family_results: Dict[str, Dict],
    *,
    cell_id: str,
    soc_target: float = 0.95,
    soc_start: float = 0.20,
    out_path: Optional[Path] = None,
    title_suffix: str = "",
    family_order: Optional[List[str]] = None,
) -> plt.Figure:
    order = family_order or DEFAULT_FAMILIES
    families = [
        fid for fid in order
        if fid in family_results and family_results[fid].get("best_session")
    ]
    n_cols = len(families)
    if n_cols == 0:
        raise ValueError("No sessions to plot")

    fig, axes = plt.subplots(4, n_cols, figsize=(3.2 * n_cols, 10), squeeze=False)
    row_labels = ["Current (A)", "Voltage (V)", "SoC (%)", "Temperature (°C)"]

    for col, fid in enumerate(families):
        res = family_results[fid]
        session = res["best_session"]
        metrics = res["best_metrics"]
        label = get_family(fid).label
        t_min = session["time_s"] / 60.0

        header = (
            f"{label}\n"
            f"{metrics['duration_min']:.0f} min | "
            f"loss={metrics['loss']:.1f}\n"
            f"{'feasible' if metrics['feasible'] else 'infeasible'}"
        )
        if metrics.get("constraint_mode") == "energy":
            header += (
                f"\nE={metrics['energy_delivered_j']:.0f}/"
                f"{metrics['energy_required_j']:.0f} J"
            )
        axes[0, col].set_title(header, fontsize=9)

        axes[0, col].plot(t_min, _charge_current_plot(session["current_a"]), color="C0", lw=1.2)
        axes[1, col].plot(t_min, session["voltage_v"], color="C1", lw=1.2)
        axes[1, col].axhline(4.2, color="gray", ls="--", lw=0.8, alpha=0.7)
        axes[2, col].plot(t_min, session["soc"] * 100.0, color="C2", lw=1.2)
        axes[2, col].axhline(soc_target * 100.0, color="gray", ls="--", lw=0.8, alpha=0.7)
        axes[2, col].axhline(soc_start * 100.0, color="C2", ls=":", lw=0.8, alpha=0.5)
        axes[3, col].plot(t_min, session["temperature_c"], color="C3", lw=1.2)
        axes[3, col].axhline(TEMP_LOW_C, color="green", ls="--", lw=0.8, alpha=0.6)
        axes[3, col].axhline(TEMP_HIGH_C, color="green", ls="--", lw=0.8, alpha=0.6)
        axes[3, col].axhspan(TEMP_LOW_C, TEMP_HIGH_C, color="green", alpha=0.08)

        for row in range(4):
            axes[row, col].grid(True, alpha=0.3)
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row])

    for col in range(n_cols):
        axes[3, col].set_xlabel("Time (min)")

    sup = f"Best charging profiles — {cell_id}  (temp + time reward)"
    if title_suffix:
        sup += f"\n{title_suffix}"
    fig.suptitle(sup, fontsize=11, y=1.01)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def _rest_intervals_min(
    time_s: np.ndarray,
    current_a: np.ndarray,
    *,
    eps_a: float = 0.02,
) -> List[Tuple[float, float]]:
    """Return (t_start_min, t_end_min) spans where |I| ≈ 0 (rest)."""
    t_min = np.asarray(time_s, dtype=np.float64) / 60.0
    rest = np.abs(np.asarray(current_a, dtype=np.float64)) <= eps_a
    if rest.size == 0:
        return []
    spans: List[Tuple[float, float]] = []
    in_rest = False
    start_idx = 0
    for k, is_rest in enumerate(rest):
        if is_rest and not in_rest:
            in_rest = True
            start_idx = k
        elif not is_rest and in_rest:
            in_rest = False
            end_idx = max(start_idx, k - 1)
            spans.append((float(t_min[start_idx]), float(t_min[end_idx])))
    if in_rest:
        spans.append((float(t_min[start_idx]), float(t_min[-1])))
    return spans


def plot_optimized_profile(
    session: Dict,
    metrics: Dict,
    *,
    family_label: str,
    params: Optional[Dict] = None,
    soc_target: float = 0.95,
    soc_start: float = 0.20,
    out_path: Optional[Path] = None,
) -> plt.Figure:
    """Four-panel trajectory plot for a single optimized charging session."""
    t_min = session["time_s"] / 60.0
    i_plot = _charge_current_plot(session["current_a"])
    rest_spans = _rest_intervals_min(session["time_s"], session["current_a"])
    t_end = float(t_min[-1]) if t_min.size else 0.0
    soc_end_pct = float(session["soc"][-1] * 100.0) if session["soc"].size else 0.0

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 10), sharex=True)
    row_labels = ["Current (A)", "Voltage (V)", "SoC (%)", "Temperature (°C)"]

    feasible = metrics.get("feasible", False)
    if metrics.get("reward_mode") == "hybrid_qloss":
        header = (
            f"Optimized charging profile\n"
            f"Family: {family_label}\n"
            f"Reward = {metrics['total_reward']:.3f}  |  "
            f"ΔSoC = {metrics.get('soc_delta', 0):.3f}  |  "
            f"Loss index = {metrics.get('qloss_total', 0):.4g}  |  "
            f"Time = {metrics['duration_min']:.0f} min  |  "
            f"Peak T = {metrics['peak_temperature']:.1f}°C\n"
            f"{'Feasible' if feasible else 'Infeasible'}"
        )
    else:
        header = (
            f"Optimized charging profile\n"
            f"Family: {family_label}\n"
            f"Reward = {metrics['total_reward']:.1f}  |  "
            f"Time = {metrics['duration_min']:.0f} min  |  "
            f"Peak T = {metrics['peak_temperature']:.1f}°C\n"
            f"{'Feasible' if feasible else 'Infeasible'}"
        )
    if metrics.get("constraint_mode") == "energy":
        header += (
            f"  |  E={metrics.get('energy_delivered_j', 0):.0f}/"
            f"{metrics.get('energy_required_j', 0):.0f} J"
        )
    axes[0].set_title(header, fontsize=10, fontweight="bold", loc="left")

    for ax in axes:
        for t0, t1 in rest_spans:
            ax.axvspan(t0, t1, color="#94a3b8", alpha=0.18, zorder=0)
        ax.axvline(t_end, color="#9333ea", ls="--", lw=1.0, alpha=0.85, zorder=1)
        ax.grid(True, alpha=0.3)

    axes[0].plot(t_min, i_plot, color="C0", lw=1.4, drawstyle="steps-post", label="Charge current")
    axes[0].set_ylabel(row_labels[0])
    if rest_spans:
        axes[0].fill_between([], [], color="#94a3b8", alpha=0.35, label="Rest interval")

    axes[1].plot(t_min, session["voltage_v"], color="C1", lw=1.2)
    axes[1].axhline(4.2, color="gray", ls="--", lw=0.8, alpha=0.7)
    axes[1].set_ylabel(row_labels[1])

    axes[2].plot(t_min, session["soc"] * 100.0, color="C2", lw=1.2)
    axes[2].axhline(soc_target * 100.0, color="gray", ls="--", lw=0.8, alpha=0.7,
                    label=f"SoC target ({soc_target * 100:.0f}%)")
    axes[2].axhline(soc_start * 100.0, color="C2", ls=":", lw=0.8, alpha=0.5)
    axes[2].scatter([t_end], [soc_end_pct], color="#9333ea", s=45, zorder=5, label="Stop")
    axes[2].annotate(
        f"Stop\n{metrics.get('end_reason', '')}",
        (t_end, soc_end_pct),
        textcoords="offset points",
        xytext=(-40, 8),
        fontsize=8,
        color="#9333ea",
    )
    axes[2].set_ylabel(row_labels[2])

    axes[3].plot(t_min, session["temperature_c"], color="C3", lw=1.2)
    axes[3].axhline(TEMP_LOW_C, color="green", ls="--", lw=0.8, alpha=0.6)
    axes[3].axhline(TEMP_HIGH_C, color="green", ls="--", lw=0.8, alpha=0.6)
    axes[3].axhspan(TEMP_LOW_C, TEMP_HIGH_C, color="green", alpha=0.08,
                    label=f"Optimal band ({TEMP_LOW_C:.0f}–{TEMP_HIGH_C:.0f} °C)")
    axes[3].set_ylabel(row_labels[3])
    axes[3].set_xlabel("Time (min)")

    if params and params.get("family_id") == "pulsed":
        ic = params.get("i_charge", 0.0)
        on = params.get("pulse_on_min", 0.0)
        rest = params.get("pulse_rest_min", 0.0)
        param_note = f"Pulsed: {ic:.2f} A  |  ON {on:.2f} min  |  REST {rest:.2f} min"
        axes[0].text(
            0.02, 0.97, param_note, transform=axes[0].transAxes,
            fontsize=8, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#eef4ff", edgecolor="#93b4e8", alpha=0.95),
        )

    axes[0].legend(loc="upper right", fontsize=8)
    axes[2].legend(loc="lower right", fontsize=8)
    axes[3].legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def _cell_from_results_meta(meta: Dict) -> "CellConfig":
    from Constrained_BO.config import CellConfig, get_cell_config
    from Constrained_BO.profile_catalog import ProfileBounds

    cell = get_cell_config(meta["cell"])
    overrides: Dict = {}
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


def rebuild_family_results_from_json(
    payload: Dict,
    *,
    device: str = "auto",
) -> Dict[str, Dict]:
    """Re-simulate best params per family so trajectories can be plotted from JSON."""
    from Constrained_BO.config import CellConfig  # noqa: F401
    from Constrained_BO.profiles import ProfileParams, set_profile_bounds
    from Constrained_BO.simulator import ChargingSimulator

    meta = payload["meta"]
    cell = _cell_from_results_meta(meta)
    if cell.profile_bounds is not None:
        set_profile_bounds(cell.profile_bounds)

    simulator = ChargingSimulator.from_cell(cell, device=device)
    simulator.decision_interval_info = meta.get("decision_interval_selection", {})

    rebuilt: Dict[str, Dict] = {}
    for fid, entry in payload["families"].items():
        params_dict = entry.get("best_params")
        if not params_dict:
            continue
        family = get_family(fid)
        vals = {k: v for k, v in params_dict.items() if k != "family_id"}
        # Always go through from_dict so multi-step ΔI / SoC gaps stay enforced.
        params = family.from_dict({k: float(v) for k, v in vals.items()})
        session = simulator.simulate(cell.start_state, params, family=family)
        rebuilt[fid] = {
            **entry,
            "best_session": session,
            "best_metrics": entry.get("best_metrics", {}),
        }
    return rebuilt


def plot_best_profiles_from_json(
    results_path: Path,
    *,
    out_path: Optional[Path] = None,
    device: str = "auto",
) -> plt.Figure:
    """Load ``constrained_bo_results.json`` and plot all families with best sessions."""
    with open(results_path) as f:
        payload = json.load(f)
    meta = payload["meta"]
    family_results = rebuild_family_results_from_json(payload, device=device)

    n_random = meta.get("n_random")
    method = meta.get("method", "random_search")
    title = f"{method}"
    if n_random is not None:
        title += f", n={n_random}"
    title += f", max {meta.get('max_duration_min', 150):.0f} min"
    if meta.get("constraint_mode") == "energy":
        title += f", energy ≥ {meta.get('energy_fraction', 0):.0%} of pack"

    out_path = out_path or Path(results_path).parent / "best_profiles.png"
    return plot_best_profiles(
        family_results,
        cell_id=meta["cell"],
        soc_target=float(meta.get("soc_target", 0.95)),
        soc_start=float(meta.get("soc_start", 0.20)),
        out_path=out_path,
        title_suffix=title,
        family_order=meta.get("families"),
    )


if __name__ == "__main__":
    import argparse

    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Plot reward curves or best profiles from JSON")
    parser.add_argument(
        "--results",
        type=Path,
        help="Regenerate best_profiles.png from constrained_bo_results.json",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for --rewards-only (default: Constrained_BO/data/rewards)",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rewards-only", action="store_true", help="Only plot reward curves")
    args = parser.parse_args()

    if args.results is not None:
        fig = plot_best_profiles_from_json(
            args.results, out_path=args.out, device=args.device,
        )
        plt.close(fig)
        out = args.out or args.results.parent / "best_profiles.png"
        print(f"Wrote {out}")
    elif args.rewards_only or args.results is None:
        plot_reward_curves(args.out_dir)
