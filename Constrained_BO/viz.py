"""Plot reward curves and best charging profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

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
from Constrained_BO.profiles import get_family

REWARDS_OUT_DIR = Path(__file__).resolve().parent / "data" / "rewards"


def _charge_current_plot(i_a: np.ndarray) -> np.ndarray:
    """Display magnitude (positive = charge)."""
    return -np.asarray(i_a, dtype=np.float64)


def plot_temperature_reward(out_path: Path | None = None) -> plt.Figure:
    t = np.linspace(0, TEMP_MAX_C, 500)
    r = np.array([temperature_reward(x) for x in t])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t, r, color="#2563eb", lw=2.2)
    ax.axvspan(TEMP_LOW_C, TEMP_HIGH_C, color="#22c55e", alpha=0.12, label="Optimal band (15–35 °C)")
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
        "Reward = 1.5 for 15 °C ≤ T ≤ 35 °C.\n"
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


def plot_reward_curves(out_dir: Path | None = None) -> None:
    out_dir = out_dir or REWARDS_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("temperature_reward.png", plot_temperature_reward),
        ("time_reward.png", plot_time_reward),
        ("rewards_combined.png", plot_rewards_combined),
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
) -> plt.Figure:
    families = [fid for fid in family_results if family_results[fid].get("best_session")]
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
        axes[3, col].axhline(15.0, color="green", ls="--", lw=0.8, alpha=0.6)
        axes[3, col].axhline(35.0, color="green", ls="--", lw=0.8, alpha=0.6)
        axes[3, col].axhspan(15.0, 35.0, color="green", alpha=0.08)

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


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    plot_reward_curves()
