"""Reporting/visualization for the Phase-1 hybrid degradation model.

All Q_loss values plotted here are a **Relative Capacity-Loss Index** (a
dimensionless, physically-motivated ranking signal), not a calibrated
"Capacity Fade (%)" — see README "Hybrid degradation methodology" for the
full caveat. The literature coefficients used (Eq. 2 calendar; Eq. 7 /
Table 7 cyclic) have not been fit to NASA RW9-RW12 or LFP aging data.

Generates four figures:

  Figure 1 — Calendar degradation contour (SoC x Temperature) using Eq. (2).
             Pure closed-form; no simulator/BDT required.

  Figure 2 — Cyclic degradation curves (Q_cyclic vs Ah throughput) at
             C/2, 2C, 6C, 10C using Eq. (7) / Table 7. Pure closed-form;
             no simulator/BDT required.

  Figure 3 — Cumulative calendar / cyclic / total degradation along the
             charging session, for every optimized profile in a
             ``constrained_bo_results.json``. Requires re-simulating each
             profile's best params through the original BDT checkpoint
             (via ``Constrained_BO.viz.rebuild_family_results_from_json``),
             since per-step trajectories are not stored in the results JSON.
             If the checkpoint is unavailable in the current environment,
             this figure is skipped with a clear message (Figures 1/2/4
             still complete).

  Figure 4 — Equal-energy comparison table across charging strategies,
             built directly from the ``best_metrics`` already recorded in
             the results JSON (no BDT required): duration, peak T, mean/max
             C-rate, Ah throughput, Q_calendar, Q_cyclic, Q_total, delivered
             vs. required energy.

Usage:
    python -m Constrained_BO.degradation_report \\
        --results Constrained_BO/results/RW9/constrained_bo_results.json \\
        --out-dir Constrained_BO/results/degradation_report
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from Constrained_BO.hybrid_degradation import (
    TABLE7_C_RATES,
    CalendarDegradation,
    CyclicDegradation,
    HybridDegradationParameters,
    compute_step_degradation,
)

QLOSS_CAPTION = (
    "Q_loss values are a Relative Capacity-Loss Index (dimensionless ranking "
    "signal), not a calibrated Capacity Fade (%)."
)

TABLE7_LABELS = ["C/2", "2C", "6C", "10C"]


# --------------------------------------------------------------------------- #
# Figure 1 — Calendar degradation contour (Eq. 2): SoC x Temperature
# --------------------------------------------------------------------------- #

def plot_calendar_contour(
    out_path: Optional[Path] = None,
    *,
    params: Optional[HybridDegradationParameters] = None,
    t_hours: float = 720.0,
    soc_range=(0.0, 1.0),
    temp_range_c=(0.0, 50.0),
    n: int = 120,
) -> plt.Figure:
    """Fig. 1: Q_calendar(SoC, T) at fixed elapsed time, using Eq. (2)."""
    params = params or HybridDegradationParameters()
    cal = CalendarDegradation(params)

    socs = np.linspace(soc_range[0], soc_range[1], n)
    temps = np.linspace(temp_range_c[0], temp_range_c[1], n)
    soc_grid, temp_grid = np.meshgrid(socs, temps)
    q_grid = np.zeros_like(soc_grid)
    for r in range(soc_grid.shape[0]):
        for c in range(soc_grid.shape[1]):
            q_grid[r, c] = cal.q_loss(soc_grid[r, c], temp_grid[r, c], t_hours)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    cs = ax.contourf(soc_grid * 100.0, temp_grid, q_grid, levels=30, cmap="inferno")
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label("Q_calendar — Relative Capacity-Loss Index (dimensionless)")
    ax.set_xlabel("Storage SoC (%)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(
        f"Figure 1 — Calendar degradation contour (Eq. 2), t = {t_hours:.0f} h",
        fontweight="bold",
    )
    ax.text(
        0.5, -0.14, QLOSS_CAPTION, transform=ax.transAxes, fontsize=8,
        ha="center", color="#5b6573", style="italic",
    )
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Figure 2 — Cyclic degradation curves (Eq. 7 / Table 7): Q_cyclic vs Ah
# --------------------------------------------------------------------------- #

def plot_cyclic_curves(
    out_path: Optional[Path] = None,
    *,
    params: Optional[HybridDegradationParameters] = None,
    temperature_c: float = 25.0,
    ah_max: float = 20.0,
    n: int = 200,
) -> plt.Figure:
    """Fig. 2: Q_cyclic vs Ah throughput at C/2, 2C, 6C, 10C, using Eq. (7)/Table 7."""
    params = params or HybridDegradationParameters()
    cyc = CyclicDegradation(params)

    ah = np.linspace(0.0, ah_max, n)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    colors = ["#2563eb", "#16a34a", "#f59e0b", "#ef4444"]
    for c_rate, label, color in zip(TABLE7_C_RATES, TABLE7_LABELS, colors):
        q = np.array([cyc.q_loss(a, temperature_c, float(c_rate)) for a in ah])
        ax.plot(ah, q, label=label, color=color, lw=2.0)

    ax.set_xlabel("Ah throughput (Ah)")
    ax.set_ylabel("Q_cyclic — Relative Capacity-Loss Index (dimensionless)")
    ax.set_title(
        f"Figure 2 — Cyclic degradation curves (Eq. 7 / Table 7), T = {temperature_c:.0f} °C",
        fontweight="bold",
    )
    ax.legend(title="C-rate", loc="upper left")
    ax.grid(True, alpha=0.35)
    ax.text(
        0.5, -0.14, QLOSS_CAPTION, transform=ax.transAxes, fontsize=8,
        ha="center", color="#5b6573", style="italic",
    )
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Figure 3 — Cumulative degradation per optimized profile
# --------------------------------------------------------------------------- #

class BDTUnavailableError(RuntimeError):
    """Raised when a results JSON's BDT checkpoint cannot be loaded for re-simulation."""


def plot_cumulative_degradation(
    results_path: Path,
    out_path: Optional[Path] = None,
    *,
    device: str = "auto",
    params: Optional[HybridDegradationParameters] = None,
) -> plt.Figure:
    """Fig. 3: cumulative Q_calendar / Q_cyclic / Q_total vs time, per family.

    Re-simulates each family's best params through the original BDT checkpoint
    (trajectories are not persisted in the results JSON). Raises
    ``BDTUnavailableError`` if the checkpoint referenced in the JSON's meta
    cannot be loaded in this environment.
    """
    from Constrained_BO.profiles import DEFAULT_FAMILIES, get_family
    from Constrained_BO.viz import rebuild_family_results_from_json

    with open(results_path) as f:
        payload = json.load(f)

    try:
        family_results = rebuild_family_results_from_json(payload, device=device)
    except Exception as exc:  # BDT checkpoint missing/unreadable in this environment
        raise BDTUnavailableError(
            f"Could not re-simulate sessions from {results_path} "
            f"(BDT checkpoint unavailable): {exc}"
        ) from exc

    order = payload["meta"].get("families", DEFAULT_FAMILIES)
    families = [fid for fid in order if fid in family_results]
    if not families:
        raise BDTUnavailableError(f"No rebuildable family sessions in {results_path}")

    params = params or HybridDegradationParameters()
    fig, axes = plt.subplots(1, len(families), figsize=(4.6 * len(families), 4.8), squeeze=False)
    axes = axes[0]

    for col, fid in enumerate(families):
        session = family_results[fid]["best_session"]
        step = compute_step_degradation(session, params=params)
        t_min = session["time_s"] / 60.0
        cum_cal = np.cumsum(step["delta_qloss_calendar"])
        cum_cyc = np.cumsum(step["delta_qloss_cyclic"])
        cum_tot = cum_cal + cum_cyc

        ax = axes[col]
        ax.plot(t_min, cum_cal, label="Calendar", color="#2563eb", lw=1.6)
        ax.plot(t_min, cum_cyc, label="Cyclic", color="#ef4444", lw=1.6)
        ax.plot(t_min, cum_tot, label="Total", color="#16a34a", lw=2.2, ls="--")
        ax.set_title(get_family(fid).label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (min)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Cumulative Q_loss\n(Relative Capacity-Loss Index)")
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Figure 3 — Cumulative degradation per optimized profile — {payload['meta'].get('cell', '?')}",
        fontsize=12, fontweight="bold", y=1.03,
    )
    fig.text(0.5, -0.02, QLOSS_CAPTION, fontsize=8, ha="center", color="#5b6573", style="italic")
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------- #
# Figure 4 — Equal-energy comparison table (no BDT required)
# --------------------------------------------------------------------------- #

_TABLE_COLUMNS = [
    ("family_label", "Profile"),
    ("duration_min", "Time (min)"),
    ("energy_delivered_j", "E delivered (J)"),
    ("energy_required_j", "E required (J)"),
    ("mean_c_rate", "Mean C-rate"),
    ("max_c_rate", "Max C-rate"),
    ("ah_throughput", "Ah throughput"),
    ("efc", "EFC"),
    ("peak_voltage", "Peak V"),
    ("peak_temperature", "Peak T (°C)"),
    ("qloss_calendar", "Q_calendar (index)"),
    ("qloss_cyclic", "Q_cyclic (index)"),
    ("qloss_total", "Q_total (index)"),
    ("feasible", "Feasible"),
]


def _equal_energy_rows(payload: Dict) -> List[Dict]:
    rows = []
    order = payload["meta"].get("families", list(payload["families"].keys()))
    for fid in order:
        entry = payload["families"].get(fid)
        if not entry:
            continue
        m = entry.get("best_metrics") or {}
        rows.append({
            "family_label": entry.get("family_label", fid),
            "duration_min": m.get("duration_min"),
            "energy_delivered_j": m.get("energy_delivered_j"),
            "energy_required_j": m.get("energy_required_j"),
            "mean_c_rate": m.get("nominal_c_rate"),
            "max_c_rate": m.get("max_c_rate"),
            "ah_throughput": m.get("ah_throughput"),
            "efc": m.get("efc"),
            "peak_voltage": m.get("peak_voltage"),
            "peak_temperature": m.get("peak_temperature"),
            "qloss_calendar": m.get("qloss_calendar"),
            "qloss_cyclic": m.get("qloss_cyclic"),
            "qloss_total": m.get("qloss_total"),
            "feasible": m.get("feasible"),
        })
    return rows


def _fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key == "feasible":
        return "yes" if value else "no"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if key in ("qloss_calendar", "qloss_cyclic", "qloss_total"):
            return f"{value:.4g}"
        if key in ("mean_c_rate", "max_c_rate", "efc"):
            return f"{value:.3f}"
        if key in ("energy_delivered_j", "energy_required_j"):
            return f"{value:.0f}"
        return f"{value:.2f}"
    return str(value)


def plot_equal_energy_table(
    results_path: Path,
    out_path: Optional[Path] = None,
    *,
    csv_out_path: Optional[Path] = None,
) -> plt.Figure:
    """Fig. 4: equal-energy comparison table across families (same session json,
    hence same energy/SoC requirement, current bounds, and time budget)."""
    with open(results_path) as f:
        payload = json.load(f)
    rows = _equal_energy_rows(payload)
    if not rows:
        raise ValueError(f"No family results found in {results_path}")

    headers = [label for _, label in _TABLE_COLUMNS]
    keys = [key for key, _ in _TABLE_COLUMNS]
    cell_text = [[_fmt(k, row.get(k)) for k in keys] for row in rows]

    fig_h = 0.6 + 0.45 * len(rows)
    fig, ax = plt.subplots(figsize=(1.35 * len(headers), fig_h))
    ax.axis("off")
    table = ax.table(cellText=cell_text, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)
    for c in range(len(headers)):
        table[0, c].set_facecolor("#E8EEF5")
        table[0, c].set_text_props(fontweight="bold")

    meta = payload["meta"]
    constraint_note = f"constraint_mode={meta.get('constraint_mode')}"
    if meta.get("constraint_mode") == "energy":
        constraint_note += f", energy_fraction={meta.get('energy_fraction')}"
    else:
        constraint_note += f", soc_target={meta.get('soc_target')}"
    ax.set_title(
        f"Figure 4 — Equal-energy comparison — {meta.get('cell', '?')}  ({constraint_note})",
        fontsize=11, fontweight="bold", pad=14,
    )
    fig.text(0.5, 0.01, QLOSS_CAPTION, fontsize=8, ha="center", color="#5b6573", style="italic")
    fig.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if csv_out_path:
        csv_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(cell_text)
    return fig


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def generate_all(
    out_dir: Path,
    *,
    results_path: Optional[Path] = None,
    device: str = "auto",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fig1 = plot_calendar_contour(out_dir / "fig1_calendar_contour.png")
    plt.close(fig1)
    print(f"Wrote {out_dir / 'fig1_calendar_contour.png'}")

    fig2 = plot_cyclic_curves(out_dir / "fig2_cyclic_curves.png")
    plt.close(fig2)
    print(f"Wrote {out_dir / 'fig2_cyclic_curves.png'}")

    if results_path is None:
        print("No --results JSON given: skipping Figure 3 (cumulative degradation) "
              "and Figure 4 (equal-energy table).")
        return

    if not Path(results_path).is_file():
        print(f"--results {results_path} not found: skipping Figure 3 and Figure 4.")
        return

    try:
        fig3 = plot_cumulative_degradation(
            Path(results_path), out_dir / "fig3_cumulative_degradation.png", device=device,
        )
        plt.close(fig3)
        print(f"Wrote {out_dir / 'fig3_cumulative_degradation.png'}")
    except BDTUnavailableError as exc:
        print(f"Skipped Figure 3 (cumulative degradation): {exc}")

    try:
        fig4 = plot_equal_energy_table(
            Path(results_path),
            out_dir / "fig4_equal_energy_table.png",
            csv_out_path=out_dir / "fig4_equal_energy_table.csv",
        )
        plt.close(fig4)
        print(f"Wrote {out_dir / 'fig4_equal_energy_table.png'}")
        print(f"Wrote {out_dir / 'fig4_equal_energy_table.csv'}")
    except Exception as exc:
        print(f"Skipped Figure 4 (equal-energy table): {exc}")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Generate the 4 hybrid-degradation report figures (Phase 1).",
    )
    parser.add_argument(
        "--results", type=Path, default=None,
        help="constrained_bo_results.json to source Figures 3 and 4 from "
             "(Figures 1 and 2 are always closed-form and need no results file).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path(__file__).resolve().parent / "results" / "degradation_report",
        help="Output directory for the generated figures.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    generate_all(args.out_dir, results_path=args.results, device=args.device)


if __name__ == "__main__":
    main()
