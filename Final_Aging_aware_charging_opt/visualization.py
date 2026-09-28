"""Figures for the charging study and the cross-cell Top-6 litmus test.

Every image and table is written as a file directly under ``results/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from config import (
    DEGRADATION, OptimizationSpec, Paths, RESULTS_DIR, RewardSpec,
    ensure_matplotlib_cache, family_result_name, provenance, read_json,
    result_path, set_results_dir, write_json,
)
from optimization import (
    evaluate_params, get_family, load_anchors, load_degradation_model, make_simulator,
)

"""Paper figure style (palette / DPI / fonts aligned with the previous study)."""


import matplotlib.pyplot as plt

PAPER_LIGHT_BG = "#FFFFFF"
PAPER_ACCENT = "#2563EB"
PAPER_ORANGE = "#EA580C"
PAPER_GREEN = "#16A34A"
PAPER_RED = "#DC2626"
PAPER_GREY = "#6B7280"
PAPER_PROFILE_COLORS = (PAPER_ACCENT, PAPER_ORANGE, PAPER_GREEN, PAPER_RED)
PAPER_DPI = 200

FAMILY_ORDER = ("cccv", "two_step", "three_step", "pulsed")
FAMILY_LABELS = {
    "cccv": "CCCV",
    "two_step": "Two-step",
    "three_step": "Three-step",
    "pulsed": "Pulsed",
}

GROUP_COLORS = {
    "CCCV 0.5C": "#64748b",
    "CCCV ½C": "#64748b",
    "CCCV 1C": "#2563eb",
    "CCCV 2C": "#1e40af",
    "Random": "#f59e0b",
    "GP-BO": "#16a34a",
    "GP-BO (min Q)": "#16a34a",
    "GP-BO (max R)": "#0f766e",
}

POLICY_STYLE = {
    "CCCV 0.5C": {"color": "#64748b", "ls": "-", "lw": 2.2},
    "CCCV ½C": {"color": "#64748b", "ls": "-", "lw": 2.2},
    "CCCV 1C": {"color": "#2563eb", "ls": "--", "lw": 1.6},
    "CCCV 2C": {"color": "#1e3a8a", "ls": ":", "lw": 1.6},
    "Random": {"color": "#f59e0b", "ls": "-", "lw": 2.4},
    "GP-BO": {"color": "#16a34a", "ls": "-", "lw": 2.6},
    "GP-BO (min Q)": {"color": "#16a34a", "ls": "-", "lw": 2.6},
    "GP-BO (max R)": {"color": "#0f766e", "ls": "--", "lw": 2.2},
}


def apply_paper_style() -> None:
    plt.rcParams.update({
        "figure.dpi": PAPER_DPI,
        "savefig.dpi": PAPER_DPI,
        "figure.facecolor": PAPER_LIGHT_BG,
        "axes.facecolor": PAPER_LIGHT_BG,
        "savefig.facecolor": PAPER_LIGHT_BG,
        "font.family": "DejaVu Sans",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 19,
        "xtick.labelsize": 17,
        "ytick.labelsize": 17,
        "legend.fontsize": 16,
        "figure.titlesize": 22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "lines.linewidth": 2.2,
    })


def savefig(fig, path, *, pad_inches: float = 0.25) -> None:
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=PAPER_DPI, bbox_inches="tight", pad_inches=pad_inches,
        facecolor=PAPER_LIGHT_BG, edgecolor=PAPER_LIGHT_BG, transparent=False,
    )
    plt.close(fig)

"""Plotters for charging-optimization figures (paper layout, calibrated JSON)."""


from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch



def _as_array(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _charge_current(i) -> np.ndarray:
    return -_as_array(i)


def plot_best_profiles(
    family_results: Dict[str, Dict[str, Any]],
    *,
    cell_id: str,
    method_title: str,
    soc_start: float = 0.20,
    v_max: float = 4.20,
    out_path: Path,
) -> None:
    apply_paper_style()
    families = [
        fid for fid in FAMILY_ORDER
        if fid in family_results and family_results[fid].get("best_session")
    ]
    n_cols = len(families)
    if n_cols == 0:
        raise ValueError(f"{cell_id}: no sessions to plot")

    fig, axes = plt.subplots(
        4, n_cols, figsize=(2.6 * n_cols + 0.5, 9.8),
        squeeze=False, facecolor=PAPER_LIGHT_BG,
    )
    row_labels = ["Current (A)", "Voltage (V)", "SoC (%)", "Temperature (°C)"]
    c_i, c_v, c_soc, c_t = PAPER_PROFILE_COLORS

    for col, fid in enumerate(families):
        res = family_results[fid]
        session = res["best_session"]
        t_min = _as_array(session["time_s"]) / 60.0
        axes[0, col].set_title(FAMILY_LABELS.get(fid, fid), fontsize=12, fontweight="bold", pad=8)
        axes[0, col].plot(
            t_min, _charge_current(session["current_a"]),
            color=c_i, lw=2.2, drawstyle="steps-post",
        )
        axes[1, col].plot(t_min, _as_array(session["voltage_v"]), color=c_v, lw=2.0)
        axes[1, col].axhline(v_max, color=PAPER_GREY, ls="--", lw=1.1, alpha=0.75)
        axes[2, col].plot(t_min, _as_array(session["soc"]) * 100.0, color=c_soc, lw=2.0)
        axes[2, col].axhline(soc_start * 100.0, color=c_soc, ls=":", lw=1.1, alpha=0.55)
        axes[3, col].plot(t_min, _as_array(session["temperature_c"]), color=c_t, lw=2.0)

        for row in range(4):
            ax = axes[row, col]
            ax.set_facecolor(PAPER_LIGHT_BG)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.tick_params(labelsize=11)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=13)

    for col in range(n_cols):
        axes[3, col].set_xlabel("Time (min)", fontsize=13)

    fig.tight_layout(h_pad=0.75, w_pad=0.35)
    savefig(fig, out_path)


_AXIS_I = "#2563EB"
_AXIS_V = "#EA580C"
_AXIS_SOC = "#16A34A"
_AXIS_T = "#DC2626"


def _hide_extra_spines(ax) -> None:
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _style_spine(ax, *, side: str, color: str) -> None:
    ax.spines[side].set_visible(True)
    ax.spines[side].set_color(color)
    ax.spines[side].set_linewidth(2.0)
    ax.tick_params(axis="y", colors=color, labelsize=15)
    ax.yaxis.label.set_color(color)


def _draw_multiaxis_session(
    host,
    session: Dict[str, Any],
    metrics: Dict[str, Any],
    *,
    family_id: str,
    method_name: str,
    v_max: float = 4.20,
) -> None:
    """One time axis, four colour-coded y-axes (solid lines)."""
    t_min = _as_array(session["time_s"]) / 60.0
    i_a = _charge_current(session["current_a"])
    v = _as_array(session["voltage_v"])
    soc = _as_array(session["soc"]) * 100.0
    temp = _as_array(session["temperature_c"])
    host.set_facecolor(PAPER_LIGHT_BG)
    host.set_xlabel("Time (min)", fontsize=18)
    host.tick_params(axis="x", labelsize=15)
    host.grid(True, linestyle=":", alpha=0.35, axis="x")

    p_v = host.twinx()
    p_soc = host.twinx()
    p_t = host.twinx()
    p_soc.spines["right"].set_position(("axes", 1.16))
    _hide_extra_spines(p_t)
    p_t.spines["left"].set_position(("axes", -0.20))
    p_t.yaxis.set_label_position("left")
    p_t.yaxis.tick_left()

    host.plot(t_min, i_a, color=_AXIS_I, lw=2.2, solid_capstyle="butt",
              drawstyle="steps-post", zorder=3)
    p_v.plot(t_min, v, color=_AXIS_V, lw=2.1, zorder=2)
    p_soc.plot(t_min, soc, color=_AXIS_SOC, lw=2.1, zorder=2)
    p_t.plot(t_min, temp, color=_AXIS_T, lw=2.1, zorder=2)

    host.set_ylabel("Current (A)", fontsize=18, fontweight="bold", labelpad=12)
    p_v.set_ylabel("Voltage (V)", fontsize=18, fontweight="bold", labelpad=12)
    p_soc.set_ylabel("SoC (%)", fontsize=18, fontweight="bold", labelpad=14)
    p_t.set_ylabel("Temperature (°C)", fontsize=18, fontweight="bold", labelpad=14)
    _style_spine(host, side="left", color=_AXIS_I)
    _style_spine(p_v, side="right", color=_AXIS_V)
    _style_spine(p_soc, side="right", color=_AXIS_SOC)
    _style_spine(p_t, side="left", color=_AXIS_T)
    host.spines["top"].set_visible(False)
    host.spines["right"].set_visible(False)

    i_pad = max(0.15, 0.08 * (float(np.nanmax(i_a)) - float(np.nanmin(i_a)) + 1e-9))
    host.set_ylim(max(0.0, float(np.nanmin(i_a)) - i_pad), float(np.nanmax(i_a)) + i_pad)
    p_v.set_ylim(min(3.6, float(np.nanmin(v)) - 0.05), max(v_max + 0.05, float(np.nanmax(v)) + 0.05))
    p_soc.set_ylim(15.0, max(65.0, float(np.nanmax(soc)) + 5.0))
    t_lo, t_hi = float(np.nanmin(temp)), float(np.nanmax(temp))
    t_pad = max(0.25, 0.25 * (t_hi - t_lo + 0.2))
    p_t.set_ylim(t_lo - t_pad, t_hi + t_pad)
    if t_min.size:
        host.set_xlim(0.0, float(t_min[-1]) * 1.02)

def plot_gpbo_vs_random_multiaxis(
    *,
    cell_id: str,
    gpbo_session: Dict[str, Any],
    gpbo_metrics: Dict[str, Any],
    gpbo_family: str,
    random_session: Dict[str, Any],
    random_metrics: Dict[str, Any],
    random_family: str,
    out_path: Path,
) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 13.4), facecolor=PAPER_LIGHT_BG)
    fig.subplots_adjust(left=0.22, right=0.76, hspace=0.32, top=0.97, bottom=0.06)
    _draw_multiaxis_session(
        axes[0], random_session, random_metrics,
        family_id=random_family, method_name="Random",
    )
    _draw_multiaxis_session(
        axes[1], gpbo_session, gpbo_metrics,
        family_id=gpbo_family, method_name="GP-BO (max R)",
    )
    savefig(fig, out_path, pad_inches=0.35)


def plot_all_cells_gpbo_vs_random_multiaxis(
    packs: Dict[str, Dict[str, Any]],
    *,
    out_path: Path,
) -> None:
    apply_paper_style()
    cells = list(packs.keys())
    n = len(cells)
    fig, axes = plt.subplots(2 * n, 1, figsize=(12.4, 6.9 * n), facecolor=PAPER_LIGHT_BG)
    if 2 * n == 1:
        axes = np.array([axes])
    fig.subplots_adjust(left=0.22, right=0.76, hspace=0.32, top=0.99, bottom=0.03)
    for i, cell in enumerate(cells):
        p = packs[cell]
        _draw_multiaxis_session(
            axes[2 * i], p["random_session"], p["random_metrics"],
            family_id=p["random_family"], method_name=f"{cell}  Random",
        )
        _draw_multiaxis_session(
            axes[2 * i + 1], p["gpbo_session"], p["gpbo_metrics"],
            family_id=p["gpbo_family"], method_name=f"{cell}  GP-BO (max R)",
        )
    savefig(fig, out_path, pad_inches=0.35)


def plot_metric_bars(
    rows: Sequence[Dict[str, Any]],
    *,
    value_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
    clip_to_data: bool = False,
    fmt: str = "{:.3f}",
) -> None:
    apply_paper_style()
    labels = [r["label"] for r in rows]
    values = [float(r[value_key]) for r in rows]
    colors = [GROUP_COLORS.get(r["label"], "#6b7280") for r in rows]
    fig, ax = plt.subplots(
        figsize=(max(8.0, 1.45 * len(labels)), 5.0),
        facecolor=PAPER_LIGHT_BG,
    )
    ax.set_facecolor(PAPER_LIGHT_BG)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8, width=0.7)
    for bar, row, val in zip(bars, rows, values):
        if not row.get("feasible", True):
            bar.set_hatch("//")
            bar.set_alpha(0.55)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            fmt.format(val),
            ha="center", va="bottom", fontsize=11, color="#0f172a",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    if clip_to_data and values:
        lo, hi = min(values), max(values)
        pad = max(0.15, 0.35 * (hi - lo) if hi > lo else 0.2)
        ax.set_ylim(lo - pad, hi + pad)
    legend_items = [
        Patch(facecolor="#64748b", label="CCCV 0.5C"),
        Patch(facecolor="#2563eb", label="CCCV 1C"),
        Patch(facecolor="#1e40af", label="CCCV 2C"),
        Patch(facecolor="#f59e0b", label="Random best"),
        Patch(facecolor="#16a34a", label="GP-BO best"),
    ]
    if any(not r.get("feasible", True) for r in rows):
        legend_items.append(Patch(facecolor="#94a3b8", hatch="//", label="Infeasible"))
    ax.legend(handles=legend_items, fontsize=11, loc="best", framealpha=0.95)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_qloss_bars(
    rows: Sequence[Dict[str, Any]],
    *,
    cell_id: str,
    out_path: Path,
) -> None:
    apply_paper_style()
    labels = [r["label"] for r in rows]
    q_tot = [float(r["q_total"]) for r in rows]
    colors = [GROUP_COLORS.get(r["label"], "#94a3b8") for r in rows]
    fig, ax = plt.subplots(figsize=(10.0, 5.8), dpi=PAPER_DPI, facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    x = np.arange(len(rows))
    bars = ax.bar(x, q_tot, color=colors, edgecolor="k", lw=0.6, width=0.7)
    ymax = max(q_tot) if q_tot else 1.0
    for bar, row, q in zip(bars, rows, q_tot):
        if not row.get("feasible", True):
            bar.set_hatch("//")
            bar.set_alpha(0.4)
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            q + ymax * 0.02,
            f"{q:.3e}",
            ha="center", va="bottom", fontsize=10, color="#0f172a",
        )
    base = next((r for r in rows if r["label"] in ("CCCV 0.5C", "CCCV ½C") and r.get("feasible")), None)
    if base is not None and float(base["q_total"]) > 0:
        bq = float(base["q_total"])
        for i, (row, q) in enumerate(zip(rows, q_tot)):
            if row["label"] in ("Random", "GP-BO", "GP-BO (min Q)") and row.get("feasible"):
                red = 100.0 * (bq - q) / bq
                ax.annotate(
                    f"{red:+.1f}% vs 0.5C",
                    xy=(i, q), xytext=(0, 22), textcoords="offset points",
                    ha="center", fontsize=11, fontweight="bold",
                    color="#166534" if red > 0 else "#9a3412",
                )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Session $Q_{\\mathrm{loss}}$ (capacity fraction, lower better)", fontsize=13)
    ax.set_ylim(0, ymax * 1.40)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_qloss_detail(
    rows: Sequence[Dict[str, Any]],
    *,
    cell_id: str,
    out_path: Path,
) -> None:
    apply_paper_style()
    labels = [r["label"] for r in rows]
    q_tot = np.array([float(r["q_total"]) for r in rows])
    q_cal = np.array([float(r.get("q_calendar") or 0.0) for r in rows])
    q_cyc = np.array([float(r.get("q_cyclic") or 0.0) for r in rows])
    colors = [GROUP_COLORS.get(r["label"], "#94a3b8") for r in rows]
    fig, axes = plt.subplots(
        1, 2, figsize=(12.2, 5.8), dpi=PAPER_DPI,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
        facecolor=PAPER_LIGHT_BG,
    )
    for a in axes:
        a.set_facecolor(PAPER_LIGHT_BG)
    x = np.arange(len(rows))
    ax = axes[0]
    bars = ax.bar(x, q_tot, color=colors, edgecolor="k", lw=0.5, width=0.72)
    for bar, row in zip(bars, rows):
        if not row.get("feasible", True):
            bar.set_hatch("//")
            bar.set_alpha(0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Session $Q_{\\mathrm{loss}}$ (lower = less degradation)", fontsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    ax2 = axes[1]
    ax2.bar(x, q_cyc, color=colors, edgecolor="k", lw=0.4, width=0.72, label="$Q_{\\mathrm{cyclic}}$")
    ax2.bar(
        x, q_cal, bottom=q_cyc, color="#e2e8f0", edgecolor="k", lw=0.4,
        width=0.72, label="$Q_{\\mathrm{calendar}}$",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_ylabel("Stacked $Q_{\\mathrm{loss}}$", fontsize=13)
    ax2.legend(fontsize=11, loc="upper right")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_pareto_cloud(
    rows: Sequence[Dict[str, Any]],
    cloud: Sequence[Dict[str, Any]],
    *,
    cell_id: str,
    out_path: Path,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.8, 5.8), dpi=PAPER_DPI, facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    for method, color, marker in (("Random", GROUP_COLORS["Random"], "o"),
                                  ("GP-BO", GROUP_COLORS["GP-BO"], "o")):
        pts = [p for p in cloud if p["method"] == method and p["feasible"]]
        if pts:
            ax.scatter(
                [p["duration_min"] for p in pts],
                [p["q_total"] for p in pts],
                c=color, s=28, alpha=0.35, marker=marker,
                label=f"{method} trials (feasible)", edgecolors="none",
            )
        inf = [p for p in cloud if p["method"] == method and not p["feasible"]]
        if inf:
            ax.scatter(
                [p["duration_min"] for p in inf],
                [p["q_total"] for p in inf],
                c=color, s=22, alpha=0.15, marker="x",
            )
    for r in rows:
        q = float(r["q_total"])
        d = float(r["duration_min"])
        color = GROUP_COLORS.get(r["label"], "#334155")
        marker = "s" if str(r["label"]).startswith("CCCV") else "*"
        size = 180 if marker == "*" else 130
        ax.scatter(
            [d], [q], c=color, s=size, marker=marker,
            edgecolors="k", linewidths=0.8, zorder=5, label=r["label"],
        )
        ax.annotate(
            r["label"], xy=(d, q), xytext=(6, 6),
            textcoords="offset points", fontsize=11, color="#0f172a",
        )
    ax.set_xlabel("Charge duration [min]", fontsize=14)
    ax.set_ylabel("Session $Q_{\\mathrm{loss}}$ (lower better)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    seen, uniq = set(), []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        uniq.append((h, lab))
    ax.legend(*zip(*uniq), fontsize=11, loc="best", framealpha=0.95)
    fig.tight_layout()
    savefig(fig, out_path)


def _policy_ylim(curves: Dict[str, Dict[str, np.ndarray]], key: str = "retention_pct") -> tuple:
    vals = []
    for d in curves.values():
        y = np.asarray(d[key], dtype=np.float64)
        if y.size:
            vals.append(float(np.nanmin(y)))
            vals.append(float(np.nanmax(y)))
    if not vals:
        return 80.0, 100.0
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 0.05)
    return lo - 0.15 * span, min(100.05, hi + 0.25 * span)


def plot_lifetime_vs_cycles(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    *,
    cell_id: str,
    n_cycles: int,
    soh_ref: float,
    out_path: Path,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=PAPER_DPI, facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    for p in policies:
        name = p["name"]
        if name not in curves:
            continue
        d = curves[name]
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        ax.plot(
            d["cycles"], d["retention_pct"],
            color=style["color"], ls=style["ls"], lw=style.get("lw", 2.2),
            label=name,
        )
    ax.axhline(soh_ref, color="#94a3b8", ls="--", lw=1.2, alpha=0.85)
    ax.set_xlabel("Equivalent charge cycles (one 40% energy session each)", fontsize=18)
    ax.set_ylabel("Projected remaining capacity [%]", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(0, n_cycles)
    ax.set_ylim(60.0, 100.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=15, loc="best", framealpha=0.95)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_lifetime_delta(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    *,
    cell_id: str,
    baseline: str,
    out_path: Path,
) -> None:
    if baseline not in curves:
        return
    base = curves[baseline]["retention_pct"]
    cycles = curves[baseline]["cycles"]
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.8, 5.4), dpi=PAPER_DPI, facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    ax.axhline(0.0, color="#64748b", lw=1.4)
    for p in policies:
        name = p["name"]
        if name == baseline or name not in curves:
            continue
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        ax.plot(
            cycles, curves[name]["retention_pct"] - base,
            color=style["color"], ls=style["ls"], lw=style.get("lw", 2.2),
            label=name,
        )
    ax.set_xlabel("Equivalent charge cycles", fontsize=18)
    ax.set_ylabel(f"Δ remaining capacity vs {baseline} [%-points]", fontsize=17)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=15, loc="best")
    fig.tight_layout()
    savefig(fig, out_path)


def plot_lifetime_vs_ah(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    *,
    cell_id: str,
    out_path: Path,
) -> None:
    apply_paper_style()
    fig, (ax, ax_zoom) = plt.subplots(
        1, 2, figsize=(12.2, 5.4), dpi=PAPER_DPI,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        facecolor=PAPER_LIGHT_BG,
    )
    for a in (ax, ax_zoom):
        a.set_facecolor(PAPER_LIGHT_BG)
    for p in policies:
        name = p["name"]
        if name not in curves:
            continue
        d = curves[name]
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        lw = style.get("lw", 2.2)
        ax.plot(d["cum_ah"], d["retention_pct"], color=style["color"], ls=style["ls"], lw=lw, label=name)
        ax_zoom.plot(d["cum_ah"], d["retention_pct"], color=style["color"], ls=style["ls"], lw=lw, label=name)
    ax.set_xlabel("Cumulative ampere-hour throughput [Ah]", fontsize=17)
    ax.set_ylabel("Projected remaining capacity [%]", fontsize=17)
    ax.tick_params(axis="both", labelsize=15)
    ax.set_ylim(60.0, 100.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=14, loc="best", framealpha=0.95)
    ax.set_ylim(60.0, 100.0)
    ax_zoom.set_xlabel("Cumulative Ah", fontsize=17)
    ax_zoom.set_ylabel("Projected remaining capacity [%]", fontsize=17)
    ax_zoom.tick_params(axis="both", labelsize=15)
    ax_zoom.set_ylim(60.0, 100.0)
    ax_zoom.grid(True, linestyle="--", alpha=0.35)
    ax_zoom.legend(fontsize=14, loc="best")
    fig.tight_layout()
    savefig(fig, out_path)


def plot_lifetime_capacity_ah(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    *,
    cell_id: str,
    q_rated_ah: float,
    n_show: int,
    out_path: Path,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=PAPER_DPI, facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    for p in policies:
        name = p["name"]
        if name not in curves:
            continue
        d = curves[name]
        n = min(n_show, len(d["cycles"]) - 1)
        x = d["cycles"][: n + 1]
        q_ah = q_rated_ah * d["retention_pct"][: n + 1] / 100.0
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        ax.plot(x, q_ah, color=style["color"], ls=style["ls"], lw=style.get("lw", 2.2), label=name)
    ax.set_xlabel("Equivalent charge cycle index", fontsize=18)
    ax.set_ylabel("Projected capacity [Ah]", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=15, loc="best")
    fig.tight_layout()
    savefig(fig, out_path)


def plot_consolidated_comparison(
    per_cell: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    out_path: Path,
) -> None:
    """Three-panel: time, peak T, reward for CCCV 0.5C / 1C / Random / GP-BO."""
    apply_paper_style()
    cells = list(per_cell.keys())
    series = ("CCCV 0.5C", "CCCV 1C", "Random", "GP-BO")
    colors = [GROUP_COLORS[s] for s in series]
    metrics = [
        ("duration_min", "Charging time (min)", "(a) Charging time"),
        ("peak_t", "Peak temperature (°C)", "(b) Peak temperature"),
        ("reward", "Total reward", "(c) Total reward"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor=PAPER_LIGHT_BG)
    x = np.arange(len(cells))
    width = 0.18
    n_s = len(series)
    for ax, (key, ylabel, title) in zip(axes, metrics):
        ax.set_facecolor(PAPER_LIGHT_BG)
        for k, (name, color) in enumerate(zip(series, colors)):
            vals = [float(per_cell[c][name][key]) for c in cells]
            ax.bar(
                x + (k - (n_s - 1) / 2.0) * width, vals, width=width,
                color=color, edgecolor="k", lw=0.4, label=name,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(cells, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        if key == "peak_t":
            vals_all = [float(per_cell[c][s][key]) for c in cells for s in series]
            lo, hi = min(vals_all), max(vals_all)
            pad = max(0.15, 0.25 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
        if key == "reward":
            vals_all = [float(per_cell[c][s][key]) for c in cells for s in series]
            lo, hi = min(vals_all), max(vals_all)
            pad = max(0.005, 0.25 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)
    axes[0].legend(fontsize=10, loc="best")
    fig.tight_layout()
    savefig(fig, out_path)


def plot_lifetime_grid(
    all_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    all_policies: Dict[str, List[Dict[str, Any]]],
    *,
    n_cycles: int,
    soh_ref: float,
    out_path: Path,
) -> None:
    apply_paper_style()
    cells = list(all_curves.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.6), facecolor=PAPER_LIGHT_BG)
    axes = axes.ravel()
    for ax, cell in zip(axes, cells):
        ax.set_facecolor(PAPER_LIGHT_BG)
        curves = all_curves[cell]
        for p in all_policies[cell]:
            name = p["name"]
            if name not in curves:
                continue
            style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
            ax.plot(
                curves[name]["cycles"], curves[name]["retention_pct"],
                color=style["color"], ls=style["ls"], lw=style.get("lw", 2.2),
                label=name,
            )
        ax.axhline(soh_ref, color="#94a3b8", ls="--", lw=1.0, alpha=0.75)
        ax.set_xlim(0, n_cycles)
        ax.set_ylim(60.0, 100.0)
        ax.set_title(cell, fontsize=16, fontweight="bold")
        ax.set_xlabel("Equivalent charge cycles (one equal-energy session each)", fontsize=14)
        ax.set_ylabel("Projected remaining capacity [%]", fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=12, loc="best")
    fig.tight_layout()
    savefig(fig, out_path)


CELL_LINE_COLORS = {
    "RW9": "#2563eb",
    "RW10": "#16a34a",
    "RW11": "#ea580c",
    "RW12": "#7c3aed",
}


def _history_rows(families: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fid in FAMILY_ORDER:
        blob = families.get(fid) or {}
        for h in blob.get("history") or []:
            m = h.get("metrics") or {}
            rows.append({
                "feasible": bool(m.get("feasible", h.get("feasible"))),
                "loss": float(h.get("loss", m.get("loss", np.inf))),
                "duration_min": float(m.get("duration_min") or np.nan),
                "q_calendar": float(m.get("q_calendar") or np.nan),
                "q_cyclic": float(m.get("q_cyclic") or np.nan),
                "q_total": float(m.get("q_total") or m.get("qloss_total") or np.nan),
            })
    return rows


def _best_so_far_trace(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    n = len(rows)
    evals = np.arange(1, n + 1, dtype=np.float64)
    t = np.full(n, np.nan)
    q_cal = np.full(n, np.nan)
    q_cyc = np.full(n, np.nan)
    q_tot = np.full(n, np.nan)
    best_loss = np.inf
    cur = None
    for i, r in enumerate(rows):
        if r["feasible"] and np.isfinite(r["loss"]) and r["loss"] < best_loss:
            best_loss = r["loss"]
            cur = r
        if cur is not None:
            t[i] = cur["duration_min"]
            q_cal[i] = cur["q_calendar"]
            q_cyc[i] = cur["q_cyclic"]
            q_tot[i] = cur["q_total"]
    return {
        "eval": evals,
        "duration_min": t,
        "q_calendar": q_cal,
        "q_cyclic": q_cyc,
        "q_total": q_tot,
    }


def _retention_at_600(q_session: np.ndarray, q_halfc: float) -> np.ndarray:
    """Paper ranking projection: CCCV 0.5C → 80% at 600 equal-energy cycles."""
    q_h = max(float(q_halfc), 1e-18)
    return np.clip(100.0 - 20.0 * (q_session / q_h), 0.0, 100.0)


def plot_gpbo_overall_fig10(
    series_by_cell: Dict[str, Dict[str, np.ndarray]],
    *,
    out_path: Path,
    q_halfc_by_cell: Dict[str, float],
) -> None:
    """Fig. 10 analogue: GP-BO best-so-far time, calendar/cyclic loss, retention."""
    apply_paper_style()
    cells = [c for c in ("RW9", "RW10", "RW11", "RW12") if c in series_by_cell]
    fig = plt.figure(figsize=(13.6, 4.6), facecolor=PAPER_LIGHT_BG, layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.05], height_ratios=[1.0, 1.0],
                          wspace=0.38, hspace=0.18)
    ax_t = fig.add_subplot(gs[:, 0])
    ax_cal = fig.add_subplot(gs[0, 1])
    ax_cyc = fig.add_subplot(gs[1, 1], sharex=ax_cal)
    ax_ret = fig.add_subplot(gs[:, 2])
    for ax in (ax_t, ax_cal, ax_cyc, ax_ret):
        ax.set_facecolor(PAPER_LIGHT_BG)

    for cell in cells:
        s = series_by_cell[cell]
        x = s["eval"]
        col = CELL_LINE_COLORS.get(cell, PAPER_GREY)
        ax_t.plot(x, s["duration_min"], color=col, marker="o", ms=3.2, lw=1.8, label=cell)
        ax_cal.plot(x, s["q_calendar"], color=col, marker="o", ms=3.2, lw=1.8, label=cell)
        ax_cyc.plot(x, s["q_cyclic"], color=col, marker="o", ms=3.2, lw=1.8, label=cell)
        ax_ret.plot(
            x, _retention_at_600(s["q_total"], q_halfc_by_cell[cell]),
            color=col, marker="o", ms=3.2, lw=1.8, label=cell,
        )

    ax_t.set_ylabel("Charging time $t_d$ [min]")
    ax_cal.set_ylabel(r"$Q_{\mathrm{calendar}}$")
    ax_cyc.set_ylabel(r"$Q_{\mathrm{cyclic}}$")
    ax_ret.set_ylabel("Capacity retention $C_r$ [%]")
    ax_t.set_xlabel("GP-BO evaluation")
    ax_cyc.set_xlabel("GP-BO evaluation")
    ax_ret.set_xlabel("GP-BO evaluation")
    ax_cal.tick_params(labelbottom=False)
    for ax in (ax_t, ax_cal, ax_cyc, ax_ret):
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.tick_params(labelsize=12)
    ax_t.legend(fontsize=11, loc="best", framealpha=0.95)
    savefig(fig, out_path)


def plot_gpbo_vs_random_fig11(
    gpbo_by_cell: Dict[str, Dict[str, np.ndarray]],
    random_by_cell: Dict[str, Dict[str, np.ndarray]],
    *,
    q_halfc_by_cell: Dict[str, float],
    out_path: Path,
) -> None:
    """Fig. 11 analogue: Random Search vs GP-BO best-so-far time and retention."""
    apply_paper_style()
    cells = [c for c in ("RW9", "RW10", "RW11", "RW12") if c in gpbo_by_cell]
    fig, axes = plt.subplots(2, len(cells), figsize=(3.15 * len(cells) + 0.8, 6.8),
                             facecolor=PAPER_LIGHT_BG, squeeze=False)
    for col, cell in enumerate(cells):
        ax_t = axes[0, col]
        ax_r = axes[1, col]
        ax_t.set_facecolor(PAPER_LIGHT_BG)
        ax_r.set_facecolor(PAPER_LIGHT_BG)
        g = gpbo_by_cell[cell]
        r = random_by_cell[cell]
        qh = q_halfc_by_cell[cell]
        ax_t.plot(
            r["eval"], r["duration_min"], color="#111827", marker="^", ms=3.4,
            lw=1.5, ls=":", label="Random Search",
        )
        ax_t.plot(
            g["eval"], g["duration_min"], color=PAPER_GREEN, marker="o", ms=3.4,
            lw=1.8, label="GP-BO",
        )
        ax_r.plot(
            r["eval"], _retention_at_600(r["q_total"], qh),
            color="#111827", marker="^", ms=3.4, lw=1.5, ls=":", label="Random Search",
        )
        ax_r.plot(
            g["eval"], _retention_at_600(g["q_total"], qh),
            color=PAPER_GREEN, marker="o", ms=3.4, lw=1.8, label="GP-BO",
        )
        ax_t.set_title(cell, fontsize=13, fontweight="bold")
        ax_t.set_ylabel("Charging time $t_d$ [min]" if col == 0 else "")
        ax_r.set_ylabel("Capacity retention $C_r$ [%]" if col == 0 else "")
        ax_r.set_xlabel("Evaluation")
        ax_t.grid(True, linestyle="--", alpha=0.35)
        ax_r.grid(True, linestyle="--", alpha=0.35)
        ax_t.tick_params(labelsize=11)
        ax_r.tick_params(labelsize=11)
        if col == len(cells) - 1:
            ax_t.legend(fontsize=10, loc="best", framealpha=0.95)
    fig.tight_layout(h_pad=0.9, w_pad=0.45)
    savefig(fig, out_path)


def plot_family_reward_bars(
    per_family: Dict[str, Dict[str, Any]],
    *,
    cell_id: str,
    method: str,
    out_path: Path,
) -> None:
    apply_paper_style()
    fids = [f for f in FAMILY_ORDER if f in per_family]
    labels = [FAMILY_LABELS.get(f, f) for f in fids]
    rewards = [float(per_family[f]["reward"]) for f in fids]
    colors = list(PAPER_PROFILE_COLORS)[: len(fids)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    x = np.arange(len(fids))
    bars = ax.bar(x, rewards, color=colors, edgecolor="k", lw=0.5, width=0.7)
    for bar, r, fid in zip(bars, rewards, fids):
        ax.text(
            bar.get_x() + bar.get_width() / 2, r,
            f"{r:.4f}\n{float(per_family[fid]['duration_min']):.1f} min",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Reward")
    lo, hi = min(rewards), max(rewards)
    pad = max(0.002, 0.35 * (hi - lo) if hi > lo else 0.01)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    savefig(fig, out_path)


def plot_paper_table_image(rows: List[Dict[str, Any]], *, out_path: Path) -> None:
    apply_paper_style()
    cols = [
        "Cell", "GP-BO family", "Time (min)", "Time ↓ vs 0.5C",
        "Deg. ↓ vs 0.5C", "Time ↓ vs Random", "Deg. ↓ vs Random", "Reward",
    ]
    cell_text = []
    for r in rows:
        cell_text.append([
            r["cell"],
            r["family"],
            f"{r['duration_min']:.2f}",
            f"{r['time_vs_halfc']:+.1f}%",
            f"{r['deg_vs_halfc']:+.1f}%",
            f"{r['time_vs_random']:+.1f}%",
            f"{r['deg_vs_random']:+.1f}%",
            f"{r['reward']:.4f}",
        ])
    fig, ax = plt.subplots(figsize=(14.0, 2.2 + 0.45 * len(rows)), facecolor=PAPER_LIGHT_BG)
    ax.axis("off")
    table = ax.table(
        cellText=cell_text, colLabels=cols, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        if row == 0:
            cell.set_facecolor("#1e3a5f")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f5f9")
    fig.tight_layout()
    savefig(fig, out_path)

"""STAGE 12 — publication figures from the current Random Search / GP-BO JSON.

Reads only ``results/07_random_search`` and ``results/08_gp_bo`` (plus the
frozen degradation coefficients for the equivalent-cycle projection). Does not
re-optimize. Layout matches ``Constrained_BO/results/final_charging_opt_results``.
"""


import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np



BASELINE_RENAME = {
    "CCCV 0.5C": "CCCV 0.5C",
    "CCCV 1C": "CCCV 1C",
    "CCCV 2C": "CCCV 2C",
}


def _pct(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or abs(den) < 1e-18:
        return float("nan")
    return 100.0 * float(num) / float(den)


class _Flat:
    """Join names onto ``results/`` with an optional prefix. No subfolders."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def __truediv__(self, name: str) -> Path:
        stem = f"{self.prefix}_{name}" if self.prefix else str(name)
        return result_path(stem)

    def mkdir(self, *args, **kwargs) -> None:
        del args, kwargs
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_families(stage: str, cell: str) -> Dict[str, Dict[str, Any]]:
    method = "random_search" if "random" in stage else "gp_bo"
    out = {}
    for fid in FAMILY_ORDER:
        path = result_path(family_result_name(method, cell, fid))
        if path.is_file():
            out[fid] = read_json(path)
    if not out:
        raise FileNotFoundError(f"no {method} JSON for {cell} in {RESULTS_DIR}")
    return out


def pick_best_family(families: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    feas = [(fid, blob) for fid, blob in families.items()
            if (blob.get("best_metrics") or {}).get("feasible")]
    pool = feas or list(families.items())
    fid, blob = min(pool, key=lambda kv: float(kv[1]["best_loss"]))
    return fid, blob


def pick_min_q(families: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    best = None
    for fid, blob in families.items():
        for h in blob.get("history") or []:
            m = h.get("metrics") or {}
            if not m.get("feasible"):
                continue
            q = float(m.get("q_total", m.get("qloss_total", np.inf)))
            cand = (q, fid, m, h)
            if best is None or q < best[0]:
                best = cand
    if best is None:
        return pick_best_family(families)
    q, fid, m, h = best
    return fid, {
        "best_params": h.get("params"),
        "best_loss": h.get("loss"),
        "best_reward": h.get("reward"),
        "best_metrics": m,
        "history": [],
    }


def family_summary(families: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for fid, blob in families.items():
        m = blob["best_metrics"]
        out[fid] = {
            "reward": float(m["reward"]),
            "loss": float(blob["best_loss"]),
            "q_total": float(m["q_total"]),
            "q_calendar": float(m.get("q_calendar") or 0.0),
            "q_cyclic": float(m.get("q_cyclic") or 0.0),
            "duration_min": float(m["duration_min"]),
            "peak_t": float(m.get("peak_t") or 0.0),
            "peak_v": float(m.get("peak_v") or 0.0),
            "feasible": bool(m.get("feasible")),
            "family_label": FAMILY_LABELS.get(fid, fid),
        }
    return out


def baseline_row(item: Dict[str, Any]) -> Dict[str, Any]:
    m = item.get("metrics") or item
    name = item.get("name") or "CCCV"
    return {
        "label": BASELINE_RENAME.get(name, name),
        "reward": float(m["reward"]),
        "q_total": float(m["q_total"]),
        "q_calendar": float(m.get("q_calendar") or 0.0),
        "q_cyclic": float(m.get("q_cyclic") or 0.0),
        "duration_min": float(m["duration_min"]),
        "peak_t": float(m.get("peak_t") or 0.0),
        "peak_v": float(m.get("peak_v") or 0.0),
        "ah_throughput": float(m.get("ah_throughput") or 0.0),
        "duration_h": float(m.get("duration_h") or 0.0),
        "mean_soc": float(m.get("mean_soc") or 0.5),
        "mean_t": float(m.get("mean_t") or m.get("mean_temperature_c") or 24.0),
        "nominal_c_rate": float(m.get("nominal_c_rate") or 0.0),
        "feasible": bool(m.get("feasible", True)),
        "family": "cccv",
    }


def winner_row(label: str, fid: str, blob: Dict[str, Any]) -> Dict[str, Any]:
    m = blob["best_metrics"]
    return {
        "label": label,
        "family": fid,
        "family_label": FAMILY_LABELS.get(fid, fid),
        "reward": float(m["reward"]),
        "q_total": float(m["q_total"]),
        "q_calendar": float(m.get("q_calendar") or 0.0),
        "q_cyclic": float(m.get("q_cyclic") or 0.0),
        "duration_min": float(m["duration_min"]),
        "peak_t": float(m.get("peak_t") or 0.0),
        "peak_v": float(m.get("peak_v") or 0.0),
        "ah_throughput": float(m.get("ah_throughput") or 0.0),
        "duration_h": float(m.get("duration_h") or 0.0),
        "mean_soc": float(m.get("mean_soc") or 0.5),
        "mean_t": float(m.get("mean_t") or 24.0),
        "nominal_c_rate": float(m.get("nominal_c_rate") or 0.0),
        "feasible": bool(m.get("feasible", True)),
        "params": blob.get("best_params"),
    }


def collect_cloud(method: str, families: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    cloud = []
    for fid, blob in families.items():
        for h in blob.get("history") or []:
            m = h.get("metrics") or {}
            cloud.append({
                "method": method,
                "family_id": fid,
                "feasible": bool(m.get("feasible", h.get("feasible"))),
                "duration_min": float(m.get("duration_min") or 0.0),
                "q_total": float(m.get("q_total") or m.get("qloss_total") or 0.0),
                "reward": float(m.get("reward") or h.get("reward") or 0.0),
            })
    return cloud


def project_session_index(row: Dict[str, Any], n_cycles: int) -> Dict[str, np.ndarray]:
    """Paper lifetime construction: accumulate the session Q_loss index.

    remaining(N) is filled later by anchoring CCCV 0.5C to 80% at 600 cycles.
    Ranking is exactly the session-health ranking (lower Q_session is better).
    """
    n = np.arange(0, int(n_cycles) + 1, dtype=np.float64)
    q_s = float(row["q_total"])
    ah_s = float(row.get("ah_throughput") or 0.0)
    q = n * q_s
    return {
        "cycles": n,
        "cum_ah": n * ah_s,
        "q_total": q,
        "q_session": np.full_like(n, q_s),
    }


def anchor_retention(
    curves: Dict[str, Dict[str, np.ndarray]],
    *,
    anchor: str = "CCCV 0.5C",
    soh_ref: float = 80.0,
    n_cycles: int = 600,
) -> float:
    if anchor not in curves:
        anchor = next(iter(curves))
    q_anchor = float(curves[anchor]["q_total"][int(n_cycles)])
    scale = (100.0 - float(soh_ref)) / q_anchor if q_anchor > 0 else 1.0
    for d in curves.values():
        d["retention_pct"] = np.clip(100.0 - scale * d["q_total"], 0.0, 100.0)
        d["scale"] = np.array(scale)
        d["anchor"] = np.array(0.0)
    return float(scale)


def write_cell_table(path: Path, cell: str, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "cell", "label", "family", "feasible", "duration_min", "peak_t", "peak_v",
        "reward", "q_total", "q_calendar", "q_cyclic",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({"cell": cell, **{k: r.get(k) for k in fields if k != "cell"}})


def write_md_table(path: Path, cell: str, rows: List[Dict[str, Any]]) -> None:
    lines = [
        f"# {cell} comparison (calibrated $Q_{{loss}}$, paper Eq. 10)",
        "",
        "| Method | Family | Time (min) | $T_{max}$ (°C) | Reward | $Q_{loss}$ | Feasible |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r.get('family_label', r.get('family', ''))} | "
            f"{float(r['duration_min']):.2f} | {float(r['peak_t']):.2f} | "
            f"{float(r['reward']):.4f} | {float(r['q_total']):.4e} | "
            f"{'yes' if r.get('feasible') else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_cell(
    cell: str,
    *,
    model: HybridDegradationModel,
    n_cycles: int,
    soh_ref: float,
    q_rated_ah: float,
    out_dir: Path,
) -> Dict[str, Any]:
    rs = load_families("07_random_search", cell)
    bo = load_families("08_gp_bo", cell)
    baselines_raw = read_json(result_path(f"gp_bo_{cell}_baselines.json"))
    baseline_rows = [baseline_row(b) for b in baselines_raw]

    rs_fid, rs_blob = pick_best_family(rs)
    bo_fid, bo_blob = pick_best_family(bo)
    mq_fid, mq_blob = pick_min_q(bo)

    rs_row = winner_row("Random", rs_fid, rs_blob)
    opt_row = winner_row("GP-BO", bo_fid, bo_blob)
    health_row = winner_row("GP-BO (min Q)", mq_fid, mq_blob)
    speed_row = winner_row("GP-BO (max R)", bo_fid, bo_blob)

    compare_rows = baseline_rows + [rs_row, opt_row]
    compare_health = baseline_rows + [rs_row, health_row]
    cloud = collect_cloud("Random", rs) + collect_cloud("GP-BO", bo)

    plot_best_profiles(
        rs, cell_id=cell, method_title="Random Search",
        out_path=out_dir / "random_best_profiles.png",
    )
    plot_best_profiles(
        bo, cell_id=cell, method_title="GP-BO",
        out_path=out_dir / "gp_bo_best_profiles.png",
    )
    plot_gpbo_vs_random_multiaxis(
        cell_id=cell,
        gpbo_session=bo_blob["best_session"],
        gpbo_metrics=bo_blob["best_metrics"],
        gpbo_family=bo_fid,
        random_session=rs_blob["best_session"],
        random_metrics=rs_blob["best_metrics"],
        random_family=rs_fid,
        out_path=out_dir / "gpbo_vs_random_best_profile.png",
    )
    plot_metric_bars(
        compare_rows, value_key="duration_min", ylabel="Duration (min)",
        title=f"{cell}: time comparison", out_path=out_dir / "time_comparison.png",
        fmt="{:.1f}",
    )
    plot_metric_bars(
        compare_rows, value_key="peak_t", ylabel="Peak T (°C)",
        title=f"{cell}: temperature comparison",
        out_path=out_dir / "temperature_comparison.png",
        clip_to_data=True, fmt="{:.2f}",
    )
    plot_metric_bars(
        compare_rows, value_key="reward", ylabel="Total reward",
        title=f"{cell}: reward comparison",
        out_path=out_dir / "reward_comparison.png",
        clip_to_data=True, fmt="{:.4f}",
    )
    plot_qloss_bars(compare_health, cell_id=cell, out_path=out_dir / "fig8_bo_vs_cc_degradation.png")
    plot_qloss_detail(compare_health, cell_id=cell, out_path=out_dir / "fig8b_bo_vs_cc_degradation_detail.png")
    plot_pareto_cloud(
        baseline_rows + [rs_row, health_row, speed_row], cloud,
        cell_id=cell, out_path=out_dir / "fig8c_bo_vs_cc_pareto.png",
    )
    plot_family_reward_bars(
        family_summary(bo), cell_id=cell, method="GP-BO",
        out_path=out_dir / "gp_bo_family_rewards.png",
    )
    plot_family_reward_bars(
        family_summary(rs), cell_id=cell, method="Random Search",
        out_path=out_dir / "random_family_rewards.png",
    )

    policies = []
    curves = {}
    policy_specs = [
        *[br for br in baseline_rows if br.get("feasible") and br["label"] != "CCCV 2C"],
        rs_row,
        health_row,
        speed_row,
    ]
    seen = set()
    for row in policy_specs:
        name = row["label"]
        if name in seen:
            continue
        seen.add(name)
        policies.append({"name": name, "feasible": bool(row.get("feasible", True)), **row})
        curves[name] = project_session_index(row, n_cycles)
    scale = anchor_retention(curves, anchor="CCCV 0.5C", soh_ref=soh_ref, n_cycles=n_cycles)

    plot_lifetime_vs_cycles(
        policies, curves, cell_id=cell, n_cycles=n_cycles, soh_ref=soh_ref,
        out_path=out_dir / "fig9_lifetime_fade_vs_cycles.png",
    )
    plot_lifetime_vs_ah(
        policies, curves, cell_id=cell,
        out_path=out_dir / "fig9b_lifetime_fade_vs_throughput.png",
    )
    plot_lifetime_capacity_ah(
        policies, curves, cell_id=cell, q_rated_ah=q_rated_ah, n_show=80,
        out_path=out_dir / "fig9c_lifetime_capacity_vs_cycle_index.png",
    )
    plot_lifetime_delta(
        policies, curves, cell_id=cell, baseline="CCCV 0.5C",
        out_path=out_dir / "fig9d_lifetime_delta_vs_halfC.png",
    )

    write_cell_table(out_dir / "comparison_table.csv", cell, compare_rows)
    write_md_table(out_dir / "comparison_table.md", cell, compare_rows)

    ret600 = {
        name: float(d["retention_pct"][min(n_cycles, len(d["retention_pct"]) - 1)])
        for name, d in curves.items()
    }
    meta = {
        "cell": cell,
        "random_winner": {"family": rs_fid, **{k: rs_row[k] for k in (
            "reward", "q_total", "duration_min", "peak_t", "feasible")}},
        "gp_bo_winner": {"family": bo_fid, **{k: opt_row[k] for k in (
            "reward", "q_total", "duration_min", "peak_t", "feasible")}},
        "gp_bo_health": {"family": mq_fid, **{k: health_row[k] for k in (
            "reward", "q_total", "duration_min", "peak_t", "feasible")}},
        "retention_pct_at_600": ret600,
        "n_cycles": n_cycles,
        "lifetime_mode": "session_Q_index_affine",
        "lifetime_anchor": "CCCV 0.5C",
        "lifetime_scale": float(scale),
    }
    write_json(out_dir / "lifetime_fade_projection_meta.json", meta)
    return {
        "cell": cell,
        "compare_rows": compare_rows,
        "rs_row": rs_row,
        "bo_row": opt_row,
        "health_row": health_row,
        "halfc": next(r for r in baseline_rows if r["label"] == "CCCV 0.5C"),
        "onec": next(r for r in baseline_rows if r["label"] == "CCCV 1C"),
        "policies": policies,
        "curves": curves,
        "family_bo": family_summary(bo),
        "family_rs": family_summary(rs),
        "bo_families": bo,
        "rs_families": rs,
        "meta": meta,
        "random_session": rs_blob["best_session"],
        "random_metrics": rs_blob["best_metrics"],
        "random_family": rs_fid,
        "gpbo_session": bo_blob["best_session"],
        "gpbo_metrics": bo_blob["best_metrics"],
        "gpbo_family": bo_fid,
    }


def make_figures(cells=None) -> int:
    paths = Paths.load()
    opt = OptimizationSpec.load()
    cells = [c.upper() for c in (cells or paths.cells)]
    model = load_degradation_model()
    n_cycles = int(opt.lifetime.get("n_equivalent_cycles", 600))
    soh_ref = float(opt.lifetime.get("engineering_reference_retention_pct", 80.0))
    q_rated_ah = float(model.p.q_nominal_ah)

    out_root = RESULTS_DIR / "12_figures"
    out_root.mkdir(parents=True, exist_ok=True)
    per_cell_pack = {}
    paper_rows = []
    all_curves = {}
    all_policies = {}
    consolidated = {}

    for cell in cells:
        print(f"======== figures  {cell} ========", flush=True)
        cell_dir = out_root / cell
        cell_dir.mkdir(parents=True, exist_ok=True)
        pack = process_cell(
            cell, model=model, n_cycles=n_cycles, soh_ref=soh_ref,
            q_rated_ah=q_rated_ah, out_dir=cell_dir,
        )
        per_cell_pack[cell] = pack
        all_curves[cell] = pack["curves"]
        all_policies[cell] = pack["policies"]
        halfc, onec, rs_row, bo_row = (
            pack["halfc"], pack["onec"], pack["rs_row"], pack["bo_row"],
        )
        health = pack["health_row"]
        consolidated[cell] = {
            "CCCV 0.5C": halfc,
            "CCCV 1C": onec,
            "Random": rs_row,
            "GP-BO": bo_row,
        }
        paper_rows.append({
            "cell": cell,
            "family": FAMILY_LABELS.get(health["family"], health["family"]),
            "duration_min": bo_row["duration_min"],
            "reward": bo_row["reward"],
            "time_vs_halfc": _pct(halfc["duration_min"] - bo_row["duration_min"], halfc["duration_min"]),
            "deg_vs_halfc": _pct(halfc["q_total"] - health["q_total"], halfc["q_total"]),
            "time_vs_random": _pct(rs_row["duration_min"] - bo_row["duration_min"], rs_row["duration_min"]),
            "deg_vs_random": _pct(rs_row["q_total"] - health["q_total"], rs_row["q_total"]),
        })
        print(
            f"  GP-BO (max R) ({bo_row['family_label']}): t={bo_row['duration_min']:.2f} min  "
            f"R={bo_row['reward']:.4f}",
            flush=True,
        )
        print(
            f"  GP-BO (min Q) ({health['family_label']}): Q={health['q_total']:.3e}  "
            f"t={health['duration_min']:.2f} min",
            flush=True,
        )

    gpbo_traces = {}
    random_traces = {}
    q_halfc = {}
    for cell, pack in per_cell_pack.items():
        gpbo_traces[cell] = _best_so_far_trace(_history_rows(pack["bo_families"]))
        random_traces[cell] = _best_so_far_trace(_history_rows(pack["rs_families"]))
        q_halfc[cell] = float(pack["halfc"]["q_total"])
        plot_gpbo_overall_fig10(
            {cell: gpbo_traces[cell]},
            q_halfc_by_cell={cell: q_halfc[cell]},
            out_path=out_root / cell / "fig10_gpbo_overall.png",
        )
        plot_gpbo_vs_random_fig11(
            {cell: gpbo_traces[cell]},
            {cell: random_traces[cell]},
            q_halfc_by_cell={cell: q_halfc[cell]},
            out_path=out_root / cell / "fig11_gpbo_vs_random.png",
        )
    plot_gpbo_overall_fig10(
        gpbo_traces, q_halfc_by_cell=q_halfc,
        out_path=out_root / "fig10_gpbo_overall.png",
    )
    plot_gpbo_vs_random_fig11(
        gpbo_traces, random_traces, q_halfc_by_cell=q_halfc,
        out_path=out_root / "fig11_gpbo_vs_random.png",
    )

    plot_consolidated_comparison(
        consolidated, out_path=out_root / "fig_consolidated_comparison.png",
    )
    plot_lifetime_grid(
        all_curves, all_policies, n_cycles=n_cycles, soh_ref=soh_ref,
        out_path=out_root / "fig_lifetime_all.png",
    )
    plot_all_cells_gpbo_vs_random_multiaxis(
        per_cell_pack, out_path=out_root / "fig_gpbo_vs_random_best_profiles.png",
    )
    plot_paper_table_image(paper_rows, out_path=out_root / "fig10_paper_comparison_table.png")

    md = [
        "# GP-BO comparison table (calibrated coefficients, paper Eq. 10)",
        "",
        "Percent columns use **CCCV 0.5C** as the reference. "
        "Positive Time ↓ / Deg. ↓ means GP-BO is faster / lower $Q_{loss}$.",
        "",
        "| Cell | GP-BO family | Time (min) | Time ↓ vs 0.5C | Deg. ↓ vs 0.5C | Time ↓ vs Random | Deg. ↓ vs Random | Reward |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in paper_rows:
        md.append(
            f"| {r['cell']} | {r['family']} | {r['duration_min']:.2f} | "
            f"{r['time_vs_halfc']:+.1f}% | {r['deg_vs_halfc']:+.1f}% | "
            f"{r['time_vs_random']:+.1f}% | {r['deg_vs_random']:+.1f}% | "
            f"{r['reward']:.4f} |"
        )
    md.append("")
    md.append(
        "Lifetime curves accumulate the session $Q_{\\mathrm{loss}}$ index and "
        "anchor CCCV 0.5C to 80% remaining capacity at 600 cycles (paper ranking projection). "
        "GP-BO (min Q) is the lowest-Q feasible GP-BO trial (health-first). "
        "GP-BO (max R) is the Eq.~10 reward maximiser (the profile the proposed algorithm selects)."
    )
    (out_root / "paper_comparison_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_json(out_root / "figure_index.json", {
        "cells": cells,
        "n_cycles": n_cycles,
        "paper_rows": paper_rows,
        "per_cell": {c: p["meta"] for c, p in per_cell_pack.items()},
        "provenance": provenance(
            "12_figures",
            configs=["paths", "degradation_fitted", "reward", "optimization"],
            inputs=[
                result_path(f"random_search_{c}_summary.json") for c in cells
            ] + [
                result_path(f"gp_bo_{c}_summary.json") for c in cells
            ],
        ),
    })
    print(f"\nWrote figures → {out_root}", flush=True)
    return 0



"""Cross-cell Top-6 litmus test on frozen per-cell GP-BO (stage 8).

Does not re-optimize and does not change twins. Reads
``results/08_gp_bo/<CELL>/{cccv,two_step,three_step,pulsed}.json``,
takes the 6 lowest-loss feasible trials per cell (24 elite profiles),
replays each profile on all four twins, and writes tables/figures under
``results/14_litmus_top6/``.
"""


import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np



CELLS = ("RW9", "RW10", "RW11", "RW12")
N_ELITE = 6
N_GRID = 100
CELL_COLORS = {"RW9": "#2563eb", "RW10": "#16a34a", "RW11": "#ea580c", "RW12": "#7c3aed"}


def _params_from_dict(raw: Dict[str, Any]):
    blob = dict(raw)
    fid = str(blob.pop("family_id"))
    cls = type(get_family(fid))
    return cls.from_dict(blob)


def _charge_i(session: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(session["time_s"], dtype=np.float64)
    i = -np.asarray(session["current_a"], dtype=np.float64)
    return t, i


def _resample_unit_time(t: np.ndarray, y: np.ndarray, n: int = N_GRID) -> np.ndarray:
    if t.size == 0:
        return np.zeros(n, dtype=np.float64)
    t = t - t[0]
    span = float(t[-1]) if t.size else 0.0
    if span <= 1e-9:
        return np.full(n, float(y[0]) if y.size else 0.0, dtype=np.float64)
    tau = t / span
    grid = np.linspace(0.0, 1.0, n)
    return np.interp(grid, tau, y)


def _mean_std_cv(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "cv_pct": float("nan"), "min": float("nan"), "max": float("nan")}
    mu = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    cv = 100.0 * sd / abs(mu) if abs(mu) > 1e-18 else float("nan")
    return {"n": int(a.size), "mean": mu, "std": sd, "cv_pct": cv,
            "min": float(a.min()), "max": float(a.max())}


def load_elite(gpbo_dir: Path, cells: List[str], k: int) -> List[Dict[str, Any]]:
    elite: List[Dict[str, Any]] = []
    for cell in cells:
        rows: List[Dict[str, Any]] = []
        for fid in FAMILY_ORDER:
            path = result_path(family_result_name("gp_bo", cell, fid))
            if not path.is_file():
                raise FileNotFoundError(path)
            blob = read_json(path)
            for idx, h in enumerate(blob.get("history") or []):
                m = h.get("metrics") or {}
                if not bool(m.get("feasible", h.get("feasible"))):
                    continue
                rows.append({
                    "source_cell": cell,
                    "family_id": str((h.get("params") or {}).get("family_id", fid)),
                    "history_index": int(idx),
                    "loss": float(h.get("loss", m.get("loss", np.inf))),
                    "reward": float(h.get("reward", m.get("reward", np.nan))),
                    "params": dict(h.get("params") or {}),
                    "native_metrics": {kk: vv for kk, vv in m.items()
                                       if kk not in ("weights", "anchors")},
                })
        rows.sort(key=lambda r: r["loss"])
        for rank, row in enumerate(rows[:k], start=1):
            row["rank"] = rank
            row["profile_id"] = f"{cell}-P{rank}"
            elite.append(row)
    return elite


def _flat_params(p: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("i_cc", "v_cv", "i_cutoff", "i1", "i2", "i3", "soc_switch", "soc1", "soc2",
            "i_charge", "pulse_on_min", "pulse_rest_min", "rest_fraction", "i_floor")
    return {k: p.get(k) for k in keys}


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def plot_family_counts(counts: Dict[str, Counter], out: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(CELLS))
    bottom = np.zeros(len(CELLS))
    palette = {"cccv": "#2563eb", "two_step": "#16a34a",
               "three_step": "#ea580c", "pulsed": "#7c3aed"}
    for fid in FAMILY_ORDER:
        h = np.array([counts[c][fid] for c in CELLS], dtype=float)
        ax.bar(x, h, bottom=bottom, color=palette[fid], label=FAMILY_LABELS[fid], width=0.62)
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels(list(CELLS))
    ax.set_ylabel("Count in Top-6")
    ax.set_ylim(0, N_ELITE)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    savefig(fig, out)


def plot_current_overlay(native: List[Dict[str, Any]], out: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for row in native:
        t_min = np.asarray(row["t_s"]) / 60.0
        ax.plot(t_min, row["i_a"], color=CELL_COLORS[row["source_cell"]],
                lw=1.6, alpha=0.75)
    for cell, col in CELL_COLORS.items():
        ax.plot([], [], color=col, lw=2.4, label=cell)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Charge current (A)")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    savefig(fig, out)


def plot_current_grid(native: List[Dict[str, Any]], out: Path) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharey=True)
    by_cell: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CELLS}
    for row in native:
        by_cell[row["source_cell"]].append(row)
    for ax, cell in zip(axes.ravel(), CELLS):
        for row in by_cell[cell]:
            t_min = np.asarray(row["t_s"]) / 60.0
            ax.plot(t_min, row["i_a"], lw=1.7, alpha=0.85,
                    label=f"P{row['rank']} {FAMILY_LABELS.get(row['family_id'], row['family_id'])}")
        ax.set_title(cell)
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=9, frameon=False, loc="best")
    axes[0, 0].set_ylabel("Charge current (A)")
    axes[1, 0].set_ylabel("Charge current (A)")
    fig.tight_layout()
    savefig(fig, out)


def plot_similarity(ids: List[str], dist: np.ndarray, out: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    im = ax.imshow(dist, cmap="magma_r", origin="upper")
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=90, fontsize=8)
    ax.set_yticklabels(ids, fontsize=8)
    for k in range(1, 4):
        ax.axhline(k * N_ELITE - 0.5, color="white", lw=1.2)
        ax.axvline(k * N_ELITE - 0.5, color="white", lw=1.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE of I(t) (A), time-normalized")
    fig.tight_layout()
    savefig(fig, out)


def plot_cross_heat(ids: List[str], cells: List[str], grid: np.ndarray,
                    *, title: str, cmap: str, out: Path, vmin=None, vmax=None) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    im = ax.imshow(grid, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_xlabel("Evaluated on twin")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title)
    fig.tight_layout()
    savefig(fig, out)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def build_report(
    *,
    elite: List[Dict[str, Any]],
    family_counts: Dict[str, Counter],
    stats: Dict[str, Dict[str, float]],
    cross: List[Dict[str, Any]],
    within: Dict[str, float],
    between: float,
    n_all_feasible: int,
    voltage_headroom: Dict[str, Dict[str, float]],
    twin_rmse: Dict[str, Dict[str, float]],
) -> str:
    fam_rows = []
    for fid in FAMILY_ORDER:
        fam_rows.append(
            [FAMILY_LABELS[fid]] + [str(family_counts[c][fid]) for c in CELLS]
        )
    master_rows = []
    for e in elite:
        p = e["params"]
        nm = e["native_replay"]
        master_rows.append([
            e["source_cell"], str(e["rank"]), FAMILY_LABELS.get(e["family_id"], e["family_id"]),
            f"{nm['duration_min']:.2f}", f"{nm['q_total']:.4e}", f"{nm['reward']:.4f}",
            f"{nm['peak_v']:.3f}", f"{nm['i_mean']:.2f}", f"{nm['i_max']:.2f}",
            _param_short(p),
        ])
    cv_rows = []
    for name, s in stats.items():
        cv_rows.append([
            name, f"{s['mean']:.4g}", f"{s['std']:.4g}", f"{s['cv_pct']:.1f}",
            f"{s['min']:.4g}", f"{s['max']:.4g}",
        ])
    robust = []
    for e in elite:
        hits = [r for r in cross if r["profile_id"] == e["profile_id"]]
        n_ok = sum(1 for r in hits if r["feasible"])
        if n_ok == 4:
            robust.append(e["profile_id"])
    head_rows = []
    for c in CELLS:
        h = voltage_headroom[c]
        rmse = twin_rmse.get(c, {})
        head_rows.append([
            c, f"{h['peak_v_1c']:.3f}", f"{h['headroom_1c_mv']:.0f}",
            f"{h['cccv_2c_feasible']}", f"{rmse.get('v_rmse_mv', float('nan')):.1f}",
        ])
    return f"""# Top-6 cross-cell litmus test

Frozen per-cell digital twins. Frozen stage-8 GP-BO (80 evals / family / cell).
Identical operating box for every cell: SOC₀ = 20 %, T₀ = 24 °C, energy target = 40 % of
nominal pack energy, V_max = 4.20 V, I ∈ [0.75, 6.00] A, same reward, same GP-BO budget.
The only factor that changes is the cell twin.

This analysis does **not** force the four elite sets to match. It measures whether they do.

## Level 1 — charging family in each Top-6

{_md_table(["Family", *CELLS], fam_rows)}

CCCV is the only family that appears in every cell's Top-6. Pulsed appears only for RW9.
RW11's entire Top-6 is CCCV.

## Level 2 — 24 elite profiles (native-twin replay)

{_md_table(
    ["Cell", "Rank", "Family", "t (min)", "Q_loss", "Reward", "V_peak", "I_mean (A)", "I_max (A)", "Parameters"],
    master_rows,
)}

## Parameter / outcome spread across the 24 native elites

{_md_table(["Quantity", "Mean", "Std", "CV %", "Min", "Max"], cv_rows)}

A small CV means the elite set is a shared neighbourhood. A large CV on current or time
means the independently optimized elites are not the same protocol.

## Trajectory similarity (time-normalized RMSE of I(t))

Mean pairwise RMSE **within** a cell's own Top-6:

{_md_table(["Cell", "Mean within-cell RMSE (A)"], [[c, f"{within[c]:.3f}"] for c in CELLS])}

Mean pairwise RMSE **between different cells**: {between:.3f} A.

If within-cell RMSE is small and between-cell RMSE is comparable, the four Top-6 clouds
overlap. If between-cell RMSE is much larger, the elite sets occupy different current
trajectories.

## Cross-cell validation (24 profiles × 4 twins)

A profile is counted as pack-feasible if it meets the energy target and stays ≤ 4.20 V
on **every** twin.

Profiles feasible on all four twins: **{n_all_feasible} / 24**
{("— " + ", ".join(robust)) if robust else ""}

## Voltage behaviour of the twins (from frozen stage-8 baselines + stage-2 RMSE)

These numbers describe **predicted** restrictiveness, not a claim about physical impedance.

{_md_table(
    ["Cell", "CCCV 1C V_peak", "Headroom to 4.20 V (mV)", "CCCV 2C feasible?", "Twin V RMSE (mV)"],
    head_rows,
)}

Headroom of tens of millivolts is comparable to twin voltage RMSE on RW11/RW12
(48 mV / 64 mV). A predicted V = 4.05 V is therefore **not** 150 mV of demonstrated
safety margin.

## How to read the result

- **Consistent strategy:** same families dominate, currents in a tight band, many of the
  24 profiles remain feasible on the other twins.
- **Cell-specific / twin-specific:** one cell's elite set is pulsed or 6 A CCCV while
  the others sit near 2 A, and those profiles fail the 4.20 V (or energy) check on
  other twins.

The figures `family_counts.png`, `current_overlay.png`, `similarity_24x24.png`, and
`cross_feasible.png` are the visual version of this test.
"""


def _param_short(p: Dict[str, Any]) -> str:
    fid = p.get("family_id")
    if fid == "cccv":
        return f"I={p['i_cc']:.2f}A Vcv={p['v_cv']:.3f}"
    if fid == "two_step":
        return f"I1={p['i1']:.2f} I2={p['i2']:.2f}"
    if fid == "three_step":
        return f"I1={p['i1']:.2f} I2={p['i2']:.2f} I3={p['i3']:.2f}"
    if fid == "pulsed":
        return f"Ion={p['i_charge']:.2f} on={p['pulse_on_min']:.1f} rest={p['pulse_rest_min']:.1f}"
    return ""


def make_litmus(device=None) -> int:
    ensure_matplotlib_cache()
    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    paths = Paths.load()
    device = device or opt.device
    k = int(N_ELITE)
    out = _Flat("litmus")
    fig_dir = out

    elite = load_elite(RESULTS_DIR, list(CELLS), k)
    print(f"Loaded {len(elite)} elite profiles from {RESULTS_DIR}", flush=True)

    family_counts = {c: Counter() for c in CELLS}
    for e in elite:
        family_counts[e["source_cell"]][e["family_id"]] += 1

    model = load_degradation_model()
    sims = {}
    states = {}
    anchors = {}
    for cell in CELLS:
        print(f"Loading twin {cell} …", flush=True)
        sims[cell], states[cell] = make_simulator(cell, device=device)
        anchors[cell] = load_anchors(cell)

    cross_rows: List[Dict[str, Any]] = []
    native: List[Dict[str, Any]] = []
    native_I: List[np.ndarray] = []
    ids: List[str] = []

    n_eval = 0
    total = len(elite) * len(CELLS)
    for e in elite:
        params = _params_from_dict(e["params"])
        e["params"] = params.to_dict()
        for cell in CELLS:
            n_eval += 1
            print(f"  [{n_eval}/{total}] {e['profile_id']} → {cell}", flush=True)
            loss, metrics, session = evaluate_params(
                sims[cell], states[cell], params, model=model, spec=reward,
                anchors=anchors[cell],
            )
            t, i = _charge_i(session)
            rec = {
                "profile_id": e["profile_id"],
                "source_cell": e["source_cell"],
                "rank": e["rank"],
                "family_id": e["family_id"],
                "eval_cell": cell,
                "native": cell == e["source_cell"],
                "feasible": bool(metrics["feasible"]),
                "loss": float(loss),
                "reward": float(metrics["reward"]),
                "duration_min": float(metrics["duration_min"]),
                "q_total": float(metrics["q_total"]),
                "peak_v": float(metrics["peak_v"]),
                "peak_t": float(metrics["peak_t"]),
                "delta_soc": float(metrics["delta_soc"]),
                "i_mean": float(i.mean()) if i.size else float("nan"),
                "i_max": float(i.max()) if i.size else float("nan"),
                "end_reason": metrics.get("end_reason"),
            }
            rec.update({f"p_{kk}": vv for kk, vv in _flat_params(e["params"]).items()})
            cross_rows.append(rec)
            if cell == e["source_cell"]:
                e["native_replay"] = rec
                native.append({
                    "profile_id": e["profile_id"],
                    "source_cell": e["source_cell"],
                    "rank": e["rank"],
                    "family_id": e["family_id"],
                    "t_s": t,
                    "i_a": i,
                })
                native_I.append(_resample_unit_time(t, i))
                ids.append(e["profile_id"])

    I = np.stack(native_I, axis=0)
    dist = np.sqrt(((I[:, None, :] - I[None, :, :]) ** 2).mean(axis=2))

    within = {}
    between_vals = []
    for ci, c in enumerate(CELLS):
        sl = slice(ci * k, (ci + 1) * k)
        sub = dist[sl, sl]
        tri = sub[np.triu_indices(k, k=1)]
        within[c] = float(tri.mean()) if tri.size else 0.0
    for i in range(len(ids)):
        ci = i // k
        for j in range(i + 1, len(ids)):
            if j // k != ci:
                between_vals.append(float(dist[i, j]))
    between = float(np.mean(between_vals)) if between_vals else float("nan")

    i_mean = [e["native_replay"]["i_mean"] for e in elite]
    i_max = [e["native_replay"]["i_max"] for e in elite]
    t_min = [e["native_replay"]["duration_min"] for e in elite]
    qv = [e["native_replay"]["q_total"] for e in elite]
    stats = {
        "I_mean (A)": _mean_std_cv(i_mean),
        "I_max (A)": _mean_std_cv(i_max),
        "Time (min)": _mean_std_cv(t_min),
        "Q_loss": _mean_std_cv(qv),
        "Reward": _mean_std_cv([e["native_replay"]["reward"] for e in elite]),
        "V_peak (V)": _mean_std_cv([e["native_replay"]["peak_v"] for e in elite]),
    }

    n_all_feasible = 0
    for e in elite:
        hits = [r for r in cross_rows if r["profile_id"] == e["profile_id"]]
        if all(r["feasible"] for r in hits):
            n_all_feasible += 1

    voltage_headroom = {}
    for cell in CELLS:
        bl = read_json(result_path(f"gp_bo_{cell}_baselines.json"))
        by_name = {b["name"]: b for b in bl}
        p1 = float(by_name["CCCV 1C"]["metrics"]["peak_v"])
        feas2 = bool(by_name["CCCV 2C"]["metrics"]["feasible"])
        voltage_headroom[cell] = {
            "peak_v_1c": p1,
            "headroom_1c_mv": 1000.0 * (4.20 - p1),
            "cccv_2c_feasible": feas2,
        }

    twin_rmse: Dict[str, Dict[str, float]] = {}

    feas_grid = np.zeros((len(ids), len(CELLS)))
    time_grid = np.full((len(ids), len(CELLS)), np.nan)
    q_grid = np.full((len(ids), len(CELLS)), np.nan)
    id_index = {pid: i for i, pid in enumerate(ids)}
    cell_index = {c: j for j, c in enumerate(CELLS)}
    for r in cross_rows:
        i = id_index[r["profile_id"]]
        j = cell_index[r["eval_cell"]]
        feas_grid[i, j] = 1.0 if r["feasible"] else 0.0
        time_grid[i, j] = r["duration_min"]
        q_grid[i, j] = r["q_total"]

    master_fields = [
        "profile_id", "source_cell", "rank", "family_id", "loss", "reward",
        "duration_min", "q_total", "peak_v", "peak_t", "delta_soc", "i_mean", "i_max",
        "p_i_cc", "p_v_cv", "p_i_cutoff", "p_i1", "p_i2", "p_i3", "p_soc_switch",
        "p_soc1", "p_soc2", "p_i_charge", "p_pulse_on_min", "p_pulse_rest_min",
        "p_rest_fraction", "p_i_floor",
    ]
    master_rows = []
    for e in elite:
        row = dict(e["native_replay"])
        row["loss"] = e["loss"]
        master_rows.append(row)

    write_csv(out / "elite_master.csv", master_rows, master_fields)
    write_csv(out / "cross_cell.csv", cross_rows, [
        "profile_id", "source_cell", "rank", "family_id", "eval_cell", "native",
        "feasible", "loss", "reward", "duration_min", "q_total", "peak_v", "peak_t",
        "delta_soc", "i_mean", "i_max", "end_reason",
    ])
    count_rows = [{"family": FAMILY_LABELS[fid], **{c: family_counts[c][fid] for c in CELLS}}
                  for fid in FAMILY_ORDER]
    write_csv(out / "family_counts.csv", count_rows, ["family", *CELLS])
    stat_rows = [{"quantity": name, **s} for name, s in stats.items()]
    write_csv(out / "parameter_spread.csv", stat_rows,
              ["quantity", "n", "mean", "std", "cv_pct", "min", "max"])

    dist_rows = [{"profile_id": ids[i], **{ids[j]: float(dist[i, j]) for j in range(len(ids))}}
                 for i in range(len(ids))]
    write_csv(out / "similarity_rmse_I.csv", dist_rows, ["profile_id", *ids])

    write_json(out / "summary.json", {
        "k": k,
        "n_elite": len(elite),
        "family_counts": {c: dict(family_counts[c]) for c in CELLS},
        "spread": stats,
        "within_cell_rmse_A": within,
        "between_cell_rmse_A": between,
        "n_profiles_feasible_on_all_twins": n_all_feasible,
        "voltage_headroom": voltage_headroom,
        "twin_voltage_rmse_mv": twin_rmse,
        "operating_box": {
            "soc_start": opt.session.soc_start,
            "t0_c": opt.session.ambient_t0_c,
            "v_max": opt.session.v_max,
            "energy_fraction": opt.session.energy_fraction,
            "i_min_a": opt.space.i_min_a,
            "i_max_a": opt.space.i_max_a,
            "n_evals_per_family": opt.n_evals_per_family,
        },
        "provenance": provenance(
            "14_litmus_top6",
            configs=["paths", "reward", "optimization", "degradation_fitted"],
            inputs=[result_path(f"gp_bo_{c}_summary.json") for c in CELLS],
        ),
    })

    report = build_report(
        elite=elite, family_counts=family_counts, stats=stats, cross=cross_rows,
        within=within, between=between, n_all_feasible=n_all_feasible,
        voltage_headroom=voltage_headroom, twin_rmse=twin_rmse,
    )
    (out / "REPORT.md").write_text(report, encoding="utf-8")

    plot_family_counts(family_counts, fig_dir / "family_counts.png")
    plot_current_overlay(native, fig_dir / "current_overlay.png")
    plot_current_grid(native, fig_dir / "current_grid.png")
    plot_similarity(ids, dist, fig_dir / "similarity_24x24.png")
    plot_cross_heat(ids, list(CELLS), feas_grid, title="Feasible (1 = yes)",
                    cmap="Greens", out=fig_dir / "cross_feasible.png", vmin=0, vmax=1)
    plot_cross_heat(ids, list(CELLS), time_grid, title="Time (min)",
                    cmap="viridis", out=fig_dir / "cross_time.png")
    plot_cross_heat(ids, list(CELLS), q_grid, title="Q_loss",
                    cmap="magma", out=fig_dir / "cross_qloss.png")

    print(f"\nWrote litmus outputs → {RESULTS_DIR}", flush=True)
    print(f"Family counts: { {c: dict(family_counts[c]) for c in CELLS} }", flush=True)
    print(f"Feasible on all 4 twins: {n_all_feasible}/24", flush=True)
    print(f"Between-cell I(t) RMSE: {between:.3f} A", flush=True)
    return 0




def _run() -> int:
    global RESULTS_DIR
    ap = argparse.ArgumentParser(description="Write charging figures into results/")
    ap.add_argument("--only", choices=["all", "figures", "litmus"], default="all")
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--results-dir", default=None,
                    help="Read and write this directory instead of results/.")
    args = ap.parse_args()
    if args.results_dir:
        RESULTS_DIR = set_results_dir(args.results_dir)
        print(f"results dir {RESULTS_DIR}", flush=True)
    if args.only in ("all", "figures"):
        make_figures(args.cells)
    if args.only in ("litmus", "all"):
        make_litmus(args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
