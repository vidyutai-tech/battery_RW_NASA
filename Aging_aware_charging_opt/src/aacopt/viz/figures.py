"""Plotters for charging-optimization figures (paper layout, calibrated JSON)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from aacopt.viz.style import (
    FAMILY_LABELS,
    FAMILY_ORDER,
    GROUP_COLORS,
    PAPER_DPI,
    PAPER_GREEN,
    PAPER_GREY,
    PAPER_LIGHT_BG,
    PAPER_PROFILE_COLORS,
    POLICY_STYLE,
    apply_paper_style,
    savefig,
)


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
        metrics = res["best_metrics"]
        t_min = _as_array(session["time_s"]) / 60.0
        label = FAMILY_LABELS.get(fid, fid)
        header = (
            f"{label}\n"
            f"{float(metrics['duration_min']):.1f} min  ·  "
            f"{'feasible' if metrics['feasible'] else 'infeasible'}\n"
            f"R={float(metrics['reward']):.4f}  Q={float(metrics['q_total']):.3e}"
        )
        axes[0, col].set_title(header, fontsize=11, fontweight="bold", pad=8, linespacing=1.35)
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

    fig.suptitle(
        f"Best charging profiles — {cell_id} ({method_title})",
        fontsize=16, fontweight="bold", y=0.99,
    )
    fig.tight_layout(h_pad=0.75, w_pad=0.35, rect=(0, 0, 1, 0.97))
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
    fam = FAMILY_LABELS.get(family_id, family_id)
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

    header = (
        f"{method_name}  ·  {fam}\n"
        f"{float(metrics['duration_min']):.1f} min   "
        f"R={float(metrics['reward']):.4f}   "
        f"Q={float(metrics['q_total']):.3e}"
    )
    host.set_title(header, fontsize=18, fontweight="bold", pad=12, loc="left")


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
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 14.2), facecolor=PAPER_LIGHT_BG)
    fig.subplots_adjust(left=0.22, right=0.76, hspace=0.52, top=0.93, bottom=0.06)
    _draw_multiaxis_session(
        axes[0], random_session, random_metrics,
        family_id=random_family, method_name="Random",
    )
    _draw_multiaxis_session(
        axes[1], gpbo_session, gpbo_metrics,
        family_id=gpbo_family, method_name="GP-BO (max R)",
    )
    fig.suptitle(
        f"{cell_id}: best equal-energy profiles",
        fontsize=20, fontweight="bold", y=0.985,
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
    fig.subplots_adjust(left=0.22, right=0.76, hspace=0.55, top=0.97, bottom=0.03)
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
    fig.suptitle(
        "Best charging profiles: Random Search vs GP-BO (max R)",
        fontsize=20, fontweight="bold", y=0.995,
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
    ax.set_title(title, fontweight="bold", fontsize=15)
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
            if row["label"] in ("Random", "GP-BO") and row.get("feasible"):
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
    ax.set_title(
        f"Charging policy vs degradation — {cell_id}\n"
        "CCCV 0.5C / 1C / 2C   vs   Random best   vs   GP-BO best",
        fontsize=14, fontweight="bold",
    )
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
    ax.set_title(f"{cell_id}: degradation per equal-energy session", fontsize=13, fontweight="bold")
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
    ax2.set_title("Calendar vs cyclic split", fontsize=13, fontweight="bold")
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
    ax.set_title(
        f"{cell_id}: search cloud + baselines (equal-energy when feasible)\n"
        "Lower-left is better (faster + less degradation).",
        fontsize=14, fontweight="bold",
    )
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
    ax.set_xlabel("Equivalent charge cycles (one 40% energy session each)", fontsize=13)
    ax.set_ylabel("Projected remaining capacity [%]", fontsize=14)
    ax.set_title(
        f"{cell_id}: projected remaining capacity\n"
        f"(session $Q_{{loss}}$ accumulated · CCCV 0.5C anchored to {soh_ref:.0f}% at {n_cycles} cycles)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, n_cycles)
    ax.set_ylim(60.0, 100.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=12, loc="best", framealpha=0.95)
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
    ax.set_xlabel("Equivalent charge cycles", fontsize=14)
    ax.set_ylabel(f"Δ remaining capacity vs {baseline} [%-points]", fontsize=13)
    ax.set_title(f"{cell_id}: capacity retention vs {baseline}", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=12, loc="best")
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
    ax.set_xlabel("Cumulative ampere-hour throughput [Ah]", fontsize=13)
    ax.set_ylabel("Projected remaining capacity [%]", fontsize=13)
    ax.set_title(f"{cell_id}: fade vs throughput", fontsize=13, fontweight="bold")
    ax.set_ylim(60.0, 100.0)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=11, loc="best", framealpha=0.95)
    ax.set_ylim(60.0, 100.0)
    ax_zoom.set_xlabel("Cumulative Ah", fontsize=13)
    ax_zoom.set_ylabel("Projected remaining capacity [%]", fontsize=13)
    ax_zoom.set_title("Policy comparison", fontsize=13, fontweight="bold")
    ax_zoom.set_ylim(60.0, 100.0)
    ax_zoom.grid(True, linestyle="--", alpha=0.35)
    ax_zoom.legend(fontsize=11, loc="best")
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
    ax.set_xlabel("Equivalent charge cycle index", fontsize=14)
    ax.set_ylabel("Projected capacity [Ah]", fontsize=14)
    ax.set_title(f"{cell_id}: capacity vs cycle (unscaled model projection)", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()
    savefig(fig, out_path)


def plot_consolidated_comparison(
    per_cell: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    out_path: Path,
) -> None:
    """Three-panel: time, peak T, reward for CCCV 0.5C / Random / GP-BO."""
    apply_paper_style()
    cells = list(per_cell.keys())
    series = ("CCCV 0.5C", "Random", "GP-BO")
    colors = [GROUP_COLORS[s] for s in series]
    metrics = [
        ("duration_min", "Charging time (min)", "(a) Charging time"),
        ("peak_t", "Peak temperature (°C)", "(b) Peak temperature"),
        ("reward", "Total reward", "(c) Total reward"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), facecolor=PAPER_LIGHT_BG)
    x = np.arange(len(cells))
    width = 0.24
    for ax, (key, ylabel, title) in zip(axes, metrics):
        ax.set_facecolor(PAPER_LIGHT_BG)
        for k, (name, color) in enumerate(zip(series, colors)):
            vals = [float(per_cell[c][name][key]) for c in cells]
            ax.bar(x + (k - 1) * width, vals, width=width, color=color, edgecolor="k", lw=0.4, label=name)
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
    fig.suptitle(
        "CCCV 0.5C vs Random Search vs GP-BO under identical energy-delivery constraints",
        fontsize=14, fontweight="bold", y=1.02,
    )
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
        ax.set_title(cell, fontsize=14, fontweight="bold")
        ax.set_xlabel("Equivalent charge cycles (one equal-energy session each)")
        ax.set_ylabel("Projected remaining capacity [%]")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9, loc="best")
    fig.suptitle(
        "Projected remaining battery capacity over 600 repeated equal-energy charging cycles\n"
        r"(session $Q_{\mathrm{loss}}$ accumulated; CCCV 0.5C anchored to 80% at 600 cycles)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
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
    ax.set_title(f"{cell_id}: best feasible reward by family ({method})", fontweight="bold")
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
    ax.set_title(
        "GP-BO vs CCCV 0.5C and Random Search (calibrated $Q_{\\mathrm{loss}}$, paper Eq. 10)",
        fontsize=13, fontweight="bold", pad=12,
    )
    fig.tight_layout()
    savefig(fig, out_path)
