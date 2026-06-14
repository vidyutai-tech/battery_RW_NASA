"""Priority 3 — Pareto front plots and tabular exports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charging_opt.charging_profile_family import FAMILY_LABELS
from charging_opt.family_reporting import PLOT_RC, _param_str
from charging_opt.io_utils import resolve_writable_path
from charging_opt.pareto_analysis import (
    ParetoAnalysisResult,
    ParetoCandidate,
    analyze_results_payload,
    extract_feasible_candidates,
    pareto_front,
)

TAG_COLORS = {
    "fastest": "#2166ac",
    "lifetime": "#1b7837",
    "balanced": "#762a83",
}
TAG_MARKERS = {
    "fastest": ">",
    "lifetime": "s",
    "balanced": "D",
}


def _degradation_value(c: ParetoCandidate, key: str) -> float:
    val = getattr(c, key, None)
    if val is None or (isinstance(val, float) and val != val):
        return float(c.sei_per_pct_soc)
    return float(val)


def _family_colors(family_ids: Iterable[str]) -> Dict[str, str]:
    ids = sorted(set(family_ids))
    cmap = plt.get_cmap("tab10")
    return {fid: cmap(i % 10) for i, fid in enumerate(ids)}


def _scatter_pareto_ax(
    ax,
    all_feasible: List[ParetoCandidate],
    front: List[ParetoCandidate],
    *,
    x_attr: str,
    y_attr: str,
    xlabel: str,
    ylabel: str,
    title: str,
    tagged: Optional[Dict[str, ParetoCandidate]] = None,
) -> None:
    colors = _family_colors(c.family_id for c in all_feasible)
    for c in all_feasible:
        y_val = _degradation_value(c, y_attr) if y_attr in ("sei_per_pct_soc", "capacity_fade_pct") else getattr(c, y_attr)
        ax.scatter(
            getattr(c, x_attr), y_val,
            c=[colors[c.family_id]], s=36, alpha=0.5, edgecolors="none",
        )
    if front:
        fx = [getattr(c, x_attr) for c in front]
        fy = [
            _degradation_value(c, y_attr) if y_attr in ("sei_per_pct_soc", "capacity_fade_pct") else getattr(c, y_attr)
            for c in front
        ]
        order = np.argsort(fx)
        ax.plot(
            np.asarray(fx)[order], np.asarray(fy)[order],
            color="#333333", lw=1.5, ls="--", alpha=0.7, label="Pareto front",
        )
        ax.scatter(fx, fy, facecolors="none", edgecolors="#333333", s=55, lw=1.2)

    if tagged:
        for tag, cand in tagged.items():
            if cand is None:
                continue
            y_val = (
                _degradation_value(cand, y_attr)
                if y_attr in ("sei_per_pct_soc", "capacity_fade_pct")
                else getattr(cand, y_attr)
            )
            ax.scatter(
                getattr(cand, x_attr), y_val,
                c=TAG_COLORS.get(tag, "black"),
                marker=TAG_MARKERS.get(tag, "o"),
                s=120, edgecolors="white", linewidths=0.8,
                label=f"{tag.capitalize()} ({cand.family_label})",
                zorder=5,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.35, linestyle="--", linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(fontsize=9, loc="best", framealpha=0.9)


def pareto_scatter_grid(
    analysis: ParetoAnalysisResult,
    all_feasible: List[ParetoCandidate],
    out_path: Path,
) -> Path:
    """Three-panel figure: duration vs SEI / V-stress / temperature."""
    plt.rcParams.update(PLOT_RC)
    tagged = {
        "fastest": analysis.tagged_global.fastest,
        "lifetime": analysis.tagged_global.lifetime,
        "balanced": analysis.tagged_global.balanced,
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    y_key = analysis.degradation_key
    y_label = analysis.degradation_label

    _scatter_pareto_ax(
        axes[0], all_feasible, analysis.pareto_front,
        x_attr="duration_min", y_attr=y_key,
        xlabel="Charge duration (min)", ylabel=y_label,
        title="Charge time vs degradation",
        tagged=tagged,
    )
    _scatter_pareto_ax(
        axes[1], all_feasible, analysis.pareto_front,
        x_attr="duration_min", y_attr="voltage_stress_v2_min",
        xlabel="Charge duration (min)", ylabel="Voltage stress (∫V²·min)",
        title="Charge time vs voltage stress",
        tagged=tagged,
    )
    _scatter_pareto_ax(
        axes[2], all_feasible, analysis.pareto_front,
        x_attr="duration_min", y_attr="temperature_penalty_c2_min",
        xlabel="Charge duration (min)", ylabel="Temperature penalty (∫°C²·min)",
        title="Charge time vs temperature penalty",
        tagged=tagged,
    )

    fig.suptitle(
        f"Pareto trade-offs — {len(all_feasible)} feasible evals, "
        f"{analysis.n_pareto_global} non-dominated",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()
    out_path = resolve_writable_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()
    return out_path


def pareto_by_family_plot(
    analysis: ParetoAnalysisResult,
    all_feasible: List[ParetoCandidate],
    out_path: Path,
) -> Path:
    """Per-family duration vs degradation metric with global tagged profiles."""
    plt.rcParams.update(PLOT_RC)
    y_key = analysis.degradation_key
    y_label = analysis.degradation_label
    obj_list = analysis.objectives
    family_ids = sorted({c.family_id for c in all_feasible})
    n = len(family_ids)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False)
    tagged = analysis.tagged_global

    for idx, fid in enumerate(family_ids):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        fam_cands = [x for x in all_feasible if x.family_id == fid]
        fam_front = pareto_front(
            fam_cands, obj_list, degradation_key=y_key,
        )
        _scatter_pareto_ax(
            ax, fam_cands, fam_front,
            x_attr="duration_min", y_attr=y_key,
            xlabel="Duration (min)", ylabel=y_label,
            title=FAMILY_LABELS.get(fid, fid),
        )
        for tag, cand in (
            ("fastest", tagged.fastest),
            ("lifetime", tagged.lifetime),
            ("balanced", tagged.balanced),
        ):
            if cand is not None and cand.family_id == fid:
                ax.scatter(
                    cand.duration_min, _degradation_value(cand, y_key),
                    c=TAG_COLORS[tag], marker=TAG_MARKERS[tag],
                    s=90, edgecolors="white", linewidths=0.6, zorder=5,
                )

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.suptitle(
        f"Per-family feasible sets (duration vs {y_label})",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()
    out_path = resolve_writable_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plt.rcdefaults()
    return out_path


def tagged_profiles_table_png(analysis: ParetoAnalysisResult, out_path: Path) -> Path:
    plt.rcParams.update(PLOT_RC)
    deg_label = analysis.degradation_label
    rows = []
    for tag in ("fastest", "lifetime", "balanced"):
        cand = getattr(analysis.tagged_global, tag)
        if cand is None:
            rows.append([tag.capitalize(), "—", "—", "—", "—", "—", "—"])
            continue
        rows.append([
            tag.capitalize(),
            cand.family_label,
            _param_str(cand.params)[:42],
            f"{cand.duration_min:.1f}",
            f"{_degradation_value(cand, analysis.degradation_key):.3f}",
            f"{cand.voltage_stress_v2_min:.2f}",
            f"{cand.loss:.2f}",
        ])

    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    headers = ["Profile", "Family", "Parameters", "Duration", deg_label, "V²·min", "Loss"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)
    ax.set_title("Reference charging profiles (Pareto analysis)", fontsize=14, pad=14)
    fig.tight_layout()
    out_path = resolve_writable_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    plt.rcdefaults()
    return out_path


PARETO_CSV_HEADERS = [
    "scope",
    "tag",
    "family_id",
    "family_label",
    "duration_min",
    "sei_per_pct_soc",
    "voltage_stress_v2_min",
    "temperature_penalty_c2_min",
    "loss",
    "peak_voltage",
    "peak_temperature",
    "parameters",
]


def pareto_csv(analysis: ParetoAnalysisResult, out_path: Path) -> Path:
    out_path = resolve_writable_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []

    for tag in ("fastest", "lifetime", "balanced"):
        cand = getattr(analysis.tagged_global, tag)
        if cand is None:
            continue
        rows.append({
            "scope": "global",
            "tag": tag,
            "family_id": cand.family_id,
            "family_label": cand.family_label,
            "duration_min": f"{cand.duration_min:.2f}",
            "sei_per_pct_soc": f"{cand.sei_per_pct_soc:.4f}",
            "voltage_stress_v2_min": f"{cand.voltage_stress_v2_min:.4f}",
            "temperature_penalty_c2_min": f"{cand.temperature_penalty_c2_min:.4f}",
            "loss": f"{cand.loss:.4f}",
            "peak_voltage": f"{cand.peak_voltage:.4f}",
            "peak_temperature": f"{cand.peak_temperature:.2f}",
            "parameters": _param_str(cand.params),
        })

    for c in analysis.pareto_front:
        rows.append({
            "scope": "pareto_front",
            "tag": "",
            "family_id": c.family_id,
            "family_label": c.family_label,
            "duration_min": f"{c.duration_min:.2f}",
            "sei_per_pct_soc": f"{c.sei_per_pct_soc:.4f}",
            "voltage_stress_v2_min": f"{c.voltage_stress_v2_min:.4f}",
            "temperature_penalty_c2_min": f"{c.temperature_penalty_c2_min:.4f}",
            "loss": f"{c.loss:.4f}",
            "peak_voltage": f"{c.peak_voltage:.4f}",
            "peak_temperature": f"{c.peak_temperature:.2f}",
            "parameters": _param_str(c.params),
        })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PARETO_CSV_HEADERS)
        w.writeheader()
        w.writerows(rows)
    return out_path


def save_pareto_json(analysis: ParetoAnalysisResult, out_path: Path) -> Path:
    out_path = resolve_writable_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(analysis.to_dict(), f, indent=2)
    return out_path


def export_pareto_artifacts(
    data: dict,
    pareto_plots_dir: Path,
    *,
    models_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Build analysis from family JSON payload and write plots + tables."""
    pareto_plots_dir = Path(pareto_plots_dir)
    pareto_plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(models_dir) if models_dir else pareto_plots_dir.parent.parent / "models"

    analysis = analyze_results_payload(data)
    all_feasible = extract_feasible_candidates(
        data.get("families") or {},
        degradation_key=analysis.degradation_key,
    )

    written: dict[str, Path] = {}
    written["pareto_json"] = save_pareto_json(
        analysis, models_dir / "pareto_analysis.json",
    )
    written["pareto_csv"] = pareto_csv(analysis, models_dir / "pareto_profiles.csv")
    written["scatter_grid"] = pareto_scatter_grid(
        analysis, all_feasible, pareto_plots_dir / "pareto_tradeoffs.png",
    )
    written["by_family"] = pareto_by_family_plot(
        analysis, all_feasible, pareto_plots_dir / "pareto_by_family.png",
    )
    written["tagged_table"] = tagged_profiles_table_png(
        analysis, pareto_plots_dir / "pareto_reference_profiles.png",
    )
    return written
