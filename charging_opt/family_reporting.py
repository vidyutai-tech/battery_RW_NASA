"""Reporting helpers for multi-family charging profile optimization (Priority 1)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from charging_opt.charging_profile_family import FAMILY_LABELS, ProfileParams, get_family
from charging_opt.family_optimizer import FamilyOptimizationResult
from charging_opt.io_utils import resolve_writable_path
from charging_opt.lifetime_reward import aggregate_lifetime_reward
from charging_opt.pareto_analysis import (
    degradation_summary,
    degradation_value,
    format_degradation_value,
    resolve_pareto_config,
)
from charging_opt.profile_simulator import ProfileSimulator

# Readable defaults for publication-style figures
PLOT_RC = {
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
}

CSV_HEADERS = [
    "family_id",
    "family_label",
    "feasible",
    "objective_mode",
    "loss",
    "sei_term",
    "fade_term",
    "time_term",
    "temperature_term",
    "voltage_stress_term",
    "duration_min",
    "degradation_key",
    "degradation_value",
    "sei_per_pct_soc",
    "capacity_fade_pct",
    "sei_proxy",
    "voltage_stress_v2_min",
    "temperature_penalty_c2_min",
    "peak_voltage",
    "peak_temperature",
    "soc_end",
    "end_reason",
    "parameters",
]


def _param_str(values: Mapping[str, float]) -> str:
    skip = {"family_id"}
    return ", ".join(
        f"{k}={v:.4g}" for k, v in values.items() if k not in skip
    )


def _infer_constraints_from_results(
    results: Dict[str, FamilyOptimizationResult],
) -> dict:
    for r in results.values():
        comp = r.best_metrics.get("components") or {}
        mode = comp.get("objective_mode")
        if mode == "physics_degradation":
            return {"objective_mode": "physics"}
        if mode == "legacy":
            return {"objective_mode": "legacy"}
    return {"objective_mode": "composite"}


def _degradation_config(constraints: Optional[dict]) -> tuple[str, str]:
    _, key, label = resolve_pareto_config(constraints)
    return key, label


def _degradation_value(metrics: dict, key: str) -> float:
    return degradation_value(metrics, key)


def _format_degradation(value: float, key: str) -> str:
    return format_degradation_value(value, key)


def _loss_subtitle(metrics: dict) -> str:
    comp = metrics.get("components") or {}
    mode = comp.get("objective_mode", "composite")
    if comp.get("infeasible") is not False:
        return ""
    common = (
        f"  |  V²·min={metrics.get('voltage_stress_v2_min', float('nan')):.2f}  "
        f"°C²·min={metrics.get('temperature_penalty_c2_min', float('nan')):.2f}\n"
    )
    if mode == "physics_degradation":
        return (
            common
            + f"  loss terms: fade={comp.get('fade_term', float('nan')):.3f}  "
            f"time={comp.get('time_term', float('nan')):.2f}  "
            f"temp={comp.get('temperature_term', float('nan')):.2f}  "
            f"V={comp.get('voltage_stress_term', float('nan')):.2f}"
        )
    if mode == "composite":
        return (
            common
            + f"  loss terms: SEI={comp.get('sei_term', float('nan')):.1f}  "
            f"time={comp.get('time_term', float('nan')):.2f}  "
            f"temp={comp.get('temperature_term', float('nan')):.2f}  "
            f"V={comp.get('voltage_stress_term', float('nan')):.2f}"
        )
    return ""


def _objective_from_constraints(constraints: dict) -> tuple:
    from charging_opt.lifetime_reward import LifetimeWeights, ObjectiveMode

    mode: ObjectiveMode = constraints.get("objective_mode", "composite")
    w = constraints.get("weights") or {}
    if mode == "legacy":
        weights = LifetimeWeights.legacy()
    else:
        weights = LifetimeWeights(
            sei=float(w.get("w_sei", 1.0)),
            time=float(w.get("w_time", 0.02)),
            temperature=float(w.get("w_temperature", 0.05)),
            voltage_stress=float(w.get("w_voltage_stress", 0.08)),
        )
    return (
        weights,
        mode,
        float(constraints.get("v_ref_stress", 4.0)),
        float(constraints.get("t_comfort_c", 35.0)),
    )


def _format_spec_summary(session: dict) -> str:
    spec = session.get("profile_spec") or {}
    parts = []
    for key in sorted(spec.keys()):
        if key == "family_id":
            continue
        val = spec[key]
        if isinstance(val, (int, float)):
            parts.append(f"{key}={val:.3g}")
    return "  |  ".join(parts[:6])


def family_plot(
    session: dict,
    metrics: dict,
    title: str,
    out_path: Path,
    *,
    constraints: Optional[dict] = None,
) -> Path:
    """
    Practical I / V / SoC figure with separate y-scales (not shared I+V axis).

    Voltage may look flat when the cell enters CV or hits the ~4.2 V ceiling —
    that is physical, not a plotting bug. The voltage panel uses a tight y-limit
    so the actual trajectory is visible.
    """
    plt.rcParams.update(PLOT_RC)
    t_min = np.asarray(session["time_s"], dtype=float) / 60.0
    i_a = np.maximum(-np.asarray(session["current_a"], dtype=float), 0.0)
    v = np.asarray(session["voltage_v"], dtype=float)
    soc_pct = np.asarray(session["soc"], dtype=float) * 100.0
    s0 = session["initial_state"]

    fig, axes = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.05], "hspace": 0.10},
    )
    ax_i, ax_v, ax_s = axes

    ax_i.plot(t_min, i_a, color="#2166ac", lw=2.2, solid_capstyle="round")
    ax_i.set_ylabel("Current (A)")
    ax_i.set_ylim(0.0, max(0.5, float(i_a.max()) * 1.12))
    ax_i.grid(alpha=0.35, linestyle="--", linewidth=0.6)

    ax_v.plot(t_min, v, color="#b2182b", lw=2.0)
    v_lo, v_hi = float(v.min()), float(v.max())
    v_pad = max(0.015, (v_hi - v_lo) * 0.12)
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_ylim(v_lo - v_pad, v_hi + v_pad)
    ax_v.grid(alpha=0.35, linestyle="--", linewidth=0.6)
    if v_hi - v_lo < 0.08 and v_hi >= 4.05:
        ax_v.text(
            0.01, 0.94,
            f"CV / V-ceiling hold ≈ {v_hi:.3f} V (BDT plateau)",
            transform=ax_v.transAxes, va="top", fontsize=10, color="#666666",
        )

    ax_s.plot(t_min, soc_pct, color="#1a1a1a", lw=2.0)
    ax_s.set_ylabel("SoC (%)")
    ax_s.set_xlabel("Time (min)")
    ax_s.set_ylim(max(0, float(soc_pct[0]) - 5), 105)
    ax_s.grid(alpha=0.35, linestyle="--", linewidth=0.6)

    feasible = metrics.get("feasible", False)
    loss = metrics.get("loss", float("nan"))
    if constraints is None:
        constraints = {}
        comp = metrics.get("components") or {}
        if comp.get("objective_mode") == "physics_degradation":
            constraints = {"objective_mode": "physics"}
    deg_key, deg_label = _degradation_config(constraints)
    deg_val = _degradation_value(metrics, deg_key)
    summary = (
        f"{metrics['duration_min']:.1f} min  ·  "
        f"{deg_label}={_format_degradation(deg_val, deg_key)}  ·  "
        f"loss={loss:.2f}"
    )
    if not feasible:
        summary += "  ·  infeasible"
    fig.suptitle(f"{title}\n{summary}", fontsize=14, y=0.97)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.09, right=0.97)
    out_path = resolve_writable_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    plt.rcdefaults()
    return out_path


def comparison_table_png(
    results: Dict[str, FamilyOptimizationResult],
    out_path: Path,
    *,
    constraints: Optional[dict] = None,
) -> Path:
    plt.rcParams.update(PLOT_RC)
    if constraints is None:
        constraints = _infer_constraints_from_results(results)
    deg_key, deg_label = _degradation_config(constraints)
    rows = []
    for fid, r in sorted(results.items(), key=lambda kv: kv[1].best_loss):
        m = r.best_metrics
        deg_val = _degradation_value(m, deg_key)
        rows.append([
            r.family_label,
            _param_str(r.best_params.values)[:40],
            f"{m.get('duration_min', float('nan')):.1f}",
            _format_degradation(deg_val, deg_key),
            f"{m.get('voltage_stress_v2_min', float('nan')):.1f}",
            f"{r.best_loss:.1f}",
            "yes" if m.get("feasible") else "no",
        ])

    fig, ax = plt.subplots(figsize=(12, max(4.0, 0.55 * len(rows) + 1.5)))
    ax.axis("off")
    headers = ["Family", "Parameters", "Duration", deg_label, "V²·min", "Loss", "OK"]
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    ax.set_title("Multi-family charging profile comparison", fontsize=15, pad=18)
    fig.tight_layout()
    out_path = resolve_writable_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    plt.rcdefaults()
    return out_path


def comparison_table_csv(
    results: Dict[str, FamilyOptimizationResult],
    out_path: Path,
    *,
    constraints: Optional[dict] = None,
) -> Path:
    out_path = resolve_writable_path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if constraints is None:
        constraints = _infer_constraints_from_results(results)
    deg_key, _ = _degradation_config(constraints)
    rows = sorted(results.values(), key=lambda r: r.best_loss)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        w.writeheader()
        for r in rows:
            m = r.best_metrics
            comp = m.get("components") or {}
            deg_val = _degradation_value(m, deg_key)
            w.writerow({
                "family_id": r.family_id,
                "family_label": r.family_label,
                "feasible": m.get("feasible", False),
                "objective_mode": comp.get("objective_mode", ""),
                "loss": f"{r.best_loss:.4f}",
                "sei_term": f"{comp.get('sei_term', float('nan')):.4f}",
                "fade_term": f"{comp.get('fade_term', float('nan')):.4f}",
                "time_term": f"{comp.get('time_term', float('nan')):.4f}",
                "temperature_term": f"{comp.get('temperature_term', float('nan')):.4f}",
                "voltage_stress_term": f"{comp.get('voltage_stress_term', float('nan')):.4f}",
                "duration_min": f"{m.get('duration_min', float('nan')):.2f}",
                "degradation_key": deg_key,
                "degradation_value": _format_degradation(deg_val, deg_key),
                "sei_per_pct_soc": f"{m.get('sei_per_pct_soc', float('nan')):.4f}",
                "capacity_fade_pct": f"{m.get('capacity_fade_pct', float('nan')):.4f}",
                "sei_proxy": f"{m.get('sei_proxy', float('nan')):.2f}",
                "voltage_stress_v2_min": f"{m.get('voltage_stress_v2_min', float('nan')):.4f}",
                "temperature_penalty_c2_min": f"{m.get('temperature_penalty_c2_min', float('nan')):.4f}",
                "peak_voltage": f"{m.get('peak_voltage', float('nan')):.4f}",
                "peak_temperature": f"{m.get('peak_temperature', float('nan')):.2f}",
                "soc_end": f"{m.get('soc_end', float('nan')):.4f}",
                "end_reason": m.get("end_reason", ""),
                "parameters": _param_str(r.best_params.values),
            })
    return out_path


def export_family_artifacts(
    results: Dict[str, FamilyOptimizationResult],
    plots_dir: Path,
    *,
    csv_path: Optional[Path] = None,
    constraints: Optional[dict] = None,
) -> dict[str, Path]:
    """Write per-family PNGs, comparison table PNG, and optional CSV."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if constraints is None:
        constraints = _infer_constraints_from_results(results)
    written: dict[str, Path] = {}

    for fid, r in results.items():
        p = family_plot(
            r.best_session,
            r.best_metrics,
            r.family_label,
            plots_dir / f"best_{fid}.png",
            constraints=constraints,
        )
        written[f"plot_{fid}"] = p

    written["comparison_png"] = comparison_table_png(
        results,
        plots_dir / "profile_family_comparison.png",
        constraints=constraints,
    )
    if csv_path is not None:
        written["comparison_csv"] = comparison_table_csv(
            results, csv_path, constraints=constraints,
        )
    return written


def load_results_payload(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def rehydrate_results_from_json(
    data: dict,
    simulator: ProfileSimulator,
    initial_state: dict,
    *,
    soc_target: float,
    max_duration_min: Optional[float],
    family_ids: Optional[Iterable[str]] = None,
) -> Dict[str, FamilyOptimizationResult]:
    """Rebuild FamilyOptimizationResult objects (re-simulating best params for plots)."""
    constraints = data.get("constraints", {})
    weights, objective_mode, v_ref_stress, t_comfort_c = _objective_from_constraints(constraints)
    ids = list(family_ids) if family_ids else list(data.get("families", {}).keys())
    out: Dict[str, FamilyOptimizationResult] = {}
    for fid in ids:
        entry = data["families"][fid]
        params = ProfileParams.from_dict(entry["best_params"])
        family = get_family(fid)
        session = simulator.simulate_params(initial_state, params, family=family)
        _, metrics = aggregate_lifetime_reward(
            session,
            soc_target=soc_target,
            max_duration_min=max_duration_min,
            weights=weights,
            objective_mode=objective_mode,
            v_ref_stress=v_ref_stress,
            t_comfort_c=t_comfort_c,
        )
        out[fid] = FamilyOptimizationResult(
            family_id=fid,
            family_label=entry.get("family_label", FAMILY_LABELS.get(fid, fid)),
            best_params=params,
            best_session=session,
            best_metrics=metrics,
            best_loss=float(entry.get("best_loss", metrics.get("loss", 1e6))),
            history=entry.get("history", []),
            skopt_result=None,
        )
    return out
