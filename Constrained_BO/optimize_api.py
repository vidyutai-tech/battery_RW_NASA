"""Shared API for CLI and Streamlit: run GP-BO + random search under one constraint set."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from Constrained_BO.bayesian_optimizer import DEFAULT_ACQ_FUNC, optimize_all_families_gp_bo
from Constrained_BO.config import (
    SOC_START,
    energy_fraction_for,
    finetune_frac_for,
    get_cell_config,
)
from Constrained_BO.objective import (
    QLOSS_TERMINOLOGY,
    QLOSS_TERMINOLOGY_NOTE,
    RESULT_METRIC_UNITS,
    energy_required_j,
    evaluate_session,
    full_capacity_joules,
)
from Constrained_BO.ocv import ocv_curve_path
from Constrained_BO.profiles import (
    DEFAULT_FAMILIES,
    CCCVFamily,
    ProfileParams,
    TwoStepFamily,
    get_family,
    set_profile_bounds,
)
from Constrained_BO.simulator import ChargingSimulator
from Constrained_BO.viz import plot_best_profiles

DEFAULT_CC_CURRENTS_A = (0.5, 1.0, 2.0, 3.0, 4.0)
PAPER_N_CALLS = 120
PAPER_N_INITIAL = 15
PAPER_N_RANDOM = 120
PAPER_ACQ = "EI"
PAPER_ELITE_TOP_K = 5


def _optimize_all_families_random(**kwargs):
    from Constrained_BO.run import _optimize_all_families_random as _fn
    return _fn(**kwargs)


METRIC_COLS = [
    ("method", "Method"),
    ("family_label", "Profile family"),
    ("feasible", "Feasible"),
    ("total_reward", "Total reward"),
    ("soc_reward", "SoC reward"),
    ("qloss_penalty", "Qloss penalty"),
    ("time_penalty", "Time penalty"),
    ("duration_min", "Duration (min)"),
    ("peak_temperature", "Peak T (°C)"),
    ("mean_temperature", "Mean T (°C)"),
    ("energy_delivered_j", "E delivered (J)"),
    ("energy_required_j", "E required (J)"),
    ("loss", "Loss"),
    ("qloss_calendar", "Q_calendar (index)"),
    ("qloss_cyclic", "Q_cyclic (index)"),
    ("qloss_total", "Q_total (index)"),
    ("end_reason", "End reason"),
    ("best_params", "Best params"),
]

# Primary Streamlit metric tabs (order matters).
PRIMARY_CHART_METRICS = [
    ("total_reward", "Total reward (higher better)"),
    ("duration_min", "Duration (min)"),
    ("peak_temperature", "Peak T (°C)"),
    ("energy_delivered_j", "Energy delivered (J)"),
    ("loss", "Loss (lower better)"),
]

DEGRADATION_CHART_METRICS = [
    ("qloss_total", "Q_total (index)"),
    ("qloss_calendar", "Q_calendar (index)"),
    ("qloss_cyclic", "Q_cyclic (index)"),
]


def build_cell(
    cell_id: str,
    *,
    energy_fraction: Optional[float] = None,
    soc_mode: bool = False,
    soc_target: Optional[float] = None,
    max_duration_min: Optional[float] = None,
    decision_interval_s: Optional[int] = None,
    auto_decision_interval: bool = True,
    refit_ocv: bool = False,
):
    """Resolve CellConfig with the same energy-default rules as ``run.py``."""
    cell_id = cell_id.upper()
    if soc_mode:
        efrac = None
    elif energy_fraction is not None:
        efrac = float(energy_fraction)
    else:
        efrac = energy_fraction_for(cell_id)

    cell = get_cell_config(cell_id, refit_ocv=refit_ocv)
    cell = cell.with_run_overrides(
        soc_target=soc_target,
        energy_fraction=efrac,
        max_duration_min=max_duration_min,
        decision_interval_s=decision_interval_s,
        auto_decision_interval=auto_decision_interval and decision_interval_s is None,
    )
    if cell.profile_bounds is not None:
        set_profile_bounds(cell.profile_bounds)
    return cell


def _strip_sessions(results: Dict[str, Dict]) -> Dict[str, Dict]:
    out = {}
    for fid, res in results.items():
        out[fid] = {k: v for k, v in res.items() if k != "best_session"}
    return out


def _meta_for(
    cell,
    *,
    method: str,
    reward_mode: str,
    reward_weights: Dict[str, float],
    families: Sequence[str],
    simulator: ChargingSimulator,
    seed: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cell_id = cell.cell_id.upper()
    meta: Dict[str, Any] = {
        "cell": cell_id,
        "bdt_ckpt": str(cell.bdt_ckpt),
        "finetune_fraction": finetune_frac_for(cell_id) if cell_id not in ("RW9",) else None,
        "soc_start": cell.start_state.get("soc", SOC_START),
        "ocv_curve": str(ocv_curve_path(cell_id)),
        "start_state": cell.start_state,
        "soc_target": cell.soc_target,
        "max_duration_min": cell.max_duration_min,
        "constraint_mode": cell.constraint_mode,
        "v_nom": cell.v_nom,
        "method": method,
        "reward_mode": reward_mode,
        "reward_weights": dict(reward_weights),
        "families": list(families),
        "decision_interval_s": simulator.decision_interval_s,
        "decision_interval_selection": simulator.decision_interval_info,
        "seed": seed,
        "qloss_terminology": QLOSS_TERMINOLOGY,
        "qloss_terminology_note": QLOSS_TERMINOLOGY_NOTE,
        "metric_units": RESULT_METRIC_UNITS,
    }
    if cell.profile_bounds is not None:
        meta["profile_bounds"] = cell.profile_bounds.to_dict()
    if cell.constraint_mode == "energy":
        meta["energy_fraction"] = cell.energy_fraction
        meta["energy_full_j"] = full_capacity_joules(cell.q_rated_as, cell.v_nom)
        meta["energy_required_j"] = energy_required_j(
            cell.q_rated_as, cell.energy_fraction, cell.v_nom,
        )
    if extra:
        meta.update(extra)
    return meta


def run_optimization(
    cell,
    *,
    method: str = "gp_bo",
    families: Optional[Sequence[str]] = None,
    device: str = "auto",
    seed: int = 42,
    reward_mode: str = "hybrid_qloss",
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    z: float = 0.55,
    n_calls: int = 40,
    n_initial: int = 10,
    acq_func: str = DEFAULT_ACQ_FUNC,
    n_random: int = 80,
    elite_histories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    elite_top_k: int = PAPER_ELITE_TOP_K,
    simulator: Optional[ChargingSimulator] = None,
    qloss_cap: Optional[float] = None,
    qloss_cap_scale: float = 80.0,
    duration_loss_weight: float = 1e-3,
) -> Tuple[Dict[str, Any], Dict[str, Any], ChargingSimulator]:
    """
    Run one optimizer (gp_bo or random_search).

    Returns (payload_without_sessions, family_results_with_sessions, simulator).
    """
    families = list(families or DEFAULT_FAMILIES)
    if simulator is None:
        simulator = ChargingSimulator.from_cell(cell, device=device)

    w_time_use = float(w_time)
    if reward_mode == "legacy_temp_time" and abs(w_time_use - 0.1) < 1e-15:
        w_time_use = 1.0

    rw = {
        "w_soc": w_soc,
        "w_qloss": w_qloss,
        "w_time": w_time_use,
        "w_temperature": w_temperature,
        "z": z,
    }

    if method == "gp_bo":
        results = optimize_all_families_gp_bo(
            simulator,
            cell.start_state,
            families=families,
            n_calls=n_calls,
            n_initial_points=n_initial,
            seed=seed,
            reward_mode=reward_mode,  # type: ignore[arg-type]
            w_time=w_time_use,
            w_temperature=w_temperature,
            w_soc=w_soc,
            w_qloss=w_qloss,
            z=z,
            acq_func=acq_func,
            elite_histories=elite_histories,
            elite_top_k=elite_top_k,
            qloss_cap=qloss_cap,
            qloss_cap_scale=qloss_cap_scale,
            duration_loss_weight=duration_loss_weight,
        )
        extra = {
            "n_calls": n_calls,
            "n_initial": n_initial,
            "acq_func": acq_func,
            "elite_top_k": elite_top_k if elite_histories else 0,
            "qloss_cap": qloss_cap,
            "duration_loss_weight": duration_loss_weight,
        }
    elif method == "random_search":
        results = _optimize_all_families_random(
            simulator=simulator,
            initial_state=cell.start_state,
            families=families,
            n_random=n_random,
            seed=seed,
            reward_mode=reward_mode,
            w_time=w_time_use,
            w_temperature=w_temperature,
            w_soc=w_soc,
            w_qloss=w_qloss,
            z=z,
            qloss_cap=qloss_cap,
            qloss_cap_scale=qloss_cap_scale,
            duration_loss_weight=duration_loss_weight,
        )
        extra = {
            "n_random": n_random,
            "qloss_cap": qloss_cap,
            "duration_loss_weight": duration_loss_weight,
        }
    else:
        raise ValueError(f"Unknown method {method!r}")

    meta = _meta_for(
        cell,
        method=method,
        reward_mode=reward_mode,
        reward_weights=rw,
        families=families,
        simulator=simulator,
        seed=seed,
        extra=extra,
    )
    payload = {"meta": meta, "families": _strip_sessions(results)}
    return payload, results, simulator


def results_to_dataframe(
    payload: Dict[str, Any],
    *,
    method_label: Optional[str] = None,
) -> pd.DataFrame:
    rows = []
    method = method_label or payload["meta"].get("method", "?")
    order = payload["meta"].get("families", list(payload["families"].keys()))
    for fid in order:
        entry = payload["families"].get(fid) or {}
        m = entry.get("best_metrics") or {}
        rows.append({
            "method": method,
            "family_id": fid,
            "family_label": entry.get("family_label", fid),
            "feasible": m.get("feasible"),
            "loss": m.get("loss"),
            "total_reward": m.get("total_reward"),
            "soc_reward": m.get("soc_reward"),
            "qloss_penalty": m.get("qloss_penalty"),
            "time_penalty": m.get("time_penalty"),
            "duration_min": m.get("duration_min"),
            "energy_delivered_j": m.get("energy_delivered_j"),
            "energy_required_j": m.get("energy_required_j"),
            "energy_shortfall_j": m.get("energy_shortfall_j"),
            "soc_start": m.get("soc_start"),
            "soc_end": m.get("soc_end"),
            "peak_voltage": m.get("peak_voltage"),
            "peak_temperature": m.get("peak_temperature"),
            "mean_temperature": m.get("mean_temperature"),
            "ah_throughput": m.get("ah_throughput"),
            "nominal_c_rate": m.get("nominal_c_rate"),
            "max_c_rate": m.get("max_c_rate"),
            "efc": m.get("efc"),
            "qloss_calendar": m.get("qloss_calendar"),
            "qloss_cyclic": m.get("qloss_cyclic"),
            "qloss_total": m.get("qloss_total"),
            "end_reason": m.get("end_reason"),
            "best_params": entry.get("best_params"),
        })
    return pd.DataFrame(rows)


def comparison_dataframe(
    bo_payload: Dict[str, Any],
    random_payload: Dict[str, Any],
) -> pd.DataFrame:
    return pd.concat(
        [
            results_to_dataframe(bo_payload, method_label="GP-BO"),
            results_to_dataframe(random_payload, method_label="Random search"),
        ],
        ignore_index=True,
    )


def winner_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-family winner on feasible total_reward (higher better); ties allowed."""
    rows = []
    if df.empty or "family_id" not in df.columns:
        return pd.DataFrame(rows)
    for fid, g in df.groupby("family_id", sort=False):
        label = str(g["family_label"].iloc[0])
        feas = g[g["feasible"] == True]  # noqa: E712
        pool = feas if not feas.empty else g
        best = pool.loc[pool["total_reward"].astype(float).idxmax()]
        runners = []
        for _, r in pool.iterrows():
            runners.append({
                "method": r["method"],
                "reward": float(r["total_reward"]),
                "duration_min": float(r["duration_min"]) if r.get("duration_min") is not None else None,
            })
        # Count unique best reward
        top_val = float(best["total_reward"])
        winners = [r["method"] for r in runners if abs(r["reward"] - top_val) < 1e-9]
        rows.append({
            "family_id": fid,
            "family_label": label,
            "winner": ", ".join(winners),
            "best_reward": top_val,
            "gp_bo_reward": float(g.loc[g["method"] == "GP-BO", "total_reward"].iloc[0])
            if (g["method"] == "GP-BO").any() else None,
            "random_reward": float(g.loc[g["method"] == "Random search", "total_reward"].iloc[0])
            if (g["method"] == "Random search").any() else None,
        })
    return pd.DataFrame(rows)


def winner_counts(summary: pd.DataFrame) -> Dict[str, int]:
    counts = {"GP-BO": 0, "Random search": 0, "Tie": 0}
    if summary.empty:
        return counts
    for _, row in summary.iterrows():
        w = str(row["winner"])
        if "GP-BO" in w and "Random" in w:
            counts["Tie"] += 1
        elif w == "GP-BO":
            counts["GP-BO"] += 1
        elif "Random" in w:
            counts["Random search"] += 1
        else:
            counts["Tie"] += 1
    return counts


def dataframe_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "comparison") -> bytes:
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            export = df.copy()
            if "best_params" in export.columns:
                export["best_params"] = export["best_params"].map(
                    lambda x: str(x) if x is not None else ""
                )
            export.to_excel(writer, index=False, sheet_name=sheet_name)
            units = pd.DataFrame(
                [{"metric": k, "unit": v} for k, v in RESULT_METRIC_UNITS.items()]
            )
            units.to_excel(writer, index=False, sheet_name="metric_units")
    except ImportError:
        buf = io.BytesIO(df.to_csv(index=False).encode("utf-8"))
    return buf.getvalue()


def figure_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def plot_profiles_png(
    family_results: Dict[str, Any],
    cell,
    *,
    title_suffix: str,
) -> bytes:
    import matplotlib.pyplot as plt

    fig = plot_best_profiles(
        family_results,
        cell_id=cell.cell_id,
        soc_target=cell.soc_target,
        soc_start=cell.start_state.get("soc", SOC_START),
        out_path=None,
        title_suffix=title_suffix,
    )
    data = figure_to_png_bytes(fig)
    plt.close(fig)
    return data


def _cc_params(current_a: float) -> ProfileParams:
    return ProfileParams(
        family_id=TwoStepFamily.family_id,
        values={"i1": float(current_a), "i2": float(current_a), "soc_switch": 0.1},
    )


def _cccv_params(
    current_a: float,
    *,
    v_cv: float = 4.2,
    i_cutoff: float = 0.01,
) -> ProfileParams:
    """Classic CCCV: constant current ``current_a`` then CV hold at ``v_cv``."""
    return CCCVFamily.from_dict({
        "i_cc": float(current_a),
        "v_cv": float(v_cv),
        "i_cutoff": float(i_cutoff),
    })


def evaluate_cc_baselines(
    cell,
    simulator: ChargingSimulator,
    *,
    currents_a: Sequence[float] = DEFAULT_CC_CURRENTS_A,
    reward_mode: str = "hybrid_qloss",
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    z: float = 0.55,
) -> List[Dict[str, Any]]:
    """Evaluate fixed-CC baselines under the same constraint / reward as the optimizers."""
    from Constrained_BO.compare_constant_current import _format_profile, _metrics_row

    rows: List[Dict[str, Any]] = []
    family = TwoStepFamily()
    for current_a in currents_a:
        params = _cc_params(float(current_a))
        session = simulator.simulate(cell.start_state, params, family=family)
        _, metrics = evaluate_session(
            session,
            reward_mode=reward_mode,  # type: ignore[arg-type]
            w_soc=w_soc,
            w_qloss=w_qloss,
            w_time=w_time,
            w_temperature=w_temperature,
            z=z,
        )
        row = _metrics_row(
            "CC",
            f"CC {current_a:g}",
            metrics,
            profile=_format_profile(method="CC", current_a=float(current_a)),
            current_a=float(current_a),
            family_id=family.family_id,
            family_label=family.label,
            params=params.to_dict(),
        )
        rows.append(row)
    return rows


def evaluate_cccv_baselines(
    cell,
    simulator: ChargingSimulator,
    *,
    currents_a: Sequence[float] = DEFAULT_CC_CURRENTS_A,
    v_cv: Optional[float] = None,
    i_cutoff: float = 0.01,
    reward_mode: str = "hybrid_qloss",
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    z: float = 0.55,
) -> List[Dict[str, Any]]:
    """Evaluate classic CCCV baselines (CC phase then CV taper) under the same constraint/reward.

    Unlike ``evaluate_cc_baselines`` (flat current only), this uses ``CCCVFamily``:
    charge at ``i_cc`` until ``v_cv``, then hold voltage while current steps down to
    ``i_cutoff``.
    """
    from Constrained_BO.compare_constant_current import _format_profile, _metrics_row

    if cell.profile_bounds is not None:
        set_profile_bounds(cell.profile_bounds)
    v_hold = float(v_cv) if v_cv is not None else float(getattr(cell, "v_max", 4.2))
    rows: List[Dict[str, Any]] = []
    family = CCCVFamily()
    for current_a in currents_a:
        params = _cccv_params(float(current_a), v_cv=v_hold, i_cutoff=float(i_cutoff))
        session = simulator.simulate(cell.start_state, params, family=family)
        _, metrics = evaluate_session(
            session,
            reward_mode=reward_mode,  # type: ignore[arg-type]
            w_soc=w_soc,
            w_qloss=w_qloss,
            w_time=w_time,
            w_temperature=w_temperature,
            z=z,
        )
        row = _metrics_row(
            "CCCV",
            f"CCCV {current_a:g}",
            metrics,
            profile=_format_profile(
                method="CCCV",
                current_a=float(current_a),
                family_id=family.family_id,
                family_label=family.label,
                params=params.to_dict(),
            ),
            current_a=float(current_a),
            family_id=family.family_id,
            family_label=family.label,
            params=params.to_dict(),
        )
        rows.append(row)
    return rows


def _opt_row_from_payload(
    payload: Dict[str, Any],
    *,
    method_label: str,
) -> Optional[Dict[str, Any]]:
    """Best overall family entry as a baseline-comparison-style row."""
    from Constrained_BO.compare_constant_current import _format_profile, _metrics_row

    families = payload.get("families") or {}
    best = None
    for fid, entry in families.items():
        m = entry.get("best_metrics") or {}
        if not m:
            continue
        cand = (bool(m.get("feasible")), -float(m.get("loss", 1e9)))
        if best is None or cand > best[0]:
            best = (cand, fid, entry, m)
    if best is None:
        return None
    _, fid, entry, m = best
    family_label = entry.get("family_label") or get_family(fid).label
    # e.g. "GP-BO (Pulsed charge/rest)" — family name on the bar axis
    axis_label = f"{method_label} ({family_label})"
    return _metrics_row(
        method_label,
        axis_label,
        m,
        profile=_format_profile(
            method=method_label,
            family_id=fid,
            family_label=family_label,
            params=entry.get("best_params"),
            short=False,
        ),
        family_id=fid,
        family_label=family_label,
        params=entry.get("best_params"),
    )


def build_baseline_comparison_rows(
    cell,
    simulator: ChargingSimulator,
    *,
    bo_payload: Optional[Dict[str, Any]] = None,
    random_payload: Optional[Dict[str, Any]] = None,
    currents_a: Sequence[float] = DEFAULT_CC_CURRENTS_A,
    reward_kwargs: Optional[Dict[str, Any]] = None,
    use_cccv: bool = True,
) -> List[Dict[str, Any]]:
    rw = reward_kwargs or {}
    eval_kw = dict(
        currents_a=currents_a,
        reward_mode=rw.get("reward_mode", "hybrid_qloss"),
        w_soc=float(rw.get("w_soc", 1.0)),
        w_qloss=float(rw.get("w_qloss", 1.0)),
        w_time=float(rw.get("w_time", 0.1)),
        w_temperature=float(rw.get("w_temperature", 1.0)),
        z=float(rw.get("z", 0.55)),
    )
    rows = (
        evaluate_cccv_baselines(cell, simulator, **eval_kw)
        if use_cccv
        else evaluate_cc_baselines(cell, simulator, **eval_kw)
    )
    if bo_payload is not None:
        opt = _opt_row_from_payload(bo_payload, method_label="GP-BO")
        if opt is not None:
            # Use "Optimized" color path for plots that key off method name.
            opt_plot = dict(opt)
            opt_plot["method"] = "Optimized"
            # Keep family-aware label from _opt_row_from_payload
            rows.append(opt_plot)
    if random_payload is not None:
        rnd = _opt_row_from_payload(random_payload, method_label="Random")
        if rnd is not None:
            rnd_plot = dict(rnd)
            rnd_plot["method"] = "Random"
            rows.append(rnd_plot)
    return rows


def baseline_rows_to_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    keep = [
        "method", "label", "profile", "duration_min", "peak_temperature",
        "mean_temperature", "total_reward", "soc_reward", "qloss_penalty",
        "time_penalty", "qloss_total", "feasible", "end_reason",
        "energy_delivered_j",
    ]
    flat = []
    for r in rows:
        item = {k: r.get(k) for k in keep if k != "energy_delivered_j"}
        item["energy_delivered_j"] = (r.get("metrics") or {}).get(
            "energy_delivered_j", r.get("energy_delivered_j"),
        )
        flat.append(item)
    return pd.DataFrame(flat)


def plot_baseline_bar_png(
    rows: List[Dict[str, Any]],
    *,
    value_key: str,
    ylabel: str,
    title: str,
) -> bytes:
    """Hima-style bar chart; CC + Optimized/Random (exclude high-current infeasible CC from plot)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from Constrained_BO.compare_constant_current import PLOT_CC_CURRENTS_A, _infeasible_bar_label

    plot_currents = set(PLOT_CC_CURRENTS_A)
    cc = sorted(
        (r for r in rows if r["method"] == "CC" and r.get("current_a") in plot_currents),
        key=lambda r: float(r["current_a"]),
    )
    others = [r for r in rows if r["method"] in ("Optimized", "Random", "GP-BO", "Random search")]
    plot_rows = cc + others
    if not plot_rows:
        plot_rows = list(rows)

    color_map = {
        "CC": "#2563eb",
        "Optimized": "#9333ea",
        "GP-BO": "#9333ea",
        "Random": "#ea580c",
        "Random search": "#ea580c",
    }
    labels = [r["label"] for r in plot_rows]
    values = [float(r[value_key]) for r in plot_rows]
    colors = [color_map.get(r["method"], "#6b7280") for r in plot_rows]

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, row in zip(bars, plot_rows):
        if not row.get("feasible", True):
            bar.set_hatch("//")
            bar.set_alpha(0.55)
            y = bar.get_height()
            span = abs(ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
            y_anchor = y + 0.02 * span if y >= 0 else y - 0.04 * span
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y_anchor,
                _infeasible_bar_label(row),
                ha="center",
                va="bottom" if y >= 0 else "top",
                fontsize=8,
                color="#555555",
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    legend_items = [
        Patch(facecolor="#2563eb", label="CC baseline"),
        Patch(facecolor="#9333ea", label="GP-BO best"),
        Patch(facecolor="#ea580c", label="Random best"),
    ]
    if any(not r.get("feasible", True) for r in plot_rows):
        legend_items.append(
            Patch(facecolor="white", edgecolor="#666", hatch="//", label="Infeasible"),
        )
    ax.legend(handles=legend_items, loc="best", fontsize=8)
    fig.tight_layout()
    data = figure_to_png_bytes(fig)
    plt.close(fig)
    return data


def generate_degradation_report_figures(
    out_dir: Path,
    *,
    results_path: Optional[Path] = None,
    device: str = "auto",
) -> Dict[str, Path]:
    """Write degradation report figures; return map of name → path for those that exist."""
    import matplotlib.pyplot as plt

    from Constrained_BO.degradation_report import (
        BDTUnavailableError,
        plot_calendar_contour,
        plot_cyclic_curves,
        plot_cumulative_degradation,
        plot_equal_energy_table,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    fig1 = plot_calendar_contour(out_dir / "fig1_calendar_contour.png")
    plt.close(fig1)
    paths["fig1_calendar_contour"] = out_dir / "fig1_calendar_contour.png"

    fig2 = plot_cyclic_curves(out_dir / "fig2_cyclic_curves.png")
    plt.close(fig2)
    paths["fig2_cyclic_curves"] = out_dir / "fig2_cyclic_curves.png"

    if results_path is None or not Path(results_path).is_file():
        return paths

    try:
        fig3 = plot_cumulative_degradation(
            Path(results_path), out_dir / "fig3_cumulative_degradation.png", device=device,
        )
        plt.close(fig3)
        paths["fig3_cumulative_degradation"] = out_dir / "fig3_cumulative_degradation.png"
    except BDTUnavailableError:
        pass

    try:
        fig4 = plot_equal_energy_table(
            Path(results_path),
            out_dir / "fig4_equal_energy_table.png",
            csv_out_path=out_dir / "fig4_equal_energy_table.csv",
        )
        plt.close(fig4)
        paths["fig4_equal_energy_table"] = out_dir / "fig4_equal_energy_table.png"
        csv_p = out_dir / "fig4_equal_energy_table.csv"
        if csv_p.is_file():
            paths["fig4_equal_energy_table_csv"] = csv_p
    except Exception:
        pass

    return paths


def save_run_artifacts(
    out_dir: Path,
    *,
    bo_payload: Dict[str, Any],
    random_payload: Dict[str, Any],
    bo_results: Dict[str, Any],
    random_results: Dict[str, Any],
    cell,
    comparison_df: pd.DataFrame,
    baseline_rows: Optional[List[Dict[str, Any]]] = None,
    device: str = "auto",
) -> Dict[str, Path]:
    """Write JSON / PNG / Excel under ``out_dir``; return path map."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    bo_json = out_dir / "gp_bo_results.json"
    rnd_json = out_dir / "random_search_results.json"
    bo_json.write_text(json.dumps(bo_payload, indent=2))
    rnd_json.write_text(json.dumps(random_payload, indent=2))
    paths["gp_bo_json"] = bo_json
    paths["random_json"] = rnd_json

    efrac = cell.energy_fraction
    bo_title = f"GP-BO, {bo_payload['meta'].get('reward_mode')}"
    rnd_title = f"random search, {random_payload['meta'].get('reward_mode')}"
    if cell.constraint_mode == "energy" and efrac is not None:
        bo_title += f", energy ≥ {efrac:.0%} of pack"
        rnd_title += f", energy ≥ {efrac:.0%} of pack"

    bo_png = out_dir / "gp_bo_best_profiles.png"
    rnd_png = out_dir / "random_best_profiles.png"
    bo_png.write_bytes(plot_profiles_png(bo_results, cell, title_suffix=bo_title))
    rnd_png.write_bytes(plot_profiles_png(random_results, cell, title_suffix=rnd_title))
    paths["gp_bo_png"] = bo_png
    paths["random_png"] = rnd_png

    xlsx = out_dir / "bo_vs_random_comparison.xlsx"
    xlsx.write_bytes(dataframe_to_excel_bytes(comparison_df))
    paths["excel"] = xlsx
    comparison_df.to_csv(out_dir / "bo_vs_random_comparison.csv", index=False)
    paths["csv"] = out_dir / "bo_vs_random_comparison.csv"

    if baseline_rows:
        bdf = baseline_rows_to_dataframe(baseline_rows)
        bdf.to_csv(out_dir / "baseline_comparison.csv", index=False)
        paths["baseline_csv"] = out_dir / "baseline_comparison.csv"
        for key, ylabel, title, fname in (
            ("total_reward", "Total reward", "Reward comparison", "reward_comparison.png"),
            ("duration_min", "Duration (min)", "Time comparison", "time_comparison.png"),
            ("peak_temperature", "Peak T (°C)", "Temperature comparison", "temperature_comparison.png"),
        ):
            png = plot_baseline_bar_png(baseline_rows, value_key=key, ylabel=ylabel, title=title)
            p = out_dir / fname
            p.write_bytes(png)
            paths[fname] = p

    deg_dir = out_dir / "degradation_report"
    deg_paths = generate_degradation_report_figures(
        deg_dir, results_path=bo_json, device=device,
    )
    paths.update({f"deg_{k}": v for k, v in deg_paths.items()})
    return paths
