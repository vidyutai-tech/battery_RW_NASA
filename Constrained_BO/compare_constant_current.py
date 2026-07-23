#!/usr/bin/env python3
"""Compare constant-current baselines against optimized charging profiles."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Constrained_BO.config import energy_fraction_for, get_cell_config
from Constrained_BO.objective import evaluate_session, reward_kwargs_from_meta
from Constrained_BO.profile_catalog import ProfileBounds
from Constrained_BO.profiles import ProfileParams, TwoStepFamily, get_family, set_profile_bounds
from Constrained_BO.simulator import ChargingSimulator
from Constrained_BO.viz import plot_optimized_profile

DEFAULT_CC_CURRENTS_A = (0.5, 1.0, 2.0, 3.0, 4.0)
PLOT_CC_CURRENTS_A = (0.5, 1.0)  # feasible baselines shown in bar charts
_COMPARE_OUTPUTS = (
    "baseline_results.json",
    "reward_comparison.png",
    "time_comparison.png",
    "temperature_comparison.png",
    "optimized_profile.png",
    "comparison_table.csv",
    "comparison_table.md",
)


def _is_writable_out_dir(d: Path) -> bool:
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


def _resolve_out_dir(requested: Path | None, cell_id: str) -> Path:
    import getpass

    if requested is not None:
        return Path(requested)

    default = Path(__file__).resolve().parent / "results" / cell_id / "baseline_comparison"
    if _is_writable_out_dir(default):
        return default

    user = getpass.getuser()
    fallback = (
        Path(__file__).resolve().parent
        / "results"
        / "_compare"
        / user
        / cell_id
        / "baseline_comparison"
    )
    print(
        f"Warning: {default} is not writable; writing to {fallback} instead.\n"
        f"  To reuse {default}, run: sudo chown -R $USER {default.parent}"
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
            f"Re-run with --out-dir, or run: sudo chown -R $USER {path.parent}"
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


def _cell_from_meta(meta: Dict[str, Any]):
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
    return ProfileParams(family_id=d["family_id"], values=vals)


def _cc_params(current_a: float) -> ProfileParams:
    """Fixed CC via two-step family with i1 == i2 (no SoC switching)."""
    return ProfileParams(
        family_id=TwoStepFamily.family_id,
        values={"i1": float(current_a), "i2": float(current_a), "soc_switch": 0.1},
    )


def _evaluate(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    params: ProfileParams,
    *,
    reward_kwargs: Optional[Dict[str, Any]] = None,
    w_time: float = 0.1,
    w_temperature: float = 1.0,
) -> Tuple[Dict, Dict]:
    family = get_family(params.family_id)
    session = simulator.simulate(initial_state, params, family=family)
    kwargs = dict(reward_kwargs or {})
    if not kwargs:
        kwargs = {"w_time": w_time, "w_temperature": w_temperature}
    _, metrics = evaluate_session(session, **kwargs)
    return session, metrics


def _metrics_row(
    method: str,
    label: str,
    metrics: Dict[str, Any],
    *,
    profile: str,
    current_a: Optional[float] = None,
    family_id: Optional[str] = None,
    family_label: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = {
        "method": method,
        "label": label,
        "profile": profile,
        "current_a": current_a,
        "family_id": family_id,
        "family_label": family_label,
        "params": params,
        "duration_min": metrics["duration_min"],
        "duration_s": metrics["duration_s"],
        "peak_temperature": metrics["peak_temperature"],
        "mean_temperature": metrics["mean_temperature"],
        "temperature_reward": metrics.get("temperature_reward", 0.0),
        "time_reward": metrics.get("time_reward", 0.0),
        "soc_reward": metrics.get("soc_reward", 0.0),
        "qloss_total": metrics.get("qloss_total", 0.0),
        "qloss_calendar": metrics.get("qloss_calendar", 0.0),
        "qloss_cyclic": metrics.get("qloss_cyclic", 0.0),
        "time_penalty": metrics.get("time_penalty", 0.0),
        "total_reward": metrics["total_reward"],
        "feasible": metrics["feasible"],
        "end_reason": metrics["end_reason"],
        "metrics": {k: v for k, v in metrics.items() if k != "reward_weights"},
    }
    return row


def _format_profile(
    *,
    method: str,
    current_a: Optional[float] = None,
    family_id: Optional[str] = None,
    family_label: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    short: bool = False,
) -> str:
    if method == "CC" and current_a is not None:
        return f"CC {current_a:g} A"

    params = params or {}
    if family_id == "pulsed":
        ic = float(params.get("i_charge", 0.0))
        if short:
            return f"Pulsed ({ic:.2f} A)"
        on = float(params.get("pulse_on_min", 0.0))
        rest = float(params.get("pulse_rest_min", 0.0))
        return f"Pulsed — {ic:.2f} A ON / {rest:.2f} min REST / {on:.2f} min ON"

    if family_id == "cccv":
        ic = float(params.get("i_cc", 0.0))
        return f"CCCV ({ic:.2f} A)" if short else f"CCCV — {ic:.2f} A CC"
    if family_id == "two_step":
        i1 = float(params.get("i1", 0.0))
        i2 = float(params.get("i2", 0.0))
        return f"2-step ({i1:.2f}/{i2:.2f} A)" if short else f"2-step — {i1:.2f} A → {i2:.2f} A"
    if family_id == "three_step":
        i1 = float(params.get("i1", 0.0))
        return f"3-step ({i1:.2f} A)" if short else f"3-step — i1={i1:.2f} A"

    return family_label or (get_family(family_id).label if family_id else "Optimized")


def _find_results_path(cell_id: str, energy_fraction: Optional[float]) -> Optional[Path]:
    base = Path(__file__).resolve().parent / "results"
    candidates = [
        base / "hima" / cell_id / "constrained_bo_results.json",
        base / "_run" / "hima" / cell_id / "constrained_bo_results.json",
        base / cell_id / "constrained_bo_results.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        with open(path) as f:
            meta = json.load(f)["meta"]
        if meta.get("cell", "").upper() != cell_id.upper():
            continue
        if energy_fraction is not None:
            frac = meta.get("energy_fraction")
            if frac is None or abs(float(frac) - float(energy_fraction)) > 1e-6:
                continue
        return path
    return None


def _load_or_run_optimizer(
    *,
    cell_id: str,
    energy_fraction: float,
    results_path: Optional[Path],
    run_optimizer: bool,
    device: str,
) -> Tuple[Path, Dict[str, Any]]:
    if results_path is not None:
        path = Path(results_path)
        if not path.is_file():
            raise FileNotFoundError(f"Results not found: {path}")
    elif run_optimizer:
        opt_out = Path(__file__).resolve().parent / "results" / cell_id
        opt_out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "Constrained_BO.run",
            "--cell", cell_id,
            "--energy-fraction", str(energy_fraction),
            "--out-dir", str(opt_out),
            "--device", device,
        ]
        print(f"Running optimizer: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        path = opt_out / "constrained_bo_results.json"
    else:
        path = _find_results_path(cell_id, energy_fraction)
        if path is None:
            raise FileNotFoundError(
                f"No optimization results for {cell_id} at energy_fraction={energy_fraction}. "
                "Run Constrained_BO.run first or pass --results / --run-optimizer."
            )
        print(f"Loaded optimizer results from {path}")

    with open(path) as f:
        payload = json.load(f)
    return path, payload


_FAMILY_TIEBREAK_ORDER = ("pulsed", "three_step", "two_step", "cccv")


def _best_optimized(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    families = payload["families"]
    best_loss = min(families[fid]["best_loss"] for fid in families)
    tied = [
        fid for fid in families
        if abs(float(families[fid]["best_loss"]) - float(best_loss)) <= 1e-6
    ]
    best_fid = tied[0]
    for pref in _FAMILY_TIEBREAK_ORDER:
        if pref in tied:
            best_fid = pref
            break
    entry = families[best_fid]
    return best_fid, entry["best_params"], entry["best_metrics"]


def _print_table(rows: List[Dict[str, Any]]) -> None:
    sorted_rows = sorted(rows, key=lambda r: r["total_reward"], reverse=True)
    headers = (
        "Method", "Profile", "Time (min)", "Peak Temp",
        "Temp Reward", "Time Reward", "Total Reward", "Feasible",
    )
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in sorted_rows:
        profile = row.get("profile_short") or row["profile"]
        feasible = "yes" if row["feasible"] else "no"
        print(
            f"| {row['label']:9s} | {profile:22s} | "
            f"{row['duration_min']:10.1f} | {row['peak_temperature']:9.2f} | "
            f"{row['temperature_reward']:11.3f} | {row['time_reward']:11.2f} | "
            f"{row['total_reward']:12.3f} | {feasible:>8s} |"
        )
    print()
    n_feasible = sum(1 for r in rows if r["feasible"])
    if n_feasible < len(rows):
        print(
            f"Note: {len(rows) - n_feasible} profile(s) did not meet the energy target "
            f"({n_feasible}/{len(rows)} feasible). Compare feasible rows for a fair baseline."
        )


def _table_export_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda r: r["total_reward"], reverse=True)
    export = []
    for row in sorted_rows:
        export.append({
            "method": row["method"],
            "label": row["label"],
            "profile": row["profile"],
            "profile_short": row.get("profile_short", row["profile"]),
            "current_a": "" if row["current_a"] is None else row["current_a"],
            "family": row.get("family_label") or row.get("family_id") or "",
            "time_min": row["duration_min"],
            "peak_temp_c": row["peak_temperature"],
            "mean_temp_c": row["mean_temperature"],
            "temp_reward": row["temperature_reward"],
            "time_reward": row["time_reward"],
            "total_reward": row["total_reward"],
            "feasible": row["feasible"],
            "end_reason": row["end_reason"],
        })
    return export


def _write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    export = _table_export_rows(rows)
    fieldnames = list(export[0].keys()) if export else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export)


def _write_comparison_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    export = _table_export_rows(rows)
    headers = [
        "Method", "Profile", "Time (min)", "Peak Temp (°C)",
        "Temp Reward", "Time Reward", "Total Reward", "Feasible",
    ]
    lines = [
        "# Baseline comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in export:
        feasible = "yes" if row["feasible"] else "no"
        lines.append(
            f"| {row['label']} | {row['profile_short']} | "
            f"{row['time_min']:.1f} | {row['peak_temp_c']:.2f} | "
            f"{row['temp_reward']:.3f} | {row['time_reward']:.2f} | "
            f"{row['total_reward']:.3f} | {feasible} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _bar_chart_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """CC 0.5 / 1.0 + Optimized only (exclude infeasible high-current CC from plots)."""
    plot_currents = set(PLOT_CC_CURRENTS_A)
    cc = sorted(
        (r for r in rows if r["method"] == "CC" and r.get("current_a") in plot_currents),
        key=lambda r: r["current_a"],
    )
    opt = [r for r in rows if r["method"] == "Optimized"]
    return cc + opt


def _infeasible_bar_label(row: Dict[str, Any]) -> str:
    """Short (2–3 word) annotation for hatched infeasible bars."""
    end = str(row.get("end_reason", "")).lower()
    if "time" in end:
        return "Too slow"
    shortfall = float(row.get("metrics", {}).get("energy_shortfall_j", 0.0) or 0.0)
    if shortfall > 1.0:
        return "Low energy"
    if "voltage" in end or "v_max" in end:
        return "V limit"
    return "Infeasible"


def _plot_bar_comparison(
    rows: List[Dict[str, Any]],
    *,
    value_key: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    labels = [r["label"] for r in rows]
    values = [r[value_key] for r in rows]
    colors = ["#2563eb" if r["method"] == "CC" else "#9333ea" for r in rows]

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, row in zip(bars, rows):
        if not row["feasible"]:
            bar.set_hatch("//")
            bar.set_alpha(0.55)
            y = bar.get_height()
            y_anchor = y + (0.02 * abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
            if y < 0:
                y_anchor = y - (0.04 * abs(ax.get_ylim()[1] - ax.get_ylim()[0]))
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
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#2563eb", label="CC baseline"),
        Patch(facecolor="#9333ea", label="Optimized"),
    ]
    if any(not row["feasible"] for row in rows):
        legend_items.append(
            Patch(facecolor="white", edgecolor="#666", hatch="//", label="Infeasible"),
        )
    ax.legend(handles=legend_items, loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare constant-current baselines vs optimized charging",
    )
    parser.add_argument("--cell", default="RW9")
    parser.add_argument(
        "--energy-fraction",
        type=float,
        default=None,
        help="Fraction of pack energy to deliver (default: per-cell, usually 0.40)",
    )
    parser.add_argument(
        "--currents",
        type=float,
        nargs="+",
        default=list(DEFAULT_CC_CURRENTS_A),
        help="Constant charge currents to evaluate (A)",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Path to constrained_bo_results.json (default: auto-discover)",
    )
    parser.add_argument(
        "--run-optimizer",
        action="store_true",
        help="Run Constrained_BO.run before comparison if results are missing",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: Constrained_BO/results/<cell>/baseline_comparison)",
    )
    args = parser.parse_args()

    cell_id = args.cell.upper()
    energy_fraction = (
        args.energy_fraction
        if args.energy_fraction is not None
        else energy_fraction_for(cell_id)
    )
    results_path, payload = _load_or_run_optimizer(
        cell_id=cell_id,
        energy_fraction=energy_fraction,
        results_path=args.results,
        run_optimizer=args.run_optimizer,
        device=args.device,
    )
    meta = payload["meta"]

    cell = _cell_from_meta(meta)
    if cell.profile_bounds is not None:
        set_profile_bounds(cell.profile_bounds)

    w_time = float(meta["reward_weights"]["w_time"])
    w_temperature = float(meta["reward_weights"].get("w_temperature", 1.0))
    reward_kwargs = reward_kwargs_from_meta(meta)
    simulator = ChargingSimulator.from_cell(cell, device=args.device)
    simulator.decision_interval_info = meta.get("decision_interval_selection", {})

    cc_rows: List[Dict[str, Any]] = []
    cc_family = TwoStepFamily()
    for current_a in args.currents:
        params = _cc_params(current_a)
        session, metrics = _evaluate(
            simulator,
            cell.start_state,
            params,
            reward_kwargs=reward_kwargs,
        )
        cc_rows.append(_metrics_row(
            "CC",
            f"CC {current_a:g}",
            metrics,
            profile=_format_profile(method="CC", current_a=float(current_a)),
            current_a=float(current_a),
            family_id=cc_family.family_id,
            family_label=cc_family.label,
            params=params.to_dict(),
        ))
        cc_rows[-1]["profile_short"] = cc_rows[-1]["profile"]

    best_fid, best_params, _stored_metrics = _best_optimized(payload)
    opt_params = _params_from_dict(best_params)
    opt_session, opt_metrics = _evaluate(
        simulator,
        cell.start_state,
        opt_params,
        reward_kwargs=reward_kwargs,
    )
    opt_family = get_family(best_fid)
    opt_label = opt_family.label
    opt_profile = _format_profile(
        method="Optimized",
        family_id=best_fid,
        family_label=opt_label,
        params=best_params,
        short=False,
    )
    opt_profile_short = _format_profile(
        method="Optimized",
        family_id=best_fid,
        family_label=opt_label,
        params=best_params,
        short=True,
    )
    optimized_row = _metrics_row(
        "Optimized",
        "Optimized",
        opt_metrics,
        profile=opt_profile,
        current_a=None,
        family_id=best_fid,
        family_label=opt_label,
        params=best_params,
    )
    optimized_row["profile_short"] = opt_profile_short

    all_rows = cc_rows + [optimized_row]
    comparison_table = sorted(all_rows, key=lambda r: r["total_reward"], reverse=True)

    out_dir = _resolve_out_dir(args.out_dir, cell_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "cell": cell_id,
        "energy_fraction": meta.get("energy_fraction"),
        "constraint_mode": meta.get("constraint_mode"),
        "optimizer_results": str(results_path),
        "cc_currents_a": list(args.currents),
        "cc_profile": {
            "family_id": TwoStepFamily.family_id,
            "description": "Fixed CC via two_step with i1=i2 and soc_switch=0.1",
        },
        "cc_baselines": cc_rows,
        "optimized": optimized_row,
        "optimized_session_meta": {
            "family_id": best_fid,
            "family_label": opt_label,
            "params": best_params,
            "duration_s": opt_metrics["duration_s"],
            "end_reason": opt_metrics["end_reason"],
        },
        "comparison_table": comparison_table,
    }
    json_path = out_dir / "baseline_results.json"
    _write_json(json_path, summary)

    plot_rows = _bar_chart_rows(cc_rows + [optimized_row])
    _plot_bar_comparison(
        plot_rows,
        value_key="total_reward",
        ylabel="Total Reward",
        title=f"Total Reward — {cell_id} (energy {meta.get('energy_fraction', 0):.0%})",
        out_path=out_dir / "reward_comparison.png",
    )
    _plot_bar_comparison(
        plot_rows,
        value_key="duration_min",
        ylabel="Charging Time (minutes)",
        title=f"Charging Time — {cell_id} (energy {meta.get('energy_fraction', 0):.0%})",
        out_path=out_dir / "time_comparison.png",
    )
    _plot_bar_comparison(
        plot_rows,
        value_key="peak_temperature",
        ylabel="Peak Temperature (°C)",
        title=f"Peak Temperature — {cell_id} (energy {meta.get('energy_fraction', 0):.0%})",
        out_path=out_dir / "temperature_comparison.png",
    )

    profile_fig = plot_optimized_profile(
        opt_session,
        opt_metrics,
        family_label=opt_label,
        params=best_params,
        soc_target=float(meta.get("soc_target", cell.soc_target)),
        soc_start=float(meta.get("soc_start", cell.start_state.get("soc", 0.2))),
        out_path=out_dir / "optimized_profile.png",
    )
    plt.close(profile_fig)

    _write_comparison_csv(out_dir / "comparison_table.csv", all_rows)
    _write_comparison_md(out_dir / "comparison_table.md", all_rows)

    print(f"\n=== Constant-current baseline comparison ({cell_id}) ===")
    print(f"Energy fraction: {meta.get('energy_fraction')}")
    print(f"Decision interval: {simulator.decision_interval_s} s")
    print(f"Optimizer source: {results_path}")
    print(f"Best optimized: {opt_label} ({best_fid})")
    _print_table(all_rows)

    print(f"Wrote {json_path}")
    print(f"Wrote {out_dir / 'reward_comparison.png'}")
    print(f"Wrote {out_dir / 'time_comparison.png'}")
    print(f"Wrote {out_dir / 'temperature_comparison.png'}")
    print(f"Wrote {out_dir / 'optimized_profile.png'}")
    print(f"Wrote {out_dir / 'comparison_table.csv'}")
    print(f"Wrote {out_dir / 'comparison_table.md'}")


if __name__ == "__main__":
    main()
