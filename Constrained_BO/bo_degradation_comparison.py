"""One-plot degradation comparison: CC (½C / 1C / 2C) vs GP-BO vs Random.

Uses the same energy/SoC constraint and hybrid Q_loss index as the optimizer
runs. Q_loss is a Relative Capacity-Loss Index for one charging session — not
a multi-year measured % fade.

Usage
-----
    python -m Constrained_BO.bo_degradation_comparison \\
        --bo Constrained_BO/results/ui_runs_1/RW9/gp_bo_results.json \\
        --random Constrained_BO/results/ui_runs_1/RW9/random_search_results.json \\
        --out-dir Constrained_BO/results/grounded_figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Constrained_BO.compare_constant_current import _format_profile, _metrics_row
from Constrained_BO.config import Q_RATED_AH, energy_fraction_for
from Constrained_BO.objective import evaluate_session
from Constrained_BO.optimize_api import build_cell, evaluate_cc_baselines
from Constrained_BO.profiles import get_family, set_profile_bounds
from Constrained_BO.simulator import ChargingSimulator

ROOT = Path(__file__).resolve().parents[1]

# C-rate → amperes for NASA RW (Q_rated = 2.2 Ah → 1C = 2.2 A)
DEFAULT_C_RATES = (0.5, 1.0, 2.0)

COLORS = {
    "CC ½C": "#64748b",
    "CC 1C": "#2563eb",
    "CC 2C": "#1e40af",
    "Random": "#f59e0b",
    "GP-BO": "#16a34a",
}


def _c_rate_label(c_rate: float) -> str:
    if abs(c_rate - 0.5) < 1e-9:
        return "CC ½C"
    if abs(c_rate - 1.0) < 1e-9:
        return "CC 1C"
    if abs(c_rate - 2.0) < 1e-9:
        return "CC 2C"
    return f"CC {c_rate:g}C"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _cell_from_meta(meta: Dict[str, Any]):
    cell_id = str(meta["cell"]).upper()
    efrac = meta.get("energy_fraction")
    if efrac is None and meta.get("constraint_mode") == "energy":
        # recover from start_state / soc window if stored in reward path
        efrac = energy_fraction_for(cell_id)
    # Prefer explicit energy from energy_required if present on families later;
    # ui meta usually has energy via cell rebuild:
    soc_mode = meta.get("constraint_mode") == "soc"
    cell = build_cell(
        cell_id,
        energy_fraction=None if soc_mode else float(
            meta.get("energy_fraction") or energy_fraction_for(cell_id)
        ),
        soc_mode=soc_mode,
        soc_target=float(meta["soc_target"]) if meta.get("soc_target") is not None else None,
        max_duration_min=float(meta["max_duration_min"]) if meta.get("max_duration_min") else None,
        decision_interval_s=meta.get("decision_interval_s"),
        auto_decision_interval=meta.get("decision_interval_s") is None,
    )
    # Match saved start temperature / SoC when present
    ss = meta.get("start_state") or {}
    if ss:
        cell.start_state = {**cell.start_state, **{k: float(v) for k, v in ss.items()}}
    if cell.profile_bounds is not None:
        set_profile_bounds(cell.profile_bounds)
    return cell


def _reward_kwargs(meta: Dict[str, Any]) -> Dict[str, Any]:
    rw = meta.get("reward_weights") or {}
    return {
        "reward_mode": meta.get("reward_mode", "hybrid_qloss"),
        "w_soc": float(rw.get("w_soc", 1.0)),
        "w_qloss": float(rw.get("w_qloss", 1.0)),
        "w_time": float(rw.get("w_time", 0.1)),
        "w_temperature": float(rw.get("w_temperature", 1.0)),
        "z": float(rw.get("z", 0.55)),
    }


def _pick_best_family(
    payload: Dict[str, Any],
    *,
    by: str = "reward",
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Pick best trial across families.

    by='reward' — feasible-first, then min optimizer loss (default BO winner).
    by='qloss'  — feasible-first, then min session Q_loss index.
    """
    best = None
    best_params: Dict[str, Any] = {}
    for fid, entry in (payload.get("families") or {}).items():
        hist = entry.get("history") or []
        if hist:
            pool = hist
        else:
            m = entry.get("best_metrics") or {}
            if not m:
                continue
            pool = [{
                "feasible": m.get("feasible"),
                "loss": m.get("loss"),
                "metrics": m,
                "params": entry.get("best_params") or {},
                "family_id": fid,
            }]
        for h in pool:
            m = h.get("metrics") or {}
            if not m:
                continue
            feas = bool(h.get("feasible", m.get("feasible")))
            if by == "qloss":
                score = (feas, -float(m.get("qloss_total", 1e9)))
            else:
                score = (feas, -float(h.get("loss", m.get("loss", 1e9))))
            if best is None or score > best[0]:
                best = (score, fid, entry, m)
                best_params = dict(h.get("params") or entry.get("best_params") or {})
    if best is None:
        raise ValueError("No family metrics in optimizer payload")
    _, fid, entry, m = best
    return fid, entry, m, best_params


def _params_from_values(fid: str, values: Dict[str, Any]):
    raw = dict(values or {})
    raw.pop("family_id", None)
    family = get_family(fid)
    return family.from_dict({k: float(v) for k, v in raw.items()}), family


def _eval_optimizer_best(
    cell,
    simulator: ChargingSimulator,
    payload: Dict[str, Any],
    *,
    method_label: str,
    reward_kwargs: Dict[str, Any],
    by: str = "reward",
) -> Dict[str, Any]:
    fid, entry, _, values = _pick_best_family(payload, by=by)
    params, family = _params_from_values(fid, values)
    session = simulator.simulate(cell.start_state, params, family=family)
    _, metrics = evaluate_session(session, **reward_kwargs)
    family_label = entry.get("family_label") or family.label
    suffix = "min Q" if by == "qloss" else family_label
    label = f"{method_label}\n({suffix})"
    return _metrics_row(
        method_label,
        label,
        metrics,
        profile=_format_profile(
            method=method_label,
            family_id=fid,
            family_label=family_label,
            params=params.to_dict(),
            short=False,
        ),
        family_id=fid,
        family_label=family_label,
        params=params.to_dict(),
    )


def _history_points(payload: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
    pts: List[Dict[str, Any]] = []
    for fid, entry in (payload.get("families") or {}).items():
        for h in entry.get("history") or []:
            m = h.get("metrics") or {}
            if not m:
                continue
            pts.append({
                "method": method,
                "family_id": fid,
                "feasible": bool(h.get("feasible", m.get("feasible"))),
                "qloss_total": float(m.get("qloss_total") or 0.0),
                "duration_min": float(m.get("duration_min") or 0.0),
                "peak_temperature": float(m.get("peak_temperature") or 0.0),
                "total_reward": float(m.get("total_reward") or 0.0),
            })
    return pts


def build_comparison_rows(
    bo_path: Path,
    random_path: Path,
    *,
    c_rates: Sequence[float] = DEFAULT_C_RATES,
    q_rated_ah: float = Q_RATED_AH,
    device: str = "cpu",
    optimizer_by: str = "reward",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    bo = _load_json(bo_path)
    rnd = _load_json(random_path)
    meta = bo.get("meta") or rnd.get("meta") or {}
    cell = _cell_from_meta(meta)
    # Prefer energy_fraction stored on a family metric if meta lacks it
    if meta.get("energy_fraction") is None and cell.constraint_mode == "energy":
        for entry in (bo.get("families") or {}).values():
            ef = (entry.get("best_metrics") or {}).get("energy_fraction")
            if ef is not None:
                cell = cell.with_run_overrides(energy_fraction=float(ef))
                break
    rw = _reward_kwargs(meta)
    simulator = ChargingSimulator.from_cell(cell, device=device)

    currents = [float(c) * float(q_rated_ah) for c in c_rates]
    cc_rows = evaluate_cc_baselines(cell, simulator, currents_a=currents, **rw)

    rows: List[Dict[str, Any]] = []
    for c_rate, row in zip(c_rates, cc_rows):
        amps = float(c_rate) * float(q_rated_ah)
        short = _c_rate_label(float(c_rate))
        row = dict(row)
        row["axis_label"] = f"{short}\n({amps:g} A)"
        row["group"] = short
        row["c_rate"] = float(c_rate)
        rows.append(row)

    for path, method, group in (
        (random_path, "Random", "Random"),
        (bo_path, "GP-BO", "GP-BO"),
    ):
        payload = _load_json(path)
        row = _eval_optimizer_best(
            cell, simulator, payload,
            method_label=method, reward_kwargs=rw, by=optimizer_by,
        )
        row = dict(row)
        row["axis_label"] = row["label"]
        row["group"] = group
        row["c_rate"] = None
        rows.append(row)

    cloud = _history_points(rnd, "Random") + _history_points(bo, "GP-BO")

    info = {
        "cell": cell.cell_id,
        "constraint_mode": cell.constraint_mode,
        "energy_fraction": cell.energy_fraction,
        "soc_target": cell.soc_target,
        "q_rated_ah": q_rated_ah,
        "c_rates": list(c_rates),
        "currents_a": currents,
        "reward_mode": rw["reward_mode"],
        "start_state": cell.start_state,
        "bdt_ckpt": str(cell.bdt_ckpt),
        "optimizer_selection": optimizer_by,
    }
    return rows, info, cloud


def _save_csv(rows: List[Dict[str, Any]], path: Path, info: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "group", "label", "axis_label", "c_rate", "current_a", "feasible",
        "qloss_total", "qloss_calendar", "qloss_cyclic", "duration_min",
        "peak_temperature", "mean_temperature", "total_reward", "end_reason",
        "profile",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fields}
            if out.get("current_a") is None and r.get("c_rate") is not None:
                out["current_a"] = float(r["c_rate"]) * float(info["q_rated_ah"])
            w.writerow(out)


def plot_simple_one_axis(
    rows: List[Dict[str, Any]],
    info: Dict[str, Any],
    out_path: Path,
) -> None:
    """Clean single-axis bar chart (main deliverable)."""
    labels = [r["axis_label"] for r in rows]
    q_tot = [float(r.get("qloss_total") or 0.0) for r in rows]
    feasible = [bool(r.get("feasible")) for r in rows]
    colors = [COLORS.get(r["group"], "#94a3b8") for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=150)
    x = np.arange(len(rows))
    bars = ax.bar(x, q_tot, color=colors, edgecolor="k", lw=0.6, width=0.7)
    ymax = max(q_tot) if q_tot else 1.0
    for bar, ok, q, r in zip(bars, feasible, q_tot, rows):
        if not ok:
            bar.set_hatch("//")
            bar.set_alpha(0.4)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                q + ymax * 0.02,
                "infeasible\n(Vmax / shortfall)",
                ha="center", va="bottom", fontsize=7, color="#b91c1c",
            )
        else:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                q + ymax * 0.015,
                f"{q:.4f}",
                ha="center", va="bottom", fontsize=8, color="#0f172a",
            )

    # % vs feasible CC ½C
    base_q = None
    for r, q, ok in zip(rows, q_tot, feasible):
        if r["group"] == "CC ½C" and ok:
            base_q = q
            break
    if base_q and base_q > 0:
        for i, (r, q, ok) in enumerate(zip(rows, q_tot, feasible)):
            if r["group"] in ("GP-BO", "Random") and ok:
                red = 100.0 * (base_q - q) / base_q
                ax.annotate(
                    f"{red:+.1f}% vs ½C",
                    xy=(i, q),
                    xytext=(0, 22),
                    textcoords="offset points",
                    ha="center", fontsize=8,
                    color="#166534" if red > 0 else "#9a3412",
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Session degradation Q_loss (index, lower better)")
    cell = info.get("cell", "?")
    efrac = info.get("energy_fraction")
    mode = info.get("constraint_mode")
    if mode == "energy" and efrac is not None:
        constraint = f"{100 * float(efrac):.0f}% energy target"
    else:
        constraint = f"SoC → {info.get('soc_target')}"
    ax.set_title(
        f"Charging policy vs degradation — {cell}\n"
        f"CC ½C / 1C / 2C   vs   Random best   vs   GP-BO best\n"
        f"({constraint}; same BDT twin + hybrid calendar/cyclic model)"
    )
    ax.set_ylim(0, ymax * 1.35)
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor=COLORS["CC ½C"], edgecolor="k", label="CC ½C (1.1 A)"),
        Patch(facecolor=COLORS["CC 1C"], edgecolor="k", label="CC 1C (2.2 A)"),
        Patch(facecolor=COLORS["CC 2C"], edgecolor="k", label="CC 2C (4.4 A)"),
        Patch(facecolor=COLORS["Random"], edgecolor="k", label="Random best"),
        Patch(facecolor=COLORS["GP-BO"], edgecolor="k", label="GP-BO best"),
        Patch(facecolor="#94a3b8", edgecolor="k", hatch="//", label="Infeasible (energy)"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="upper left", framealpha=0.95)

    fig.text(
        0.5, -0.06,
        "NASA RW Q_rated = 2.2 Ah → ½C=1.1 A, 1C=2.2 A, 2C=4.4 A. "
        "1C/2C hit Vmax before delivering the energy target (hatched). "
        "Q_loss = Relative Capacity-Loss Index for one session (not multi-year % fade).",
        ha="center", fontsize=7.5, color="#475569", style="italic",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_degradation_comparison(
    rows: List[Dict[str, Any]],
    info: Dict[str, Any],
    out_path: Path,
) -> None:
    """Detail: Q_total bars + stacked calendar/cyclic."""
    labels = [r["axis_label"] for r in rows]
    q_tot = np.array([float(r.get("qloss_total") or 0.0) for r in rows])
    q_cal = np.array([float(r.get("qloss_calendar") or 0.0) for r in rows])
    q_cyc = np.array([float(r.get("qloss_cyclic") or 0.0) for r in rows])
    feasible = [bool(r.get("feasible")) for r in rows]
    colors = [COLORS.get(r["group"], "#94a3b8") for r in rows]
    durations = [float(r.get("duration_min") or 0.0) for r in rows]
    peak_t = [float(r.get("peak_temperature") or 0.0) for r in rows]

    fig, axes = plt.subplots(
        1, 2, figsize=(11.5, 5.2), dpi=140,
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    ax = axes[0]
    x = np.arange(len(rows))
    bars = ax.bar(x, q_tot, color=colors, edgecolor="k", lw=0.5, width=0.72)
    for i, (bar, ok) in enumerate(zip(bars, feasible)):
        if not ok:
            bar.set_hatch("//")
            bar.set_alpha(0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Session Q_loss index (lower = less degradation)")
    cell = info.get("cell", "?")
    efrac = info.get("energy_fraction")
    mode = info.get("constraint_mode", "?")
    title_extra = (
        f"energy {100 * float(efrac):.0f}%" if efrac is not None and mode == "energy"
        else f"SoC→{info.get('soc_target')}"
    )
    ax.set_title(f"{cell}: degradation per charge session ({title_extra})")
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    ax2.bar(x, q_cyc, color=colors, edgecolor="k", lw=0.4, width=0.72, label="Q_cyclic")
    ax2.bar(
        x, q_cal, bottom=q_cyc, color="#e2e8f0", edgecolor="k", lw=0.4,
        width=0.72, label="Q_calendar",
    )
    for i, ok in enumerate(feasible):
        if not ok:
            ax2.patches[i].set_hatch("//")
            ax2.patches[i].set_alpha(0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Stacked Q_loss index")
    ax2.set_title("Calendar vs cyclic split")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, axis="y", alpha=0.3)

    lines = []
    for r, d, t, ok in zip(rows, durations, peak_t, feasible):
        tag = "OK" if ok else "INF"
        lines.append(
            f"{r['group']}: Q={float(r['qloss_total']):.4g}  "
            f"t={d:.1f} min  peakT={t:.1f}°C  [{tag}]"
        )
    fig.text(
        0.5, -0.02,
        "  |  ".join(lines) + "\n"
        "Q_loss = Relative Capacity-Loss Index for one equal-energy session; "
        "hatched = constraint-infeasible.",
        ha="center", fontsize=7.5, color="#475569",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_cloud(
    rows: List[Dict[str, Any]],
    cloud: List[Dict[str, Any]],
    info: Dict[str, Any],
    out_path: Path,
) -> None:
    """Duration vs Q_loss: all BO/Random trials + CC / winners highlighted."""
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=150)

    for method, color, marker in (
        ("Random", COLORS["Random"], "o"),
        ("GP-BO", COLORS["GP-BO"], "o"),
    ):
        pts = [p for p in cloud if p["method"] == method and p["feasible"]]
        if not pts:
            continue
        ax.scatter(
            [p["duration_min"] for p in pts],
            [p["qloss_total"] for p in pts],
            c=color, s=22, alpha=0.35, marker=marker,
            label=f"{method} trials (feasible)", edgecolors="none",
        )
        inf = [p for p in cloud if p["method"] == method and not p["feasible"]]
        if inf:
            ax.scatter(
                [p["duration_min"] for p in inf],
                [p["qloss_total"] for p in inf],
                c=color, s=18, alpha=0.15, marker="x",
            )

    for r in rows:
        q = float(r.get("qloss_total") or 0.0)
        d = float(r.get("duration_min") or 0.0)
        ok = bool(r.get("feasible"))
        color = COLORS.get(r["group"], "#334155")
        marker = "s" if str(r["group"]).startswith("CC") else "*"
        size = 160 if marker == "*" else 110
        ax.scatter(
            [d], [q], c=color, s=size, marker=marker,
            edgecolors="k", linewidths=0.8, zorder=5,
            label=r["group"] + ("" if ok else " (infeas.)"),
        )
        ax.annotate(
            r["group"],
            xy=(d, q),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="#0f172a",
        )

    ax.set_xlabel("Charge duration [min]")
    ax.set_ylabel("Session Q_loss index (lower better)")
    cell = info.get("cell", "?")
    efrac = info.get("energy_fraction")
    ax.set_title(
        f"{cell}: search cloud + baselines (equal-energy when feasible)\n"
        f"Lower-left is better (faster + less degradation). "
        f"Energy fraction={efrac}"
    )
    ax.grid(True, alpha=0.3)
    # de-duplicate legend
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq = []
    for h, lab in zip(handles, labels):
        if lab in seen:
            continue
        seen.add(lab)
        uniq.append((h, lab))
    ax.legend(*zip(*uniq), fontsize=8, loc="best", framealpha=0.95)
    fig.text(
        0.5, -0.03,
        "Stars = optimizer winners; squares = CC ½C/1C/2C. "
        "Infeasible CC still plotted at their truncated session Q_loss (do not deliver full energy).",
        ha="center", fontsize=7.5, color="#475569", style="italic",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bo",
        type=Path,
        default=ROOT / "Constrained_BO/results/ui_runs_1/RW9/gp_bo_results.json",
    )
    ap.add_argument(
        "--random",
        type=Path,
        default=ROOT / "Constrained_BO/results/ui_runs_1/RW9/random_search_results.json",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Constrained_BO/results/grounded_figures",
    )
    ap.add_argument(
        "--select",
        choices=("reward", "qloss"),
        default="reward",
        help="How to pick GP-BO / Random winners (default: optimizer reward/loss)",
    )
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows, info, cloud = build_comparison_rows(
        args.bo, args.random, device=args.device, optimizer_by=args.select,
    )
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    _save_csv(rows, out / "bo_vs_cc_degradation.csv", info)
    (out / "bo_vs_cc_degradation_meta.json").write_text(json.dumps(info, indent=2, default=str))
    plot_simple_one_axis(rows, info, out / "fig8_bo_vs_cc_degradation.png")
    plot_degradation_comparison(rows, info, out / "fig8b_bo_vs_cc_degradation_detail.png")
    plot_pareto_cloud(rows, cloud, info, out / "fig8c_bo_vs_cc_pareto.png")

    print("Comparison rows:")
    for r in rows:
        print(
            f"  {r['group']:10s}  Q_total={float(r['qloss_total']):.6f}  "
            f"feas={r['feasible']}  t={float(r['duration_min']):.1f} min  "
            f"peakT={float(r['peak_temperature']):.2f}°C  "
            f"end={r.get('end_reason')}"
        )
    print(f"Wrote → {out / 'fig8_bo_vs_cc_degradation.png'}")
    print(f"Wrote → {out / 'fig8c_bo_vs_cc_pareto.png'}")


if __name__ == "__main__":
    main()
