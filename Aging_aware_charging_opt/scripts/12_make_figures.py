#!/usr/bin/env python
"""STAGE 12 — publication figures from the current Random Search / GP-BO JSON.

Reads only ``results/07_random_search`` and ``results/08_gp_bo`` (plus the
frozen degradation coefficients for the equivalent-cycle projection). Does not
re-optimize. Layout matches ``Constrained_BO/results/final_charging_opt_results``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.config import (  # noqa: E402
    OptimizationSpec, Paths, provenance, read_json, stage_dir, write_json,
)
from aacopt.degradation import HybridDegradationModel  # noqa: E402
from aacopt.evaluate import load_degradation_model  # noqa: E402
from aacopt.viz.figures import (  # noqa: E402
    plot_all_cells_gpbo_vs_random_multiaxis, plot_best_profiles,
    plot_consolidated_comparison, plot_family_reward_bars,
    plot_gpbo_vs_random_multiaxis, plot_lifetime_capacity_ah,
    plot_lifetime_delta, plot_lifetime_grid, plot_lifetime_vs_ah,
    plot_lifetime_vs_cycles, plot_metric_bars, plot_paper_table_image,
    plot_pareto_cloud, plot_qloss_bars, plot_qloss_detail,
)
from aacopt.viz.style import FAMILY_LABELS, FAMILY_ORDER  # noqa: E402

BASELINE_RENAME = {
    "CCCV 0.5C": "CCCV 0.5C",
    "CCCV 1C": "CCCV 1C",
    "CCCV 2C": "CCCV 2C",
}


def _pct(num: float, den: float) -> float:
    if den is None or not np.isfinite(den) or abs(den) < 1e-18:
        return float("nan")
    return 100.0 * float(num) / float(den)


def load_families(stage: str, cell: str) -> Dict[str, Dict[str, Any]]:
    d = stage_dir(stage, create=False) / cell
    out = {}
    for fid in FAMILY_ORDER:
        path = d / f"{fid}.json"
        if path.is_file():
            out[fid] = read_json(path)
    if not out:
        raise FileNotFoundError(f"no family JSON under {d}")
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
    baselines_raw = read_json(stage_dir("08_gp_bo", create=False) / cell / "baselines.json")
    baseline_rows = [baseline_row(b) for b in baselines_raw]

    rs_fid, rs_blob = pick_best_family(rs)
    bo_fid, bo_blob = pick_best_family(bo)
    mq_fid, mq_blob = pick_min_q(bo)

    rs_row = winner_row("Random", rs_fid, rs_blob)
    opt_row = winner_row("GP-BO", bo_fid, bo_blob)
    health_row = winner_row("GP-BO", mq_fid, mq_blob)
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
        "policies": policies,
        "curves": curves,
        "family_bo": family_summary(bo),
        "family_rs": family_summary(rs),
        "meta": meta,
        "random_session": rs_blob["best_session"],
        "random_metrics": rs_blob["best_metrics"],
        "random_family": rs_fid,
        "gpbo_session": bo_blob["best_session"],
        "gpbo_metrics": bo_blob["best_metrics"],
        "gpbo_family": bo_fid,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=None)
    args = ap.parse_args()

    paths = Paths.load()
    opt = OptimizationSpec.load()
    cells = [c.upper() for c in (args.cells or paths.cells)]
    model = load_degradation_model()
    n_cycles = int(opt.lifetime.get("n_equivalent_cycles", 600))
    soh_ref = float(opt.lifetime.get("engineering_reference_retention_pct", 80.0))
    q_rated_ah = float(model.p.q_nominal_ah)

    out_root = stage_dir("12_figures")
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
        halfc, rs_row, bo_row = pack["halfc"], pack["rs_row"], pack["bo_row"]
        health = pack["health_row"]
        consolidated[cell] = {
            "CCCV 0.5C": halfc,
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
            f"  GP-BO health ({health['family_label']}): Q={health['q_total']:.3e}  "
            f"t={health['duration_min']:.2f} min",
            flush=True,
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
        "The GP-BO lifetime line is the lowest-Q feasible GP-BO trial (health-first). "
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
                stage_dir("07_random_search", create=False) / c / "summary.json"
                for c in cells
            ] + [
                stage_dir("08_gp_bo", create=False) / c / "summary.json"
                for c in cells
            ],
        ),
    })
    print(f"\nWrote figures → {out_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
