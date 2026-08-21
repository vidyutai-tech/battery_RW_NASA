"""Improved GP-BO final export → ``Constrained_BO/results/final_charging_opt_results``.

Pipeline per cell
-----------------
1. Random search with degradation-aligned weights (``w_time=0``, elevated ``w_qloss``).
2. Set soft ``qloss_cap`` = best feasible Random session Q.
3. GP-BO with the same weights + ``qloss_cap`` (must match/beat Random on Q)
   and a tiny duration tie-break so it still prefers faster profiles among equals.
4. Write profile grids (families kept distinct via ``from_dict`` step constraints),
   fig8*/fig9*, comparison tables.

Usage
-----
    python -m Constrained_BO.export_final_charging_opt_results --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Constrained_BO.bo_degradation_comparison import (
    PAPER_C_RATES,
    _pick_best_family,
    _save_csv,
    build_comparison_rows,
    plot_degradation_comparison,
    plot_pareto_cloud,
    plot_simple_one_axis,
)
from Constrained_BO.config import energy_fraction_for
from Constrained_BO.lifetime_fade_projection import (
    POLICY_STYLE,
    _collect_policies,
    _load_measured_cell,
    plot_delta_vs_halfc,
    plot_lifetime_vs_ah,
    plot_lifetime_vs_ref_style,
    project_fade,
    save_projection_csv,
)
from Constrained_BO.optimize_api import (
    build_baseline_comparison_rows,
    build_cell,
    comparison_dataframe,
    dataframe_to_excel_bytes,
    paper_cccv_currents_a,
    plot_baseline_bar_png,
    results_to_dataframe,
    run_optimization,
)
from Constrained_BO.simulator import ChargingSimulator
from Constrained_BO.viz import (
    PAPER_DPI,
    PAPER_LIGHT_BG,
    apply_paper_style,
    plot_best_profiles,
    rebuild_family_results_from_json,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "Constrained_BO" / "results" / "final_charging_opt_results"
UI_ROOT = ROOT / "Constrained_BO" / "results" / "ui_runs"
ANCHOR_N = 400

# Keep paper-like speed while soft-constraining Q ≤ Random-best.
IMPROVED_W_SOC = 1.0
IMPROVED_W_QLOSS = 2.0          # mild boost vs paper default 1.0
IMPROVED_W_TIME = 0.1           # keep time pressure (do NOT set 0)
IMPROVED_Z = 0.55
IMPROVED_DURATION_LOSS_WEIGHT = 1e-3
IMPROVED_QLOSS_CAP_SCALE = 120.0  # strong soft penalty if Q > Random


def _pct(num: float, den: float) -> Optional[float]:
    if den is None or not np.isfinite(den) or abs(den) < 1e-12:
        return None
    if num is None or not np.isfinite(num):
        return None
    return 100.0 * float(num) / float(den)


def _best_feasible_qloss(payload: Dict[str, Any]) -> float:
    """Session Q of Random's reward-best feasible trial (competitive baseline).

    Using absolute min-Q across all Random trials can lock GP-BO onto a very
    slow gentle profile; reward-best matches the paper comparison baseline.
    """
    _, _, m, _ = _pick_best_family(payload, by="reward")
    return float(m["qloss_total"])


def _try_plot_profiles(results, cell, out_path: Path, title_suffix: str) -> None:
    try:
        plot_best_profiles(
            results,
            cell_id=cell.cell_id,
            soc_target=float(cell.soc_target),
            soc_start=float(cell.start_state.get("soc", 0.2)),
            out_path=out_path,
            title_suffix=title_suffix,
        )
    except Exception as exc:  # pragma: no cover
        print(f"  (skip profile plot {out_path.name}: {exc})")


def _assert_family_structure(results: Dict[str, Any]) -> None:
    """Soft checks so multi-step / pulsed winners keep their visual identity."""
    for fid, entry in results.items():
        vals = (entry.get("best_params") or {}).get("values") or entry.get("best_params") or {}
        if not isinstance(vals, dict):
            continue
        if fid == "two_step":
            i1 = float(vals.get("i1", 0.0))
            i2 = float(vals.get("i2", 0.0))
            if abs(i1 - i2) < 0.2:
                print(f"  WARN [{fid}]: i1≈i2 ({i1:.3f},{i2:.3f}) — may look flat")
        elif fid == "three_step":
            i1 = float(vals.get("i1", 0.0))
            i2 = float(vals.get("i2", 0.0))
            i3 = float(vals.get("i3", 0.0))
            if abs(i1 - i2) < 0.2 and abs(i2 - i3) < 0.2:
                print(f"  WARN [{fid}]: currents nearly flat — may look like CCCV")
        elif fid == "pulsed":
            ich = float(vals.get("i_charge", vals.get("i_on", 0.0)) or 0.0)
            if ich < 0.5:
                print(f"  WARN [{fid}]: low pulse current {ich:.3f} A")


def run_improved_pair(
    cell_id: str,
    out_dir: Path,
    *,
    n_calls: int,
    n_initial: int,
    n_random: int,
    device: str,
    seed: int,
    energy_fraction: Optional[float],
    resume: bool,
) -> Tuple[Path, Path]:
    cell = build_cell(cell_id, energy_fraction=energy_fraction, soc_mode=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    rnd_path = out_dir / "random_search_results.json"
    bo_path = out_dir / "gp_bo_results.json"

    print(
        f"\n=== {cell_id}  improved GP-BO  "
        f"w_qloss={IMPROVED_W_QLOSS} w_time={IMPROVED_W_TIME}  "
        f"energy={cell.energy_fraction}"
    )

    if resume and rnd_path.is_file() and bo_path.is_file():
        meta = json.loads(bo_path.read_text()).get("meta") or {}
        rw = meta.get("reward_weights") or {}
        if (
            abs(float(rw.get("w_qloss", -1)) - IMPROVED_W_QLOSS) < 1e-9
            and abs(float(rw.get("w_time", -1)) - IMPROVED_W_TIME) < 1e-9
            and meta.get("qloss_cap") is not None
        ):
            print("  Resume: improved JSONs already present → skip optimization")
            return bo_path, rnd_path
        print("  Existing JSONs are not improved settings → re-running")

    sim = ChargingSimulator.from_cell(cell, device=device)
    common_kw = dict(
        reward_mode="hybrid_qloss",
        w_soc=IMPROVED_W_SOC,
        w_qloss=IMPROVED_W_QLOSS,
        w_time=IMPROVED_W_TIME,
        z=IMPROVED_Z,
        duration_loss_weight=IMPROVED_DURATION_LOSS_WEIGHT,
        simulator=sim,
    )

    if resume and rnd_path.is_file():
        rnd_payload = json.loads(rnd_path.read_text())
        rw = (rnd_payload.get("meta") or {}).get("reward_weights") or {}
        if (
            abs(float(rw.get("w_qloss", -1)) - IMPROVED_W_QLOSS) < 1e-9
            and abs(float(rw.get("w_time", -1)) - IMPROVED_W_TIME) < 1e-9
        ):
            print(f"  Resume: reusing {rnd_path.name}")
        else:
            rnd_path.unlink(missing_ok=True)

    if not rnd_path.is_file():
        t0 = time.time()
        print(f"  Random (n={n_random}/family, w_time={IMPROVED_W_TIME}, w_qloss={IMPROVED_W_QLOSS})…")
        rnd_payload, rnd_results, sim = run_optimization(
            cell,
            method="random_search",
            device=device,
            seed=seed,
            n_random=n_random,
            **common_kw,
        )
        rnd_path.write_text(json.dumps(rnd_payload, indent=2, default=str))
        _assert_family_structure(rnd_results)
        _try_plot_profiles(
            rnd_results, cell, out_dir / "random_best_profiles.png",
            f"random search, hybrid_qloss, w_qloss={IMPROVED_W_QLOSS}, w_time=0",
        )
        print(f"  Random done in {time.time() - t0:.1f}s")
    else:
        rnd_payload = json.loads(rnd_path.read_text())
        # Rebind simulator from cell (common_kw already has it).
        pass

    q_cap = _best_feasible_qloss(rnd_payload)
    print(f"  qloss_cap (Random best Q) = {q_cap:.6g}")

    elite = {
        fid: (entry.get("history") or [])
        for fid, entry in (rnd_payload.get("families") or {}).items()
    }

    t1 = time.time()
    print(
        f"  GP-BO (n_calls={n_calls}, qloss_cap={q_cap:.5f}, "
        f"w_qloss={IMPROVED_W_QLOSS})…"
    )
    bo_payload, bo_results, _ = run_optimization(
        cell,
        method="gp_bo",
        device=device,
        seed=seed + 1,
        n_calls=n_calls,
        n_initial=n_initial,
        elite_histories=elite,
        elite_top_k=5,
        qloss_cap=q_cap,
        qloss_cap_scale=IMPROVED_QLOSS_CAP_SCALE,
        **common_kw,
    )
    # Stamp improvement meta for resume detection / paper notes.
    bo_payload["meta"]["improvement"] = {
        "mode": "qloss_cap_vs_random",
        "w_qloss": IMPROVED_W_QLOSS,
        "w_time": IMPROVED_W_TIME,
        "qloss_cap": q_cap,
        "duration_loss_weight": IMPROVED_DURATION_LOSS_WEIGHT,
    }
    bo_path.write_text(json.dumps(bo_payload, indent=2, default=str))
    _assert_family_structure(bo_results)
    _try_plot_profiles(
        bo_results, cell, out_dir / "gp_bo_best_profiles.png",
        f"GP-BO, hybrid_qloss, w_qloss={IMPROVED_W_QLOSS}, "
        f"qloss_cap={q_cap:.5f}",
    )
    print(f"  GP-BO done in {time.time() - t1:.1f}s")

    # Quick head-to-head on session Q / time
    _, _, rnd_m, _ = _pick_best_family(rnd_payload, by="reward")
    _, _, bo_m, _ = _pick_best_family(bo_payload, by="reward")
    print(
        f"  Reward-best  Random: t={float(rnd_m['duration_min']):.2f} min  "
        f"Q={float(rnd_m['qloss_total']):.5f}  R={float(rnd_m['total_reward']):.4f}"
    )
    print(
        f"  Reward-best  GP-BO:   t={float(bo_m['duration_min']):.2f} min  "
        f"Q={float(bo_m['qloss_total']):.5f}  R={float(bo_m['total_reward']):.4f}"
    )
    return bo_path, rnd_path


def _plot_lifetime_fair(
    policies: List[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    info: Dict[str, Any],
    *,
    scale: float,
    out_path: Path,
    anchor_n: int = ANCHOR_N,
) -> None:
    """Lifetime plot: solid = equal-energy feasible; dotted = infeasible ghosts."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.2, 5.0), facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    feas_pols = [p for p in policies if p.get("feasible")]
    ghosts = [p for p in policies if not p.get("feasible")]

    for p in feas_pols:
        name = p["name"]
        c = curves.get(name)
        if c is None:
            continue
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        ax.plot(
            c["cycles"], c["remaining_pct"],
            color=style["color"], ls=style.get("ls", "-"),
            lw=style.get("lw", 2.2), label=name,
        )
    for p in ghosts:
        name = p["name"]
        c = curves.get(name)
        if c is None:
            continue
        ax.plot(
            c["cycles"], c["remaining_pct"],
            color="#94a3b8", ls=":", lw=1.6, alpha=0.7,
            label=f"{name} (infeasible)",
        )

    ax.axhline(80.0, color="#475569", ls="--", lw=1.1, alpha=0.7)
    ax.axvline(anchor_n, color="#475569", ls=":", lw=1.1, alpha=0.5)
    ax.set_xlabel("Equivalent charge cycles (one equal-energy session each)", fontsize=14)
    ax.set_ylabel("Projected remaining capacity [%]", fontsize=14)
    cell = info.get("cell", "")
    ax.set_title(
        f"{cell}: capacity fade — equal-energy lifetime ranking\n"
        f"(hybrid calendar+cyclic; solid = feasible)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, max((c["cycles"][-1] for c in curves.values()), default=600))
    ax.set_ylim(60, 102)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight", facecolor=PAPER_LIGHT_BG)
    plt.close(fig)


def _pick_best_cccv_baseline(by_pol: Dict[str, Dict[str, Any]]) -> str:
    """Percent-column baseline: prefer CCCV ½C, else CCCV 1C (paper narrative)."""
    for name in ("CCCV ½C", "CCCV 1C"):
        p = by_pol.get(name)
        if p and p.get("feasible"):
            return name
    if by_pol.get("CC ½C", {}).get("feasible"):
        return "CC ½C"
    if by_pol.get("CC 1C", {}).get("feasible"):
        return "CC 1C"
    return "Random"


def _comparison_table(
    session_rows: List[Dict[str, Any]],
    policies: List[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    info: Dict[str, Any],
    *,
    anchor_n: int = ANCHOR_N,
) -> List[Dict[str, Any]]:
    """Build the all_cells-style comparison rows from lifetime curves + session metrics."""
    by_pol = {p["name"]: p for p in policies}
    # Also merge feasibility/Q/time from session_rows for CCCV baselines.
    for r in session_rows:
        g = str(r.get("group") or "")
        if not g:
            continue
        if g not in by_pol:
            by_pol[g] = {
                "name": g,
                "feasible": bool(r.get("feasible")),
                "qloss_total": float(r.get("qloss_total") or np.nan),
                "duration_min": float(r.get("duration_min") or 0.0),
            }
        else:
            by_pol[g]["feasible"] = bool(r.get("feasible", by_pol[g].get("feasible")))
            if r.get("qloss_total") is not None:
                by_pol[g]["qloss_total"] = float(r["qloss_total"])
            if r.get("duration_min") is not None:
                by_pol[g]["duration_min"] = float(r["duration_min"])

    base_name = _pick_best_cccv_baseline(by_pol)
    if base_name not in by_pol and "Random" in by_pol:
        base_name = "Random"
    base_q = float(by_pol[base_name]["qloss_total"]) if base_name in by_pol else None
    base_t = float(by_pol[base_name]["duration_min"]) if base_name in by_pol else None
    base_curve = curves.get(base_name)
    if base_curve is not None:
        idx = int(np.argmin(np.abs(base_curve["cycles"] - anchor_n)))
        base_ret = float(base_curve["remaining_pct"][idx])
    else:
        base_ret = 80.0

    sess_by = {r.get("group") or r.get("method") or r.get("label"): r for r in session_rows}
    # Normalize keys used by build_comparison_rows
    for r in session_rows:
        name = r.get("group") or r.get("label") or r.get("method")
        if name:
            sess_by[str(name).replace("\n", " ").strip()] = r

    order = ["CCCV ½C", "CCCV 1C", "Random", "GP-BO", "GP-BO (min Q)"]
    # Map session row labels → canonical names
    alias = {}
    for r in session_rows:
        g = str(r.get("group") or "")
        lab = str(r.get("label") or "")
        text = g + " " + lab
        if "CCCV" in text:
            if "½C" in text or "0.5" in text:
                alias[g] = "CCCV ½C"
            elif "1C" in text or "1.0" in text:
                alias[g] = "CCCV 1C"
            elif "2C" in text or "2.0" in text:
                alias[g] = "CCCV 2C"
        elif "½C" in g or "0.5C" in g or "½C" in lab:
            alias[g] = "CC ½C"
        elif g.startswith("CC") and ("1C" in g or "1.0" in g):
            alias[g] = "CC 1C"
        elif g.startswith("CC") and ("2C" in g or "2.0" in g):
            alias[g] = "CC 2C"
        elif g.startswith("Random"):
            alias[g] = "Random"
        elif "min Q" in g or "min Q" in lab:
            alias[g] = "GP-BO (min Q)"
        elif g.startswith("GP-BO"):
            alias[g] = "GP-BO"

    rows_out: List[Dict[str, Any]] = []
    # Prefer policies for fade; session_rows for energy/peak when present.
    names = []
    for n in order:
        if n in by_pol or n == "GP-BO (min Q)":
            names.append(n)
    # Ensure CC from session even if filtered from policies
    for r in session_rows:
        canon = alias.get(str(r.get("group") or ""), None)
        if canon and canon not in names:
            names.append(canon)

    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        p = by_pol.get(name)
        # Find matching session row
        s = None
        for r in session_rows:
            g = str(r.get("group") or "")
            if alias.get(g) == name or g == name:
                s = r
                break
        feas = bool((p or s or {}).get("feasible", True))
        if name == "GP-BO (min Q)" and p is None:
            continue
        dur = float((p or s or {}).get("duration_min") or 0.0)
        q = float((p or s or {}).get("qloss_total") or np.nan)
        peak = float((s or p or {}).get("peak_temperature") or 0.0)
        sm = (s or {}).get("metrics") or {}
        e_del = float(sm.get("energy_delivered_j") or (s or {}).get("energy_delivered_j") or 0.0)
        e_full = float(sm.get("energy_full_j") or (s or {}).get("energy_full_j") or 0.0)
        efrac = float(sm.get("energy_fraction") or info.get("energy_fraction") or 0.0)
        if e_full <= 0:
            e_req = float(sm.get("energy_required_j") or 0.0)
            if e_req > 0 and efrac > 0:
                e_full = e_req / efrac
        energy_pct = _pct(e_del, e_full) if e_full > 0 else None
        if energy_pct is None and feas and efrac > 0:
            energy_pct = 100.0 * efrac

        curve = curves.get(name)
        if curve is not None and feas:
            idx = int(np.argmin(np.abs(curve["cycles"] - anchor_n)))
            ret = float(curve["remaining_pct"][idx])
            fade = 100.0 - ret
            life_imp = ret - base_ret
        else:
            ret = fade = life_imp = None

        if feas and base_q and np.isfinite(q):
            deg_imp = _pct(base_q - q, base_q)
        else:
            deg_imp = None
        if feas and base_t and base_t > 0:
            time_saved = _pct(base_t - dur, base_t)
        else:
            time_saved = None

        status = "feasible" if feas else "infeasible (energy target not met)"
        rows_out.append({
            "Method": name,
            "Charging Time (min)": round(dur, 2) if np.isfinite(dur) else None,
            "Energy Delivered (%)": None if energy_pct is None else round(energy_pct, 2),
            "Capacity Fade (%)": None if fade is None else round(fade, 2),
            "Capacity Retention (%)": None if ret is None else round(ret, 2),
            "Lifetime Improvement (%)": None if life_imp is None else round(life_imp, 2),
            "Degradation Improvement (%)": None if deg_imp is None else round(deg_imp, 2),
            "Time Saved (%)": None if time_saved is None else round(time_saved, 2),
            "Peak Temperature": round(peak, 2) if peak else None,
            "Feasible": feas,
            "Feasibility Status": status,
            "Baseline for %": base_name,
        })
    return rows_out


def _write_table_csv(rows: List[Dict[str, Any]], path: Path, cell_id: str) -> None:
    if not rows:
        return
    fieldnames = ["cell"] + list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({"cell": cell_id, **r})


def _export_ui(
    cell,
    out_dir: Path,
    bo_payload: Dict[str, Any],
    rnd_payload: Dict[str, Any],
    bo_results: Dict[str, Any],
    rnd_results: Dict[str, Any],
    device: str,
) -> None:
    """Write UI-style artifacts into ``out_dir`` and mirror under ``ui_runs``."""
    ui_dir = UI_ROOT / cell.cell_id
    ui_dir.mkdir(parents=True, exist_ok=True)
    for src, dst in [
        (out_dir / "gp_bo_results.json", ui_dir / "gp_bo_results.json"),
        (out_dir / "random_search_results.json", ui_dir / "random_search_results.json"),
        (out_dir / "gp_bo_best_profiles.png", ui_dir / "gp_bo_best_profiles.png"),
        (out_dir / "random_best_profiles.png", ui_dir / "random_best_profiles.png"),
    ]:
        if src.is_file():
            shutil.copy2(src, dst)

    try:
        cdf = comparison_dataframe(bo_payload, rnd_payload)
        for dest in (out_dir, ui_dir):
            cdf.to_csv(dest / "bo_vs_random_comparison.csv", index=False)
            (dest / "bo_vs_random_comparison.xlsx").write_bytes(
                dataframe_to_excel_bytes(cdf)
            )
    except Exception as exc:
        print(f"  (comparison xlsx skipped: {exc})")

    try:
        sim = ChargingSimulator.from_cell(cell, device=device)
        plot_currents = paper_cccv_currents_a()
        rows = build_baseline_comparison_rows(
            cell,
            sim,
            bo_payload=bo_payload,
            random_payload=rnd_payload,
            currents_a=plot_currents,
        )
        for dest in (out_dir, ui_dir):
            pd.DataFrame(rows).to_csv(dest / "baseline_comparison.csv", index=False)
        bar_specs = (
            ("total_reward", "Total reward", "Reward comparison", "reward_comparison.png"),
            ("duration_min", "Duration (min)", "Time comparison", "time_comparison.png"),
            ("peak_temperature", "Peak T (°C)", "Temperature comparison", "temperature_comparison.png"),
            ("total_reward", "Total reward", f"{cell.cell_id}: baseline comparison", "baseline_comparison.png"),
        )
        for key, ylabel, title, fname in bar_specs:
            png = plot_baseline_bar_png(
                rows,
                value_key=key,
                ylabel=ylabel,
                title=title,
                plot_currents_a=plot_currents,
            )
            for dest in (out_dir, ui_dir):
                (dest / fname).write_bytes(png)
        print(f"  Wrote bar charts → {out_dir.name}/ + ui_runs/{cell.cell_id}/")
    except Exception as exc:
        print(f"  (baseline bars skipped: {exc})")


def export_cell(
    cell_id: str,
    *,
    device: str,
    n_calls: int,
    n_initial: int,
    n_random: int,
    seed: int,
    resume: bool,
    figures_only: bool,
    energy_fraction: Optional[float] = None,
) -> List[Dict[str, Any]]:
    cell_id = cell_id.upper()
    out_dir = OUT_ROOT / cell_id
    out_dir.mkdir(parents=True, exist_ok=True)
    efrac = float(energy_fraction) if energy_fraction is not None else energy_fraction_for(cell_id)

    if figures_only:
        bo_path = out_dir / "gp_bo_results.json"
        rnd_path = out_dir / "random_search_results.json"
        if not bo_path.is_file() or not rnd_path.is_file():
            raise FileNotFoundError(f"Missing improved JSONs under {out_dir}")
    else:
        bo_path, rnd_path = run_improved_pair(
            cell_id, out_dir,
            n_calls=n_calls, n_initial=n_initial, n_random=n_random,
            device=device, seed=seed, energy_fraction=efrac, resume=resume,
        )

    print(f"=== {cell_id}  sources:\n  {bo_path}\n  {rnd_path}")

    rows, info, cloud = build_comparison_rows(
        bo_path, rnd_path, device=device, c_rates=PAPER_C_RATES,
    )
    # Also evaluate min-Q GP-BO row for the table
    from Constrained_BO.bo_degradation_comparison import (
        _cell_from_meta,
        _eval_optimizer_best,
        _load_json,
        _reward_kwargs,
    )
    bo = _load_json(bo_path)
    rnd = _load_json(rnd_path)
    meta = bo.get("meta") or {}
    cell = _cell_from_meta(meta)
    rw = _reward_kwargs(meta)
    sim = ChargingSimulator.from_cell(cell, device=device)
    try:
        minq_row = _eval_optimizer_best(
            cell, sim, bo, method_label="GP-BO", reward_kwargs=rw, by="qloss",
        )
        minq_row["group"] = "GP-BO (min Q)"
        minq_row["axis_label"] = "GP-BO\n(min Q)"
        minq_row["label"] = minq_row["axis_label"]
        # Avoid duplicate if already present
        if not any("min Q" in str(r.get("group", "")) for r in rows):
            rows = list(rows) + [minq_row]
    except Exception as exc:
        print(f"  (min-Q row skipped: {exc})")

    COLORS = {
        "CCCV ½C": "#64748b",
        "CCCV 1C": "#2563eb",
        "Random": "#f59e0b",
        "GP-BO": "#16a34a",
        "GP-BO (min Q)": "#86efac",
    }
    # Monkey-patch plot colors for min-Q bar if needed — plots use module COLORS.
    import Constrained_BO.bo_degradation_comparison as _bdc
    _bdc.COLORS.setdefault("GP-BO (min Q)", "#86efac")

    _save_csv(rows, out_dir / "bo_vs_cc_degradation.csv", info)
    (out_dir / "bo_vs_cc_degradation_meta.json").write_text(
        json.dumps(info, indent=2, default=str),
    )
    plot_simple_one_axis(rows, info, out_dir / "fig8_bo_vs_cc_degradation.png")
    plot_degradation_comparison(rows, info, out_dir / "fig8b_bo_vs_cc_degradation_detail.png")
    plot_pareto_cloud(rows, cloud, info, out_dir / "fig8c_bo_vs_cc_pareto.png")

    # Lifetime: reward-best (primary) + min-Q (for table row fade only).
    policies, life_info = _collect_policies(
        bo_path, rnd_path, device=device, include_infeasible_cc=True,
        gpbo_select="reward", c_rates=PAPER_C_RATES,
    )
    _, curves, scale = project_fade(policies)
    policies_mq, _ = _collect_policies(
        bo_path, rnd_path, device=device, include_infeasible_cc=True,
        gpbo_select="min_q", c_rates=PAPER_C_RATES,
    )
    _, curves_mq, _ = project_fade(policies_mq)
    # Expose min-Q GP-BO under a distinct curve key for the comparison table.
    if "GP-BO" in curves_mq:
        curves = dict(curves)
        curves["GP-BO (min Q)"] = curves_mq["GP-BO"]
        mq_pol = next((p for p in policies_mq if p["name"] == "GP-BO"), None)
        if mq_pol is not None:
            policies = list(policies) + [{**mq_pol, "name": "GP-BO (min Q)"}]
    measured = _load_measured_cell(cell_id)
    _plot_lifetime_fair(
        policies, curves, life_info, scale=scale,
        out_path=out_dir / "fig9_lifetime_fade_vs_cycles.png",
        anchor_n=ANCHOR_N,
    )
    plot_delta_vs_halfc(
        policies, curves, life_info,
        out_path=out_dir / "fig9d_lifetime_delta_vs_halfC.png",
    )
    plot_lifetime_vs_ah(
        policies, curves, life_info, measured,
        out_path=out_dir / "fig9b_lifetime_fade_vs_throughput.png",
    )
    plot_lifetime_vs_ref_style(
        policies, curves, life_info, measured,
        out_path=out_dir / "fig9c_lifetime_capacity_vs_cycle_index.png",
    )
    save_projection_csv(policies, curves, out_dir / "lifetime_fade_projection.csv")
    (out_dir / "lifetime_fade_projection_meta.json").write_text(
        json.dumps({**life_info, "scale": scale, "gpbo_select": "reward"}, indent=2, default=str),
    )

    table = _comparison_table(rows, policies, curves, {**info, **life_info}, anchor_n=ANCHOR_N)
    _write_table_csv(table, out_dir / "comparison_table.csv", cell_id)
    try:
        pd.DataFrame([{"cell": cell_id, **r} for r in table]).to_excel(
            out_dir / "comparison_table.xlsx", index=False,
        )
    except Exception as exc:
        print(f"  (comparison xlsx skipped: {exc})")

    # Always regenerate GP-BO / Random best-profile grids (paper fonts/theme).
    for name in ("gp_bo_best_profiles.png", "random_best_profiles.png"):
        try:
            payload = bo if name.startswith("gp_bo") else rnd
            results = rebuild_family_results_from_json(payload, device=device)
            _assert_family_structure(results)
            suffix = (
                f"GP-BO  ·  hybrid_qloss  ·  w_qloss={IMPROVED_W_QLOSS}  ·  qloss_cap"
                if name.startswith("gp_bo")
                else (
                    f"random search  ·  hybrid_qloss  ·  "
                    f"w_qloss={IMPROVED_W_QLOSS}, w_time={IMPROVED_W_TIME}"
                )
            )
            plot_best_profiles(
                results,
                cell_id=cell.cell_id,
                soc_target=float(cell.soc_target),
                soc_start=float(cell.start_state.get("soc", 0.2)),
                out_path=out_dir / name,
                title_suffix=suffix,
            )
            print(f"  Wrote {name}")
        except Exception as exc:
            print(f"  (profile plot {name}: {exc})")

    _export_ui(cell, out_dir, bo, rnd, {}, {}, device)
    print(f"  Wrote → {out_dir}")
    return [{"cell": cell_id, **r} for r in table]


def _plot_summary_table(all_rows: List[Dict[str, Any]], out_path: Path) -> None:
    # Compact GP-BO vs baselines summary for fig10
    cells = sorted({r["cell"] for r in all_rows})
    lines = [
        ["Cell", "Energy", "GP-BO t", "Time↓ vs base", "Deg↓ vs base", "Time↓ vs Rnd", "Deg↓ vs Rnd", "Baseline"],
    ]
    for cell in cells:
        subset = [r for r in all_rows if r["cell"] == cell]
        gp = next((r for r in subset if r["Method"] == "GP-BO"), None)
        rnd = next((r for r in subset if r["Method"] == "Random"), None)
        if gp is None:
            continue
        base = gp.get("Baseline for %") or "CCCV ½C"
        base_row = next((r for r in subset if r["Method"] == base), None)
        e = gp.get("Energy Delivered (%)")
        t_vs_base = gp.get("Time Saved (%)")
        d_vs_base = gp.get("Degradation Improvement (%)")
        t_vs_rnd = None
        d_vs_rnd = None
        if rnd and gp.get("Charging Time (min)") and rnd.get("Charging Time (min)"):
            t_vs_rnd = _pct(
                float(rnd["Charging Time (min)"]) - float(gp["Charging Time (min)"]),
                float(rnd["Charging Time (min)"]),
            )
        if rnd and gp.get("Capacity Fade (%)") is not None and rnd.get("Capacity Fade (%)") is not None:
            # approx via fade
            d_vs_rnd = _pct(
                float(rnd["Capacity Fade (%)"]) - float(gp["Capacity Fade (%)"]),
                float(rnd["Capacity Fade (%)"]),
            )
        def fmt(v):
            if v is None:
                return "—"
            return f"{v:.1f}%" if isinstance(v, float) else str(v)
        lines.append([
            cell,
            f"{e:.0f}%" if isinstance(e, (int, float)) else "—",
            f"{gp.get('Charging Time (min)', '—')}",
            fmt(t_vs_base),
            fmt(d_vs_base),
            fmt(t_vs_rnd),
            fmt(d_vs_rnd),
            base if base_row and base_row.get("Feasible") else "none (all CC infeasible)" if base == "Random" else base,
        ])

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(12.5, 1.5 + 0.55 * len(lines)), facecolor=PAPER_LIGHT_BG)
    ax.axis("off")
    table = ax.table(cellText=lines[1:], colLabels=lines[0], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.55)
    ax.set_title(
        "GP-BO vs best CCCV / Random — energy target, time saved, degradation improvement",
        fontsize=14, pad=14, fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight", facecolor=PAPER_LIGHT_BG)
    plt.close(fig)


def _write_paper_md(all_rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Slim Table-4 style markdown: CCCV ½C | CCCV 1C | Random | GP-BO."""
    cells = sorted({r["cell"] for r in all_rows})
    lines = [
        "# GP-BO comparison table",
        "",
        "Baselines are classic **CCCV (CC→CV at Vmax)** at **½C and 1C**.",
        "Percent columns use **CCCV ½C** as the reference (falls back to 1C if ½C is infeasible).",
        "",
        "| Cell | Energy | GP-BO time (min) | Time ↓ vs CCCV ½C | Deg. ↓ vs CCCV ½C | Time ↓ vs Random | Deg. ↓ vs Random | Baseline |",
        "|------|--------|------------------|-------------------|-------------------|------------------|------------------|----------|",
    ]
    for cell in cells:
        subset = [r for r in all_rows if r["cell"] == cell]
        gp = next((r for r in subset if r["Method"] == "GP-BO"), None)
        rnd = next((r for r in subset if r["Method"] == "Random"), None)
        if gp is None:
            continue
        base = gp.get("Baseline for %") or "CCCV ½C"
        e = gp.get("Energy Delivered (%)")
        t_vs_base = gp.get("Time Saved (%)")
        d_vs_base = gp.get("Degradation Improvement (%)")
        t_vs_rnd = d_vs_rnd = None
        if rnd and gp.get("Charging Time (min)") and rnd.get("Charging Time (min)"):
            t_vs_rnd = _pct(
                float(rnd["Charging Time (min)"]) - float(gp["Charging Time (min)"]),
                float(rnd["Charging Time (min)"]),
            )
        if (
            rnd
            and gp.get("Capacity Fade (%)") is not None
            and rnd.get("Capacity Fade (%)") is not None
        ):
            d_vs_rnd = _pct(
                float(rnd["Capacity Fade (%)"]) - float(gp["Capacity Fade (%)"]),
                float(rnd["Capacity Fade (%)"]),
            )

        def fmt(v):
            if v is None:
                return "—"
            return f"{v:.1f}%" if isinstance(v, float) else str(v)

        lines.append(
            "| {cell} | {e} | {t} | {tb} | {db} | {tr} | {dr} | {base} |".format(
                cell=cell,
                e=f"{e:.0f}%" if isinstance(e, (int, float)) else "—",
                t=gp.get("Charging Time (min)", "—"),
                tb=fmt(t_vs_base),
                db=fmt(d_vs_base),
                tr=fmt(t_vs_rnd),
                dr=fmt(d_vs_rnd),
                base=base,
            )
        )
    lines.extend(
        [
            "",
            "**Reading guide**",
            "",
            "- **Energy**: delivered fraction of full pack (same 40% target on all cells).",
            "- **Time ↓**: \\((t_{\\mathrm{base}}-t_{\\mathrm{GPBO}})/t_{\\mathrm{base}}\\). Positive = faster than baseline.",
            "- **Deg. ↓**: \\((Q_{\\mathrm{base}}-Q_{\\mathrm{GPBO}})/Q_{\\mathrm{base}}\\). Positive = *less* session degradation than baseline.",
            "- Hitting 4.2 V enters the CV phase; it does not mark the baseline infeasible.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=["RW9", "RW10", "RW11", "RW12"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-calls", type=int, default=80)
    ap.add_argument("--n-initial", type=int, default=15)
    ap.add_argument("--n-random", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument(
        "--energy-fraction",
        type=float,
        default=0.40,
        help="Same energy target for all cells (default 0.40).",
    )
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_tables: List[Dict[str, Any]] = []
    for cell_id in args.cells:
        table = export_cell(
            cell_id,
            device=args.device,
            n_calls=args.n_calls,
            n_initial=args.n_initial,
            n_random=args.n_random,
            seed=args.seed,
            resume=not args.no_resume,
            figures_only=args.figures_only,
            energy_fraction=args.energy_fraction,
        )
        all_tables.extend(table)

    # Aggregate
    agg_path = OUT_ROOT / "all_cells_comparison_table.csv"
    if all_tables:
        fieldnames = list(all_tables[0].keys())
        with agg_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_tables)
        _plot_summary_table(all_tables, OUT_ROOT / "fig10_paper_comparison_table.png")
        # paper md/csv slim view
        pd.DataFrame(all_tables).to_csv(OUT_ROOT / "paper_comparison_table.csv", index=False)
        _write_paper_md(all_tables, OUT_ROOT / "paper_comparison_table.md")

    (OUT_ROOT / "README.txt").write_text(
        "Final charging optimization results (paper + UI)\n"
        "================================================\n"
        "Per-cell folders RW9–RW12 contain fig8*/fig9*, comparison_table.*,\n"
        "GP-BO/random JSON+PNG, UI-style bar charts (time/temp/reward), and lifetime CSVs.\n"
        "\n"
        "* Same energy target for all cells (default ``--energy-fraction 0.40``).\n"
        f"* w_qloss={IMPROVED_W_QLOSS}, w_time={IMPROVED_W_TIME}.\n"
        "* Soft qloss_cap = Random reward-best Q (GP-BO must match/beat Random on Q).\n"
        "* Baselines = classic CCCV (CC→CV at Vmax) at **½C and 1C** (1.1 A / 2.2 A).\n"
        "* Paper % columns use **CCCV ½C** as the reference baseline.\n"
        "* Hitting 4.2 V enters CV (not infeasible); energy target still applies.\n"
        "* GP-BO / Random best-profile PNGs are regenerated from saved JSON.\n"
        "* Also writes time_comparison.png, temperature_comparison.png, reward_comparison.png.\n"
        "\n"
        "UI mirrors: Constrained_BO/results/ui_runs/{cell}/\n"
        "Regenerate figures (reuse existing JSON):\n"
        "  python -m Constrained_BO.export_final_charging_opt_results "
        "--device cpu --figures-only --energy-fraction 0.40\n"
        "Full re-optimize:\n"
        "  python -m Constrained_BO.export_final_charging_opt_results "
        "--device cuda --energy-fraction 0.40\n",
        encoding="utf-8",
    )
    print(f"\nDone → {OUT_ROOT}")


if __name__ == "__main__":
    main()
