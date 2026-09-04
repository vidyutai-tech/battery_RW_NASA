"""Shared driver for Random Search and GP-BO (identical space / budget / objective)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from aacopt.config import (
    OptimizationSpec, Paths, RewardSpec, file_hash, provenance, stage_dir, write_json,
)
from aacopt.evaluate import (
    cccv_params, evaluate_params, jsonable_family_result, load_anchors,
    load_degradation_model, make_simulator,
)
from aacopt.optimizers import optimize_family_gp_bo, optimize_family_random
from aacopt.profiles import bind_search


def evaluate_baselines(simulator, initial_state, *, model, spec, anchors) -> List[Dict[str, Any]]:
    opt = OptimizationSpec.load()
    rows = []
    for b in opt.baselines:
        params = cccv_params(float(b["c_rate"]), v_cv=float(b["v_cv"]), i_cutoff=float(b["i_cutoff"]))
        loss, metrics, session = evaluate_params(
            simulator, initial_state, params, model=model, spec=spec, anchors=anchors,
        )
        rows.append({
            "name": b["name"],
            "c_rate": float(b["c_rate"]),
            "params": params.to_dict(),
            "loss": float(loss),
            "metrics": {k: v for k, v in metrics.items() if k not in ("weights",)},
            "end_reason": session.get("end_reason"),
            "duration_s": int(session["current_a"].size),
        })
    return rows


def run_cell(
    cell: str,
    method: str,
    *,
    device: str = "auto",
    families: Optional[List[str]] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    if method not in ("random_search", "gp_bo"):
        raise ValueError(method)
    bind_search()
    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    paths = Paths.load()
    model = load_degradation_model()
    anchors = load_anchors(cell)
    simulator, state = make_simulator(cell, device=device)
    fams = list(families or opt.families)
    stage = "07_random_search" if method == "random_search" else "08_gp_bo"
    out_dir = stage_dir(stage) / cell.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    family_results: Dict[str, Any] = {}
    t0 = time.perf_counter()
    for fid in fams:
        fam_path = out_dir / f"{fid}.json"
        if resume and fam_path.is_file():
            print(f"  [{cell} {method}] skip existing {fid}", flush=True)
            from aacopt.config import read_json
            family_results[fid] = read_json(fam_path)
            continue
        seed = opt.seed_for(cell, fid, method)
        print(f"  [{cell} {method}] {fid}  n={opt.n_evals_per_family}  seed={seed}", flush=True)
        ft = time.perf_counter()
        if method == "random_search":
            result = optimize_family_random(
                simulator, state, fid, model=model, spec=reward, anchors=anchors,
                n_calls=opt.n_evals_per_family, seed=seed,
            )
        else:
            result = optimize_family_gp_bo(
                simulator, state, fid, model=model, spec=reward, anchors=anchors,
                n_calls=opt.n_evals_per_family,
                n_initial_points=int(opt.gp_bo["n_initial_points"]),
                seed=seed,
                acq_func=str(opt.gp_bo.get("acq_func", "EI")),
                acq_xi=float(opt.gp_bo.get("acq_xi", 0.01)),
                noise=float(opt.gp_bo.get("noise", 1e-6)),
            )
        result["wall_clock_s"] = float(time.perf_counter() - ft)
        payload = jsonable_family_result(result, include_session=True)
        write_json(fam_path, payload)
        family_results[fid] = payload
        m = result["best_metrics"]
        print(
            f"    best R={m['reward']:.4f}  loss={result['best_loss']:.4f}  "
            f"feas={m['feasible']}  t={m['duration_min']:.2f} min  "
            f"Q={m['q_total']:.3e}  ΔSOC={m['delta_soc']:.3f}  [{m['end_reason']}]",
            flush=True,
        )

    baselines = evaluate_baselines(simulator, state, model=model, spec=reward, anchors=anchors)
    write_json(out_dir / "baselines.json", baselines)

    feasible_any = any(
        (family_results[f].get("best_metrics") or {}).get("feasible") for f in fams
        if f in family_results
    )
    summary = {
        "cell": cell.upper(),
        "method": method,
        "families": fams,
        "n_evals_per_family": opt.n_evals_per_family,
        "device": str(simulator.bdt.device),
        "checkpoint": str(paths.checkpoint(cell)),
        "checkpoint_sha256": file_hash(paths.checkpoint(cell)),
        "start_state": state,
        "v_nom": simulator.v_nom,
        "anchors": anchors,
        "reward": reward.to_dict(),
        "baselines": baselines,
        "best_by_family": {
            fid: {
                "best_params": family_results[fid]["best_params"],
                "best_loss": family_results[fid]["best_loss"],
                "best_reward": family_results[fid]["best_reward"],
                "feasible": family_results[fid]["best_metrics"]["feasible"],
                "q_total": family_results[fid]["best_metrics"]["q_total"],
                "duration_min": family_results[fid]["best_metrics"].get(
                    "duration_min", family_results[fid]["best_metrics"]["duration_h"] * 60.0
                ),
                "peak_t": family_results[fid]["best_metrics"].get("peak_t"),
                "end_reason": family_results[fid]["best_metrics"]["end_reason"],
            }
            for fid in fams if fid in family_results
        },
        "any_feasible": bool(feasible_any),
        "wall_clock_s": float(time.perf_counter() - t0),
        "provenance": provenance(stage, configs=["paths", "degradation_fitted", "reward", "optimization"],
                                 inputs=[paths.checkpoint(cell)]),
    }
    write_json(out_dir / "summary.json", summary)
    return summary
