"""Pack-level GP-BO: one shared profile, scored on every cell twin.

A trial is pack-feasible only if every cell meets the energy target and stays
at or below the 4.20 V ceiling. Pack loss is the worst-cell loss (minimax).

Each twin is rolled out independently from the same parameter vector θ, so CV
taper may differ slightly cell to cell. That is a conservative pack charger:
the protocol must be legal on every cell. A fully coupled series pack (one
shared I(t) limited by the first cell to hit Vmax) is stricter and is not
modelled here.

Does not write into results/07_random_search or results/08_gp_bo.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from aacopt.config import (
    OptimizationSpec, Paths, RewardSpec, file_hash, provenance, stage_dir, write_json,
)
from aacopt.evaluate import (
    cccv_params, evaluate_params, jsonable_family_result, jsonable_session,
    load_anchors, load_degradation_model, make_simulator,
)
from aacopt.optimizers.gp_bo import DUPLICATE_JITTER_SCALE, _vector_key, ensure_twin_eval
from aacopt.profiles import ProfileParams, bind_search, get_family
from aacopt.viz.figures import plot_best_profiles
from aacopt.viz.style import FAMILY_LABELS, FAMILY_ORDER

_KEEP_METRICS = (
    "reward", "loss", "feasible", "delta_soc", "q_total", "q_calendar", "q_cyclic",
    "duration_min", "peak_v", "peak_t", "mean_t", "end_reason", "penalty",
    "e_norm", "ah_throughput", "mean_charge_c_rate",
)


def _compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in _KEEP_METRICS:
        if k not in metrics:
            continue
        v = metrics[k]
        if isinstance(v, (np.generic,)):
            v = v.item()
        out[k] = v
    return out


def evaluate_params_pack(
    members: Dict[str, Tuple[Any, Dict[str, float], Dict[str, float]]],
    params: ProfileParams,
    *,
    model,
    spec: RewardSpec,
) -> Tuple[float, Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Simulate one θ on every cell. Return (pack_loss, pack_metrics, sessions)."""
    per_cell: Dict[str, Dict[str, Any]] = {}
    sessions: Dict[str, Dict[str, Any]] = {}
    for cell, (simulator, state, anchors) in members.items():
        loss, metrics, session = evaluate_params(
            simulator, state, params, model=model, spec=spec, anchors=anchors,
        )
        per_cell[cell] = _compact_metrics(metrics)
        per_cell[cell]["loss"] = float(loss)
        sessions[cell] = session

    cells = list(per_cell)
    pack_feasible = all(bool(per_cell[c]["feasible"]) for c in cells)
    pack_loss = max(float(per_cell[c]["loss"]) for c in cells)
    pack_reward = min(float(per_cell[c]["reward"]) for c in cells)
    worst = max(cells, key=lambda c: float(per_cell[c]["loss"]))
    pack_metrics = {
        "reward": float(pack_reward),
        "loss": float(pack_loss),
        "feasible": bool(pack_feasible),
        "worst_cell": worst,
        "duration_min": float(per_cell[worst]["duration_min"]),
        "q_total": float(max(per_cell[c]["q_total"] for c in cells)),
        "q_calendar": float(max(per_cell[c].get("q_calendar", 0.0) for c in cells)),
        "q_cyclic": float(max(per_cell[c].get("q_cyclic", 0.0) for c in cells)),
        "peak_v": float(max(per_cell[c]["peak_v"] for c in cells)),
        "peak_t": float(max(per_cell[c]["peak_t"] for c in cells)),
        "delta_soc": float(min(per_cell[c]["delta_soc"] for c in cells)),
        "end_reason": per_cell[worst].get("end_reason"),
        "per_cell": per_cell,
    }
    return float(pack_loss), pack_metrics, sessions


def optimize_family_gp_bo_pack(
    members: Dict[str, Tuple[Any, Dict[str, float], Dict[str, float]]],
    family_id: str,
    *,
    model,
    spec: RewardSpec,
    n_calls: int,
    n_initial_points: int,
    seed: int,
    acq_func: str = "EI",
    acq_xi: float = 0.01,
    noise: float = 1e-6,
) -> Dict[str, Any]:
    from skopt import gp_minimize

    for _simulator, _state, _anchors in members.values():
        ensure_twin_eval(_simulator)
    family = get_family(family_id)
    cls = type(family)
    history: List[Dict[str, Any]] = []
    seen = set()
    rng = np.random.default_rng(int(seed))
    space = cls.search_space()
    lo = [float(d.low) for d in space]
    hi = [float(d.high) for d in space]

    def _jitter(x: List[float]) -> List[float]:
        key = _vector_key(x)
        if key not in seen:
            return list(x)
        out = list(x)
        for _ in range(8):
            for i, (a, b) in enumerate(zip(lo, hi)):
                span = max(b - a, 1e-9)
                out[i] = float(np.clip(out[i] + rng.normal(0.0, DUPLICATE_JITTER_SCALE * span), a, b))
            if _vector_key(out) not in seen:
                return out
        return out

    def objective(x: List[float]) -> float:
        for _simulator, _state, _anchors in members.values():
            ensure_twin_eval(_simulator)
        x = _jitter(list(x))
        seen.add(_vector_key(x))
        params = cls.from_vector(x)
        pack_loss, pack_metrics, _ = evaluate_params_pack(
            members, params, model=model, spec=spec,
        )
        history.append({
            "family_id": family_id,
            "params": params.to_dict(),
            "loss": float(pack_loss),
            "reward": float(pack_metrics["reward"]),
            "feasible": bool(pack_metrics["feasible"]),
            "worst_cell": pack_metrics["worst_cell"],
            "metrics": {k: v for k, v in pack_metrics.items() if k != "per_cell"},
            "per_cell": pack_metrics["per_cell"],
        })
        return float(pack_loss)

    seeds = list(cls.seed_points())
    n_seed = len(seeds)
    n_random_extra = max(0, int(n_initial_points) - n_seed)
    n_calls_eff = max(int(n_calls), n_seed + n_random_extra)
    gp_kwargs: Dict[str, Any] = {
        "func": objective,
        "dimensions": space,
        "n_calls": n_calls_eff,
        "n_initial_points": n_random_extra,
        "x0": seeds if seeds else None,
        "random_state": int(seed),
        "acq_func": acq_func,
        "noise": float(noise),
    }
    if acq_func == "EI":
        gp_kwargs["xi"] = float(acq_xi)
    skopt_result = gp_minimize(**gp_kwargs)

    feasible = [h for h in history if h["feasible"]]
    pool = feasible if feasible else history
    best = min(pool, key=lambda h: h["loss"])
    best_params = cls.from_dict(best["params"])
    for _simulator, _state, _anchors in members.values():
        ensure_twin_eval(_simulator)
    pack_loss, pack_metrics, sessions = evaluate_params_pack(
        members, best_params, model=model, spec=spec,
    )
    worst = pack_metrics["worst_cell"]
    return {
        "family_id": family_id,
        "family_label": family.label,
        "method": "gp_bo_pack",
        "objective": "minimax_worst_cell_loss",
        "cells": list(members),
        "best_params": best_params.to_dict(),
        "best_loss": float(pack_loss),
        "best_reward": float(pack_metrics["reward"]),
        "best_metrics": pack_metrics,
        "best_session": sessions[worst],
        "best_sessions_by_cell": {c: jsonable_session(s) for c, s in sessions.items()},
        "history": history,
        "n_evaluated": len(history),
        "n_calls": n_calls_eff,
        "n_seed_points": n_seed,
        "n_initial_points": int(n_initial_points),
        "acq_func": acq_func,
        "seed": int(seed),
        "skopt_fun": float(skopt_result.fun) if skopt_result is not None else None,
        "warm_start_from_random_search": False,
    }


def evaluate_pack_baselines(
    members: Dict[str, Tuple[Any, Dict[str, float], Dict[str, float]]],
    *,
    model,
    spec: RewardSpec,
) -> List[Dict[str, Any]]:
    opt = OptimizationSpec.load()
    rows = []
    for b in opt.baselines:
        params = cccv_params(float(b["c_rate"]), v_cv=float(b["v_cv"]), i_cutoff=float(b["i_cutoff"]))
        pack_loss, pack_metrics, _ = evaluate_params_pack(
            members, params, model=model, spec=spec,
        )
        rows.append({
            "name": b["name"],
            "c_rate": float(b["c_rate"]),
            "params": params.to_dict(),
            "loss": float(pack_loss),
            "reward": float(pack_metrics["reward"]),
            "feasible": bool(pack_metrics["feasible"]),
            "worst_cell": pack_metrics["worst_cell"],
            "metrics": {k: v for k, v in pack_metrics.items() if k != "per_cell"},
            "per_cell": pack_metrics["per_cell"],
        })
    return rows


def _param_key(params: Dict[str, Any], ndigits: int = 4) -> tuple:
    items = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, (int, float, np.generic)):
            items.append((k, round(float(v), ndigits)))
        else:
            items.append((k, v))
    return tuple(items)


def build_elite_top6(family_results: Dict[str, Dict[str, Any]], *, k: int = 6) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fid, blob in family_results.items():
        for h in blob.get("history") or []:
            rows.append({
                "family_id": fid,
                "params": h.get("params") or {},
                "loss": float(h.get("loss", np.inf)),
                "reward": float(h.get("reward", -np.inf)),
                "feasible": bool(h.get("feasible")),
                "worst_cell": h.get("worst_cell"),
                "metrics": h.get("metrics") or {},
                "per_cell": h.get("per_cell") or {},
            })
    feasible = [r for r in rows if r["feasible"]]
    pool = feasible if feasible else rows
    pool.sort(key=lambda r: r["loss"])
    elite: List[Dict[str, Any]] = []
    seen = set()
    for r in pool:
        key = (r["family_id"], _param_key(r["params"]))
        if key in seen:
            continue
        seen.add(key)
        elite.append(r)
        if len(elite) >= k:
            break
    for i, r in enumerate(elite, 1):
        r["rank"] = i
    return elite


def elite_markdown(elite: List[Dict[str, Any]], cells: Sequence[str]) -> str:
    lines = [
        "# Pack GP-BO elite set (minimax / worst-cell loss)",
        "",
        "One shared profile per row. Pack-feasible means **every** cell met the",
        "energy target and stayed at or below 4.20 V. Time and Q_loss are",
        "shown per cell so ranking inside the set can shuffle without changing",
        "the waveform.",
        "",
        "| Rank | Family | Pack loss | Worst cell | " + " | ".join(
            f"{c} t (min)" for c in cells
        ) + " | " + " | ".join(
            f"{c} Q" for c in cells
        ) + " | Feasible |",
        "|---|---|---|---|---" + "|---" * (2 * len(cells)) + "|---|",
    ]
    for r in elite:
        pc = r.get("per_cell") or {}
        times = []
        qs = []
        for c in cells:
            m = pc.get(c) or {}
            t = m.get("duration_min")
            q = m.get("q_total")
            times.append(f"{t:.2f}" if isinstance(t, (int, float)) else "—")
            qs.append(f"{q:.3e}" if isinstance(q, (int, float)) else "—")
        lines.append(
            f"| {r.get('rank', '')} | {FAMILY_LABELS.get(r['family_id'], r['family_id'])} "
            f"| {r['loss']:.4f} | {r.get('worst_cell', '')} | "
            + " | ".join(times) + " | " + " | ".join(qs)
            + f" | {'yes' if r['feasible'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    for r in elite:
        lines.append(f"- **#{r.get('rank')}** `{r['family_id']}`: `{r['params']}`")
    lines.append("")
    return "\n".join(lines)


def run_pack(
    *,
    cells: Optional[List[str]] = None,
    families: Optional[List[str]] = None,
    device: str = "auto",
    resume: bool = True,
    n_evals: Optional[int] = None,
    stage: str = "08_gp_bo_pack",
) -> Dict[str, Any]:
    bind_search()
    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    paths = Paths.load()
    model = load_degradation_model()
    cells = [c.upper() for c in (cells or paths.cells)]
    fams = list(families or opt.families)
    n_calls = int(n_evals if n_evals is not None else opt.n_evals_per_family)
    out_dir = stage_dir(stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    members: Dict[str, Tuple[Any, Dict[str, float], Dict[str, float]]] = {}
    member_meta = {}
    for cell in cells:
        simulator, state = make_simulator(cell, device=device)
        anchors = load_anchors(cell)
        members[cell] = (simulator, state, anchors)
        member_meta[cell] = {
            "checkpoint": str(paths.checkpoint(cell)),
            "checkpoint_sha256": file_hash(paths.checkpoint(cell)),
            "start_state": state,
            "v_nom": simulator.v_nom,
            "anchors": anchors,
            "device": str(simulator.bdt.device),
        }
        print(f"  loaded twin {cell}  device={simulator.bdt.device}", flush=True)

    family_results: Dict[str, Any] = {}
    t0 = time.perf_counter()
    for fid in fams:
        fam_path = out_dir / f"{fid}.json"
        if resume and fam_path.is_file():
            from aacopt.config import read_json
            print(f"  [pack gp_bo] skip existing {fid}", flush=True)
            family_results[fid] = read_json(fam_path)
            continue
        seed = opt.seed_for_pack(fid, "gp_bo")
        n_initial = min(int(opt.gp_bo["n_initial_points"]), n_calls)
        print(f"  [pack gp_bo] {fid}  n={n_calls}  n_initial={n_initial}  seed={seed}  cells={cells}", flush=True)
        ft = time.perf_counter()
        result = optimize_family_gp_bo_pack(
            members, fid, model=model, spec=reward,
            n_calls=n_calls,
            n_initial_points=n_initial,
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
            f"    pack R_min={m['reward']:.4f}  loss={result['best_loss']:.4f}  "
            f"feas={m['feasible']}  worst={m['worst_cell']}  "
            f"t_worst={m['duration_min']:.2f} min  Q_max={m['q_total']:.3e}",
            flush=True,
        )

    baselines = evaluate_pack_baselines(members, model=model, spec=reward)
    write_json(out_dir / "baselines.json", baselines)

    elite = build_elite_top6(family_results, k=6)
    write_json(out_dir / "elite_top6.json", elite)
    (out_dir / "elite_top6.md").write_text(elite_markdown(elite, cells), encoding="utf-8")

    plot_families = {}
    for fid in fams:
        blob = family_results.get(fid) or {}
        if blob.get("best_session"):
            plot_families[fid] = blob
    if plot_families:
        plot_best_profiles(
            plot_families,
            cell_id="PACK",
            method_title="Pack GP-BO (minimax)",
            soc_start=float(opt.session.soc_start),
            v_max=float(opt.session.v_max),
            out_path=out_dir / "pack_family_best_profiles.png",
        )

    feasible_any = any(
        (family_results[f].get("best_metrics") or {}).get("feasible") for f in fams
        if f in family_results
    )
    summary = {
        "mode": "pack",
        "objective": "minimax_worst_cell_loss",
        "method": "gp_bo",
        "cells": cells,
        "families": fams,
        "n_evals_per_family": n_calls,
        "members": member_meta,
        "reward": reward.to_dict(),
        "baselines": [
            {k: v for k, v in b.items() if k != "per_cell"} | {"per_cell": b.get("per_cell")}
            for b in baselines
        ],
        "best_by_family": {
            fid: {
                "best_params": family_results[fid]["best_params"],
                "best_loss": family_results[fid]["best_loss"],
                "best_reward": family_results[fid]["best_reward"],
                "feasible": family_results[fid]["best_metrics"]["feasible"],
                "worst_cell": family_results[fid]["best_metrics"].get("worst_cell"),
                "q_total": family_results[fid]["best_metrics"]["q_total"],
                "duration_min": family_results[fid]["best_metrics"]["duration_min"],
                "peak_t": family_results[fid]["best_metrics"].get("peak_t"),
                "end_reason": family_results[fid]["best_metrics"].get("end_reason"),
                "per_cell": family_results[fid]["best_metrics"].get("per_cell"),
            }
            for fid in fams if fid in family_results
        },
        "elite_top6": elite,
        "any_feasible": bool(feasible_any),
        "wall_clock_s": float(time.perf_counter() - t0),
        "provenance": provenance(
            stage,
            configs=["paths", "degradation_fitted", "reward", "optimization"],
            inputs=[paths.checkpoint(c) for c in cells],
        ),
    }
    write_json(out_dir / "summary.json", summary)
    return summary
