"""Random search over one profile family, identical budget to GP-BO."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from aacopt.evaluate import evaluate_params
from aacopt.profiles import get_family
from aacopt.simulator import ChargingSimulator


def optimize_family_random(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family_id: str,
    *,
    model,
    spec,
    anchors: Dict[str, float],
    n_calls: int,
    seed: int,
) -> Dict[str, Any]:
    family = get_family(family_id)
    cls = type(family)
    rng = np.random.default_rng(int(seed))
    history: List[Dict[str, Any]] = []
    seeds = list(cls.seed_params())
    n_seed = len(seeds)

    def _run(params) -> None:
        loss, metrics, _ = evaluate_params(
            simulator, initial_state, params, model=model, spec=spec, anchors=anchors,
        )
        history.append({
            "family_id": family_id,
            "params": params.to_dict(),
            "loss": float(loss),
            "reward": float(metrics["reward"]),
            "feasible": bool(metrics["feasible"]),
            "metrics": {k: v for k, v in metrics.items() if k not in ("weights", "anchors")},
        })

    for p in seeds:
        _run(p)
        if len(history) >= n_calls:
            break
    while len(history) < n_calls:
        _run(cls.sample_random(rng))

    feasible = [h for h in history if h["feasible"]]
    pool = feasible if feasible else history
    best = min(pool, key=lambda h: h["loss"])
    best_params = cls.from_dict(best["params"])
    _, best_metrics, best_session = evaluate_params(
        simulator, initial_state, best_params, model=model, spec=spec, anchors=anchors,
    )
    return {
        "family_id": family_id,
        "family_label": family.label,
        "method": "random_search",
        "best_params": best_params.to_dict(),
        "best_loss": float(best["loss"]),
        "best_reward": float(best_metrics["reward"]),
        "best_metrics": best_metrics,
        "best_session": best_session,
        "history": history,
        "n_evaluated": len(history),
        "n_seed_points": n_seed,
        "seed": int(seed),
    }
