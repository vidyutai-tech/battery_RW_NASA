"""GP-BO over one profile family (skopt), same objective as random search."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from skopt import gp_minimize

from aacopt.evaluate import evaluate_params
from aacopt.profiles import get_family
from aacopt.simulator import ChargingSimulator

DUPLICATE_JITTER_SCALE = 1e-4


def _vector_key(x: Sequence[float], ndigits: int = 6) -> tuple:
    return tuple(round(float(v), ndigits) for v in x)


def optimize_family_gp_bo(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family_id: str,
    *,
    model,
    spec,
    anchors: Dict[str, float],
    n_calls: int,
    n_initial_points: int,
    seed: int,
    acq_func: str = "EI",
    acq_xi: float = 0.01,
    noise: float = 1e-6,
) -> Dict[str, Any]:
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
        x = _jitter(list(x))
        seen.add(_vector_key(x))
        params = cls.from_vector(x)
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
        return float(loss)

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
    _, best_metrics, best_session = evaluate_params(
        simulator, initial_state, best_params, model=model, spec=spec, anchors=anchors,
    )
    return {
        "family_id": family_id,
        "family_label": family.label,
        "method": "gp_bo",
        "best_params": best_params.to_dict(),
        "best_loss": float(best["loss"]),
        "best_reward": float(best_metrics["reward"]),
        "best_metrics": best_metrics,
        "best_session": best_session,
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
