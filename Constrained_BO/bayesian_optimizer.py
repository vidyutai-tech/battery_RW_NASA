"""
Gaussian-process Bayesian optimization over Constrained_BO profile families.

Objective (unchanged scientifically from random search):
  loss, metrics = evaluate_session(session, reward_mode=...)

Default reward is hybrid Q_loss:
  R = w_soc * ΔSoC - w_qloss * (Q_calendar + Q_cyclic) - w_time * t_h^z
  loss = -R + soft constraint penalties (SoC/energy shortfall, V ceiling)

Hard physics / ordering constraints (i2 ≤ i1, soc2 > soc1, …) are applied by
each family's ``from_dict`` / ``from_vector``, identical to random search.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from skopt import gp_minimize

from Constrained_BO.objective import RewardMode, evaluate_session
from Constrained_BO.profiles import ProfileFamily, ProfileParams, get_family
from Constrained_BO.simulator import ChargingSimulator

# Paper / UI default: EI explores better than PI (which re-queries the same points).
DEFAULT_ACQ_FUNC = "EI"
DUPLICATE_JITTER_SCALE = 1e-4


def _evaluate_params(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family: ProfileFamily,
    params: ProfileParams,
    *,
    reward_mode: RewardMode,
    w_time: float,
    w_temperature: float,
    w_soc: float,
    w_qloss: float,
    z: float,
    qloss_cap: Optional[float] = None,
    qloss_cap_scale: float = 80.0,
    duration_loss_weight: float = 1e-3,
) -> tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Single session rollout + hybrid (or legacy) objective — shared with random search."""
    session = simulator.simulate(initial_state, params, family=family)
    loss, metrics = evaluate_session(
        session,
        reward_mode=reward_mode,
        w_time=w_time,
        w_temperature=w_temperature,
        w_soc=w_soc,
        w_qloss=w_qloss,
        z=z,
        qloss_cap=qloss_cap,
        qloss_cap_scale=qloss_cap_scale,
        duration_loss_weight=duration_loss_weight,
    )
    return float(loss), metrics, session


def _select_best_history(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prefer minimum loss among feasible points; else least-bad overall.

    Feasibility is defined by evaluate_session (SoC target or energy delivery),
    not by inventing a second constraint model.
    """
    if not history:
        raise ValueError("empty BO history")
    feasible = [h for h in history if h.get("feasible")]
    pool = feasible if feasible else history
    return min(pool, key=lambda h: float(h["loss"]))


def _vector_key(x: Sequence[float], ndigits: int = 6) -> tuple:
    return tuple(round(float(v), ndigits) for v in x)


def _jitter_duplicate(
    x: List[float],
    seen: set,
    bounds_low: List[float],
    bounds_high: List[float],
    rng: np.random.Generator,
) -> List[float]:
    """If ``x`` was already evaluated, nudge it inside bounds to avoid skopt re-eval."""
    key = _vector_key(x)
    if key not in seen:
        return list(x)
    out = list(x)
    for _ in range(8):
        for i, (lo, hi) in enumerate(zip(bounds_low, bounds_high)):
            span = max(hi - lo, 1e-9)
            out[i] = float(np.clip(
                out[i] + rng.normal(0.0, DUPLICATE_JITTER_SCALE * span),
                lo, hi,
            ))
        if _vector_key(out) not in seen:
            return out
    return out


class FamilyBayesianOptimizer:
    """GP-BO for one profile family using hybrid Q_loss via ``evaluate_session``."""

    def __init__(
        self,
        simulator: ChargingSimulator,
        initial_state: Dict[str, float],
        family_id: str,
        *,
        reward_mode: RewardMode = "hybrid_qloss",
        w_time: float = 0.1,
        w_temperature: float = 1.0,
        w_soc: float = 1.0,
        w_qloss: float = 1.0,
        z: float = 0.55,
        acq_func: str = DEFAULT_ACQ_FUNC,
        random_state: int = 42,
        extra_x0: Optional[List[List[float]]] = None,
        qloss_cap: Optional[float] = None,
        qloss_cap_scale: float = 80.0,
        duration_loss_weight: float = 1e-3,
    ):
        self.simulator = simulator
        self.initial_state = dict(initial_state)
        self.family_id = family_id
        self.family = get_family(family_id)
        self.family_cls = type(self.family)
        self.reward_mode = reward_mode
        self.w_time = float(w_time)
        self.w_temperature = float(w_temperature)
        self.w_soc = float(w_soc)
        self.w_qloss = float(w_qloss)
        self.z = float(z)
        self.acq_func = acq_func
        self.random_state = int(random_state)
        self.extra_x0 = list(extra_x0 or [])
        self.qloss_cap = qloss_cap
        self.qloss_cap_scale = float(qloss_cap_scale)
        self.duration_loss_weight = float(duration_loss_weight)
        self.history: List[Dict[str, Any]] = []
        self._seen_keys: set = set()
        self._rng = np.random.default_rng(self.random_state)
        space = self.family_cls.search_space()
        self._bounds_low = [float(d.low) for d in space]
        self._bounds_high = [float(d.high) for d in space]

    def _eval_kwargs(self) -> Dict[str, Any]:
        return {
            "reward_mode": self.reward_mode,
            "w_time": self.w_time,
            "w_temperature": self.w_temperature,
            "w_soc": self.w_soc,
            "w_qloss": self.w_qloss,
            "z": self.z,
            "qloss_cap": self.qloss_cap,
            "qloss_cap_scale": self.qloss_cap_scale,
            "duration_loss_weight": self.duration_loss_weight,
        }

    def _evaluate(self, x: List[float]) -> float:
        x = _jitter_duplicate(
            list(x), self._seen_keys, self._bounds_low, self._bounds_high, self._rng,
        )
        self._seen_keys.add(_vector_key(x))
        params = self.family_cls.from_vector(x)
        loss, metrics, _session = _evaluate_params(
            self.simulator,
            self.initial_state,
            self.family,
            params,
            **self._eval_kwargs(),
        )
        self.history.append({
            "family_id": self.family_id,
            "params": params.to_dict(),
            "loss": loss,
            "feasible": bool(metrics["feasible"]),
            "metrics": metrics,
            "end_reason": metrics["end_reason"],
        })
        return loss

    def optimize(
        self,
        *,
        n_calls: int = 40,
        n_initial_points: int = 10,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        seeds = list(self.family_cls.seed_points())
        # Optional elites (e.g. top random-search vectors) appended after family seeds.
        for x in self.extra_x0:
            if x is None:
                continue
            key = _vector_key(x)
            if key in {_vector_key(s) for s in seeds}:
                continue
            seeds.append(list(x))
        n_seed = len(seeds)
        # skopt: n_initial_points are *extra* random points beyond x0.
        n_random_extra = max(0, int(n_initial_points) - len(self.family_cls.seed_points()))
        n_calls_eff = max(int(n_calls), n_seed + n_random_extra)

        gp_kwargs: Dict[str, Any] = {
            "func": self._evaluate,
            "dimensions": self.family_cls.search_space(),
            "n_calls": n_calls_eff,
            "n_initial_points": n_random_extra,
            "x0": seeds if seeds else None,
            "random_state": self.random_state,
            "acq_func": self.acq_func,
            # Small observation noise discourages exact re-queries of the same point.
            "noise": 1e-6,
        }
        if self.acq_func == "LCB":
            gp_kwargs["kappa"] = 4.0
        elif self.acq_func == "EI":
            gp_kwargs["xi"] = 0.01

        skopt_result = gp_minimize(**gp_kwargs)

        best_entry = _select_best_history(self.history)
        if not best_entry.get("feasible"):
            print(
                f"WARNING [{self.family_id}]: no feasible profile — "
                "returning least-bad infeasible candidate."
            )

        best_params = self.family_cls.from_dict(best_entry["params"])
        # Re-simulate best once so trajectories match reported metrics (no stale cache).
        _, best_metrics, best_session = _evaluate_params(
            self.simulator,
            self.initial_state,
            self.family,
            best_params,
            **self._eval_kwargs(),
        )
        best_loss = float(best_entry["loss"])
        if abs(float(best_metrics["loss"]) - best_loss) > 1e-6:
            best_loss = float(best_metrics["loss"])
            best_entry = {
                **best_entry,
                "loss": best_loss,
                "feasible": bool(best_metrics["feasible"]),
                "metrics": best_metrics,
                "end_reason": best_metrics["end_reason"],
            }

        result = {
            "family_id": self.family_id,
            "family_label": self.family.label,
            "best_params": best_params.to_dict(),
            "best_loss": best_loss,
            "best_metrics": best_metrics,
            "best_session": best_session,
            "history": list(self.history),
            "n_evaluated": len(self.history),
            "method": "gp_bo",
            "acq_func": self.acq_func,
            "n_calls": n_calls_eff,
            "n_initial_points": n_initial_points,
            "n_seed_points": n_seed,
            "n_extra_x0": len(self.extra_x0),
            "qloss_cap": self.qloss_cap,
            "skopt_fun": float(skopt_result.fun) if skopt_result is not None else None,
        }
        if callback is not None:
            callback({k: v for k, v in result.items() if k != "best_session"})
        return result


def optimize_family_gp_bo(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family_id: str,
    *,
    n_calls: int = 40,
    n_initial_points: int = 10,
    seed: int = 42,
    reward_mode: RewardMode = "hybrid_qloss",
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    z: float = 0.55,
    acq_func: str = DEFAULT_ACQ_FUNC,
    extra_x0: Optional[List[List[float]]] = None,
    qloss_cap: Optional[float] = None,
    qloss_cap_scale: float = 80.0,
    duration_loss_weight: float = 1e-3,
) -> Dict[str, Any]:
    opt = FamilyBayesianOptimizer(
        simulator,
        initial_state,
        family_id,
        reward_mode=reward_mode,
        w_time=w_time,
        w_temperature=w_temperature,
        w_soc=w_soc,
        w_qloss=w_qloss,
        z=z,
        acq_func=acq_func,
        random_state=seed,
        extra_x0=extra_x0,
        qloss_cap=qloss_cap,
        qloss_cap_scale=qloss_cap_scale,
        duration_loss_weight=duration_loss_weight,
    )
    return opt.optimize(n_calls=n_calls, n_initial_points=n_initial_points)


def _elite_vectors_from_history(
    history: List[Dict[str, Any]],
    family_cls,
    *,
    top_k: int = 5,
) -> List[List[float]]:
    """Convert top-k history entries (feasible-first, min loss) to BO vectors."""
    if not history or top_k <= 0:
        return []
    feasible = [h for h in history if h.get("feasible")]
    pool = feasible if feasible else history
    ranked = sorted(pool, key=lambda h: float(h["loss"]))[: int(top_k)]
    space = family_cls.search_space()
    lows = [float(d.low) for d in space]
    highs = [float(d.high) for d in space]
    out: List[List[float]] = []
    for h in ranked:
        params = h.get("params") or {}
        try:
            pp = family_cls.from_dict(params)
            vec = family_cls.to_vector(pp)
            # Clip into skopt bounds — from_dict can yield edge values that
            # drift slightly outside Real(low, high) after round-trips.
            clipped = [
                float(np.clip(v, lo, hi))
                for v, lo, hi in zip(vec, lows, highs)
            ]
            out.append(clipped)
        except Exception:
            continue
    return out


def optimize_all_families_gp_bo(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    *,
    families: Optional[List[str]] = None,
    n_calls: int = 40,
    n_initial_points: int = 10,
    seed: int = 42,
    reward_mode: RewardMode = "hybrid_qloss",
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    z: float = 0.55,
    acq_func: str = DEFAULT_ACQ_FUNC,
    elite_histories: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    elite_top_k: int = 5,
    qloss_cap: Optional[float] = None,
    qloss_cap_scale: float = 80.0,
    duration_loss_weight: float = 1e-3,
) -> Dict[str, Any]:
    from Constrained_BO.profiles import DEFAULT_FAMILIES

    families = families or DEFAULT_FAMILIES
    elite_histories = elite_histories or {}
    results: Dict[str, Any] = {}
    for i, fid in enumerate(families):
        family = get_family(fid)
        family_cls = type(family)
        extra = _elite_vectors_from_history(
            elite_histories.get(fid, []), family_cls, top_k=elite_top_k,
        )
        print(f"\n{'=' * 60}")
        print(f"  GP-BO family: {family.label} ({fid})")
        print(f"  reward_mode={reward_mode}  acq={acq_func}")
        print(f"  n_calls={n_calls}  n_initial={n_initial_points}")
        if extra:
            print(f"  warm-start elites from random search: {len(extra)}")
        if reward_mode == "hybrid_qloss":
            print(
                f"  hybrid weights: w_soc={w_soc} w_qloss={w_qloss} "
                f"w_time={w_time} z={z}"
            )
        if qloss_cap is not None:
            print(f"  qloss_cap={float(qloss_cap):.6g}  (soft Q ≤ Random-best)")
        print(f"{'=' * 60}")
        results[fid] = optimize_family_gp_bo(
            simulator,
            initial_state,
            fid,
            n_calls=n_calls,
            n_initial_points=n_initial_points,
            seed=seed + i * 1000,
            reward_mode=reward_mode,
            w_time=w_time,
            w_temperature=w_temperature,
            w_soc=w_soc,
            w_qloss=w_qloss,
            z=z,
            acq_func=acq_func,
            extra_x0=extra,
            qloss_cap=qloss_cap,
            qloss_cap_scale=qloss_cap_scale,
            duration_loss_weight=duration_loss_weight,
        )
    return results
