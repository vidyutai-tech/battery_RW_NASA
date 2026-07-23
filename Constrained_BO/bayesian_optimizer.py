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

from typing import Any, Callable, Dict, List, Optional

from skopt import gp_minimize

from Constrained_BO.objective import RewardMode, evaluate_session
from Constrained_BO.profiles import ProfileFamily, ProfileParams, get_family
from Constrained_BO.simulator import ChargingSimulator


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
        acq_func: str = "PI",
        random_state: int = 42,
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
        self.history: List[Dict[str, Any]] = []

    def _evaluate(self, x: List[float]) -> float:
        params = self.family_cls.from_vector(x)
        loss, metrics, _session = _evaluate_params(
            self.simulator,
            self.initial_state,
            self.family,
            params,
            reward_mode=self.reward_mode,
            w_time=self.w_time,
            w_temperature=self.w_temperature,
            w_soc=self.w_soc,
            w_qloss=self.w_qloss,
            z=self.z,
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
        seeds = self.family_cls.seed_points()
        n_seed = len(seeds)
        # skopt: n_initial_points are *extra* random points beyond x0.
        n_random_extra = max(0, int(n_initial_points) - n_seed)
        n_calls_eff = max(int(n_calls), n_seed + n_random_extra)

        gp_kwargs: Dict[str, Any] = {
            "func": self._evaluate,
            "dimensions": self.family_cls.search_space(),
            "n_calls": n_calls_eff,
            "n_initial_points": n_random_extra,
            "x0": seeds if seeds else None,
            "random_state": self.random_state,
            "acq_func": self.acq_func,
        }
        if self.acq_func == "LCB":
            gp_kwargs["kappa"] = 4.0

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
            reward_mode=self.reward_mode,
            w_time=self.w_time,
            w_temperature=self.w_temperature,
            w_soc=self.w_soc,
            w_qloss=self.w_qloss,
            z=self.z,
        )
        # Keep reported loss consistent with history selection (same objective).
        best_loss = float(best_entry["loss"])
        # If re-sim drifts numerically, trust re-sim metrics but keep BO loss for ranking.
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
    acq_func: str = "PI",
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
    )
    return opt.optimize(n_calls=n_calls, n_initial_points=n_initial_points)


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
    acq_func: str = "PI",
) -> Dict[str, Any]:
    from Constrained_BO.profiles import DEFAULT_FAMILIES

    families = families or DEFAULT_FAMILIES
    results: Dict[str, Any] = {}
    for i, fid in enumerate(families):
        family = get_family(fid)
        print(f"\n{'=' * 60}")
        print(f"  GP-BO family: {family.label} ({fid})")
        print(f"  reward_mode={reward_mode}  acq={acq_func}")
        print(f"  n_calls={n_calls}  n_initial={n_initial_points}")
        if reward_mode == "hybrid_qloss":
            print(
                f"  hybrid weights: w_soc={w_soc} w_qloss={w_qloss} "
                f"w_time={w_time} z={z}"
            )
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
        )
    return results
