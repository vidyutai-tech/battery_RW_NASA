"""
Stage 3 — Gaussian-process Bayesian optimization over profile parameters.

Uses scikit-optimize to search the low-dimensional parametric profile space.
Each evaluation runs the frozen BDT (Stage 1) and the lifetime reward (Stage 2).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from skopt import gp_minimize
from skopt.space import Real

from charging_opt.lifetime_reward import LifetimeWeights, ObjectiveMode, aggregate_lifetime_reward
from charging_opt.profile_simulator import ProfileSimulator, ProfileSpec, TAPER_STEP_A


@dataclass
class OptimizationResult:
    best_spec: ProfileSpec
    best_session: Dict
    best_reward: float
    best_metrics: Dict
    history: List[Dict]
    skopt_result: object

    def to_dict(self) -> Dict:
        return {
            "best_spec": self.best_spec.to_dict(),
            "best_reward": float(self.best_reward),
            "best_metrics": self.best_metrics,
            "history": self.history,
        }


def default_search_space(*, allow_pulsed: bool = True) -> List[Real]:
    """
    Search space for BO.

    When ``allow_pulsed`` is False, only CC-taper is searched (2 effective dims
    after fixing pulse_rest=0) — recommended for a first stable optimum.
    """
    space = [
        Real(0.75, 4.0, name="i_charge"),
        Real(5.0, 30.0, name="pulse_on_min"),
    ]
    if allow_pulsed:
        # 0 or meaningful rest (>= MIN_REST_MIN); values in (0, 5) snap to 0
        space.append(Real(0.0, 15.0, name="pulse_rest_min"))
    space.append(Real(0.75, 2.25, name="i_floor"))
    return space


def seed_points_cc_taper(
    i_levels: Optional[List[float]] = None,
    i_floor: float = 0.75,
) -> List[List[float]]:
    """Deterministic CC-taper seeds so BO always evaluates feasible baselines."""
    if i_levels is None:
        i_levels = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    seeds = []
    for i in i_levels:
        floor = 0.75 if i <= 0.75 + TAPER_STEP_A else min(i_floor, i - TAPER_STEP_A)
        seeds.append([float(i), 30.0, 0.0, float(floor)])
    return seeds


def _vector_to_spec(x: List[float], allow_pulsed: bool) -> ProfileSpec:
    vals = [float(v) for v in x]
    if allow_pulsed:
        if len(vals) != 4:
            raise ValueError(f"Expected 4 parameters, got {len(vals)}")
        return ProfileSpec.from_vector(vals)
    if len(vals) == 3:
        i_charge, pulse_on_min, i_floor = vals
        return ProfileSpec.from_vector([i_charge, pulse_on_min, 0.0, i_floor])
    if len(vals) == 4:
        return ProfileSpec.from_vector([vals[0], vals[1], 0.0, vals[3]])
    raise ValueError(f"Expected 3 or 4 parameters, got {len(vals)}")


class LifetimeBayesianOptimizer:
    """
    Find the best lifetime charging profile for one start state + BDT checkpoint.

    Swap ``bdt_path`` to adapt to a fine-tuned twin on another cell without
    retraining the optimizer — only the Stage-1 simulator changes.
    """

    def __init__(
        self,
        simulator: ProfileSimulator,
        initial_state: Dict[str, float],
        *,
        soc_target: float = 0.95,
        max_duration_min: Optional[float] = 105.0,
        weights: LifetimeWeights = LifetimeWeights(),
        objective_mode: ObjectiveMode = "composite",
        v_ref_stress: float = 4.0,
        t_comfort_c: float = 35.0,
        search_space: Optional[List[Real]] = None,
        allow_pulsed: bool = False,
        random_state: int = 42,
    ):
        self.simulator = simulator
        self.initial_state = dict(initial_state)
        self.soc_target = soc_target
        self.max_duration_min = max_duration_min
        self.weights = weights
        self.objective_mode = objective_mode
        self.v_ref_stress = v_ref_stress
        self.t_comfort_c = t_comfort_c
        self.allow_pulsed = allow_pulsed
        self.search_space = search_space or default_search_space(allow_pulsed=allow_pulsed)
        self.random_state = random_state
        self.history: List[Dict] = []

    def _evaluate(self, x: List[float]) -> float:
        spec = _vector_to_spec(x, self.allow_pulsed)
        session = self.simulator.simulate(self.initial_state, spec)
        _, metrics = aggregate_lifetime_reward(
            session,
            soc_target=self.soc_target,
            max_duration_min=self.max_duration_min,
            weights=self.weights,
            objective_mode=self.objective_mode,
            v_ref_stress=self.v_ref_stress,
            t_comfort_c=self.t_comfort_c,
        )
        loss = float(metrics.get("loss", 1e6))
        self.history.append({
            "spec": spec.to_dict(),
            "loss": loss,
            "feasible": bool(metrics.get("feasible", False)),
            "metrics": metrics,
            "end_reason": session["end_reason"],
        })
        return loss

    def _best_feasible_entry(self) -> Optional[Dict]:
        feas = [h for h in self.history if h.get("feasible")]
        if not feas:
            return None
        return min(feas, key=lambda h: h["loss"])

    def optimize(
        self,
        n_calls: int = 40,
        n_initial_points: int = 10,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> OptimizationResult:
        seeds = seed_points_cc_taper()
        if not self.allow_pulsed:
            seeds = [[s[0], s[1], s[3]] for s in seeds]

        n_seed = len(seeds)
        n_random = max(0, n_initial_points - n_seed)

        result = gp_minimize(
            self._evaluate,
            dimensions=self.search_space,
            n_calls=max(n_calls, n_seed),
            n_initial_points=n_random,
            x0=seeds,
            random_state=self.random_state,
            acq_func="EI",
        )

        entry = self._best_feasible_entry()
        if entry is None:
            entry = min(self.history, key=lambda h: h["loss"])
            print(
                "WARNING: no feasible profile (SoC target + time limit) — "
                "returning least-bad infeasible candidate. "
                "Try --max_duration_min 120, --max_minutes 150, or lower --soc_target."
            )

        best_spec = ProfileSpec.from_dict(entry["spec"])
        best_session = self.simulator.simulate(self.initial_state, best_spec)
        _, best_metrics = aggregate_lifetime_reward(
            best_session,
            soc_target=self.soc_target,
            max_duration_min=self.max_duration_min,
            weights=self.weights,
            objective_mode=self.objective_mode,
            v_ref_stress=self.v_ref_stress,
            t_comfort_c=self.t_comfort_c,
        )
        opt = OptimizationResult(
            best_spec=best_spec,
            best_session=best_session,
            best_reward=-float(entry["loss"]),
            best_metrics=best_metrics,
            history=list(self.history),
            skopt_result=result,
        )
        if callback is not None:
            callback(opt.to_dict())
        return opt


def save_optimization_result(
    result: OptimizationResult,
    out_dir: Path,
    *,
    initial_state: Dict[str, float],
    bdt_path: str,
    soc_target: float = 0.95,
    max_duration_min: Optional[float] = None,
    objective_config: Optional[Dict] = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "initial_state": initial_state,
        "bdt_checkpoint": bdt_path,
        "constraints": {
            "soc_target": soc_target,
            "max_duration_min": max_duration_min,
            **(objective_config or {}),
        },
        "best_spec": result.best_spec.to_dict(),
        "best_reward": result.best_reward,
        "best_metrics": result.best_metrics,
        "merged_profile": ProfileSimulator.merged_segments(result.best_session),
        "history": result.history,
    }
    with (out_dir / "optimization_result.json").open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    session_path = out_dir / "best_session.json"
    slim = {
        k: v for k, v in result.best_session.items()
        if k not in ("current_a", "voltage_v", "temperature_c", "soc", "time_s")
    }
    slim["n_samples"] = int(result.best_session["current_a"].size)
    with session_path.open("w") as f:
        json.dump(slim, f, indent=2, default=float)
