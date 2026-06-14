"""
Bayesian optimization over pluggable charging profile families.

ENHANCEMENTS vs original:
  1. acq_func parameter ("EI", "PI", "LCB") — Paper 3 shows PI outperforms EI
  2. use_age_conditioning flag — evaluates each candidate at multiple battery ages
     and returns the weighted mean loss, optimizing for lifetime robustness
  3. chebyshev_omega support — directed Pareto front via Chebyshev scalarization
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from skopt import gp_minimize

from charging_opt.io_utils import resolve_writable_path

from charging_opt.charging_profile_family import (
    DEFAULT_FAMILY_IDS,
    ChargingProfileFamily,
    ProfileParams,
    get_family,
)
from charging_opt.lifetime_reward import (
    LifetimeWeights,
    ObjectiveMode,
    aggregate_lifetime_reward,
    chebyshev_loss,          # NEW — added in lifetime_reward.py
)
from charging_opt.profile_simulator import ProfileSimulator
from charging_opt.thermal_management import ThermalDeratingController

# Age checkpoints for age-conditioned evaluation (Enhancement 2).
# Covers RW9→RW12 lifespan (age 0 to 1).
DEFAULT_AGE_POINTS = [0.0, 0.25, 0.50, 0.75]
DEFAULT_AGE_WEIGHTS = [0.15, 0.30, 0.35, 0.20]   # weight later life more


@dataclass
class FamilyOptimizationResult:
    family_id: str
    family_label: str
    best_params: ProfileParams
    best_session: Dict
    best_metrics: Dict
    best_loss: float
    history: List[Dict]
    skopt_result: object

    def to_dict(self) -> Dict:
        return {
            "family_id": self.family_id,
            "family_label": self.family_label,
            "best_params": self.best_params.to_dict(),
            "best_loss": float(self.best_loss),
            "best_metrics": self.best_metrics,
            "history": self.history,
        }


def _age_conditioned_loss(
    simulator: ProfileSimulator,
    params: ProfileParams,
    base_state: Dict,
    *,
    family: ChargingProfileFamily,
    age_points: List[float],
    age_weights: List[float],
    soc_target: float,
    max_duration_min: Optional[float],
    weights: LifetimeWeights,
    objective_mode: ObjectiveMode,
    v_ref_stress: float,
    t_comfort_c: float,
    chebyshev_omega: Optional[float] = None,
    thermal_controller: Optional[ThermalDeratingController] = None,
    thermal_w_comfort: float = 0.5,
    thermal_w_hard: float = 5.0,
) -> Tuple[float, Dict]:
    """
    Evaluate `params` at each age in `age_points`.

    Returns (weighted_mean_loss, summary_metrics).

    A profile that works at age=0 but degrades poorly at age=0.75
    (because Q is lower so the same current causes more normalized stress)
    is correctly penalized here. A profile that fails at ANY age gets a
    soft +50 penalty per failure on top of the aggregate loss.

    This directly addresses the gap identified vs Paper 1 (Padisala et al.),
    which adapts charging to battery age — here we optimize FOR all ages
    simultaneously using the multi-age BDT which already takes `age` as input.
    """
    w = np.asarray(age_weights, dtype=np.float64)
    w = w / w.sum()

    per_age: List[Dict] = []
    losses: List[float] = []

    for age, aw in zip(age_points, w):
        state = dict(base_state)
        state["age"] = float(age)
        state["prev_i"] = 0.0

        session = simulator.simulate_params(state, params, family=family)
        _, metrics = aggregate_lifetime_reward(
            session,
            soc_target=soc_target,
            max_duration_min=max_duration_min,
            weights=weights,
            objective_mode=objective_mode,
            v_ref_stress=v_ref_stress,
            t_comfort_c=t_comfort_c,
        )

        # Apply Chebyshev scalarization on top of feasible metrics if requested
        if chebyshev_omega is not None and metrics.get("feasible"):
            loss_val = chebyshev_loss(metrics, omega=chebyshev_omega)
        else:
            loss_val = float(metrics.get("loss", 1e6))

        if thermal_controller is not None and metrics.get("feasible"):
            t_loss = thermal_controller.temperature_loss(
                session["temperature_c"],
                w_comfort=thermal_w_comfort,
                w_hard=thermal_w_hard,
            )
            loss_val += t_loss
            metrics["temperature_derating_loss"] = t_loss
            metrics.update(thermal_controller.feasibility_check(session["temperature_c"]))
            metrics["loss"] = loss_val

        losses.append(loss_val)
        per_age.append({
            "age": age,
            "age_weight": float(aw),
            "loss": loss_val,
            "feasible": bool(metrics.get("feasible", False)),
            "duration_min": metrics.get("duration_min"),
            "sei_per_pct_soc": metrics.get("sei_per_pct_soc"),
        })

    aggregate = float(np.dot(w, losses))

    # Soft penalty for infeasible ages (not a hard wall — keeps GP landscape smooth)
    n_infeasible = sum(1 for r in per_age if not r["feasible"])
    if n_infeasible:
        aggregate += 50.0 * n_infeasible

    all_feasible = n_infeasible == 0
    # Build a representative metrics dict (weighted average of feasible ages)
    feasible_ages = [r for r in per_age if r["feasible"]]
    agg_metrics = {
        "feasible": all_feasible,
        "loss": aggregate,
        "age_results": per_age,
        "sei_per_pct_soc": (
            float(np.mean([r["sei_per_pct_soc"] for r in feasible_ages]))
            if feasible_ages else None
        ),
        "duration_min": (
            float(np.mean([r["duration_min"] for r in feasible_ages]))
            if feasible_ages else None
        ),
    }
    return aggregate, agg_metrics


class FamilyBayesianOptimizer:
    """
    GP-BO over one :class:`ChargingProfileFamily` search space.

    Key changes vs original:
      - acq_func: choose "EI", "PI", or "LCB" (Paper 3 recommends PI)
      - use_age_conditioning: evaluate at multiple ages for lifetime robustness
      - chebyshev_omega: if set, use Chebyshev scalarization instead of linear
    """

    def __init__(
        self,
        simulator: ProfileSimulator,
        family: ChargingProfileFamily,
        initial_state: Dict[str, float],
        *,
        soc_target: float = 0.95,
        max_duration_min: Optional[float] = 105.0,
        weights: LifetimeWeights = LifetimeWeights(),
        objective_mode: ObjectiveMode = "composite",
        v_ref_stress: float = 4.0,
        t_comfort_c: float = 35.0,
        random_state: int = 42,
        # Enhancement 3: acquisition function choice
        acq_func: str = "PI",
        # Enhancement 2: age-conditioned evaluation
        use_age_conditioning: bool = False,
        age_points: List[float] = DEFAULT_AGE_POINTS,
        age_weights: List[float] = DEFAULT_AGE_WEIGHTS,
        # Enhancement 4: Chebyshev scalarization
        chebyshev_omega: Optional[float] = None,
        thermal_controller: Optional[ThermalDeratingController] = None,
        thermal_w_comfort: float = 0.5,
        thermal_w_hard: float = 5.0,
    ):
        self.simulator = simulator
        self.family = family
        self.family_id = family.family_id
        self.initial_state = dict(initial_state)
        self.soc_target = soc_target
        self.max_duration_min = max_duration_min
        self.weights = weights
        self.objective_mode = objective_mode
        self.v_ref_stress = v_ref_stress
        self.t_comfort_c = t_comfort_c
        self.random_state = random_state
        self.acq_func = acq_func
        self.use_age_conditioning = use_age_conditioning
        self.age_points = list(age_points)
        self.age_weights = list(age_weights)
        self.chebyshev_omega = chebyshev_omega
        self.thermal_controller = thermal_controller
        self.thermal_w_comfort = thermal_w_comfort
        self.thermal_w_hard = thermal_w_hard
        self.search_space = family.search_space()
        self.history: List[Dict] = []

    def _apply_thermal_loss(self, session: Dict, metrics: Dict) -> float:
        loss = float(metrics.get("loss", 1e6))
        if self.thermal_controller is None or not metrics.get("feasible"):
            return loss
        t_loss = self.thermal_controller.temperature_loss(
            session["temperature_c"],
            w_comfort=self.thermal_w_comfort,
            w_hard=self.thermal_w_hard,
        )
        loss += t_loss
        metrics["temperature_derating_loss"] = t_loss
        metrics.update(self.thermal_controller.feasibility_check(session["temperature_c"]))
        metrics["loss"] = loss
        return loss

    def _evaluate(self, x: List[float]) -> float:
        params = self.family.from_vector(x)

        if self.use_age_conditioning:
            # Enhancement 2: evaluate at multiple ages
            loss, metrics = _age_conditioned_loss(
                self.simulator, params, self.initial_state,
                family=self.family,
                age_points=self.age_points,
                age_weights=self.age_weights,
                soc_target=self.soc_target,
                max_duration_min=self.max_duration_min,
                weights=self.weights,
                objective_mode=self.objective_mode,
                v_ref_stress=self.v_ref_stress,
                t_comfort_c=self.t_comfort_c,
                chebyshev_omega=self.chebyshev_omega,
                thermal_controller=self.thermal_controller,
                thermal_w_comfort=self.thermal_w_comfort,
                thermal_w_hard=self.thermal_w_hard,
            )
            self.history.append({
                "family_id": self.family_id,
                "params": params.to_dict(),
                "loss": loss,
                "feasible": bool(metrics.get("feasible", False)),
                "metrics": metrics,
                "end_reason": "age_conditioned",
                "age_conditioned": True,
            })
        else:
            # Original single-age evaluation (age from initial_state)
            session = self.simulator.simulate_params(
                self.initial_state, params, family=self.family,
            )
            _, metrics = aggregate_lifetime_reward(
                session,
                soc_target=self.soc_target,
                max_duration_min=self.max_duration_min,
                weights=self.weights,
                objective_mode=self.objective_mode,
                v_ref_stress=self.v_ref_stress,
                t_comfort_c=self.t_comfort_c,
            )
            if self.chebyshev_omega is not None and metrics.get("feasible"):
                loss = chebyshev_loss(metrics, omega=self.chebyshev_omega)
                metrics["loss"] = loss
            else:
                loss = float(metrics.get("loss", 1e6))
            loss = self._apply_thermal_loss(session, metrics)

            self.history.append({
                "family_id": self.family_id,
                "params": params.to_dict(),
                "loss": loss,
                "feasible": bool(metrics.get("feasible", False)),
                "metrics": metrics,
                "end_reason": session["end_reason"],
                "age_conditioned": False,
            })

        return loss

    def _best_feasible_entry(self) -> Optional[Dict]:
        feas = [h for h in self.history if h.get("feasible")]
        if not feas:
            return None
        return min(feas, key=lambda h: h["loss"])

    def optimize(
        self,
        n_calls: int = 30,
        n_initial_points: int = 8,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> FamilyOptimizationResult:
        seeds = self.family.seed_points()
        n_seed = len(seeds)
        n_random = max(0, n_initial_points - n_seed)

        gp_kwargs = {
            "func": self._evaluate,
            "dimensions": self.search_space,
            "n_calls": max(n_calls, n_seed),
            "n_initial_points": n_random,
            "x0": seeds if seeds else None,
            "random_state": self.random_state,
            "acq_func": self.acq_func,
        }
        if self.acq_func == "LCB":
            gp_kwargs["kappa"] = 4.0

        result = gp_minimize(**gp_kwargs)

        entry = self._best_feasible_entry()
        if entry is None:
            entry = min(self.history, key=lambda h: h["loss"])
            print(
                f"WARNING [{self.family_id}]: no feasible profile — "
                "returning least-bad infeasible candidate."
            )

        best_params = ProfileParams.from_dict(entry["params"])

        # Re-simulate best params at age=0 for reporting/plotting
        # (age-conditioned eval uses multiple ages; we want a single reference session)
        report_state = dict(self.initial_state)
        report_state["age"] = self.age_points[0] if self.use_age_conditioning else self.initial_state.get("age", 0.0)
        best_session = self.simulator.simulate_params(
            report_state, best_params, family=self.family,
        )
        _, best_metrics = aggregate_lifetime_reward(
            best_session,
            soc_target=self.soc_target,
            max_duration_min=self.max_duration_min,
            weights=self.weights,
            objective_mode=self.objective_mode,
            v_ref_stress=self.v_ref_stress,
            t_comfort_c=self.t_comfort_c,
        )

        opt = FamilyOptimizationResult(
            family_id=self.family_id,
            family_label=self.family.label,
            best_params=best_params,
            best_session=best_session,
            best_metrics=best_metrics,
            best_loss=float(entry["loss"]),
            history=list(self.history),
            skopt_result=result,
        )
        if callback is not None:
            callback(opt.to_dict())
        return opt


def optimize_families(
    simulator: ProfileSimulator,
    initial_state: Dict[str, float],
    family_ids: List[str],
    *,
    n_calls: int = 30,
    n_initial_points: int = 8,
    soc_target: float = 0.95,
    max_duration_min: Optional[float] = 105.0,
    weights: LifetimeWeights = LifetimeWeights(),
    objective_mode: ObjectiveMode = "composite",
    v_ref_stress: float = 4.0,
    t_comfort_c: float = 35.0,
    random_state: int = 42,
    # New parameters
    acq_func: str = "PI",
    use_age_conditioning: bool = False,
    age_points: List[float] = DEFAULT_AGE_POINTS,
    age_weights: List[float] = DEFAULT_AGE_WEIGHTS,
    chebyshev_omega: Optional[float] = None,
    thermal_controller: Optional[ThermalDeratingController] = None,
    thermal_w_comfort: float = 0.5,
    thermal_w_hard: float = 5.0,
    on_family_done: Optional[Callable[[Dict[str, FamilyOptimizationResult]], None]] = None,
) -> Dict[str, FamilyOptimizationResult]:
    results: Dict[str, FamilyOptimizationResult] = {}
    for fid in family_ids:
        family = get_family(fid)
        print(f"\n{'=' * 60}\n  Optimizing family: {family.label} ({fid})")
        if use_age_conditioning:
            print(f"  Age conditioning ON: ages={age_points}, weights={age_weights}")
        if chebyshev_omega is not None:
            print(f"  Chebyshev omega={chebyshev_omega:.2f}")
        if thermal_controller is not None:
            print(f"  Thermal loss: ON (comfort={thermal_controller.t_comfort_c:.1f}°C)")
        print(f"  Acquisition: {acq_func}")
        print(f"{'=' * 60}")

        opt = FamilyBayesianOptimizer(
            simulator,
            family,
            initial_state,
            soc_target=soc_target,
            max_duration_min=max_duration_min,
            weights=weights,
            objective_mode=objective_mode,
            v_ref_stress=v_ref_stress,
            t_comfort_c=t_comfort_c,
            random_state=random_state,
            acq_func=acq_func,
            use_age_conditioning=use_age_conditioning,
            age_points=age_points,
            age_weights=age_weights,
            chebyshev_omega=chebyshev_omega,
            thermal_controller=thermal_controller,
            thermal_w_comfort=thermal_w_comfort,
            thermal_w_hard=thermal_w_hard,
        )
        results[fid] = opt.optimize(
            n_calls=n_calls, n_initial_points=n_initial_points,
        )
        m = results[fid].best_metrics
        print(
            f"  Best loss={results[fid].best_loss:.2f}  "
            f"feasible={m.get('feasible')}  "
            f"dur={m.get('duration_min', float('nan')):.1f} min  "
            f"SEI/%SoC={m.get('sei_per_pct_soc', float('nan')):.1f}  "
            f"V²·min={m.get('voltage_stress_v2_min', float('nan')):.2f}"
        )
        if on_family_done is not None:
            on_family_done(results)
    return results


def _resolve_writable_json_path(path: Path, *, repo_root: Path | None = None) -> Path:
    return resolve_writable_path(path, suffix_user=True, repo_root=repo_root)


def _write_json_atomic(path: Path, payload: Dict, *, repo_root: Path | None = None) -> Path:
    path = _resolve_writable_json_path(path, repo_root=repo_root)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    tmp.replace(path)
    return path


def save_family_results(
    results: Dict[str, FamilyOptimizationResult],
    out_dir: Path,
    *,
    initial_state: Dict[str, float],
    bdt_path: str,
    soc_target: float = 0.95,
    max_duration_min: Optional[float] = None,
    objective_config: Optional[Dict] = None,
    repo_root: Path | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_families = {fid: r.to_dict() for fid, r in results.items()}
    json_path = out_dir / "family_optimization_results.json"
    resolved = _resolve_writable_json_path(json_path, repo_root=repo_root)
    if resolved.exists():
        try:
            prev = json.loads(resolved.read_text()).get("families", {})
            prev.update(merged_families)
            merged_families = prev
        except (json.JSONDecodeError, OSError):
            pass

    def _rank(entry: dict) -> float:
        loss = float(entry.get("best_loss", 1e9))
        if not entry.get("best_metrics", {}).get("feasible", False):
            return 1e9 + loss
        return loss

    best_fid = min(merged_families, key=lambda fid: _rank(merged_families[fid]))
    best_entry = merged_families[best_fid]

    payload = {
        "initial_state": initial_state,
        "bdt_checkpoint": bdt_path,
        "constraints": {
            "soc_target": soc_target,
            "max_duration_min": max_duration_min,
            **(objective_config or {}),
        },
        "families": merged_families,
        "best_overall_family": best_fid,
        "best_overall_params": best_entry["best_params"],
        "best_overall_metrics": best_entry["best_metrics"],
    }
    _write_json_atomic(json_path, payload, repo_root=repo_root)
    return _resolve_writable_json_path(json_path, repo_root=repo_root)
