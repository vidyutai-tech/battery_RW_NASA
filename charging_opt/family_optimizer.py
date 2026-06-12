"""
Bayesian optimization over pluggable charging profile families.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from skopt import gp_minimize

from charging_opt.io_utils import resolve_writable_path

from charging_opt.charging_profile_family import (
    DEFAULT_FAMILY_IDS,
    ChargingProfileFamily,
    ProfileParams,
    get_family,
)
from charging_opt.lifetime_reward import LifetimeWeights, ObjectiveMode, aggregate_lifetime_reward
from charging_opt.profile_simulator import ProfileSimulator


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


class FamilyBayesianOptimizer:
    """GP-BO over one :class:`ChargingProfileFamily` search space."""

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
        self.search_space = family.search_space()
        self.history: List[Dict] = []

    def _evaluate(self, x: List[float]) -> float:
        params = self.family.from_vector(x)
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
        loss = float(metrics.get("loss", 1e6))
        self.history.append({
            "family_id": self.family_id,
            "params": params.to_dict(),
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
        n_calls: int = 30,
        n_initial_points: int = 8,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> FamilyOptimizationResult:
        seeds = self.family.seed_points()
        n_seed = len(seeds)
        n_random = max(0, n_initial_points - n_seed)

        result = gp_minimize(
            self._evaluate,
            dimensions=self.search_space,
            n_calls=max(n_calls, n_seed),
            n_initial_points=n_random,
            x0=seeds if seeds else None,
            random_state=self.random_state,
            acq_func="EI",
        )

        entry = self._best_feasible_entry()
        if entry is None:
            entry = min(self.history, key=lambda h: h["loss"])
            print(
                f"WARNING [{self.family_id}]: no feasible profile — "
                "returning least-bad infeasible candidate."
            )

        best_params = ProfileParams.from_dict(entry["params"])
        best_session = self.simulator.simulate_params(
            self.initial_state, best_params, family=self.family,
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
    on_family_done: Optional[Callable[[Dict[str, FamilyOptimizationResult]], None]] = None,
) -> Dict[str, FamilyOptimizationResult]:
    results: Dict[str, FamilyOptimizationResult] = {}
    for fid in family_ids:
        family = get_family(fid)
        print(f"\n{'=' * 60}\n  Optimizing family: {family.label} ({fid})\n{'=' * 60}")
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
