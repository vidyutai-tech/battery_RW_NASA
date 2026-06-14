#!/usr/bin/env python3
"""
Pareto front densification — second-stage BO targeting sparse duration gaps.

Usage
-----
    venv/bin/python scripts/run_pareto_refinement.py \\
        --results_json outputs/charging_opt_user/hima/stage3_optimization/models/family_optimization_results.json \\
        --family pulsed --n_targets 3 --n_calls 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from charging_opt.artifacts import CANONICAL, resolve_bdt_ckpt
from charging_opt.charging_profile_family import get_family
from charging_opt.family_optimizer import FamilyBayesianOptimizer
from charging_opt.lifetime_reward import LifetimeWeights, aggregate_lifetime_reward
from charging_opt.pareto_refiner import duration_constrained_loss, refinement_targets_from_results
from charging_opt.profile_simulator import ProfileSimulator


def main() -> None:
    p = argparse.ArgumentParser(description="Pareto gap densification via targeted BO.")
    p.add_argument("--results_json", required=True)
    p.add_argument("--family", default="pulsed")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument("--n_targets", type=int, default=3)
    p.add_argument("--n_calls", type=int, default=20)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--soc", type=float, default=0.15)
    p.add_argument("--v0", type=float, default=3.711)
    p.add_argument("--t0", type=float, default=24.7)
    p.add_argument("--age", type=float, default=0.0)
    args = p.parse_args()

    payload = json.loads(Path(args.results_json).read_text())
    jobs = refinement_targets_from_results(payload, n_targets=args.n_targets)
    start = {"soc": args.soc, "v0": args.v0, "t0": args.t0, "age": args.age, "prev_i": 0.0}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = ProfileSimulator(
        bdt_path=resolve_bdt_ckpt(args.bdt_ckpt, root=ROOT),
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        soc_target=0.95,
    )
    family = get_family(args.family)
    refinement_results = []

    for job in jobs:
        target_dur = job["target_duration_min"]
        print(f"\nRefining toward duration ~{target_dur:.1f} min …")

        class _RefineOptimizer(FamilyBayesianOptimizer):
            def _evaluate(self, x):
                params = self.family.from_vector(x)
                session = self.simulator.simulate_params(
                    self.initial_state, params, family=self.family,
                )
                _, metrics = aggregate_lifetime_reward(
                    session,
                    soc_target=self.soc_target,
                    max_duration_min=self.max_duration_min,
                    weights=self.weights,
                )
                base = float(metrics.get("loss", 1e6))
                if metrics.get("feasible") and metrics.get("duration_min") is not None:
                    loss = duration_constrained_loss(
                        base, float(metrics["duration_min"]),
                        target_duration=target_dur,
                    )
                else:
                    loss = base
                metrics = dict(metrics)
                metrics["loss"] = loss
                metrics["target_duration_min"] = target_dur
                self.history.append({
                    "family_id": self.family_id,
                    "params": params.to_dict(),
                    "loss": loss,
                    "feasible": bool(metrics.get("feasible", False)),
                    "metrics": metrics,
                })
                return loss

        opt = _RefineOptimizer(
            sim, family, start,
            weights=LifetimeWeights(),
            acq_func="PI",
        )
        result = opt.optimize(n_calls=args.n_calls, n_initial_points=5)
        m = result.best_metrics
        refinement_results.append({
            "target_duration_min": target_dur,
            "best_loss": result.best_loss,
            "best_params": result.best_params.to_dict(),
            "duration_min": m.get("duration_min"),
            "sei_per_pct_soc": m.get("sei_per_pct_soc"),
            "feasible": m.get("feasible"),
        })
        print(
            f"  -> dur={m.get('duration_min', float('nan')):.1f} min  "
            f"SEI={m.get('sei_per_pct_soc', float('nan')):.1f}"
        )

    out_path = out_dir / "pareto_refinement.json"
    out_path.write_text(json.dumps({"jobs": jobs, "results": refinement_results}, indent=2, default=float))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
