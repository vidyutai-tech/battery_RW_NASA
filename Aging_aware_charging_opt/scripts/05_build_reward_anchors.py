#!/usr/bin/env python
"""STAGE 5 — freeze reward-normalization anchors from CCCV 1C.

Simulates the declared anchor protocol (CCCV at 1.0 C, V_cv=4.20 V) on each
cell's inherited BDT, scores it with the *frozen* calibrated degradation
model, and writes ``results/05_reward_anchors/reward_anchors.json``.

Both optimizers read this file and never recompute it.
Also stores CCCV 0.5C / 2C baselines for later comparison (not used as anchors).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.config import (
    OptimizationSpec, Paths, RewardSpec, file_hash, provenance, stage_dir, write_json,
)
from aacopt.evaluate import (
    cccv_params, evaluate_params, jsonable_session, load_degradation_model, make_simulator,
)
from aacopt.opt_driver import evaluate_baselines
from aacopt.profiles import bind_search

STAGE = "05_reward_anchors"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    bind_search()
    paths = Paths.load()
    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    model = load_degradation_model()
    cells = [c.upper() for c in (args.cells or paths.cells)]
    device = args.device or opt.device
    out_dir = stage_dir(STAGE)

    per_cell = {}
    all_ok = True
    for cell in cells:
        print(f"\n=== {cell} CCCV 1C anchor  device={device} ===", flush=True)
        sim, state = make_simulator(cell, device=device)
        params = cccv_params(1.0, v_cv=4.20, i_cutoff=0.05)
        loss, metrics, session = evaluate_params(
            sim, state, params, model=model, spec=reward,
        )
        q_ref = float(metrics["q_total"])
        t_ref = float(metrics["duration_h"])
        e_ref = float(metrics["e_delivered_j"])
        feasible = bool(metrics["feasible"])
        print(
            f"  R={metrics['reward']:.4f}  Q={q_ref:.6e}  t={metrics['duration_min']:.2f} min  "
            f"E={e_ref:.1f}/{metrics['e_required_j']:.1f} J  "
            f"ΔSOC={metrics['delta_soc']:.3f}  feas={feasible}  [{session['end_reason']}]",
            flush=True,
        )
        if not feasible or not np_finite(q_ref, t_ref, e_ref, metrics["reward"]):
            all_ok = False
        baselines = evaluate_baselines(sim, state, model=model, spec=reward, anchors=None)
        per_cell[cell] = {
            "Q_ref": q_ref,
            "t_ref_h": t_ref,
            "E_ref": e_ref,
            "E_required_j": float(metrics["e_required_j"]),
            "feasible": feasible,
            "end_reason": session.get("end_reason"),
            "params": params.to_dict(),
            "start_state": state,
            "v_nom": sim.v_nom,
            "q_rated_as": sim.q_rated_as,
            "metrics": {k: v for k, v in metrics.items() if k not in ("weights",)},
            "session": jsonable_session(session),
            "baselines": baselines,
            "checkpoint": str(paths.checkpoint(cell)),
            "checkpoint_sha256": file_hash(paths.checkpoint(cell)),
        }

    payload = {
        "note": "CCCV 1C bookkeeping only — not used to scale the reward.",
        "weights": reward.to_dict(),
        "per_cell": per_cell,
        "gate": "CCCV 1C feasible on all cells, anchors finite",
        "passed": bool(all_ok),
        "provenance": provenance(
            STAGE,
            configs=["paths", "degradation_fitted", "reward", "optimization"],
            inputs=[paths.checkpoint(c) for c in cells],
        ),
    }
    write_json(out_dir / "reward_anchors.json", payload)
    print(f"\nWrote {out_dir / 'reward_anchors.json'}")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


def np_finite(*vals) -> bool:
    import math
    return all(math.isfinite(float(v)) and float(v) > 0.0 for v in vals)


if __name__ == "__main__":
    raise SystemExit(main())
