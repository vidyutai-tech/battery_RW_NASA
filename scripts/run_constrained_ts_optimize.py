#!/usr/bin/env python3
"""
Constrained Thompson Sampling BO for one profile family (Enhancement 3b).

Usage
-----
    venv/bin/python scripts/run_constrained_ts_optimize.py \\
        --family cccv --n_calls 40 \\
        --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0
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
from charging_opt.constrained_ts_optimizer import ConstrainedTSOptimizer
from charging_opt.family_optimizer import FamilyBayesianOptimizer
from charging_opt.io_utils import current_user, user_stage3_root
from charging_opt.lifetime_reward import LifetimeWeights
from charging_opt.profile_simulator import ProfileSimulator


def main() -> None:
    p = argparse.ArgumentParser(description="Constrained-TS BO for one profile family.")
    p.add_argument("--family", default="cccv")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument("--n_calls", type=int, default=40)
    p.add_argument("--max_minutes", type=int, default=150)
    p.add_argument("--max_duration_min", type=float, default=105.0)
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--soc", type=float, default=0.15)
    p.add_argument("--v0", type=float, default=3.711)
    p.add_argument("--t0", type=float, default=24.7)
    p.add_argument("--age", type=float, default=0.0)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--age_conditioning", action="store_true")
    args = p.parse_args()

    start = {
        "soc": args.soc,
        "v0": args.v0,
        "t0": args.t0,
        "age": args.age,
        "prev_i": 0.0,
    }
    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else user_stage3_root(ROOT, current_user()).parent / "constrained_ts" / args.family
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    bdt_path = resolve_bdt_ckpt(args.bdt_ckpt, root=ROOT)
    sim = ProfileSimulator(
        bdt_path=bdt_path,
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=args.max_minutes,
        soc_target=args.soc_target,
    )
    family = get_family(args.family)

    bo = FamilyBayesianOptimizer(
        sim,
        family,
        start,
        soc_target=args.soc_target,
        max_duration_min=args.max_duration_min,
        weights=LifetimeWeights(),
        use_age_conditioning=args.age_conditioning,
    )

    bounds = [(float(dim.low), float(dim.high)) for dim in family.search_space()]
    seeds = family.seed_points()

    def objective_fn(x: list[float]) -> tuple[float, bool]:
        loss = bo._evaluate(x)
        entry = bo.history[-1]
        return loss, bool(entry.get("feasible", False))

    cts = ConstrainedTSOptimizer(bounds, objective_fn, random_state=42)
    result = cts.optimize(n_calls=args.n_calls, x0=seeds)

    payload = {
        "family_id": args.family,
        "optimizer": "constrained_ts",
        "best_x": result.best_x,
        "best_loss": result.best_loss,
        "best_feasible": result.best_feasible,
        "history": result.history,
        "initial_state": start,
    }
    out_path = out_dir / f"constrained_ts_{args.family}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))
    print(f"Best loss={result.best_loss:.2f}  feasible={result.best_feasible}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
