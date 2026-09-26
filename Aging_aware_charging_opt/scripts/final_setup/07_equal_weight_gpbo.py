#!/usr/bin/env python
"""GP-BO with equalized reward weights. Does not overwrite Arm A.

Weights make SOC, degradation, and time each about 1 on the 0.5C baseline:
  w_soc = 2.5, w_loss = 7500, w_time = 1.1
Seed is 42 for every cell and family. The whole-reward x1000 scale is off.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELLS, output_root, read_json, run_gp_bo_cell, write_json,
)

import aacopt.reward as reward_mod  # noqa: E402
from aacopt.config import OptimizationSpec, RewardSpec  # noqa: E402

W_SOC = 2.5
W_LOSS = 7500.0
W_TIME = 1.1
SEED = 42

_ORIG_LOAD = RewardSpec.load.__func__


def load_equal_reward() -> RewardSpec:
    spec = _ORIG_LOAD(RewardSpec)
    return replace(spec, w_soc=W_SOC, w_loss=W_LOSS, w_time=W_TIME)


def main() -> int:
    reward_mod.REWARD_SCALE = 1.0
    RewardSpec.load = classmethod(lambda cls: load_equal_reward())  # type: ignore[method-assign]

    opt = OptimizationSpec.load()
    device = opt.device
    root = output_root() / "07_equal_weight_gpbo"
    root.mkdir(parents=True, exist_ok=True)
    cfg = read_json(output_root(create=False) / "config_used.json")
    i_max = float(cfg["i_max_a_phase2"])
    spec = load_equal_reward()
    write_json(root / "weights.json", {
        "w_soc": W_SOC,
        "w_loss": W_LOSS,
        "w_time": W_TIME,
        "z_time": spec.z_time,
        "duration_loss_weight": spec.duration_loss_weight,
        "reward_scale": 1.0,
        "seed": SEED,
        "i_max_a": i_max,
    })
    for cell in CELLS:
        for fid in opt.families:
            seed = opt.seed_for(cell, fid, "gp_bo")
            if seed != SEED:
                raise SystemExit(f"expected seed {SEED}, got {seed} for {cell} {fid}")

    for cell in CELLS:
        print(f"\n======== equal-weight GP-BO  {cell}  seed={SEED}  i_max={i_max} ========", flush=True)
        run_gp_bo_cell(
            cell, root / cell, device=device, i_max_a=i_max,
            resume=True, include_session=False,
        )
    print(f"wrote {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
