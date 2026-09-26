#!/usr/bin/env python
"""Phase 3: reward rescaling variants. Run only if Arm A failed."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELLS, output_root, read_json, run_gp_bo_cell, run_litmus, write_json,
    make_rescale_transform, make_simulator_ckpt, apply_i_max,
)
from aacopt.config import OptimizationSpec, RewardSpec  # noqa: E402
from aacopt.evaluate import cccv_params, evaluate_params, load_anchors, load_degradation_model  # noqa: E402


def halfc_baseline_metrics(cell: str, *, device: str, i_max_a: float) -> dict:
    apply_i_max(i_max_a)
    reward = RewardSpec.load()
    model = load_degradation_model()
    sim, state = make_simulator_ckpt(cell, device=device, i_max_a=i_max_a)
    params = cccv_params(0.5, v_cv=4.20, i_cutoff=0.05)
    _, metrics, _ = evaluate_params(
        sim, state, params, model=model, spec=reward, anchors=load_anchors(cell),
    )
    return {k: v for k, v in metrics.items() if k not in ("weights", "anchors")}


def run_variant(name: str, *, drop_soc: bool, device: str, i_max: float,
                cells, resume: bool) -> dict:
    root = output_root() / "03_armA_rescaled" / name
    gp_dir = root / "gp_bo"
    lit_dir = root / "litmus"
    gp_dir.mkdir(parents=True, exist_ok=True)
    baselines = {}
    for cell in cells:
        print(f"\n======== Phase 3 {name}  {cell} ========", flush=True)
        bl = halfc_baseline_metrics(cell, device=device, i_max_a=i_max)
        baselines[cell] = bl
        print(
            f"  0.5C CCCV terms: soc={bl['soc_reward']}  q={bl['qloss_penalty']}  "
            f"t={bl['time_penalty']}  R={bl['reward']}  feas={bl['feasible']}",
            flush=True,
        )
        transform = make_rescale_transform(bl, drop_soc=drop_soc)
        run_gp_bo_cell(
            cell, gp_dir / cell, device=device, i_max_a=i_max,
            resume=resume, loss_transform=transform,
        )
    write_json(root / "halfc_baselines.json", baselines)
    payload = run_litmus(gp_dir, lit_dir, device=device, i_max_a=i_max, cells=cells)
    write_json(root / "verdict.json", payload["criteria"])
    return payload["criteria"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--force", action="store_true", help="run even if Arm A passed")
    args = ap.parse_args()

    opt = OptimizationSpec.load()
    device = args.device or opt.device
    cells = [c.upper() for c in (args.cells or CELLS)]
    root = output_root()
    cfg = read_json(root / "config_used.json")
    i_max = float(cfg["i_max_a_phase2"])

    armA = root / "02_armA" / "armA_verdict.json"
    if armA.is_file() and not args.force:
        verdict = read_json(armA)
        if verdict.get("verdict") == "PASS":
            print("Arm A PASS — skipping Phase 3 as pre-registered.", flush=True)
            skip_dir = root / "03_armA_rescaled"
            skip_dir.mkdir(parents=True, exist_ok=True)
            (skip_dir / "SKIPPED.md").write_text(
                "Phase 3 skipped because Arm A passed C1–C3.\n", encoding="utf-8"
            )
            return 0

    out = root / "03_armA_rescaled"
    out.mkdir(parents=True, exist_ok=True)
    v1 = run_variant(
        "normalize_three_terms", drop_soc=False, device=device, i_max=i_max,
        cells=cells, resume=not args.no_resume,
    )
    v2 = run_variant(
        "normalize_drop_dsoc", drop_soc=True, device=device, i_max=i_max,
        cells=cells, resume=not args.no_resume,
    )
    summary = {
        "normalize_three_terms": v1,
        "normalize_drop_dsoc": v2,
        "any_pass": v1.get("verdict") == "PASS" or v2.get("verdict") == "PASS",
    }
    write_json(out / "summary.json", summary)
    (out / "SUMMARY.md").write_text(
        "# Phase 3 reward rescaling\n\n"
        f"normalize_three_terms: **{v1.get('verdict')}**\n\n"
        f"normalize_drop_dsoc (ΔSOC term removed; energy still a hard constraint): **{v2.get('verdict')}**\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
