#!/usr/bin/env python
"""Pack-level GP-BO (does not overwrite per-cell stage 8).

One shared charging-profile parameter vector per family, scored on every
cell twin. A trial is feasible only if ALL cells meet the energy target and
stay at or below 4.20 V. The GP minimises the worst-cell loss (minimax).

Writes:
  Aging_aware_charging_opt/results/08_gp_bo_pack/
    cccv.json, two_step.json, three_step.json, pulsed.json
    baselines.json, elite_top6.json, elite_top6.md, summary.json
    pack_family_best_profiles.png

Existing results/08_gp_bo/<CELL>/ files are not read or written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.config import OptimizationSpec, Paths, ensure_matplotlib_cache
from aacopt.pack_opt import run_pack


def main() -> int:
    ensure_matplotlib_cache()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-evals", type=int, default=None,
                    help="Override evaluations per family (default: config 80).")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    paths = Paths.load()
    opt = OptimizationSpec.load()
    cells = [c.upper() for c in (args.cells or paths.cells)]
    device = args.device or opt.device
    print(f"======== Pack GP-BO  cells={cells}  device={device} ========", flush=True)
    summary = run_pack(
        cells=cells,
        families=args.families,
        device=device,
        resume=not args.no_resume,
        n_evals=args.n_evals,
    )
    elite = summary.get("elite_top6") or []
    print(f"\nany pack-feasible family: {summary['any_feasible']}", flush=True)
    print(f"elite set size: {len(elite)}  → results/08_gp_bo_pack/elite_top6.md", flush=True)
    for row in elite:
        print(
            f"  #{row.get('rank')}  {row['family_id']:<11}  "
            f"loss={row['loss']:.4f}  feas={row['feasible']}  "
            f"worst={row.get('worst_cell')}",
            flush=True,
        )
    return 0 if summary["any_feasible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
