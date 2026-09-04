#!/usr/bin/env python
"""STAGE 7 — Random Search, 80 evaluations per profile family."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.config import OptimizationSpec, Paths
from aacopt.opt_driver import run_cell


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    paths = Paths.load()
    opt = OptimizationSpec.load()
    cells = [c.upper() for c in (args.cells or paths.cells)]
    device = args.device or opt.device
    ok = True
    for cell in cells:
        print(f"\n======== Random Search  {cell} ========", flush=True)
        summary = run_cell(
            cell, "random_search", device=device,
            families=args.families, resume=not args.no_resume,
        )
        if not summary["any_feasible"]:
            ok = False
            print(f"FAIL {cell}: no feasible point")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
