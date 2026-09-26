#!/usr/bin/env python
"""Phase 2 Arm A: per-cell GP-BO with training-supported current cap."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELLS, OUTPUT_ROOT, output_root, read_json, run_gp_bo_cell, run_litmus, write_json,
)
from aacopt.config import OptimizationSpec, Paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--skip-gpbo", action="store_true")
    ap.add_argument("--skip-litmus", action="store_true")
    args = ap.parse_args()

    opt = OptimizationSpec.load()
    paths = Paths.load()
    device = args.device or opt.device
    cells = [c.upper() for c in (args.cells or CELLS)]
    root = output_root()
    cfg = read_json(root / "config_used.json")
    i_max = float(cfg["i_max_a_phase2"])
    out = root / "02_armA"
    gp_dir = out / "gp_bo"
    lit_dir = out / "litmus"
    gp_dir.mkdir(parents=True, exist_ok=True)
    lit_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_gpbo:
        for cell in cells:
            print(f"\n======== Arm A GP-BO  {cell}  i_max={i_max} ========", flush=True)
            run_gp_bo_cell(
                cell, gp_dir / cell, device=device, i_max_a=i_max,
                resume=not args.no_resume,
            )

    if not args.skip_litmus:
        payload = run_litmus(gp_dir, lit_dir, device=device, i_max_a=i_max, cells=cells)
        write_json(out / "armA_verdict.json", payload["criteria"])
        print(f"Arm A verdict: {payload['criteria']['verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
