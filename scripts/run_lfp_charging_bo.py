#!/usr/bin/env python3
"""
Run constrained charging-profile optimization for LFP (finetuned BDT).

Steps performed automatically:
  1. Fit OCV–SoC from 1C Reference discharge (if missing)
  2. Random-search over profile families (CCCV, CC-taper, pulsed, …)
  3. Write ``constrained_bo_results.json`` + ``best_profiles.png``

LFP uses **SoC target** constraint by default (not energy J), because the
cross-chemistry BDT keeps voltage near the plateau during charge — Coulomb
SoC reaches the stop target before ∫V·I dt matches an energy budget.
Use ``--energy-fraction`` only if you explicitly want energy mode.

Usage
-----
    # Full run (default: SoC 20%→45%, frac0.40 finetuned BDT)
    python3 scripts/run_lfp_charging_bo.py

    # Quick smoke test
    python3 scripts/run_lfp_charging_bo.py --n-random 20

    # Energy mode (often infeasible until LFP BDT charge dynamics improve)
    python3 scripts/run_lfp_charging_bo.py --energy-fraction 0.20 --soc-target 0.40

    # Use a specific finetune run directory
    python3 scripts/run_lfp_charging_bo.py \\
        --finetune-run outputs/lfp_finetune/finetune_percent/20260707_083400
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

# Delegate to Constrained_BO.run after patching LFP finetune path if needed.
from Constrained_BO import config as bo_config
from Constrained_BO.lfp_ocv import fit_lfp_ocv_curve, ocv_curve_path
from Constrained_BO.run import main as bo_main


def _patch_finetune_root(run_dir: Path | None) -> None:
    if run_dir is None:
        return
    run_dir = Path(run_dir)
    if (run_dir / "registry").is_dir():
        bo_config.LFP_FINETUNE_RUN = run_dir
    elif run_dir.name == "registry":
        bo_config.LFP_FINETUNE_RUN = run_dir.parent
    else:
        bo_config.LFP_FINETUNE_ROOT = run_dir


def main() -> None:
    p = argparse.ArgumentParser(description="LFP charging-profile optimization (Constrained BO)")
    p.add_argument("--mat", default=str(bo_config.LFP_MAT), help="lfp_processed.mat path")
    p.add_argument(
        "--finetune-run",
        type=Path,
        default=None,
        help="LFP finetune run root (contains registry/finetune_LFP_frac*.pt)",
    )
    p.add_argument("--refit-ocv", action="store_true", help="Re-fit LFP OCV curve")
    p.add_argument("--out-dir", type=Path, default=None, help="Output directory")
    p.add_argument("--n-random", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--energy-fraction",
        type=float,
        default=None,
        help="Fraction of pack energy (J) to deliver; disables --soc-target when set",
    )
    p.add_argument(
        "--soc-target",
        type=float,
        default=0.45,
        help="SoC stop target for classic constraint mode (default 0.45 for LFP)",
    )
    p.add_argument("--max-duration-min", type=float, default=150.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--w-soc", type=float, default=1.0)
    p.add_argument("--w-qloss", type=float, default=1.0)
    p.add_argument("--w-time", type=float, default=0.1)
    p.add_argument("--w-temperature", type=float, default=1.0)
    p.add_argument("--z", type=float, default=0.55)
    p.add_argument(
        "--reward-mode",
        choices=("hybrid_qloss", "legacy_temp_time"),
        default="hybrid_qloss",
    )
    args, extra = p.parse_known_args()

    bo_config.LFP_MAT = Path(args.mat)
    _patch_finetune_root(args.finetune_run)

    if args.refit_ocv or not ocv_curve_path("LFP").exists():
        print("Fitting LFP OCV–SoC curve …")
        fit_lfp_ocv_curve(mat_path=args.mat)

    argv = [
        "run_lfp_charging_bo.py",
        "--cell", "LFP",
        "--n-random", str(args.n_random),
        "--seed", str(args.seed),
        "--device", args.device,
        "--reward-mode", args.reward_mode,
        "--w-soc", str(args.w_soc),
        "--w-qloss", str(args.w_qloss),
        "--w-time", str(args.w_time),
        "--w-temperature", str(args.w_temperature),
        "--z", str(args.z),
        "--max-duration-min", str(args.max_duration_min),
    ]
    if args.energy_fraction is not None:
        argv.extend(["--energy-fraction", str(args.energy_fraction)])
    elif args.soc_target is not None:
        argv.extend(["--soc-target", str(args.soc_target)])
    if args.out_dir is not None:
        argv.extend(["--out-dir", str(args.out_dir)])
    if args.refit_ocv:
        argv.append("--refit-ocv")
    argv.extend(extra)

    sys.argv = argv
    bo_main()


if __name__ == "__main__":
    main()
