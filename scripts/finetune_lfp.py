#!/usr/bin/env python3
"""Fine-tune NASA BDT on LFP data (percentage sweep, two-stage)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from lfp_transfer.experiments.lfp_finetune_percent import run_lfp_finetune_percent


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune NASA BDT on LFP (lfp_processed.mat)")
    p.add_argument("--config", default="configs/lfp_finetune.yaml")
    p.add_argument(
        "--source_ckpt",
        default=None,
        help="NASA source checkpoint (twin_source_RW9.pt). "
        "Defaults to newest under outputs/twin_source/ or outputs/temp_aware/twin_source/",
    )
    p.add_argument("--out", default=None, help="Output directory")
    p.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=None,
        help="Adaptation fractions, e.g. --fractions 0.20 0.40",
    )
    args = p.parse_args()

    run_lfp_finetune_percent(
        args.config,
        source_ckpt=Path(args.source_ckpt) if args.source_ckpt else None,
        out_dir=Path(args.out) if args.out else None,
        fractions=args.fractions,
    )


if __name__ == "__main__":
    main()
