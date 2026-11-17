#!/usr/bin/env python3
"""Fine-tune pretrained twin on target cells (percentage sweep)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from rw_transfer.experiments.twin_finetune_percent import run_twin_finetune_percent


def main() -> None:
    p = argparse.ArgumentParser(description="Percentage-based transfer: fine-tune vs scratch")
    p.add_argument("--config", default=None)
    p.add_argument("--source_ckpt", default=None, help="Path to twin_source_RW9.pt")
    p.add_argument("--out", default=None)
    p.add_argument(
        "--targets", nargs="+", default=None,
        help="Target cell(s) only, e.g. --targets RW10",
    )
    args = p.parse_args()
    run_twin_finetune_percent(
        args.config,
        source_ckpt=Path(args.source_ckpt) if args.source_ckpt else None,
        out_dir=Path(args.out) if args.out else None,
        targets=args.targets,
    )


if __name__ == "__main__":
    main()
