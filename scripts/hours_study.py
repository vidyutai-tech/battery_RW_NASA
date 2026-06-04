#!/usr/bin/env python3
"""Hours-based adaptation study — minimum data for effective transfer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from rw_transfer.experiments.twin_finetune_hours import run_twin_finetune_hours


def main() -> None:
    p = argparse.ArgumentParser(description="Minimum-hours adaptation study")
    p.add_argument("--config", default=None)
    p.add_argument("--source_ckpt", default=None, help="Path to twin_source_RW9.pt")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    summary = run_twin_finetune_hours(
        args.config,
        source_ckpt=Path(args.source_ckpt) if args.source_ckpt else None,
        out_dir=Path(args.out) if args.out else None,
    )
    for line in summary.get("practical_recommendation", []):
        print(line)


if __name__ == "__main__":
    main()
