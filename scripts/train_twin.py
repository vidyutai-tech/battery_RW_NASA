#!/usr/bin/env python3
"""Train RW9 source digital twin and SOC models (Phase 1)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Avoid matplotlib cache failures on full disks (set before any mpl import).
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from rw_transfer.experiments.twin_train import run_twin_train


def main() -> None:
    p = argparse.ArgumentParser(description="Train source digital twin on RW9")
    p.add_argument("--config", default=None, help="Path to YAML config (e.g. configs/high_mape.yaml)")
    p.add_argument("--out", default=None, help="Output directory")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    args = p.parse_args()
    summary = run_twin_train(args.config, out_dir=Path(args.out) if args.out else None,
                             epochs_override=args.epochs)
    v = summary["test_metrics"].get("voltage", {})
    t = summary["test_metrics"].get("temperature", {})
    print("Twin training complete.")
    print(f"  Test voltage MAPE : {v.get('mape_pct')}%")
    print(f"  Test temp MAPE    : {t.get('mape_pct')}%")
    print("  Checkpoint        :", summary["checkpoint"])


if __name__ == "__main__":
    main()
