#!/usr/bin/env python3
"""Train SOC MLPs only (measured V/T) into an existing twin run directory."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from rw_transfer.experiments.soc_train import run_soc_train


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train SOC models on measured V/T (after digital twin is trained)",
    )
    p.add_argument(
        "--run-dir",
        required=True,
        help="Existing twin run folder, e.g. outputs/twin_source/20260601_182816",
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML config (default: configs/default.yaml)",
    )
    p.add_argument(
        "--require-twin",
        action="store_true",
        help="Fail if twin_source_*.pt is missing (SOC training never uses twin outputs)",
    )
    args = p.parse_args()

    summary = run_soc_train(
        config_path=args.config,
        out_dir=Path(args.run_dir),
        require_twin_ckpt=args.require_twin,
    )
    print("\nSOC training complete.")
    for variant, m in summary["soc_test_metrics"].items():
        print(
            f"  {variant}: test RMSE {m['rmse']:.4f}  "
            f"MAPE {m['mape_pct']:.2f}%  R² {m['r2']:.4f}"
        )
    run = Path(args.run_dir)
    print(f"  Summary : {run / 'soc_train_summary.json'}")
    print(f"  Train log: {run / 'soc_train_log.jsonl'}")
    print(f"  Curves  : {run / 'soc_train_curves.png'}")


if __name__ == "__main__":
    main()
