#!/usr/bin/env python3
"""
Profile the existing RW9 source checkpoint — no retraining.

Reads the saved .pt file, counts parameters, measures model size and inference
latency, and optionally evaluates on the held-out test set.  Writes
  <run_dir>/source_registry.json

Usage
-----
    # Profile only (no test set evaluation)
    python scripts/build_source_registry.py \\
        --run_dir outputs/twin_source/20260601_182816

    # Profile + test evaluation (reproduces the exact 20% test split)
    python scripts/build_source_registry.py \\
        --run_dir outputs/twin_source/20260601_182816 \\
        --evaluate
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import torch

from rw_transfer.config import load_config
from rw_transfer.registry import SourceModelRegistry


def main() -> None:
    p = argparse.ArgumentParser(description="Profile existing RW9 checkpoint → source_registry.json")
    p.add_argument("--run_dir",  required=True,
                   help="Source training run dir, e.g. outputs/twin_source/20260601_182816")
    p.add_argument("--config",   default=None, help="Config YAML (default: configs/default.yaml)")
    p.add_argument("--evaluate", action="store_true",
                   help="Also evaluate on held-out test set (loads data — takes ~1 min)")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    ckpt    = run_dir / "twin_source_RW9.pt"
    log     = run_dir / "twin_train_log.jsonl"

    if not ckpt.exists():
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Source checkpoint : {ckpt}")
    print(f"  Device            : {device}")

    reg = SourceModelRegistry(run_dir)

    v_met = t_met = None
    if args.evaluate:
        cfg = load_config(args.config)
        cell       = cfg["data"]["cells"]["source"]
        matlab_dir = cfg["data"]["matlab_dir"]
        decimation = int(cfg["data"].get("decimation", 1))
        twin_cfg   = cfg["twin"]
        chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
        split_cfg  = twin_cfg.get("author_split", {})
        train_frac = float(split_cfg.get("train_frac", 0.6))
        val_frac   = float(split_cfg.get("val_frac",   0.2))
        seed       = int(cfg.get("seed", 42))

        print(f"\n  Loading {cell}.mat for test evaluation …", flush=True)
        from rw_transfer.data.author_dataset import (
            AuthorChunkDataset, author_subset_to_window_batch, random_split_author_dataset,
        )
        from rw_transfer.data.author_loader import load_author_stitched_series
        from rw_transfer.metrics import metric_bundle
        from rw_transfer.training.twin_trainer import TwinTrainer, predict_twin_batch

        stitched   = load_author_stitched_series(matlab_dir, cell, decimation=decimation)
        dataset    = AuthorChunkDataset(stitched, chunk_size=chunk_size)
        _, _, test_set = random_split_author_dataset(
            dataset, train_frac=train_frac, val_frac=val_frac, seed=seed,
        )
        test_batch = author_subset_to_window_batch(test_set, max_windows=None)
        print(f"  Test windows : {len(test_batch.X):,}", flush=True)

        trainer  = TwinTrainer.load(ckpt, seq_len=chunk_size)
        v_pred, t_pred = predict_twin_batch(trainer.model, test_batch, trainer.device)
        v_met = metric_bundle(v_pred.ravel(), test_batch.Y_voltage.ravel())
        t_met = metric_bundle(t_pred.ravel(), test_batch.Y_temperature.ravel())
        print(
            f"  Voltage : RMSE={v_met['rmse']:.5f}  MAPE={v_met['mape_pct']:.3f}%  R²={v_met['r2']:.4f}"
        )
        print(
            f"  Temp    : RMSE={t_met['rmse']:.4f}  MAPE={t_met['mape_pct']:.3f}%  R²={t_met['r2']:.4f}"
        )

    cfg_seq = 150
    try:
        raw = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg_seq = int(raw.get("seq_len", 150))
    except Exception:
        pass

    reg.build_from_checkpoint(
        ckpt, device,
        seq_len=cfg_seq,
        test_voltage_metrics=v_met,
        test_temp_metrics=t_met,
        train_log_path=log if log.exists() else None,
    )
    reg.save()
    reg.print_summary()


if __name__ == "__main__":
    main()
