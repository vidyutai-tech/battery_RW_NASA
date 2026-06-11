#!/usr/bin/env python3
"""
Re-evaluate a saved finetuned checkpoint and regenerate all metrics + plots.

This script does NOT retrain anything.  It loads an existing checkpoint from
<run_dir>/registry/finetune_<target>_frac<X>.pt, runs it against the same
held-out test split (seed-fixed 20% of chunks), and writes:

  <run_dir>/plots/actual_vs_pred_<target>_frac<X>.png
  <run_dir>/plots/finetune_curves_<target>_frac<X>_stage1.png   (if log exists)
  <run_dir>/plots/finetune_curves_<target>_frac<X>_stage2.png   (if log exists)
  <run_dir>/registry/finetune_registry.json                     (updated entry)

Usage
-----
    # Single fraction
    python scripts/evaluate_finetune.py \\
        --run_dir outputs/finetune_two_stage_RW10 \\
        --source_ckpt outputs/twin_source/20260601_182816/twin_source_RW9.pt \\
        --target RW10 --fraction 0.60

    # All fractions found in registry/
    python scripts/evaluate_finetune.py \\
        --run_dir outputs/finetune_two_stage_RW10 \\
        --source_ckpt outputs/twin_source/20260601_182816/twin_source_RW9.pt \\
        --target RW10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import numpy as np
import torch

from rw_transfer.config import load_config
from rw_transfer.data.author_dataset import (
    AuthorChunkDataset,
    author_subset_to_window_batch,
    random_split_author_dataset,
)
from rw_transfer.data.author_loader import load_author_stitched_series
from rw_transfer.metrics import metric_bundle
from rw_transfer.registry import FinetuneRegistry, file_size_mb, measure_infer_ms
from rw_transfer.training.twin_trainer import TwinTrainer, predict_twin_batch
from rw_transfer.viz.plots import plot_actual_vs_predicted, plot_finetune_training_curves


def _load_test_batch(cfg, target: str):
    """Reproduce the exact test split used during finetuning (seed 42, 60/20/20)."""
    matlab_dir = cfg["data"]["matlab_dir"]
    decimation = int(cfg["data"].get("decimation", 1))
    twin_cfg   = cfg["twin"]
    chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
    split_cfg  = twin_cfg.get("author_split", {})
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac   = float(split_cfg.get("val_frac",   0.2))
    seed       = int(cfg.get("seed", 42))

    stitched = load_author_stitched_series(matlab_dir, target, decimation=decimation)
    dataset  = AuthorChunkDataset(stitched, chunk_size=chunk_size)
    _, _, test_set = random_split_author_dataset(
        dataset, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )
    return author_subset_to_window_batch(test_set, max_windows=None), chunk_size


def _evaluate_one(
    ckpt_path: Path,
    test_batch,
    chunk_size: int,
    target: str,
    fraction: float,
    *,
    plots_dir: Path,
    registry_dir: Path,
    source_ckpt: Path,
) -> None:
    print(f"\n  Evaluating  {ckpt_path.name}  ({target} {fraction:.0%}) …", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = TwinTrainer.load(ckpt_path, seq_len=chunk_size)
    trainer.model.eval()

    v_pred, t_pred = predict_twin_batch(trainer.model, test_batch, trainer.device)
    v_ref = test_batch.Y_voltage
    t_ref = test_batch.Y_temperature

    v_metrics = metric_bundle(v_pred.ravel(), v_ref.ravel())
    t_metrics = metric_bundle(t_pred.ravel(), t_ref.ravel())

    print(f"    Voltage  : RMSE={v_metrics['rmse']:.5f}  MAPE={v_metrics['mape_pct']:.3f}%  "
          f"MAE={v_metrics['mae']:.5f}  R²={v_metrics['r2']:.4f}")
    print(f"    Temp     : RMSE={t_metrics['rmse']:.4f}  MAPE={t_metrics['mape_pct']:.3f}%  "
          f"MAE={t_metrics['mae']:.4f}  R²={t_metrics['r2']:.4f}")

    size_mb  = file_size_mb(ckpt_path)
    n_params = trainer.model.n_trainable_params
    infer_ms = measure_infer_ms(trainer.model, trainer.device, seq_len=chunk_size)
    print(f"    Size={size_mb:.3f} MB  Params={n_params:,}  Infer={infer_ms:.3f} ms/chunk")

    # ── plots ─────────────────────────────────────────────────────────────────
    frac_tag = f"{target}_frac{fraction:.2f}"
    plot_actual_vs_predicted(
        v_pred.ravel(), v_ref.ravel(),
        t_pred.ravel(), t_ref.ravel(),
        plots_dir / f"actual_vs_pred_{frac_tag}.png",
        target=target,
        fraction=fraction,
    )
    print(f"    Saved → plots/actual_vs_pred_{frac_tag}.png")

    for stage in ("stage1", "stage2"):
        log = registry_dir / f"train_log_{frac_tag}_{stage}.jsonl"
        out = plots_dir / f"finetune_curves_{frac_tag}_{stage}.png"
        if log.exists():
            plot_finetune_training_curves(
                log, out,
                stage_label=f"{target} {fraction:.0%} — {stage.replace('stage','Stage ')}",
            )
            print(f"    Saved → plots/finetune_curves_{frac_tag}_{stage}.png")

    # ── update registry ────────────────────────────────────────────────────────
    reg = FinetuneRegistry(registry_dir, source_ckpt)
    reg.register_fraction(
        target=target,
        fraction=fraction,
        n_adapt=0,
        n_eval=len(test_batch.X),
        train_time_s=0.0,
        model_size_mb=size_mb,
        n_params=n_params,
        infer_ms=infer_ms,
        voltage_metrics=v_metrics,
        temperature_metrics=t_metrics,
        ckpt_path=ckpt_path,
    )
    reg.save()
    print(f"    Registry updated → {registry_dir}/finetune_registry.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Re-evaluate finetuned checkpoint(s).")
    p.add_argument("--run_dir",    required=True,  help="Finetune output dir, e.g. outputs/finetune_two_stage_RW10")
    p.add_argument("--source_ckpt", required=True, help="Path to twin_source_RW9.pt")
    p.add_argument("--target",     default="RW10", help="Target cell, e.g. RW10")
    p.add_argument("--fraction",   default=None,   type=float,
                   help="Specific fraction to re-evaluate (default: all found checkpoints)")
    p.add_argument("--config",     default=None,   help="Config YAML path (default: configs/default.yaml)")
    args = p.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["cells"]["targets"] = [args.target]

    run_dir      = Path(args.run_dir)
    plots_dir    = run_dir / "plots"
    registry_dir = run_dir / "registry"
    source_ckpt  = Path(args.source_ckpt)
    plots_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    test_batch, chunk_size = _load_test_batch(cfg, args.target)
    print(f"\n  Test batch : {len(test_batch.X):,} windows  (chunk size {chunk_size})")

    if args.fraction is not None:
        fracs = [args.fraction]
    else:
        # auto-discover all finetuned checkpoints in registry/
        pattern = f"finetune_{args.target}_frac*.pt"
        found = sorted(registry_dir.glob(pattern))
        if not found:
            print(f"  No checkpoints matching {pattern} in {registry_dir}")
            return
        fracs = []
        for p_ in found:
            stem = p_.stem  # finetune_RW10_frac0.60
            try:
                fracs.append(float(stem.split("frac")[1]))
            except (IndexError, ValueError):
                pass

    for frac in sorted(fracs):
        ckpt_path = registry_dir / f"finetune_{args.target}_frac{frac:.2f}.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: checkpoint not found: {ckpt_path} — skipping")
            continue
        _evaluate_one(
            ckpt_path, test_batch, chunk_size,
            target=args.target, fraction=frac,
            plots_dir=plots_dir,
            registry_dir=registry_dir,
            source_ckpt=source_ckpt,
        )

    # print final summary
    reg = FinetuneRegistry(registry_dir, source_ckpt)
    reg.print_summary()


if __name__ == "__main__":
    main()
