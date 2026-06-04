#!/usr/bin/env python3
"""
Generate publication-style twin + SOC figures for NASA RW experiments.

Outputs (default ``plots/``):

  digital_twin_validation.png
  digital_twin_validation_val_mean.png
  soc_estimation.png
  soc_variant_comparison.png

Usage
-----
    python scripts/visualize_twin.py
    python scripts/visualize_twin.py --ckpt outputs/twin_source/<run>/twin_source_RW9.pt
    python scripts/visualize_twin.py --ckpt_dir outputs/twin_source/<run> --out_dir plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from rw_transfer.config import load_config
from rw_transfer.data.author_dataset import AuthorChunkDataset, random_split_author_dataset
from rw_transfer.data.author_loader import load_author_stitched_series
from rw_transfer.data.series import load_battery_series
from rw_transfer.data.soc_labels import coulomb_soc_stitched_operational
from rw_transfer.data.series import slice_battery_series
from rw_transfer.training.soc_trainer import SOCTrainer, build_soc_arrays, soc_sample_indices
from rw_transfer.training.twin_trainer import TwinTrainer
from rw_transfer.viz.plots import plot_soc_prediction_series, plot_twin_training_curves
from rw_transfer.viz.twin_validation_plots import (
    compute_val_mean_trajectories,
    pick_best_validation_chunks,
    plot_digital_twin_validation,
    plot_digital_twin_validation_val_mean,
    plot_soc_estimation,
    plot_soc_variant_bars,
)


def _latest_ckpt(root: Path) -> Path:
    ckpts = sorted(root.glob("*/twin_source_RW9.pt")) if root.is_dir() else []
    if not ckpts:
        raise FileNotFoundError(f"No twin_source_RW9.pt under {root}")
    return ckpts[-1]


def run_visualize(
    config_path: str | None = None,
    ckpt_path: Path | None = None,
    out_dir: Path | None = None,
    n_panels: int = 3,
    burn_in: int = 5,
) -> None:
    cfg = load_config(config_path)
    twin_cfg = cfg["twin"]
    cell = cfg["data"]["cells"]["source"]
    matlab_dir = cfg["data"]["matlab_dir"]
    decimation = int(cfg["data"].get("decimation", 1))
    chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
    split_cfg = twin_cfg.get("author_split", {})
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))
    seed = int(cfg.get("seed", 42))

    if ckpt_path is None:
        ckpt_path = _latest_ckpt(Path(cfg["output"]["root"]) / "twin_source")
    ckpt_path = Path(ckpt_path)
    ckpt_dir = ckpt_path.parent

    if out_dir is None:
        out_dir = ckpt_dir / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  NASA RW — Twin & SOC visualization")
    print(f"{'='*60}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Output     : {out_dir}\n")

    print("  Loading stitched series …", flush=True)
    stitched = load_author_stitched_series(matlab_dir, cell, decimation=decimation)
    dataset = AuthorChunkDataset(stitched, chunk_size=chunk_size)
    train_set, val_set, test_set = random_split_author_dataset(
        dataset, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )
    print(f"  Chunks: train {len(train_set)} / val {len(val_set)} / test {len(test_set)}")

    print("  Loading twin checkpoint …", flush=True)
    trainer = TwinTrainer.load(ckpt_path)

    print("  [1] digital_twin_validation.png …", flush=True)
    samples = pick_best_validation_chunks(
        trainer, test_set, stitched, n=n_panels, burn_in=burn_in,
        age_min=0.25, age_max=0.75,
    )
    if not samples:
        samples = pick_best_validation_chunks(
            trainer, test_set, stitched, n=n_panels, burn_in=burn_in,
        )
    plot_digital_twin_validation(
        samples,
        out_dir / "digital_twin_validation.png",
        cell_id=cell,
        seq_len=chunk_size,
    )
    print(f"       Saved digital_twin_validation.png  ({len(samples)} panels)")

    print("  [1c] digital_twin_validation_val_mean.png …", flush=True)
    val_stats = compute_val_mean_trajectories(
        trainer, val_set, stitched, burn_in=burn_in, seed=seed,
    )
    if val_stats:
        plot_digital_twin_validation_val_mean(
            val_stats, out_dir / "digital_twin_validation_val_mean.png",
        )
        print("       Saved digital_twin_validation_val_mean.png")

    log_path = ckpt_dir / "twin_train_log.jsonl"
    if log_path.is_file():
        plot_twin_training_curves(log_path, out_dir / "twin_train_curves.png")
        print("       Saved twin_train_curves.png")

    print("  [SOC] Loading RW operational series for SOC plots …", flush=True)
    soc_step = cfg["data"].get("soc_step_mode", "rw_operational")
    series = load_battery_series(matlab_dir, cell, step_mode=soc_step, decimation=decimation)
    n = series.voltage_v.size
    n_train = int(n * cfg["splits"]["train_frac"])
    n_val = int(n * cfg["splits"]["val_frac"])
    test_sl = slice(n_train + n_val, n)
    test_series = slice_battery_series(series, test_sl)
    stride = int(cfg["data"].get("soc_label_stride", 200))

    soc_preds: dict = {}
    soc_results: dict = {}
    for variant in cfg["soc"]["variants"]:
        soc_ckpt = ckpt_dir / f"soc_train_{variant}.pt"
        if not soc_ckpt.is_file():
            print(f"       Skip {variant} — no {soc_ckpt.name}")
            continue
        X_te, y_te = build_soc_arrays(
            test_series,
            cfg["data"]["q_rated_as"],
            cfg["data"]["soc_q_norm"],
            variant,
            stride,
        )
        st = SOCTrainer.load(soc_ckpt)
        soc_preds[variant] = st.predict(X_te)
        soc_results[variant] = st.evaluate(X_te, y_te)
        r = soc_results[variant]
        print(f"       {variant}: test RMSE {r.get('rmse', float('nan')):.4f}  "
              f"MAPE {r.get('mape_pct', float('nan')):.2f}%")

    if soc_preds:
        soc_labels_full = coulomb_soc_stitched_operational(
            test_series,
            q_rated_as=cfg["data"]["q_rated_as"],
            q_norm=cfg["data"].get("soc_q_norm", "per_file"),
        )
        idx = soc_sample_indices(test_series, stride)
        plot_soc_estimation(
            test_series.time_s[idx],
            test_series.voltage_v[idx],
            soc_labels_full[idx],
            soc_preds,
            out_dir / "soc_estimation.png",
            cell_id=cell,
        )
        print("       Saved soc_estimation.png")

        plot_soc_prediction_series(
            test_series.time_s[idx],
            soc_labels_full[idx],
            soc_preds,
            out_dir / "soc_train_series.png",
            cell_id=cell,
        )
        print("       Saved soc_train_series.png")

        if soc_results:
            plot_soc_variant_bars(soc_results, out_dir / "soc_variant_comparison.png", cell_id=cell)
            print("       Saved soc_variant_comparison.png")

    print(f"\n{'='*60}")
    print(f"  Done — figures in {out_dir}")
    print(f"{'='*60}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="Twin + SOC visualization for NASA RW")
    p.add_argument("--config", default=None, help="YAML config path")
    p.add_argument("--ckpt", default=None, help="Path to twin_source_RW9.pt")
    p.add_argument("--ckpt_dir", default=None, help="Run directory (uses twin_source_RW9.pt inside)")
    p.add_argument("--out_dir", default=None, help="Output directory for PNGs")
    p.add_argument("--n_panels", type=int, default=3, help="Number of twin validation columns")
    p.add_argument("--burn_in", type=int, default=5, help="Burn-in steps before MAPE / plots")
    args = p.parse_args()

    ckpt = None
    if args.ckpt:
        ckpt = Path(args.ckpt)
    elif args.ckpt_dir:
        ckpt = Path(args.ckpt_dir) / "twin_source_RW9.pt"

    run_visualize(
        config_path=args.config,
        ckpt_path=ckpt,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        n_panels=args.n_panels,
        burn_in=args.burn_in,
    )


if __name__ == "__main__":
    main()
