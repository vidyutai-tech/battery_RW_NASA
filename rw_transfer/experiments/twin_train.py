"""Train source digital twin (RW9) and SOC MLP variants."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from rw_transfer.config import load_config
from rw_transfer.data.author_dataset import (
    AuthorChunkDataset,
    author_subset_to_window_batch,
    random_split_author_dataset,
)
from rw_transfer.data.author_loader import load_author_stitched_series
from rw_transfer.data.preprocess import smooth_temperature_series
from rw_transfer.data.series import BatteryTimeSeries, load_battery_series, slice_battery_series
from rw_transfer.data.windows import build_twin_windows, split_windows_by_series_fraction
from rw_transfer.experiments.logging_utils import experiment_dir, save_json
from rw_transfer.experiments.soc_train import run_soc_train
from rw_transfer.training.twin_trainer import evaluate_twin_windows, trainer_from_twin_config
from rw_transfer.viz.plots import (
    plot_twin_training_curves,
    plot_twin_predictions,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_author_twin_train(
    cfg: Dict[str, Any],
    out_dir: Path,
    epochs_override: Optional[int],
) -> Dict[str, Any]:
    """Original author pipeline: all steps, non-overlapping chunks, random split."""
    twin_cfg = cfg["twin"]
    matlab_dir = cfg["data"]["matlab_dir"]
    cell = cfg["data"]["cells"]["source"]
    decimation = int(cfg["data"].get("decimation", 1))
    chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
    split_cfg = twin_cfg.get("author_split", {})
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))
    seed = int(cfg.get("seed", 42))

    print(f"  Pipeline     : author (all steps, chunk={chunk_size}, random split)")
    print(f"  Split        : train {train_frac:.0%} / val {val_frac:.0%} / test remainder")
    print(f"  Output       : {out_dir}\n")

    print(f"  [1/5] Loading {cell}.mat (all steps, author stitch) ...", flush=True)
    author_series = load_author_stitched_series(matlab_dir, cell, decimation=decimation)
    print(f"        {author_series.n_samples:,} samples  |  {author_series.duration_hours:.1f} h"
          f"  |  {author_series.n_steps} steps")

    print(f"  [2/5] Building non-overlapping chunks ...", flush=True)
    dataset = AuthorChunkDataset(author_series, chunk_size=chunk_size)
    train_set, val_set, test_set = random_split_author_dataset(
        dataset, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )
    print(f"        {len(dataset):,} chunks  →  train {len(train_set):,} "
          f"/ val {len(val_set):,} / test {len(test_set):,}")

    epochs = epochs_override if epochs_override is not None else twin_cfg["epochs"]
    print(f"\n  [3/5] Training digital twin  (epochs={epochs}, "
          f"lr={twin_cfg['lr']}, batch={twin_cfg['batch_size']}) ...", flush=True)
    trainer = trainer_from_twin_config(twin_cfg, seq_len=chunk_size)

    fit_info = trainer.fit_author(
        train_set,
        val_set,
        epochs=epochs,
        batch_size=twin_cfg["batch_size"],
        early_stop_patience=twin_cfg["early_stop_patience"],
        plateau_patience=int(twin_cfg.get("plateau_patience", 3)),
        plateau_factor=float(twin_cfg.get("plateau_factor", 0.1)),
        log_path=out_dir / "twin_train_log.jsonl",
        num_workers=int(twin_cfg.get("num_workers", 0)),
        pearson_temp_weight=float(twin_cfg.get("pearson_temp_weight", 0.0)),
    )

    ckpt = out_dir / "twin_source_RW9.pt"
    trainer.save(ckpt)
    print(f"        Best val MAPE (V / T) : {fit_info.get('best_val_mape_v', float('nan')):.3f}%"
          f" / {fit_info.get('best_val_mape_t', float('nan')):.3f}%")
    print(f"        Best val loss         : {fit_info.get('best_val_loss', float('nan')):.6f}")
    print(f"        Epochs run            : {fit_info['epochs_run']}")

    print(f"\n  [4/5] Evaluating on held-out test chunks ...", flush=True)
    test_batch = author_subset_to_window_batch(test_set)
    test_metrics = evaluate_twin_windows(trainer.model, test_batch, trainer.device)
    v_m = test_metrics.get("voltage", {})
    t_m = test_metrics.get("temperature", {})
    print(f"        Voltage  — RMSE {v_m.get('rmse','?'):.5f}  MAPE {v_m.get('mape_pct','?'):.3f}%")
    print(f"        Temp     — RMSE {t_m.get('rmse','?'):.5f}  MAPE {t_m.get('mape_pct','?'):.3f}%")

    print(f"\n  [5/5] Generating plots ...", flush=True)
    plot_twin_training_curves(out_dir / "twin_train_log.jsonl", out_dir / "twin_train_curves.png")
    plot_twin_predictions(
        trainer.model, test_batch, trainer.device,
        out_dir / "twin_train_predictions.png", n_panels=4,
        title_prefix=f"Digital twin source ({cell}, author pipeline)",
    )

    # SOC on chronological RW operational series (transfer study still uses rw_operational)
    series = load_battery_series(
        matlab_dir, cell,
        step_mode=cfg["data"].get("soc_step_mode", "rw_operational"),
        decimation=decimation,
    )
    return {
        "pipeline": "author",
        "source_cell": cell,
        "n_chunks": len(dataset),
        "n_samples": author_series.n_samples,
        "fit": fit_info,
        "test_metrics": test_metrics,
        "checkpoint": str(ckpt),
        "series_for_soc": series,
    }


def run_twin_train(
    config_path: Optional[str] = None,
    out_dir: Optional[Path] = None,
    epochs_override: Optional[int] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    _set_seed(seed)

    matlab_dir = cfg["data"]["matlab_dir"]
    cell = cfg["data"]["cells"]["source"]
    twin_cfg = cfg["twin"]
    pipeline = str(twin_cfg.get("pipeline", "author"))

    if out_dir is None:
        out_dir = experiment_dir(cfg["output"]["root"], "twin_source")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Stage 1 — Digital Twin source training")
    print(f"{'='*60}")
    print(f"  Source cell  : {cell}")

    test_metrics_raw = None
    author_meta: Optional[Dict[str, Any]] = None

    if pipeline == "author":
        author_meta = _run_author_twin_train(cfg, out_dir, epochs_override)
        series = author_meta.pop("series_for_soc")
        fit_info = author_meta["fit"]
        test_metrics = author_meta["test_metrics"]
        ckpt = Path(author_meta["checkpoint"])
        batch_n = author_meta["n_chunks"]
        step_mode = "all"
        temp_smooth_w = 0
    else:
        step_mode = cfg["data"]["step_mode"]
        seq_len = cfg["windows"]["seq_len"]
        stride = cfg["windows"]["stride"]
        decimation = int(cfg["data"].get("decimation", 1))
        print(f"  Step mode    : {step_mode}")
        print(f"  Decimation   : {decimation}x  (keep every {decimation}th sample)")
        print(f"  Seq len      : {seq_len}    Stride: {stride}")
        print(f"  Output       : {out_dir}\n")

        print(f"  [1/5] Loading {cell}.mat ...", flush=True)
        series = load_battery_series(matlab_dir, cell, step_mode=step_mode, decimation=decimation)
        print(f"        {len(series.time_s):,} samples  |  {series.duration_hours:.1f} h total")

        temp_smooth_w = int(twin_cfg.get("temp_smooth_series_window", 0))
        series_for_windows = series
        temp_override = None
        if temp_smooth_w > 0:
            series_smooth = smooth_temperature_series(series, window=temp_smooth_w)
            temp_override = series_smooth.temperature_c
            series_for_windows = series_smooth
            print(f"        Temperature: series Savitzky-Golay (window={temp_smooth_w})", flush=True)

        print(f"  [2/5] Building sliding windows (seq={seq_len}, stride={stride}) ...", flush=True)
        batch = build_twin_windows(
            series_for_windows, seq_len=seq_len, stride=stride, temperature_c=temp_override,
        )
        splits = split_windows_by_series_fraction(
            batch, cfg["splits"]["train_frac"], cfg["splits"]["val_frac"],
        )
        print(f"        {len(batch.X):,} windows  →  train {len(splits['train'].X):,} "
              f"/ val {len(splits['val'].X):,} / test {len(splits['test'].X):,}")

        epochs = epochs_override if epochs_override is not None else twin_cfg["epochs"]
        print(f"\n  [3/5] Training digital twin  (epochs={epochs}, "
              f"lr={twin_cfg['lr']}, batch={twin_cfg['batch_size']}) ...", flush=True)
        trainer = trainer_from_twin_config(twin_cfg, seq_len=seq_len)

        fit_info = trainer.fit(
            splits["train"],
            splits["val"],
            epochs=epochs,
            batch_size=twin_cfg["batch_size"],
            early_stop_patience=twin_cfg["early_stop_patience"],
            plateau_patience=int(twin_cfg.get("plateau_patience", 20)),
            smooth_temp_targets=bool(twin_cfg.get("smooth_temp_targets", False)),
            log_path=out_dir / "twin_train_log.jsonl",
            num_workers=int(twin_cfg.get("num_workers", 0)),
        )

        ckpt = out_dir / "twin_source_RW9.pt"
        trainer.save(ckpt)
        print(f"        Best val MAPE (V / T) : {fit_info.get('best_val_mape_v', float('nan')):.3f}%"
              f" / {fit_info.get('best_val_mape_t', float('nan')):.3f}%")
        print(f"        Best val RMSE (V)   : {fit_info['best_val_voltage_rmse']:.6f} V")
        print(f"        Epochs run            : {fit_info['epochs_run']}")
        print(f"        Checkpoint saved      : {ckpt}")

        print(f"\n  [4/5] Evaluating on held-out test set ...", flush=True)
        test_metrics = evaluate_twin_windows(trainer.model, splits["test"], trainer.device)
        test_metrics_raw = None
        if splits["test"].Y_temperature_raw is not None:
            test_metrics_raw = evaluate_twin_windows(
                trainer.model, splits["test"], trainer.device,
                temperature_ref=splits["test"].Y_temperature_raw,
            )
        v_m = test_metrics.get("voltage", {})
        t_m = test_metrics.get("temperature", {})
        print(f"        Voltage  — RMSE {v_m.get('rmse','?'):.5f}  MAE {v_m.get('mae','?'):.5f}"
              f"  MAPE {v_m.get('mape_pct','?'):.3f}%  R² {v_m.get('r2','?'):.4f}")
        print(f"        Temp     — RMSE {t_m.get('rmse','?'):.5f}  MAE {v_m.get('mae','?'):.5f}"
              f"  MAPE {t_m.get('mape_pct','?'):.3f}%  R² {t_m.get('r2','?'):.4f}"
              + ("  (smoothed targets)" if temp_smooth_w > 0 else ""))
        if test_metrics_raw:
            tr = test_metrics_raw.get("temperature", {})
            print(f"        Temp raw — MAPE {tr.get('mape_pct','?'):.3f}%  RMSE {tr.get('rmse','?'):.5f}"
                  f"  (unsmoothed sensor)")

        print(f"\n  [5/5] Generating plots ...", flush=True)
        plot_twin_training_curves(out_dir / "twin_train_log.jsonl", out_dir / "twin_train_curves.png")
        plot_twin_predictions(
            trainer.model, splits["test"], trainer.device,
            out_dir / "twin_train_predictions.png", n_panels=4,
            title_prefix=f"Digital twin source ({cell})",
        )
        batch_n = len(batch.X)
        print(f"        twin_train_curves.png  ✓")
        print(f"        twin_train_predictions.png  ✓")

    soc_summary = run_soc_train(
        config_path=config_path, out_dir=out_dir, require_twin_ckpt=False,
    )
    soc_results = soc_summary["soc_test_metrics"]

    summary = {
        "source_cell": cell,
        "pipeline": pipeline,
        "step_mode": step_mode if pipeline != "author" else "all",
        "duration_hours": series.duration_hours,
        "n_windows": batch_n,
        "fit": fit_info,
        "test_metrics": test_metrics,
        "soc_test_metrics": soc_results,
        "checkpoint": str(ckpt),
    }
    if author_meta:
        summary["author"] = {k: v for k, v in author_meta.items() if k != "series_for_soc"}
    if pipeline != "author":
        summary["test_metrics_temp_raw"] = test_metrics_raw
        summary["temp_smooth_series_window"] = temp_smooth_w
    save_json(out_dir / "twin_train_summary.json", summary)
    print(f"\n{'='*60}")
    print(f"  Stage 1 complete  —  {out_dir}")
    print(f"{'='*60}\n")
    return summary
