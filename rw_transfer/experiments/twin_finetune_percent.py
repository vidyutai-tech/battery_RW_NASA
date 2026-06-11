"""Percentage-based fine-tuning — cross-battery transfer (two-stage, no scratch)."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rw_transfer.config import load_config
from rw_transfer.data.windows import WindowBatch
from rw_transfer.data.author_dataset import (
    AuthorChunkDataset,
    author_subset_to_window_batch,
    random_split_author_dataset,
    subset_author_train_by_fraction,
)
from rw_transfer.data.author_loader import load_author_stitched_series
from rw_transfer.experiments.logging_utils import append_csv_row, experiment_dir, save_json
from rw_transfer.metrics import metric_bundle
from rw_transfer.registry import FinetuneRegistry, file_size_mb, measure_infer_ms
from rw_transfer.training.twin_trainer import TwinTrainer, evaluate_twin_windows, trainer_from_twin_config
from rw_transfer.viz.plots import (
    plot_actual_vs_predicted,
    plot_finetune_percent,
    plot_finetune_training_curves,
)


# ── Shared WindowBatch adapter (used by hours study) ─────────────────────────

def _adapt(
    trainer: "TwinTrainer",
    adapt_batch: WindowBatch,
    eval_batch: WindowBatch,
    epochs: int,
    batch_size: int,
    patience: int,
) -> Dict[str, Any]:
    """
    Fine-tune a trainer on ``adapt_batch`` (WindowBatch) and evaluate on ``eval_batch``.

    Used by the hours-based study (Phase 3). Splits a 10% slice of adapt_batch
    as internal validation. Uses ``trainer.fit()`` which supports the full
    temperature-aware loss (MSE_V + MSE_T + MAPE + Pearson).
    """
    if len(adapt_batch.X) < 2:
        return {"voltage_rmse": float("nan"), "temperature_rmse": float("nan"), "skipped": True}

    n = len(adapt_batch.X)
    n_val = max(1, int(0.1 * n))
    train_b = WindowBatch(
        adapt_batch.X[:-n_val],
        adapt_batch.Y_voltage[:-n_val],
        adapt_batch.Y_temperature[:-n_val],
        adapt_batch.window_start_idx[:-n_val],
    )
    val_b = WindowBatch(
        adapt_batch.X[-n_val:],
        adapt_batch.Y_voltage[-n_val:],
        adapt_batch.Y_temperature[-n_val:],
        adapt_batch.window_start_idx[-n_val:],
    )
    trainer.fit(
        train_b, val_b,
        epochs=epochs,
        batch_size=batch_size,
        early_stop_patience=patience,
    )
    return evaluate_twin_windows(trainer.model, eval_batch, trainer.device)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_source_ckpt(cfg: Dict[str, Any], source_ckpt: Optional[Path]) -> Path:
    if source_ckpt is not None:
        return Path(source_ckpt)
    p1_root = Path(cfg["output"]["root"]) / "twin_source"
    ckpts = sorted(p1_root.glob("*/twin_source_RW9.pt")) if p1_root.is_dir() else []
    if not ckpts:
        raise FileNotFoundError(
            "Run train_twin.py first or pass --source_ckpt path/to/twin_source_RW9.pt"
        )
    return ckpts[-1]


def _adapt_two_stage(
    source_ckpt: Path,
    train_set,
    val_set,
    test_batch,
    chunk_size: int,
    twin_cfg: Dict[str, Any],
    ft_cfg: Dict[str, Any],
    *,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fine-tune from source checkpoint using two-stage temperature-aware training.

    Stage 1 — backbone frozen, output head only, temperature-biased loss.
    Stage 2 — all layers, balanced voltage + temperature + Pearson loss.
    """
    if len(train_set) < 1:
        return {
            "voltage_rmse": float("nan"),
            "temperature_rmse": float("nan"),
            "skipped": True,
        }

    trainer = TwinTrainer.load(source_ckpt, seq_len=chunk_size)
    trainer.lr = float(ft_cfg.get("finetune_lr", 5e-7))

    fit_info = trainer.fit_two_stage_author(
        train_set,
        val_set,
        stage1_epochs=int(ft_cfg.get("stage1_epochs", 150)),
        stage1_voltage_weight=float(ft_cfg.get("stage1_voltage_weight", 1.0)),
        stage1_temp_weight=float(ft_cfg.get("stage1_temp_weight", 100.0)),
        stage1_pearson_weight=float(ft_cfg.get("stage1_pearson_weight", 5.0)),
        stage1_lr=ft_cfg.get("stage1_lr"),
        stage2_epochs=int(ft_cfg.get("stage2_epochs", 500)),
        stage2_voltage_weight=float(ft_cfg.get("stage2_voltage_weight", 10.0)),
        stage2_temp_weight=float(ft_cfg.get("stage2_temp_weight", 50.0)),
        stage2_pearson_weight=float(ft_cfg.get("stage2_pearson_weight", 5.0)),
        batch_size=int(twin_cfg["batch_size"]),
        early_stop_patience=int(twin_cfg["early_stop_patience"]),
        plateau_patience=int(ft_cfg.get("plateau_patience", twin_cfg.get("plateau_patience", 3))),
        plateau_factor=float(ft_cfg.get("plateau_factor", twin_cfg.get("plateau_factor", 0.1))),
        num_workers=int(twin_cfg.get("num_workers", 0)),
        log_path=log_path,
    )

    eval_metrics = evaluate_twin_windows(trainer.model, test_batch, trainer.device)
    return {
        "voltage_rmse": eval_metrics.get("voltage_rmse", float("nan")),
        "temperature_rmse": eval_metrics.get("temperature_rmse", float("nan")),
        "voltage_metrics": eval_metrics.get("voltage", {}),
        "temperature_metrics": eval_metrics.get("temperature", {}),
        "v_pred": eval_metrics.get("v_pred"),
        "v_ref":  eval_metrics.get("v_ref"),
        "t_pred": eval_metrics.get("t_pred"),
        "t_ref":  eval_metrics.get("t_ref"),
        "fit": fit_info,
        "trainer": trainer,
    }


def _row_from_metrics(
    target: str,
    frac: float,
    n_adapt: int,
    n_eval: int,
    ft_m: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "target": target,
        "fraction": frac,
        "pipeline": "author_two_stage",
        "n_adapt_windows": n_adapt,
        "n_eval_windows": n_eval,
        "finetune_voltage_rmse": ft_m.get("voltage_rmse", float("nan")),
        "finetune_temp_rmse": ft_m.get("temperature_rmse", float("nan")),
    }


def _run_author_finetune_percent(
    cfg: Dict[str, Any],
    source_ckpt: Path,
    out_dir: Path,
    fracs: List[float],
) -> List[Dict[str, Any]]:
    """Author pipeline: two-stage temperature-aware fine-tuning at each data fraction."""
    matlab_dir = cfg["data"]["matlab_dir"]
    targets = cfg["data"]["cells"]["targets"]
    twin_cfg = cfg["twin"]
    ft_cfg = dict(cfg.get("finetune_temp", {}))
    # finetune_lr lives under phase2 in the YAML, not finetune_temp — inject it so
    # _adapt_two_stage reads the right value instead of always using the hard default.
    ft_cfg.setdefault("finetune_lr", cfg.get("phase2", {}).get("finetune_lr", 5e-7))
    chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
    split_cfg = twin_cfg.get("author_split", {})
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))
    seed = int(cfg.get("seed", 42))
    decimation = int(cfg["data"].get("decimation", 1))
    all_rows: List[Dict[str, Any]] = []

    # ── output sub-directories ────────────────────────────────────────────────
    plots_dir    = out_dir / "plots"
    registry_dir = out_dir / "registry"
    plots_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    for target in targets:
        print(
            f"\n  Two-stage finetune — {target}  "
            f"(chunk={chunk_size}, split {train_frac:.0%}/{val_frac:.0%})",
            flush=True,
        )
        stitched = load_author_stitched_series(matlab_dir, target, decimation=decimation)
        dataset = AuthorChunkDataset(stitched, chunk_size=chunk_size)
        train_set, val_set, test_set = random_split_author_dataset(
            dataset, train_frac=train_frac, val_frac=val_frac, seed=seed,
        )
        test_batch = author_subset_to_window_batch(test_set, max_windows=None)
        print(
            f"        {len(dataset):,} chunks  →  "
            f"train {len(train_set):,} / val {len(val_set):,} / test {len(test_set):,}",
            flush=True,
        )

        registry = FinetuneRegistry(registry_dir, source_ckpt)

        for frac in fracs:
            adapt_train = subset_author_train_by_fraction(train_set, frac)
            frac_tag = f"{target}_frac{frac:.2f}"
            print(
                f"\n        fraction {frac:.0%}: adapt train chunks {len(adapt_train):,}",
                flush=True,
            )

            # ── JSONL log paths (Stage 1 and Stage 2 written by fit_two_stage_author) ──
            log_path = registry_dir / f"train_log_{frac_tag}.jsonl"

            t_start = time.perf_counter()
            ft_m = _adapt_two_stage(
                source_ckpt, adapt_train, val_set, test_batch,
                chunk_size, twin_cfg, ft_cfg,
                log_path=log_path,
            )
            train_time_s = time.perf_counter() - t_start

            if ft_m.get("skipped"):
                continue

            # ── save finetuned checkpoint ──────────────────────────────────────
            ckpt_save = registry_dir / f"finetune_{frac_tag}.pt"
            trainer: TwinTrainer = ft_m["trainer"]
            trainer.save(ckpt_save)
            size_mb  = file_size_mb(ckpt_save)
            n_params = trainer.model.n_trainable_params
            infer_ms = measure_infer_ms(trainer.model, trainer.device, seq_len=chunk_size)

            # ── full metric bundle from evaluate_twin_windows ──────────────────
            v_metrics = ft_m.get("voltage_metrics", {})
            t_metrics = ft_m.get("temperature_metrics", {})

            # If metric_bundle didn't include mse yet, add it here defensively
            if "mse" not in v_metrics and "rmse" in v_metrics:
                v_metrics = dict(v_metrics)
                v_metrics["mse"] = round(v_metrics["rmse"] ** 2, 8)
            if "mse" not in t_metrics and "rmse" in t_metrics:
                t_metrics = dict(t_metrics)
                t_metrics["mse"] = round(t_metrics["rmse"] ** 2, 8)

            # ── register ───────────────────────────────────────────────────────
            fit_info = ft_m.get("fit", {})
            registry.register_fraction(
                target=target,
                fraction=frac,
                n_adapt=len(adapt_train),
                n_eval=len(test_set),
                train_time_s=train_time_s,
                model_size_mb=size_mb,
                n_params=n_params,
                infer_ms=infer_ms,
                voltage_metrics=v_metrics,
                temperature_metrics=t_metrics,
                ckpt_path=ckpt_save,
                stage1_epochs_run=fit_info.get("stage1", {}).get("epochs_run", 0),
                stage2_epochs_run=fit_info.get("stage2", {}).get("epochs_run", 0),
            )
            registry.save()

            # ── training curve plots (Stage 1 and Stage 2) ────────────────────
            s1_log = Path(str(log_path).replace(".jsonl", "_stage1.jsonl"))
            s2_log = Path(str(log_path).replace(".jsonl", "_stage2.jsonl"))
            plot_finetune_training_curves(
                s1_log,
                plots_dir / f"finetune_curves_{frac_tag}_stage1.png",
                stage_label=f"{target} {frac:.0%} — Stage 1 (head warmup)",
            )
            plot_finetune_training_curves(
                s2_log,
                plots_dir / f"finetune_curves_{frac_tag}_stage2.png",
                stage_label=f"{target} {frac:.0%} — Stage 2 (full fine-tune)",
            )

            # ── actual vs predicted plot ───────────────────────────────────────
            from rw_transfer.training.twin_trainer import predict_twin_batch
            import numpy as np

            v_pred_arr, t_pred_arr = predict_twin_batch(
                trainer.model, test_batch, trainer.device,
            )
            plot_actual_vs_predicted(
                v_pred_arr.ravel(), test_batch.Y_voltage.ravel(),
                t_pred_arr.ravel(), test_batch.Y_temperature.ravel(),
                plots_dir / f"actual_vs_pred_{frac_tag}.png",
                target=target,
                fraction=frac,
            )

            # ── print and build row ────────────────────────────────────────────
            print(
                f"        Voltage RMSE: {ft_m['voltage_rmse']:.5f} V  "
                f"MAPE: {v_metrics.get('mape_pct', float('nan')):.3f}%  "
                f"R²: {v_metrics.get('r2', float('nan')):.4f}",
                flush=True,
            )
            print(
                f"        Temp    RMSE: {ft_m['temperature_rmse']:.4f} °C  "
                f"MAPE: {t_metrics.get('mape_pct', float('nan')):.3f}%  "
                f"R²: {t_metrics.get('r2', float('nan')):.4f}",
                flush=True,
            )
            print(
                f"        Train time : {train_time_s/60:.1f} min  "
                f"Model: {size_mb:.3f} MB  "
                f"Infer: {infer_ms:.3f} ms/chunk  "
                f"Params: {n_params:,}",
                flush=True,
            )

            row = _row_from_metrics(target, frac, len(adapt_train), len(test_set), ft_m)
            all_rows.append(row)

        # ── print registry summary once per target ─────────────────────────────
        registry.print_summary()

    return all_rows


# ── Public entry point ────────────────────────────────────────────────────────

def run_twin_finetune_percent(
    config_path: Optional[str] = None,
    source_ckpt: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    fractions: Optional[List[float]] = None,
    targets: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    if targets:
        cfg["data"]["cells"]["targets"] = list(targets)
    fracs = fractions or cfg["phase2"]["data_fractions"]

    if out_dir is None:
        out_dir = experiment_dir(cfg["output"]["root"], "finetune_percent")
    out_dir = Path(out_dir)
    source_ckpt = _resolve_source_ckpt(cfg, source_ckpt)

    print(f"\n{'='*60}")
    print(f"  Stage 2 — Fine-tuning transfer (two-stage, temp-aware)")
    print(f"{'='*60}")
    print(f"  Source ckpt  : {source_ckpt}")
    print(f"  Targets      : {cfg['data']['cells']['targets']}")
    print(f"  Fractions    : {fracs}")
    print(f"  Output       : {out_dir}")
    print(f"  Plots        : {out_dir}/plots/")
    print(f"  Registry     : {out_dir}/registry/\n")

    csv_path = out_dir / "finetune_percent_results.csv"
    fields = [
        "target", "fraction", "pipeline", "n_adapt_windows", "n_eval_windows",
        "finetune_voltage_rmse", "finetune_temp_rmse",
    ]

    all_rows = _run_author_finetune_percent(cfg, source_ckpt, out_dir, fracs)

    for row in all_rows:
        append_csv_row(csv_path, row, fields)

    # ── per-target RMSE vs fraction plot (saved to plots/) ────────────────────
    for target in cfg["data"]["cells"]["targets"]:
        target_rows = [r for r in all_rows if r["target"] == target]
        if target_rows:
            plot_finetune_percent(
                target_rows, target,
                out_dir / "plots" / f"twin_finetune_percent_{target}.png",
            )

    summary = {"source_ckpt": str(source_ckpt), "pipeline": "author_two_stage", "rows": all_rows}
    save_json(out_dir / "finetune_percent_summary.json", summary)

    print(f"\n{'='*60}")
    print(f"  Stage 2 complete  —  {out_dir}")
    print(f"  Plots    → {out_dir}/plots/")
    print(f"  Registry → {out_dir}/registry/")
    print(f"{'='*60}\n")
    return summary
