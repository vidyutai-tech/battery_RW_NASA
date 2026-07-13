"""Percentage-based BDT fine-tuning on LFP data (two-stage, NASA source checkpoint)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from lfp_transfer.config import load_lfp_config
from lfp_transfer.data.lfp_loader import load_lfp_stitched_series
from rw_transfer.data.author_dataset import (
    AuthorChunkDataset,
    author_subset_to_window_batch,
    random_split_author_dataset,
    subset_author_train_by_fraction,
)
from rw_transfer.experiments.logging_utils import append_csv_row, experiment_dir, save_json
from rw_transfer.experiments.twin_finetune_percent import (
    _adapt_two_stage,
    _row_from_metrics,
)
from rw_transfer.registry import FinetuneRegistry, file_size_mb, measure_infer_ms
from rw_transfer.training.twin_trainer import TwinTrainer, predict_twin_batch
from rw_transfer.viz.plots import (
    plot_actual_vs_predicted,
    plot_finetune_percent,
    plot_finetune_training_curves,
)


def _resolve_lfp_source_ckpt(cfg: Dict[str, Any], source_ckpt: Optional[Path]) -> Path:
    if source_ckpt is not None:
        return Path(source_ckpt)

    cfg_ckpt = cfg.get("source", {}).get("checkpoint")
    if cfg_ckpt:
        project_root = Path(cfg["output"]["root"]).resolve().parents[1]
        p = Path(cfg_ckpt)
        if not p.is_absolute():
            p = project_root / p
        if p.is_file():
            return p

    project_root = Path(cfg["output"]["root"]).resolve().parents[1]
    search_roots = [
        project_root / "outputs" / "temp_aware" / "twin_source",
        project_root / "outputs" / "twin_source",
    ]
    for twin_root in search_roots:
        ckpts = sorted(twin_root.glob("*/twin_source_RW9.pt")) if twin_root.is_dir() else []
        if ckpts:
            return ckpts[-1]

    raise FileNotFoundError(
        "No NASA source checkpoint found. Train with scripts/train_twin.py or pass "
        "--source_ckpt path/to/twin_source_RW9.pt"
    )


def _run_lfp_finetune_percent(
    cfg: Dict[str, Any],
    source_ckpt: Path,
    out_dir: Path,
    fracs: List[float],
) -> List[Dict[str, Any]]:
    data_cfg = cfg["data"]
    mat_path = Path(data_cfg["lfp_mat"])
    cell_id = str(data_cfg.get("cell_id", "LFP"))
    comment_mode = str(data_cfg.get("comment_mode", "rw_only"))
    decimation = int(data_cfg.get("decimation", 1))

    twin_cfg = cfg["twin"]
    ft_cfg = dict(cfg.get("finetune_temp", {}))
    ft_cfg.setdefault("finetune_lr", cfg.get("phase2", {}).get("finetune_lr", 5e-7))
    chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
    split_cfg = twin_cfg.get("author_split", {})
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))
    seed = int(cfg.get("seed", 42))

    plots_dir = out_dir / "plots"
    registry_dir = out_dir / "registry"
    plots_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"\n  Two-stage finetune — {cell_id}  "
        f"(chunk={chunk_size}, split {train_frac:.0%}/{val_frac:.0%}, "
        f"comment_mode={comment_mode}, decimation={decimation})",
        flush=True,
    )
    stitched = load_lfp_stitched_series(
        mat_path,
        cell_id,
        comment_mode=comment_mode,
        decimation=decimation,
    )
    print(
        f"        {stitched.n_samples:,} samples  |  {stitched.duration_hours:.1f} h"
        f"  |  {stitched.n_steps} steps",
        flush=True,
    )

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
    all_rows: List[Dict[str, Any]] = []

    for frac in fracs:
        adapt_train = subset_author_train_by_fraction(train_set, frac)
        frac_tag = f"{cell_id}_frac{frac:.2f}"
        print(
            f"\n        fraction {frac:.0%}: adapt train chunks {len(adapt_train):,}",
            flush=True,
        )

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

        ckpt_save = registry_dir / f"finetune_{frac_tag}.pt"
        trainer: TwinTrainer = ft_m["trainer"]
        trainer.save(ckpt_save)
        size_mb = file_size_mb(ckpt_save)
        n_params = trainer.model.n_trainable_params
        infer_ms = measure_infer_ms(trainer.model, trainer.device, seq_len=chunk_size)

        v_metrics = ft_m.get("voltage_metrics", {})
        t_metrics = ft_m.get("temperature_metrics", {})
        if "mse" not in v_metrics and "rmse" in v_metrics:
            v_metrics = dict(v_metrics)
            v_metrics["mse"] = round(v_metrics["rmse"] ** 2, 8)
        if "mse" not in t_metrics and "rmse" in t_metrics:
            t_metrics = dict(t_metrics)
            t_metrics["mse"] = round(t_metrics["rmse"] ** 2, 8)

        fit_info = ft_m.get("fit", {})
        registry.register_fraction(
            target=cell_id,
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

        s1_log = Path(str(log_path).replace(".jsonl", "_stage1.jsonl"))
        s2_log = Path(str(log_path).replace(".jsonl", "_stage2.jsonl"))
        plot_finetune_training_curves(
            s1_log,
            plots_dir / f"finetune_curves_{frac_tag}_stage1.png",
            stage_label=f"{cell_id} {frac:.0%} — Stage 1 (head warmup)",
        )
        plot_finetune_training_curves(
            s2_log,
            plots_dir / f"finetune_curves_{frac_tag}_stage2.png",
            stage_label=f"{cell_id} {frac:.0%} — Stage 2 (full fine-tune)",
        )

        v_pred_arr, t_pred_arr = predict_twin_batch(
            trainer.model, test_batch, trainer.device,
        )
        plot_actual_vs_predicted(
            v_pred_arr.ravel(), test_batch.Y_voltage.ravel(),
            t_pred_arr.ravel(), test_batch.Y_temperature.ravel(),
            plots_dir / f"actual_vs_pred_{frac_tag}.png",
            target=cell_id,
            fraction=frac,
        )

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

        row = _row_from_metrics(cell_id, frac, len(adapt_train), len(test_set), ft_m)
        row["pipeline"] = "lfp_two_stage"
        all_rows.append(row)

    registry.print_summary()
    return all_rows


def run_lfp_finetune_percent(
    config_path: Optional[str] = None,
    source_ckpt: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    fractions: Optional[List[float]] = None,
) -> Dict[str, Any]:
    cfg = load_lfp_config(config_path)
    cell_id = str(cfg["data"].get("cell_id", "LFP"))
    fracs = fractions or cfg["phase2"]["data_fractions"]

    if out_dir is None:
        out_dir = experiment_dir(cfg["output"]["root"], "finetune_percent")
    out_dir = Path(out_dir)
    source_ckpt = _resolve_lfp_source_ckpt(cfg, source_ckpt)

    print(f"\n{'='*60}")
    print("  LFP fine-tuning — NASA BDT → LFP (two-stage, temp-aware)")
    print(f"{'='*60}")
    print(f"  Source ckpt  : {source_ckpt}")
    print(f"  LFP mat      : {cfg['data']['lfp_mat']}")
    print(f"  Cell id      : {cell_id}")
    print(f"  Fractions    : {fracs}")
    print(f"  Output       : {out_dir}")
    print(f"  Plots        : {out_dir}/plots/")
    print(f"  Registry     : {out_dir}/registry/\n")

    csv_path = out_dir / "finetune_percent_results.csv"
    fields = [
        "target", "fraction", "pipeline", "n_adapt_windows", "n_eval_windows",
        "finetune_voltage_rmse", "finetune_temp_rmse",
    ]

    all_rows = _run_lfp_finetune_percent(cfg, source_ckpt, out_dir, fracs)

    for row in all_rows:
        append_csv_row(csv_path, row, fields)

    if all_rows:
        plot_finetune_percent(
            all_rows, cell_id,
            out_dir / "plots" / f"lfp_finetune_percent_{cell_id}.png",
        )

    summary = {
        "source_ckpt": str(source_ckpt),
        "lfp_mat": cfg["data"]["lfp_mat"],
        "pipeline": "lfp_two_stage",
        "rows": all_rows,
    }
    save_json(out_dir / "finetune_percent_summary.json", summary)

    print(f"\n{'='*60}")
    print(f"  LFP fine-tune complete  —  {out_dir}")
    print(f"  Plots    → {out_dir}/plots/")
    print(f"  Registry → {out_dir}/registry/")
    print(f"{'='*60}\n")
    return summary
