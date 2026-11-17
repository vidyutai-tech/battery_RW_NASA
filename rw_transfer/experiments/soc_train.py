"""Train SOC MLPs on measured V/T (Stage 2 only — twin checkpoint optional)."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from rw_transfer.config import load_config
from rw_transfer.data.series import load_battery_series, slice_battery_series
from rw_transfer.data.soc_labels import coulomb_soc_stitched_operational
from rw_transfer.experiments.logging_utils import save_json
from rw_transfer.training.soc_trainer import SOCTrainer, build_soc_arrays, soc_sample_indices
from rw_transfer.viz.plots import (
    plot_soc_prediction_series,
    plot_soc_training_curves,
    plot_soc_variant_comparison,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_soc_train(
    config_path: Optional[str] = None,
    out_dir: Optional[Path] = None,
    *,
    require_twin_ckpt: bool = False,
) -> Dict[str, Any]:
    """
    Train SOC variants on measured V/T with per-step Coulomb labels.

    Does **not** use digital-twin predictions — only measured V, T, age, |I| from ``.mat``.

    Parameters
    ----------
    out_dir
        Experiment run directory (e.g. ``outputs/twin_source/<timestamp>``).
        SOC checkpoints and plots are written here.
    require_twin_ckpt
        If True, require ``twin_source_<cell>.pt`` in ``out_dir`` (organizational only).
    """
    cfg = load_config(config_path)
    _set_seed(int(cfg.get("seed", 42)))

    matlab_dir = cfg["data"]["matlab_dir"]
    cell = cfg["data"]["cells"]["source"]
    decimation = int(cfg["data"].get("decimation", 1))
    step_mode = cfg["data"].get("soc_step_mode", "rw_operational")
    soc_cfg = cfg["soc"]
    soc_input = str(soc_cfg.get("soc_input", "measured")).lower()
    if soc_input != "measured":
        raise ValueError(
            f"SOC training requires soc_input='measured' (got {soc_input!r}). "
            "Twin-predicted V/T is for inference / optimization only."
        )

    if out_dir is None:
        raise ValueError("out_dir is required (path to an existing or new run folder)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    twin_ckpt = out_dir / f"twin_source_{cell}.pt"
    if require_twin_ckpt and not twin_ckpt.is_file():
        raise FileNotFoundError(
            f"Digital twin checkpoint not found: {twin_ckpt}\n"
            "Train the twin first, or pass an existing --run-dir."
        )

    print(f"\n{'='*60}")
    print("  Stage 2 — SOC MLP training (measured V/T)")
    print(f"{'='*60}")
    print(f"  Cell         : {cell}")
    print(f"  Step filter  : {step_mode}")
    print(f"  Labels       : per-step Coulomb (charge / discharge / rest carry)")
    print(f"  Features     : measured V, T, age{', |I|' if 'vta_i' in soc_cfg['variants'] else ''}")
    print(f"  Output       : {out_dir}\n")

    print(f"  [1/3] Loading {cell} ({step_mode}) ...", flush=True)
    series = load_battery_series(
        matlab_dir, cell, step_mode=step_mode, decimation=decimation,
    )
    print(f"        {len(series.time_s):,} samples  |  {series.duration_hours:.1f} h", flush=True)

    soc_full = coulomb_soc_stitched_operational(
        series,
        q_rated_as=cfg["data"]["q_rated_as"],
        q_norm=cfg["data"].get("soc_q_norm", "per_file"),
    )
    print(
        f"        Label stats: min={soc_full.min():.3f}  max={soc_full.max():.3f}  "
        f"std={soc_full.std():.4f}",
        flush=True,
    )

    n = series.voltage_v.size
    n_train = int(n * cfg["splits"]["train_frac"])
    n_val = int(n * cfg["splits"]["val_frac"])
    soc_label_stride = int(cfg["data"].get("soc_label_stride", 200))
    q_norm = cfg["data"].get("soc_q_norm", "per_file")

    log_path = out_dir / "soc_train_log.jsonl"
    log_every = int(soc_cfg.get("log_every", 50))
    if log_path.is_file():
        log_path.unlink()

    print(f"\n  [2/3] Training variants (chrono split "
          f"{cfg['splits']['train_frac']:.0%}/"
          f"{cfg['splits']['val_frac']:.0%}/test) ...", flush=True)
    print(f"        Log file: {log_path}  (every {log_every} epochs)", flush=True)

    soc_results: Dict[str, Any] = {}
    soc_preds_for_plot: Dict[str, np.ndarray] = {}

    for variant in soc_cfg["variants"]:
        print(f"        variant={variant} ...", flush=True)
        X_tr, y_tr = build_soc_arrays(
            slice_battery_series(series, slice(0, n_train)),
            cfg["data"]["q_rated_as"], q_norm, variant, soc_label_stride,
        )
        X_va, y_va = build_soc_arrays(
            slice_battery_series(series, slice(n_train, n_train + n_val)),
            cfg["data"]["q_rated_as"], q_norm, variant, soc_label_stride,
        )
        X_te, y_te = build_soc_arrays(
            slice_battery_series(series, slice(n_train + n_val, n)),
            cfg["data"]["q_rated_as"], q_norm, variant, soc_label_stride,
        )
        st = SOCTrainer(
            variant=variant, hidden=soc_cfg["hidden"], lr=soc_cfg["lr"],
        )
        fit = st.fit(
            X_tr, y_tr, X_va, y_va,
            epochs=soc_cfg["epochs"],
            batch_size=soc_cfg["batch_size"],
            log_path=log_path,
            log_every=log_every,
        )
        ckpt_path = out_dir / f"soc_train_{variant}.pt"
        st.save(ckpt_path)
        soc_results[variant] = {
            **st.evaluate(X_te, y_te),
            "best_val_rmse": fit["best_val_rmse"],
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
        }
        soc_preds_for_plot[variant] = st.predict(X_te)
        r = soc_results[variant]
        print(
            f"        → test RMSE {r['rmse']:.4f}  MAPE {r['mape_pct']:.2f}%  "
            f"R² {r['r2']:.4f}  (best val RMSE {fit['best_val_rmse']:.4f})",
            flush=True,
        )

    print(f"\n  [3/3] SOC plots ...", flush=True)
    plot_soc_training_curves(log_path, out_dir / "soc_train_curves.png")
    print(f"        soc_train_curves.png      ✓")
    plot_soc_variant_comparison(
        soc_results, out_dir / "soc_train_comparison.png", cell_id=cell,
    )
    test_series = slice_battery_series(series, slice(n_train + n_val, n))
    soc_labels_plot = coulomb_soc_stitched_operational(
        test_series,
        q_rated_as=cfg["data"]["q_rated_as"],
        q_norm=q_norm,
    )
    plot_idx = soc_sample_indices(test_series, soc_label_stride)
    plot_soc_prediction_series(
        test_series.time_s[plot_idx],
        soc_labels_plot[plot_idx],
        soc_preds_for_plot,
        out_dir / "soc_train_series.png",
        cell_id=cell,
    )
    print(f"        soc_train_comparison.png  ✓")
    print(f"        soc_train_series.png      ✓")

    summary = {
        "source_cell": cell,
        "step_mode": step_mode,
        "soc_input": "measured",
        "label_method": "coulomb_soc_stitched_operational",
        "soc_q_norm": q_norm,
        "twin_checkpoint": str(twin_ckpt) if twin_ckpt.is_file() else None,
        "soc_test_metrics": soc_results,
        "soc_train_log": str(log_path),
    }
    save_json(out_dir / "soc_train_summary.json", summary)
    return summary
