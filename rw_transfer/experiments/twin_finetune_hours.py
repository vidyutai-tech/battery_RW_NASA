"""Hours-based adaptation study — minimum target data for effective transfer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rw_transfer.config import load_config
from rw_transfer.data.series import load_battery_series
from rw_transfer.data.splits import adaptation_and_eval_split, prefix_by_hours
from rw_transfer.data.windows import build_twin_windows
from rw_transfer.experiments.logging_utils import append_csv_row, experiment_dir, save_json
from rw_transfer.experiments.twin_finetune_percent import _adapt
from rw_transfer.training.twin_trainer import TwinTrainer
from rw_transfer.viz.plots import (
    plot_finetune_vs_scratch_hours,
    plot_transfer_gain_hours,
    plot_gap_closed_hours,
    plot_threshold_bar_chart,
)


def _gap_fraction(r_scratch: float, r_adapt: float, r_full: float) -> float:
    denom = r_scratch - r_full
    if not (math.isfinite(denom) and abs(denom) > 1e-9):
        return float("nan")
    return float(np.clip((r_scratch - r_adapt) / denom, 0.0, 1.0))


def _min_hours_for_threshold(
    hours: List[float], gaps: List[float], threshold: float
) -> Optional[float]:
    for h, g in sorted(zip(hours, gaps)):
        if math.isfinite(g) and g >= threshold:
            return h
    return None


def run_twin_finetune_hours(
    config_path: Optional[str] = None,
    source_ckpt: Optional[Path] = None,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    matlab_dir = cfg["data"]["matlab_dir"]
    step_mode = cfg["data"]["step_mode"]
    decimation = int(cfg["data"].get("decimation", 1))
    seq_len = cfg["windows"]["seq_len"]
    stride = cfg["windows"]["stride"]
    targets = cfg["data"]["cells"]["targets"]
    eval_tail = cfg.get("transfer", {}).get("eval_tail_frac", 0.20)
    p3 = cfg["phase3"]
    thresholds = p3["performance_thresholds"]

    if out_dir is None:
        out_dir = experiment_dir(cfg["output"]["root"], "finetune_hours")
    out_dir = Path(out_dir)

    if source_ckpt is None:
        p1_root = Path(cfg["output"]["root"]) / "twin_source"
        ckpts = sorted(p1_root.glob("*/twin_source_RW9.pt")) if p1_root.is_dir() else []
        if not ckpts:
            raise FileNotFoundError(
                "Run train_twin.py first or pass --source_ckpt path/to/twin_source_RW9.pt"
            )
        source_ckpt = ckpts[-1]

    # Build hour ladder
    hour_specs: List[tuple[str, Optional[float]]] = []
    for h in p3["adaptation_hours"]:
        if h is None:
            hour_specs.append(("full", None))
        else:
            hour_specs.append((f"{h}h", float(h)))

    csv_path = out_dir / "finetune_hours_results.csv"
    fields = [
        "target", "adaptation_label", "adaptation_hours",
        "finetune_voltage_rmse", "scratch_voltage_rmse", "full_finetune_voltage_rmse",
        "gap_fraction_finetune", "transfer_gain_rmse",
        "finetune_temperature_rmse", "scratch_temperature_rmse",
    ]
    all_rows: List[Dict[str, Any]] = []
    recommendations: Dict[str, Any] = {}

    for target in targets:
        series = load_battery_series(matlab_dir, target, step_mode=step_mode, decimation=decimation)
        adapt_pool, eval_series = adaptation_and_eval_split(series, eval_tail)
        eval_w = build_twin_windows(eval_series, seq_len=seq_len, stride=stride)

        # Ceiling: fine-tune on all of adapt_pool
        full_adapt_w = build_twin_windows(adapt_pool, seq_len=seq_len, stride=stride)
        ft_full = TwinTrainer.load(source_ckpt, seq_len=seq_len)
        full_m = _adapt(ft_full, full_adapt_w, eval_w, p3["finetune_epochs"],
                        cfg["twin"]["batch_size"], cfg["twin"]["early_stop_patience"])
        r_full = full_m.get("voltage_rmse", float("nan"))

        target_rows: List[Dict[str, Any]] = []
        hours_done: List[float] = []
        gaps: List[float] = []

        for label, h in hour_specs:
            if label == "full":
                prefix = adapt_pool
            else:
                prefix = prefix_by_hours(adapt_pool, h)
            adapt_h = prefix.duration_hours
            adapt_w = build_twin_windows(prefix, seq_len=seq_len, stride=stride)

            ft = TwinTrainer.load(source_ckpt, seq_len=seq_len)
            ft_m = _adapt(ft, adapt_w, eval_w, p3["finetune_epochs"],
                          cfg["twin"]["batch_size"], cfg["twin"]["early_stop_patience"])

            sc = TwinTrainer(seq_len=seq_len, lr=cfg["twin"]["lr"])
            sc_m = _adapt(sc, adapt_w, eval_w, p3["scratch_epochs"],
                          cfg["twin"]["batch_size"], cfg["twin"]["early_stop_patience"])

            r_ft = ft_m.get("voltage_rmse", float("nan"))
            r_sc = sc_m.get("voltage_rmse", float("nan"))
            gap = _gap_fraction(r_sc, r_ft, r_full)
            tg  = (r_sc - r_ft) if (math.isfinite(r_sc) and math.isfinite(r_ft)) else float("nan")

            row = {
                "target": target,
                "adaptation_label": label,
                "adaptation_hours": adapt_h,
                "finetune_voltage_rmse": r_ft,
                "scratch_voltage_rmse": r_sc,
                "full_finetune_voltage_rmse": r_full,
                "gap_fraction_finetune": gap,
                "transfer_gain_rmse": tg,
                "finetune_temperature_rmse": ft_m.get("temperature_rmse"),
                "scratch_temperature_rmse": sc_m.get("temperature_rmse"),
            }
            target_rows.append(row)
            all_rows.append(row)
            append_csv_row(csv_path, row, fields)
            if label != "full":
                hours_done.append(adapt_h)
                gaps.append(gap)

        # Per-target plots
        plot_finetune_vs_scratch_hours(
            target_rows, target,
            out_dir / f"twin_finetune_hours_{target}.png",
        )

        # Threshold lookup
        thresh_map = {}
        for thr in thresholds:
            key = f"hours_for_{int(thr * 100)}pct"
            thresh_map[key] = _min_hours_for_threshold(hours_done, gaps, thr)

        recommendations[target] = {
            "full_finetune_voltage_rmse": r_full,
            "thresholds": thresh_map,
            "primary_metric": p3["primary_metric"],
        }

    # Multi-target plots
    plot_transfer_gain_hours(all_rows, out_dir / "twin_finetune_gain_hours_all.png")
    plot_gap_closed_hours(all_rows, out_dir / "twin_finetune_gap_closed_all.png")
    plot_threshold_bar_chart(recommendations, out_dir / "twin_finetune_threshold_summary.png")

    practical = _build_recommendation_text(recommendations)
    summary = {
        "source_ckpt": str(source_ckpt),
        "recommendations": recommendations,
        "practical_recommendation": practical,
        "rows": all_rows,
    }
    save_json(out_dir / "finetune_hours_summary.json", summary)
    save_json(out_dir / "practical_recommendation.json",
              {"recommendations": practical, "details": recommendations})
    return summary


def _build_recommendation_text(rec: Dict[str, Any]) -> List[str]:
    lines = []
    for target, info in rec.items():
        thr = info.get("thresholds", {})
        h95 = thr.get("hours_for_95pct")
        h90 = thr.get("hours_for_90pct")
        if h95 is not None:
            lines.append(
                f"{target}: ~{h95:.1f} h of RW operational data reaches 95% of full "
                f"fine-tune voltage-RMSE improvement (ceil: {info['full_finetune_voltage_rmse']:.4f} V)."
            )
        elif h90 is not None:
            lines.append(
                f"{target}: ~{h90:.1f} h reaches 90%. "
                f"95%+ threshold not reached within tested durations."
            )
        else:
            lines.append(f"{target}: 90%+ threshold not reached within tested durations.")
    return lines
