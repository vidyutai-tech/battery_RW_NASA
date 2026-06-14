#!/usr/bin/env python3
"""
Replay measured NASA step current through the frozen BDT and plot V/T overlays.

Feeds each step's measured I(t) into ``FrozenBDTSimulator.predict_traj`` (chained
open-loop rollout — same mode as charging BO), then compares predicted vs measured
voltage and temperature.

Usage
-----
    venv/bin/python scripts/replay_nasa_step.py \\
        --cell RW9 \\
        --comment "reference charge" \\
        --max_steps 5

    venv/bin/python scripts/replay_nasa_step.py \\
        --cell RW9 \\
        --comment "charge (random walk)" \\
        --step_indices 473,1000 \\
        --out_dir outputs/replay/RW9_rw_charge

    venv/bin/python scripts/replay_nasa_step.py \\
        --cell RW10 \\
        --comment "reference charge" \\
        --ckpt outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.40.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rw_transfer.config import load_config
from rw_transfer.data.mat_loader import BatteryStep
from rw_transfer.viz.plots import ACCENT, GREY, LIGHT_BG, ORANGE, _savefig

from charging_opt.artifacts import resolve_bdt_ckpt
from charging_opt.bdt_rollout import FrozenBDTSimulator
from charging_opt.io_utils import current_user, dir_is_writable
from charging_opt.soc_utils import load_steps_with_age


def default_out_dir(cell: str, comment_slug: str) -> Path:
    user = current_user()
    base = ROOT / "outputs" / "charging_opt_user" / user / "replay" / cell.lower()
    candidate = base / comment_slug
    if dir_is_writable(candidate.parent):
        return candidate
    fallback = ROOT / "outputs" / "replay" / cell.lower() / comment_slug
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def _median_dt_seconds(time_s: np.ndarray) -> float:
    if time_s.size < 2:
        return 1.0
    dt = np.diff(time_s.astype(np.float64))
    dt = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(dt)) if dt.size else 1.0


def _mape_pct(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.abs(pred - ref) / (np.abs(ref) + eps)) * 100.0)


def _slug(text: str) -> str:
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "step"


@dataclass
class ReplayMetrics:
    step_index: int
    comment: str
    age: float
    n_samples: int
    duration_min: float
    dt_median_s: float
    i_mean_a: float
    v0_v: float
    t0_c: float
    v_rmse_v: float
    v_mape_pct: float
    v_max_abs_err_v: float
    t_rmse_c: float
    t_mape_pct: float
    dt_caveat: bool
    plot_path: str


def _comment_matches(step_comment: str, filters: Sequence[str]) -> bool:
    c = step_comment.strip().lower()
    return any(f.strip().lower() in c for f in filters)


def select_steps(
    steps: Sequence[BatteryStep],
    step_age: np.ndarray,
    *,
    comments: Sequence[str],
    step_indices: Optional[Sequence[int]] = None,
    min_samples: int = 10,
    max_steps: Optional[int] = None,
) -> List[tuple[int, BatteryStep, float]]:
    selected: List[tuple[int, BatteryStep, float]] = []
    if step_indices is not None:
        for idx in step_indices:
            if idx < 0 or idx >= len(steps):
                raise ValueError(f"step_index {idx} out of range [0, {len(steps) - 1}]")
            s = steps[idx]
            if comments and not _comment_matches(s.comment, comments):
                print(f"  WARNING: step {idx} comment {s.comment!r} "
                      f"does not match {list(comments)!r} — including anyway")
            if s.voltage_v.size >= min_samples:
                selected.append((idx, s, float(step_age[idx])))
        return selected

    for i, s in enumerate(steps):
        if comments and not _comment_matches(s.comment, comments):
            continue
        if s.voltage_v.size < min_samples:
            continue
        selected.append((i, s, float(step_age[i])))
        if max_steps is not None and len(selected) >= max_steps:
            break
    return selected


def replay_step(
    sim: FrozenBDTSimulator,
    step_index: int,
    step: BatteryStep,
    age: float,
    *,
    dt_warn_s: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, ReplayMetrics]:
    dt_med = _median_dt_seconds(step.relative_time_s)
    t_min = (step.relative_time_s - step.relative_time_s[0]) / 60.0

    v_hat, t_hat = sim.predict_traj(
        age=age,
        v0=float(step.voltage_v[0]),
        t0=float(step.temperature_c[0]),
        current_profile=step.current_a,
    )
    v_err = v_hat - step.voltage_v
    t_err = t_hat - step.temperature_c

    metrics = ReplayMetrics(
        step_index=step_index,
        comment=step.comment,
        age=age,
        n_samples=int(step.voltage_v.size),
        duration_min=float(t_min[-1]) if t_min.size else 0.0,
        dt_median_s=dt_med,
        i_mean_a=float(np.mean(step.current_a)),
        v0_v=float(step.voltage_v[0]),
        t0_c=float(step.temperature_c[0]),
        v_rmse_v=float(np.sqrt(np.mean(v_err ** 2))),
        v_mape_pct=_mape_pct(v_hat, step.voltage_v),
        v_max_abs_err_v=float(np.max(np.abs(v_err))),
        t_rmse_c=float(np.sqrt(np.mean(t_err ** 2))),
        t_mape_pct=_mape_pct(t_hat, step.temperature_c),
        dt_caveat=bool(dt_med > dt_warn_s),
        plot_path="",
    )
    return v_hat, t_hat, metrics


def plot_replay_overlay(
    step: BatteryStep,
    v_hat: np.ndarray,
    t_hat: np.ndarray,
    metrics: ReplayMetrics,
    out_path: Path,
    *,
    cell_id: str,
) -> None:
    t_min = (step.relative_time_s - step.relative_time_s[0]) / 60.0
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True, facecolor=LIGHT_BG)
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)

    axes[0].plot(t_min, step.current_a, color=ACCENT, lw=1.6, label="Measured I(t)")
    axes[0].set_ylabel("Current (A)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(
        f"Input — measured current replay  |  I_mean={metrics.i_mean_a:.3f} A",
        fontsize=9,
    )

    axes[1].plot(t_min, step.voltage_v, color=GREY, ls="--", lw=2.0, label="Measured")
    axes[1].plot(t_min, v_hat, color=ACCENT, lw=1.8, label="BDT predicted")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_title(
        f"Voltage  RMSE={metrics.v_rmse_v:.4f} V  MAPE={metrics.v_mape_pct:.2f}%",
        fontsize=9,
    )

    axes[2].plot(t_min, step.temperature_c, color=GREY, ls="--", lw=2.0, label="Measured")
    axes[2].plot(t_min, t_hat, color=ORANGE, lw=1.8, label="BDT predicted")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Time (min)")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_title(
        f"Temperature  RMSE={metrics.t_rmse_c:.3f} °C  MAPE={metrics.t_mape_pct:.2f}%",
        fontsize=9,
    )

    caveat = ""
    if metrics.dt_caveat:
        caveat = f"  |  dt≈{metrics.dt_median_s:.1f}s (BDT trained ~1 Hz — interpret with care)"

    fig.suptitle(
        f"BDT replay — {cell_id}  step {metrics.step_index}  "
        f"\"{metrics.comment}\"  age={metrics.age:.3f}{caveat}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, out_path)
    plt.close(fig)


def _parse_step_indices(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Replay NASA step I(t) through frozen BDT.")
    p.add_argument("--config", default=None)
    p.add_argument("--cell", default="RW9")
    p.add_argument(
        "--comment",
        default="reference charge",
        help="Substring filter on step comment (comma-separated for multiple)",
    )
    p.add_argument("--ckpt", default=None, help="BDT checkpoint (default: canonical RW9 source)")
    p.add_argument("--out_dir", default=None, help="Output directory for PNG/CSV/JSON")
    p.add_argument(
        "--step_indices",
        default=None,
        help="Comma-separated step indices (overrides --max_steps scan order)",
    )
    p.add_argument("--max_steps", type=int, default=5, help="Max steps when scanning by comment")
    p.add_argument("--min_samples", type=int, default=10)
    p.add_argument("--dt_warn_s", type=float, default=1.5, help="Warn when median dt exceeds this")
    args = p.parse_args()

    cfg = load_config(args.config)
    matlab_dir = cfg["data"]["matlab_dir"]
    cell = args.cell.upper()
    if not cell.startswith("RW"):
        cell = f"RW{cell}"

    ckpt = resolve_bdt_ckpt(args.ckpt, root=ROOT)
    comments = [c.strip() for c in args.comment.split(",") if c.strip()]
    step_indices = _parse_step_indices(args.step_indices)

    slug = _slug(comments[0] if len(comments) == 1 else "multi_comment")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else default_out_dir(cell, slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cell: {cell}")
    print(f"Checkpoint: {ckpt}")
    print(f"Comment filter: {comments or '(all)'}")
    print(f"Output: {out_dir}\n")

    steps, step_age = load_steps_with_age(matlab_dir, cell)
    selected = select_steps(
        steps,
        step_age,
        comments=comments,
        step_indices=step_indices,
        min_samples=args.min_samples,
        max_steps=None if step_indices else args.max_steps,
    )
    if not selected:
        raise SystemExit(
            f"No steps matched comment={comments!r} with n>={args.min_samples}. "
            "Try --comment or --step_indices."
        )

    print(f"Replaying {len(selected)} step(s) …")
    sim = FrozenBDTSimulator(ckpt)
    rows: List[ReplayMetrics] = []

    for step_index, step, age in selected:
        v_hat, t_hat, metrics = replay_step(
            sim, step_index, step, age, dt_warn_s=args.dt_warn_s,
        )
        fname = (
            f"replay_{_slug(metrics.comment)}_step{step_index:05d}_"
            f"age{metrics.age:.3f}.png"
        )
        plot_path = plots_dir / fname
        plot_replay_overlay(step, v_hat, t_hat, metrics, plot_path, cell_id=cell)
        metrics.plot_path = str(plot_path.relative_to(out_dir))
        rows.append(metrics)
        flag = " [dt caveat]" if metrics.dt_caveat else ""
        print(
            f"  step {step_index:5d}  n={metrics.n_samples:5d}  "
            f"dt={metrics.dt_median_s:.2f}s  V_RMSE={metrics.v_rmse_v:.4f}  "
            f"T_RMSE={metrics.t_rmse_c:.3f}{flag}  -> {plot_path.name}"
        )

    summary_path = out_dir / "replay_summary.json"
    csv_path = out_dir / "replay_summary.csv"
    payload = {
        "cell": cell,
        "checkpoint": str(ckpt),
        "comments": comments,
        "n_steps": len(rows),
        "dt_warn_s": args.dt_warn_s,
        "steps": [asdict(r) for r in rows],
    }
    with summary_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    fieldnames = list(asdict(rows[0]).keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    v_rmses = [r.v_rmse_v for r in rows]
    t_rmses = [r.t_rmse_c for r in rows]
    print(f"\nSummary ({len(rows)} steps):")
    print(f"  V RMSE: median={np.median(v_rmses):.4f} V  "
          f"max={np.max(v_rmses):.4f} V")
    print(f"  T RMSE: median={np.median(t_rmses):.3f} °C  "
          f"max={np.max(t_rmses):.3f} °C")
    n_caveat = sum(r.dt_caveat for r in rows)
    if n_caveat:
        print(f"  dt caveat: {n_caveat}/{len(rows)} steps have median dt > {args.dt_warn_s}s")
    print(f"\nWrote {summary_path}")
    print(f"Wrote {csv_path}")
    print(f"Plots -> {plots_dir}/")


if __name__ == "__main__":
    main()
