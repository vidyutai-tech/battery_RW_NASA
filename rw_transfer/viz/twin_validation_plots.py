"""
Publication-style digital twin and SOC figures (aligned with main-repo visualize.py).

  * digital_twin_validation.png — measured vs predicted V/T on best test chunks
  * digital_twin_validation_val_mean.png — mean trajectories over validation chunks
  * soc_estimation.png — Coulomb vs MLP variants (time + V–SOC scatter)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import savgol_filter
from torch.utils.data import Subset

from rw_transfer.data.author_dataset import AuthorChunkDataset, random_split_author_dataset
from rw_transfer.data.author_loader import AuthorStitchedSeries, load_author_stitched_series
from rw_transfer.data.series import BatteryTimeSeries, load_battery_series
from rw_transfer.data.soc_labels import coulomb_soc_from_voltage_anchor
from rw_transfer.training.soc_trainer import SOCTrainer, build_soc_arrays
from rw_transfer.training.twin_trainer import TwinTrainer
from rw_transfer.viz.plots import (
    ACCENT,
    GREY,
    LIGHT_BG,
    ORANGE,
    SOC_VARIANT_COLORS,
    SOC_VARIANT_LABELS,
    _savefig,
)


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


@torch.no_grad()
def _predict_chunk(
    trainer: TwinTrainer,
    state: torch.Tensor,
    action: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    out = trainer.model.forward_author(
        state.unsqueeze(0).to(trainer.device),
        action.unsqueeze(0).to(trainer.device),
    )
    v = out[0, :, 0].cpu().numpy()
    t = out[0, :, 1].cpu().numpy()
    return v, t


def pick_best_validation_chunks(
    trainer: TwinTrainer,
    test_set: Subset,
    stitched: AuthorStitchedSeries,
    n: int = 3,
    burn_in: int = 5,
    age_min: float = 0.0,
    age_max: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Score test chunks by post-burn-in MAPE; return the ``n`` lowest-error windows.
    """
    base: AuthorChunkDataset = test_set.dataset
    cs = base.chunk_size
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for idx in test_set.indices:
        state, action, next_state = base[idx]
        rel_age = float(state[0].item())
        if rel_age < age_min or rel_age > age_max:
            continue

        v0 = float(state[1].item())
        t0 = float(state[2].item())
        start = int(idx) * cs
        end = start + cs + 1
        if end > stitched.voltage_v.size:
            continue

        v_act = next_state[:, 0].numpy()
        t_act = next_state[:, 1].numpy()
        try:
            v_pred, t_pred = _predict_chunk(trainer, state, action)
        except Exception:
            continue

        st = min(max(burn_in, 0), len(v_act) - 1)
        mape_v = _mape_pct(v_pred[st:], v_act[st:])
        mape_t = _mape_pct(t_pred[st:], t_act[st:])
        score = mape_v + mape_t

        time_win = stitched.non_relative_time_s[start:end]
        dt = _median_dt_seconds(time_win)
        t_min = (time_win - time_win[0]) / 60.0

        scored.append((
            score,
            {
                "chunk_idx": int(idx),
                "rel_age": rel_age,
                "v0": v0,
                "t0": t0,
                "start_sample": start,
                "burn_in": st,
                "v_actual": v_act,
                "t_actual": t_act,
                "v_pred": v_pred,
                "t_pred": t_pred,
                "t_minutes": t_min[1:],
                "dt_s": dt,
                "mape_v": mape_v,
                "mape_t": mape_t,
            },
        ))

    scored.sort(key=lambda x: x[0])
    return [item[1] for item in scored[:n]]


def plot_digital_twin_validation(
    samples: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    cell_id: str = "RW9",
    seq_len: int = 150,
    title_suffix: str = "",
) -> None:
    """
    2×N grid: voltage (top), temperature (bottom), measured vs digital twin.

    Matches main-repo ``plot_digital_twin`` layout (time in minutes, burn-in MAPE).
    """
    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 6), facecolor=LIGHT_BG)
    if n == 1:
        axes = axes[:, np.newaxis]

    for col, samp in enumerate(samples):
        st = samp["burn_in"]
        T = len(samp["v_actual"])
        t_axis = samp["t_minutes"][st:] if len(samp["t_minutes"]) >= T - st else (
            np.arange(T - st, dtype=np.float64) * samp["dt_s"] / 60.0
        )

        v_a = samp["v_actual"][st:]
        v_p = samp["v_pred"][st:]
        t_a = samp["t_actual"][st:]
        t_p = samp["t_pred"][st:]

        ax = axes[0, col]
        ax.set_facecolor(LIGHT_BG)
        ax.plot(t_axis, v_a, color=GREY, linestyle="--", label="Measured", alpha=0.85, lw=1.8)
        ax.plot(t_axis, v_p, color=ACCENT, label="Digital Twin predicted", lw=2.0)
        ax.set_title(
            f"Chunk {samp['chunk_idx']}  —  Relative age = {samp['rel_age']:.3f}\n"
            f"Voltage MAPE = {samp['mape_v']:.2f}%  (steps {st + 1}–{T})",
            fontsize=9,
        )
        ax.set_ylabel("Voltage (V)" if col == 0 else "")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85)
        ax.set_ylim(2.8, 4.5)

        wl = min(21, len(t_p) if len(t_p) % 2 == 1 else len(t_p) - 1)
        wl = max(wl, 5)
        if len(t_p) >= wl:
            t_smooth = savgol_filter(t_p, window_length=wl, polyorder=min(3, wl - 1))
        else:
            t_smooth = t_p

        ax = axes[1, col]
        ax.set_facecolor(LIGHT_BG)
        ax.plot(t_axis, t_a, color=GREY, linestyle="--", label="Measured", alpha=0.85, lw=1.8)
        ax.plot(t_axis, t_p, color=ORANGE, linestyle=":", alpha=0.4, linewidth=1.0)
        ax.plot(t_axis, t_smooth, color=ORANGE, linestyle="-",
                label="Digital Twin predicted", linewidth=2.0)
        ax.set_title(
            f"Temperature MAPE = {samp['mape_t']:.2f}%  (steps {st + 1}–{T})",
            fontsize=9,
        )
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Temperature (°C)" if col == 0 else "")
        ax.legend(fontsize=7.5, loc="upper right", framealpha=0.85)

    suffix = f"  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Digital Twin — measured vs predicted  "
        f"(first {seq_len} steps, {n} held-out test chunks, {cell_id}){suffix}",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, out_path)


@torch.no_grad()
def compute_val_mean_trajectories(
    trainer: TwinTrainer,
    val_set: Subset,
    stitched: AuthorStitchedSeries,
    *,
    burn_in: int = 5,
    max_windows: int = 400,
    seed: int = 42,
) -> Optional[Dict[str, np.ndarray]]:
    """Mean measured vs predicted V/T over validation chunks (post burn-in)."""
    base: AuthorChunkDataset = val_set.dataset
    cs = base.chunk_size
    indices = list(val_set.indices)
    if not indices:
        return None

    rng = np.random.default_rng(seed)
    if len(indices) > max_windows:
        indices = rng.choice(indices, size=max_windows, replace=False).tolist()

    st = min(max(burn_in, 0), cs - 1)
    n_vis = cs - st
    sums_v_m = np.zeros(n_vis, dtype=np.float64)
    sums_v_p = np.zeros(n_vis, dtype=np.float64)
    sums_t_m = np.zeros(n_vis, dtype=np.float64)
    sums_t_p = np.zeros(n_vis, dtype=np.float64)
    sse_v: List[float] = []
    sse_t: List[float] = []
    used = 0
    dt_med = 1.0

    for idx in indices:
        state, action, next_state = base[idx]
        v_act = next_state[:, 0].numpy()
        t_act = next_state[:, 1].numpy()
        try:
            v_pred, t_pred = _predict_chunk(trainer, state, action)
        except Exception:
            continue

        start = int(idx) * cs
        time_win = stitched.non_relative_time_s[start : start + cs + 1]
        dt_med = _median_dt_seconds(time_win)

        va, ta = v_act[st:], t_act[st:]
        vp, tp = v_pred[st:], t_pred[st:]
        sums_v_m += va
        sums_v_p += vp
        sums_t_m += ta
        sums_t_p += tp
        sse_v.extend(np.abs(vp - va) / (np.abs(va) + 1e-8))
        sse_t.extend(np.abs(tp - ta) / (np.abs(ta) + 1e-8))
        used += 1

    if used == 0:
        return None

    inv = 1.0 / used
    t_axis = np.arange(n_vis, dtype=np.float64) * dt_med / 60.0
    return {
        "n_windows_used": used,
        "t_axis_minutes": t_axis,
        "v_meas_mean": sums_v_m * inv,
        "v_pred_mean": sums_v_p * inv,
        "t_meas_mean": sums_t_m * inv,
        "t_pred_mean": sums_t_p * inv,
        "pooled_mape_v_pct": float(np.mean(sse_v) * 100.0),
        "pooled_mape_t_pct": float(np.mean(sse_t) * 100.0),
        "burn_in_steps": st,
    }


def plot_digital_twin_validation_val_mean(
    stats: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Mean V/T on validation chunks (main-repo figure 1c style)."""
    t_ax = stats["t_axis_minutes"]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True, facecolor=LIGHT_BG)
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)

    axes[0].plot(t_ax, stats["v_meas_mean"], color=GREY, linestyle="--", lw=2.0, label="Measured")
    axes[0].plot(t_ax, stats["v_pred_mean"], color=ACCENT, lw=2.2, label="Digital Twin predicted")
    axes[0].set_ylabel("Voltage (V)")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title(f"Voltage MAPE = {stats['pooled_mape_v_pct']:.2f}%", fontsize=9)

    npt = len(t_ax)
    wl = min(21, npt if npt % 2 == 1 else npt - 1)
    wl = max(wl, 5)
    po = min(3, max(1, wl - 2))
    t_pred_smooth = savgol_filter(stats["t_pred_mean"], window_length=wl, polyorder=po)

    axes[1].plot(t_ax, stats["t_meas_mean"], color=GREY, linestyle="--", lw=2.0, label="Measured")
    axes[1].plot(t_ax, t_pred_smooth, color=ORANGE, lw=2.2, label="Digital Twin predicted")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Temperature (°C)")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title(f"Temperature MAPE = {stats['pooled_mape_t_pct']:.2f}%", fontsize=9)

    fig.suptitle(
        f"Digital Twin — mean measured vs predicted "
        f"({stats['n_windows_used']} validation chunks)",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_soc_estimation(
    time_s: np.ndarray,
    voltage_v: np.ndarray,
    soc_labels: np.ndarray,
    soc_preds: Dict[str, np.ndarray],
    out_path: Path,
    *,
    cell_id: str = "RW9",
    max_points: int = 8000,
    primary_variant: str = "vta",
) -> None:
    """
    Two-panel SOC figure: time-series (Coulomb + MLPs) and SOC vs voltage scatter.

    ``time_s``, ``voltage_v``, ``soc_labels``, and each prediction array must share
    the same length (e.g. from :func:`soc_sample_indices`).
    """
    n = min(max_points, len(time_s), len(voltage_v), len(soc_labels))
    for pred in soc_preds.values():
        n = min(n, len(pred))
    if n < 10:
        return

    t_h = (time_s[:n].astype(np.float64) - time_s[0]) / 3600.0
    volt = voltage_v[:n]
    labels_pct = soc_labels[:n] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=LIGHT_BG)
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)

    ax = axes[0]
    ax.plot(t_h, labels_pct, color=GREEN, lw=2.0, label="Coulomb counting (labels)")
    for variant in ("v_only", "vta", "vta_i"):
        if variant not in soc_preds:
            continue
        pred = soc_preds[variant][:n] * 100.0
        lw = 2.2 if variant == primary_variant else 1.5
        ls = "-" if variant == primary_variant else "--"
        ax.plot(
            t_h, pred, color=SOC_VARIANT_COLORS[variant], linestyle=ls, linewidth=lw,
            label=SOC_VARIANT_LABELS.get(variant, variant),
        )
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("State of Charge (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.set_title(f"{cell_id} — SOC vs time", fontsize=10, fontweight="bold")

    ax = axes[1]
    ax.scatter(volt, labels_pct, s=6, alpha=0.45, color=GREEN, label="Coulomb", zorder=2)
    for variant in ("v_only", "vta", "vta_i"):
        if variant not in soc_preds:
            continue
        pred = soc_preds[variant][:n] * 100.0
        ax.scatter(
            volt, pred, s=6, alpha=0.4,
            color=SOC_VARIANT_COLORS[variant],
            label=SOC_VARIANT_LABELS.get(variant, variant),
            zorder=3 if variant == primary_variant else 1,
        )
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("SOC (%)")
    ax.legend(fontsize=8, markerscale=2, framealpha=0.9)
    ax.set_title("SOC vs voltage", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"SOC estimation — Coulomb labels vs measured-feature MLPs ({cell_id})",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_soc_variant_bars(
    results: Dict[str, Dict[str, Any]],
    out_path: Path,
    cell_id: str = "RW9",
) -> None:
    from rw_transfer.viz.plots import plot_soc_variant_comparison
    plot_soc_variant_comparison(results, out_path, cell_id=cell_id)
