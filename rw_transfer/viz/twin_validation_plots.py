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
    GREEN,
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


def _savgol_display(arr: np.ndarray, *, max_window: int = 21) -> np.ndarray:
    """Light Savitzky–Golay smoothing for plot display (sensor quantization)."""
    arr = np.asarray(arr, dtype=np.float64)
    n = arr.size
    if n < 5:
        return arr
    wl = min(max_window, n if n % 2 == 1 else n - 1)
    wl = max(wl, 5)
    po = min(3, wl - 2)
    return savgol_filter(arr, window_length=wl, polyorder=po)


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


def _chunk_pulse_metrics(
    v_actual: np.ndarray,
    i_actual: np.ndarray,
    *,
    burn_in: int = 5,
    current_threshold_a: float = 0.05,
) -> Tuple[float, float, int]:
    """Return ``(pulse_score, voltage_range, current_transitions)`` after burn-in."""
    v = np.asarray(v_actual, dtype=np.float64)[burn_in:]
    i = np.asarray(i_actual, dtype=np.float64)[burn_in:]
    if v.size == 0:
        return 0.0, 0.0, 0
    v_range = float(np.max(v) - np.min(v))
    active = np.abs(i) > current_threshold_a
    transitions = int(np.sum(np.diff(active.astype(int)) != 0)) if active.size > 1 else 0
    pulse_score = v_range * max(transitions, 1)
    return pulse_score, v_range, transitions


def _build_chunk_sample(
    trainer: TwinTrainer,
    base: AuthorChunkDataset,
    stitched: AuthorStitchedSeries,
    idx: int,
    *,
    burn_in: int,
    i_actual: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Shared chunk dict for validation plot helpers."""
    cs = base.chunk_size
    state, action, next_state = base[idx]
    rel_age = float(state[0].item())
    v0 = float(state[1].item())
    t0 = float(state[2].item())
    start = int(idx) * cs
    end = start + cs + 1
    if end > stitched.voltage_v.size:
        return None

    v_act = next_state[:, 0].numpy()
    t_act = next_state[:, 1].numpy()
    try:
        v_pred, t_pred = _predict_chunk(trainer, state, action)
    except Exception:
        return None

    st = min(max(burn_in, 0), len(v_act) - 1)
    mape_v = _mape_pct(v_pred[st:], v_act[st:])
    mape_t = _mape_pct(t_pred[st:], t_act[st:])

    time_win = stitched.non_relative_time_s[start:end]
    dt = _median_dt_seconds(time_win)
    t_min = (time_win - time_win[0]) / 60.0

    pulse_score, v_range, transitions = _chunk_pulse_metrics(
        v_act, i_actual, burn_in=st,
    )
    mean_abs_i = float(np.mean(np.abs(i_actual[st:]))) if i_actual.size > st else 0.0

    return {
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
        "pulse_score": pulse_score,
        "v_range": v_range,
        "current_transitions": transitions,
        "mean_abs_current_a": mean_abs_i,
    }


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
        state, _, _ = base[idx]
        rel_age = float(state[0].item())
        if rel_age < age_min or rel_age > age_max:
            continue

        start = int(idx) * cs
        end = start + cs + 1
        if end > stitched.current_a.size:
            continue

        i_actual = stitched.current_a[start + 1 : end]
        sample = _build_chunk_sample(
            trainer, base, stitched, int(idx), burn_in=burn_in, i_actual=i_actual,
        )
        if sample is None:
            continue

        scored.append((sample["mape_v"] + sample["mape_t"], sample))

    scored.sort(key=lambda x: x[0])
    return [item[1] for item in scored[:n]]


def pick_pulsed_validation_chunks(
    trainer: TwinTrainer,
    test_set: Subset,
    stitched: AuthorStitchedSeries,
    n: int = 3,
    burn_in: int = 5,
    age_min: float = 0.25,
    age_max: float = 0.75,
    min_voltage_range_v: float = 0.30,
    min_current_transitions: int = 4,
) -> List[Dict[str, Any]]:
    """
    Select test chunks with pulsed current and visible voltage swings.

    Picks up to ``n`` windows spread across age bins (when possible) so panels
    cover early/mid/late life within the requested age band.
    """
    base: AuthorChunkDataset = test_set.dataset
    cs = base.chunk_size
    candidates: List[Dict[str, Any]] = []

    for idx in test_set.indices:
        state, _, _ = base[idx]
        rel_age = float(state[0].item())
        if rel_age < age_min or rel_age > age_max:
            continue

        start = int(idx) * cs
        end = start + cs + 1
        if end > stitched.current_a.size:
            continue

        i_actual = stitched.current_a[start + 1 : end]
        sample = _build_chunk_sample(
            trainer, base, stitched, int(idx), burn_in=burn_in, i_actual=i_actual,
        )
        if sample is None:
            continue
        if sample["v_range"] < min_voltage_range_v:
            continue
        if sample["current_transitions"] < min_current_transitions:
            continue
        candidates.append(sample)

    if not candidates:
        return []

    candidates.sort(key=lambda s: s["pulse_score"], reverse=True)

    if n <= 1 or len(candidates) <= n:
        return candidates[:n]

    age_lo, age_hi = age_min, age_max
    bin_edges = np.linspace(age_lo, age_hi, n + 1)
    picked: List[Dict[str, Any]] = []
    used_idx: set[int] = set()

    for b in range(n):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bin_cands = [
            s for s in candidates
            if lo <= s["rel_age"] < hi or (b == n - 1 and s["rel_age"] <= hi)
        ]
        for samp in bin_cands:
            if samp["chunk_idx"] in used_idx:
                continue
            picked.append(samp)
            used_idx.add(samp["chunk_idx"])
            break

    for samp in candidates:
        if len(picked) >= n:
            break
        if samp["chunk_idx"] in used_idx:
            continue
        picked.append(samp)
        used_idx.add(samp["chunk_idx"])

    return picked[:n]


def pick_active_validation_chunks(
    trainer: TwinTrainer,
    test_set: Subset,
    stitched: AuthorStitchedSeries,
    n: int = 3,
    burn_in: int = 5,
    age_min: float = 0.25,
    age_max: float = 0.75,
    min_mean_current_a: float = 0.5,
    min_voltage_range_v: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Lowest-MAPE test chunks with non-trivial current (excludes rest / near-idle).

    Useful for cross-chemistry transfer where rest windows yield misleadingly
    low MAPE while predictions oscillate on flat measured voltage.
    """
    base: AuthorChunkDataset = test_set.dataset
    cs = base.chunk_size
    scored: List[Tuple[float, Dict[str, Any]]] = []

    for idx in test_set.indices:
        state, _, _ = base[idx]
        rel_age = float(state[0].item())
        if rel_age < age_min or rel_age > age_max:
            continue

        start = int(idx) * cs
        end = start + cs + 1
        if end > stitched.current_a.size:
            continue

        i_actual = stitched.current_a[start + 1 : end]
        sample = _build_chunk_sample(
            trainer, base, stitched, int(idx), burn_in=burn_in, i_actual=i_actual,
        )
        if sample is None:
            continue
        if sample["mean_abs_current_a"] < min_mean_current_a:
            continue
        if sample["v_range"] < min_voltage_range_v:
            continue

        scored.append((sample["mape_v"] + sample["mape_t"], sample))

    scored.sort(key=lambda x: x[0])
    return [item[1] for item in scored[:n]]


def plot_digital_twin_validation(
    samples: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    cell_id: str = "RW9",
    seq_len: int = 150,
    title_suffix: str = "",
    voltage_ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """
    2×N grid: voltage (top), temperature (bottom), measured vs digital twin.

    Matches main-repo ``plot_digital_twin`` layout (time in minutes, burn-in MAPE).
    """
    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(5.6 * n, 7.0), facecolor=LIGHT_BG)
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
        # Display: faint raw prediction + Savitzky–Golay overlay (MAPE stays on raw).
        ax.plot(t_axis, v_a, color=GREY, linestyle=":", alpha=0.35, lw=1.0, zorder=1)
        ax.plot(
            t_axis, _savgol_display(v_a), color=GREY, linestyle="--",
            label="Measured", alpha=0.85, lw=2.2, zorder=2,
        )
        ax.plot(t_axis, v_p, color=ACCENT, linestyle=":", alpha=0.4, lw=1.2, zorder=3)
        ax.plot(
            t_axis, _savgol_display(v_p), color=ACCENT, linestyle="-",
            label="Digital Twin predicted", lw=2.4, zorder=4,
        )
        ax.set_title(
            f"Chunk {samp['chunk_idx']}  —  Relative age = {samp['rel_age']:.3f}\n"
            f"Voltage MAPE = {samp['mape_v']:.2f}%  (steps {st + 1}–{T})",
            fontsize=15, fontweight="bold",
        )
        ax.set_ylabel("Voltage (V)" if col == 0 else "", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=13, loc="lower right", framealpha=0.85)
        if voltage_ylim is not None:
            ax.set_ylim(*voltage_ylim)
        else:
            v_lo = min(float(np.min(v_a)), float(np.min(v_p)))
            v_hi = max(float(np.max(v_a)), float(np.max(v_p)))
            pad = max(0.05, 0.05 * (v_hi - v_lo))
            ax.set_ylim(v_lo - pad, v_hi + pad)

        t_meas_smooth = _savgol_display(t_a)
        t_pred_smooth = _savgol_display(t_p)

        ax = axes[1, col]
        ax.set_facecolor(LIGHT_BG)
        ax.plot(t_axis, t_a, color=ORANGE, linestyle=":", alpha=0.35, linewidth=1.0, zorder=1)
        ax.plot(
            t_axis, t_meas_smooth, color=GREY, linestyle="--",
            label="Measured", alpha=0.85, lw=2.2, zorder=2,
        )
        ax.plot(t_axis, t_p, color=ORANGE, linestyle=":", alpha=0.4, linewidth=1.2, zorder=3)
        ax.plot(
            t_axis, t_pred_smooth, color=ORANGE, linestyle="-",
            label="Digital Twin predicted", linewidth=2.4, zorder=4,
        )
        ax.set_title(
            f"Temperature MAPE = {samp['mape_t']:.2f}%  (steps {st + 1}–{T})",
            fontsize=15, fontweight="bold",
        )
        ax.set_xlabel("Time (min)", fontsize=16)
        ax.set_ylabel("Temperature (°C)" if col == 0 else "", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=13, loc="upper right", framealpha=0.85)

    suffix = f"  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Digital Twin — measured vs predicted  "
        f"(first {seq_len} steps, {n} held-out test chunks, {cell_id}){suffix}",
        fontsize=18,
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
    pulsed_only: bool = False,
    min_voltage_range_v: float = 0.30,
    min_current_transitions: int = 4,
    min_mean_current_a: float = 0.0,
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
        start = int(idx) * cs
        end = start + cs + 1
        i_actual = stitched.current_a[start + 1 : end]
        if pulsed_only:
            _, v_range, transitions = _chunk_pulse_metrics(
                v_act, i_actual, burn_in=st,
            )
            if v_range < min_voltage_range_v or transitions < min_current_transitions:
                continue
        if min_mean_current_a > 0.0:
            mean_i = float(np.mean(np.abs(i_actual[st:])))
            if mean_i < min_mean_current_a:
                continue
        try:
            v_pred, t_pred = _predict_chunk(trainer, state, action)
        except Exception:
            continue

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
    *,
    title_suffix: str = "",
) -> None:
    """Mean V/T on validation chunks (main-repo figure 1c style)."""
    t_ax = stats["t_axis_minutes"]
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True, facecolor=LIGHT_BG)
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)

    axes[0].plot(t_ax, stats["v_meas_mean"], color=GREY, linestyle=":", alpha=0.35, lw=1.0)
    axes[0].plot(
        t_ax, _savgol_display(stats["v_meas_mean"]),
        color=GREY, linestyle="--", lw=2.4, label="Measured",
    )
    axes[0].plot(t_ax, stats["v_pred_mean"], color=ACCENT, linestyle=":", alpha=0.4, lw=1.2)
    axes[0].plot(
        t_ax, _savgol_display(stats["v_pred_mean"]),
        color=ACCENT, lw=2.6, label="Digital Twin predicted",
    )
    axes[0].set_ylabel("Voltage (V)", fontsize=16)
    axes[0].tick_params(labelsize=14)
    axes[0].legend(fontsize=13, loc="lower right")
    axes[0].set_title(f"Voltage MAPE = {stats['pooled_mape_v_pct']:.2f}%", fontsize=15, fontweight="bold")

    t_meas_smooth = _savgol_display(stats["t_meas_mean"])
    t_pred_smooth = _savgol_display(stats["t_pred_mean"])

    axes[1].plot(t_ax, stats["t_meas_mean"], color=ORANGE, linestyle=":", alpha=0.35, lw=1.0)
    axes[1].plot(t_ax, t_meas_smooth, color=GREY, linestyle="--", lw=2.4, label="Measured")
    axes[1].plot(t_ax, t_pred_smooth, color=ORANGE, lw=2.6, label="Digital Twin predicted")
    axes[1].set_xlabel("Time (min)", fontsize=16)
    axes[1].set_ylabel("Temperature (°C)", fontsize=16)
    axes[1].tick_params(labelsize=14)
    axes[1].legend(fontsize=13, loc="upper right")
    axes[1].set_title(f"Temperature MAPE = {stats['pooled_mape_t_pct']:.2f}%", fontsize=15, fontweight="bold")

    suffix = f"  {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"Digital Twin — mean measured vs predicted "
        f"({stats['n_windows_used']} validation chunks){suffix}",
        fontsize=17,
        fontweight="bold",
    )
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_voltage_error_panel(
    samples: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    cell_id: str = "RW9",
) -> None:
    """Paper panel: voltage residual (pred−meas) shows absolute noise is mV-scale."""
    if not samples:
        return
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8), facecolor=LIGHT_BG, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, samp in zip(axes, samples):
        ax.set_facecolor(LIGHT_BG)
        st = samp["burn_in"]
        T = len(samp["v_actual"])
        t_axis = samp["t_minutes"][st:] if len(samp["t_minutes"]) >= T - st else (
            np.arange(T - st, dtype=np.float64) * samp["dt_s"] / 60.0
        )
        err_mv = 1000.0 * (samp["v_pred"][st:] - samp["v_actual"][st:])
        ax.axhline(0.0, color=GREY, lw=1.2)
        ax.plot(t_axis, err_mv, color=ACCENT, lw=1.5, alpha=0.85)
        ax.plot(t_axis, _savgol_display(err_mv), color="#1e3a8a", lw=2.4, label="Smoothed residual")
        mae = float(np.mean(np.abs(err_mv)))
        ax.set_title(
            f"Chunk {samp['chunk_idx']}  |  MAE = {mae:.1f} mV\n"
            f"Voltage MAPE = {samp['mape_v']:.2f}%",
            fontsize=13, fontweight="bold",
        )
        ax.set_xlabel("Time (min)", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=11, loc="upper right", framealpha=0.85)
    axes[0].set_ylabel("Voltage residual (mV)\npred − measured", fontsize=14)
    fig.suptitle(
        f"{cell_id}: voltage prediction residual (held-out test chunks)",
        fontsize=15, fontweight="bold",
    )
    fig.text(
        0.5, -0.02,
        "High-frequency twin jitter is typically a few mV; MAPE remains on raw predictions.",
        ha="center", fontsize=11, color="#64748b", style="italic",
    )
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_cross_cell_twin_summary(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    footnote: str = "RW9 = source twin; RW10–RW12 = finetuned models.",
) -> None:
    """Bar comparison of pooled val MAPE across cells (paper summary)."""
    if not rows:
        return
    labels = []
    for r in rows:
        cell = str(r.get("cell", "?"))
        frac = r.get("fraction")
        if frac is None:
            labels.append(cell)
        else:
            labels.append(f"{cell}\n{float(frac):.0%}")
    v_mape = [float(r["mape_v"]) for r in rows]
    t_mape = [float(r["mape_t"]) for r in rows]
    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(9.0, 1.25 * len(labels)), 5.2), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    ax.bar(x - w / 2, v_mape, w, color=ACCENT, edgecolor="k", lw=0.4, label="Voltage MAPE")
    ax.bar(x + w / 2, t_mape, w, color=ORANGE, edgecolor="k", lw=0.4, label="Temperature MAPE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Validation MAPE (%)", fontsize=14)
    ax.set_title(
        "Digital Twin accuracy across NASA RW cells\n(pooled validation windows)",
        fontsize=15, fontweight="bold",
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12)
    for i, (vv, tt) in enumerate(zip(v_mape, t_mape)):
        ax.text(i - w / 2, vv + 0.02, f"{vv:.2f}", ha="center", va="bottom", fontsize=11)
        ax.text(i + w / 2, tt + 0.02, f"{tt:.2f}", ha="center", va="bottom", fontsize=11)
    fig.text(0.5, -0.04, footnote, ha="center", fontsize=11, color="#64748b", style="italic")
    plt.tight_layout()
    _savefig(fig, out_path)


def plot_finetune_fraction_summary(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Grouped bars: V/T MAPE vs adapt fraction for RW10–RW12."""
    cells = sorted({r["cell"] for r in rows if r.get("fraction") is not None})
    fracs = sorted({float(r["fraction"]) for r in rows if r.get("fraction") is not None})
    if not cells or not fracs:
        return
    by = {(r["cell"], float(r["fraction"])): r for r in rows if r.get("fraction") is not None}
    x = np.arange(len(fracs))
    w = 0.22
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor=LIGHT_BG, sharey=False)
    colors = {"RW10": "#2563eb", "RW11": "#16a34a", "RW12": "#ea580c"}
    for ax, key, ylab, title in (
        (axes[0], "mape_v", "Voltage MAPE (%)", "Voltage MAPE vs adapt fraction"),
        (axes[1], "mape_t", "Temperature MAPE (%)", "Temperature MAPE vs adapt fraction"),
    ):
        ax.set_facecolor(LIGHT_BG)
        for i, cell in enumerate(cells):
            vals = [float(by[(cell, f)][key]) if (cell, f) in by else np.nan for f in fracs]
            ax.bar(
                x + (i - (len(cells) - 1) / 2) * w,
                vals,
                w,
                color=colors.get(cell, GREY),
                edgecolor="k",
                lw=0.3,
                label=cell,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"{f:.0%}" for f in fracs], fontsize=12)
        ax.set_xlabel("Finetune adapt-data fraction", fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=11)
    fig.suptitle(
        "Finetuned digital twin: effect of adapt-data fraction (20% / 40% / 60%)",
        fontsize=15, fontweight="bold",
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
    ax.set_xlabel("Time (h)", fontsize=14)
    ax.set_ylabel("State of Charge (%)", fontsize=14)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.set_title(f"{cell_id} — SOC vs time", fontsize=14, fontweight="bold")

    ax = axes[1]
    ax.scatter(volt, labels_pct, s=8, alpha=0.45, color=GREEN, label="Coulomb", zorder=2)
    for variant in ("v_only", "vta", "vta_i"):
        if variant not in soc_preds:
            continue
        pred = soc_preds[variant][:n] * 100.0
        ax.scatter(
            volt, pred, s=8, alpha=0.4,
            color=SOC_VARIANT_COLORS[variant],
            label=SOC_VARIANT_LABELS.get(variant, variant),
            zorder=3 if variant == primary_variant else 1,
        )
    ax.set_xlabel("Voltage (V)", fontsize=14)
    ax.set_ylabel("SOC (%)", fontsize=14)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, markerscale=2, framealpha=0.9)
    ax.set_title("SOC vs voltage", fontsize=14, fontweight="bold")

    fig.suptitle(
        f"SOC estimation — Coulomb labels vs measured-feature MLPs ({cell_id})",
        fontsize=15,
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
