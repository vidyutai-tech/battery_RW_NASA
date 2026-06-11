"""Publication-ready plots — theme matches main repo visualize.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── shared visual style (mirrors main repo visualize.py) ─────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "lines.linewidth": 1.8,
})

ACCENT   = "#2563EB"
ORANGE   = "#EA580C"
GREEN    = "#16A34A"
PURPLE   = "#7C3AED"
GREY     = "#6B7280"
RED      = "#DC2626"
LIGHT_BG = "#F8FAFC"

CELL_COLORS = {"RW9": ACCENT, "RW10": ORANGE, "RW11": GREEN, "RW12": PURPLE}

SOC_VARIANT_COLORS = {"v_only": GREY, "vta": ACCENT, "vta_i": GREEN}
SOC_VARIANT_LABELS = {"v_only": "V only", "vta": "VTA", "vta_i": "VTA + |I|"}

FINETUNE_COLOR = ACCENT
SCRATCH_COLOR  = ORANGE
FULL_COLOR     = GREEN


def _savefig(fig: plt.Figure, path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", **kwargs)
    plt.close(fig)


# ── Training curves ───────────────────────────────────────────────────────────

def plot_twin_training_curves(log_path: Path, out_path: Path) -> None:
    """Training loss + val V/T RMSE over epochs from JSONL log."""
    log_path = Path(log_path)
    if not log_path.is_file():
        return
    epochs, train_loss, val_v, val_t = [], [], [], []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            epochs.append(row["epoch"])
            train_loss.append(row.get("train_loss"))
            val_v.append(row.get("val_voltage_rmse"))
            val_t.append(row.get("val_temp_rmse"))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), facecolor=LIGHT_BG)
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)

    axes[0].plot(epochs, train_loss, color=ACCENT, lw=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Training loss (weighted MSE)")

    axes[1].plot(epochs, val_v, color=ORANGE, lw=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (V)")
    axes[1].set_title("Val voltage RMSE")

    has_t = any(v is not None and np.isfinite(v) for v in val_t)
    if has_t:
        axes[2].plot(epochs, val_t, color=GREEN, lw=1.5)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("RMSE (°C)")
        axes[2].set_title("Val temperature RMSE")
    else:
        axes[2].set_visible(False)

    fig.suptitle("Digital twin — training curves", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_soc_training_curves(log_path: Path, out_path: Path) -> None:
    """Train loss + val RMSE / MAPE per epoch (one row per SOC variant)."""
    log_path = Path(log_path)
    if not log_path.is_file():
        return

    by_variant: Dict[str, Dict[str, list]] = {}
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            var = row.get("variant", "vta")
            bucket = by_variant.setdefault(
                var, {"epoch": [], "train_loss": [], "val_rmse": [], "val_mape_pct": []},
            )
            bucket["epoch"].append(row["epoch"])
            bucket["train_loss"].append(row.get("train_loss"))
            bucket["val_rmse"].append(row.get("val_rmse"))
            bucket["val_mape_pct"].append(row.get("val_mape_pct"))

    if not by_variant:
        return

    order = [v for v in ("v_only", "vta", "vta_i") if v in by_variant]
    order += [v for v in by_variant if v not in order]
    n_rows = len(order)

    fig, axes = plt.subplots(
        n_rows, 3, figsize=(13, 3.2 * n_rows), facecolor=LIGHT_BG, squeeze=False,
    )
    for row_i, var in enumerate(order):
        data = by_variant[var]
        ep = data["epoch"]
        color = SOC_VARIANT_COLORS.get(var, ACCENT)
        label = SOC_VARIANT_LABELS.get(var, var)
        for col_i, (key, ylabel, title) in enumerate([
            ("train_loss", "MSE", "Train loss"),
            ("val_rmse", "RMSE", "Val RMSE"),
            ("val_mape_pct", "MAPE (%)", "Val MAPE"),
        ]):
            ax = axes[row_i, col_i]
            ax.set_facecolor(LIGHT_BG)
            ax.plot(ep, data[key], color=color, lw=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            if col_i == 0:
                ax.set_title(f"{label} — {title}", fontsize=9, fontweight="bold")
            else:
                ax.set_title(title, fontsize=9)

    fig.suptitle("SOC MLP — training curves (measured V/T)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_path)


# ── Twin prediction panels ────────────────────────────────────────────────────

def plot_twin_predictions(
    model,
    batch,
    device,
    out_path: Path,
    n_panels: int = 4,
    title_prefix: str = "Digital twin",
) -> None:
    from rw_transfer.training.twin_trainer import predict_twin_batch

    out_path = Path(out_path)
    if len(batch.X) == 0:
        return
    v_pred, t_pred = predict_twin_batch(model, batch, device)
    n = min(n_panels, len(batch.X))
    idxs = np.linspace(0, len(batch.X) - 1, n, dtype=int)
    T = batch.Y_voltage.shape[1]
    steps = np.arange(T)

    fig, axes = plt.subplots(n, 2, figsize=(11, 2.8 * n), facecolor=LIGHT_BG)
    if n == 1:
        axes = np.array([axes])
    for ax in axes.flat:
        ax.set_facecolor(LIGHT_BG)

    for row_ax, i in zip(axes, idxs):
        row_ax[0].plot(steps, batch.Y_voltage[i], color=GREY, lw=1.4, label="Measured")
        row_ax[0].plot(steps, v_pred[i], color=ACCENT, lw=1.6, ls="--", label="Predicted")
        row_ax[0].set_ylabel("Voltage (V)")
        row_ax[0].legend(fontsize=8, framealpha=0.6)
        v_rmse = float(np.sqrt(np.mean((v_pred[i] - batch.Y_voltage[i]) ** 2)))
        row_ax[0].set_title(f"Window {i}  |  RMSE = {v_rmse:.4f} V", fontsize=9)

        row_ax[1].plot(steps, batch.Y_temperature[i], color=GREY, lw=1.4, label="Measured")
        row_ax[1].plot(steps, t_pred[i], color=ORANGE, lw=1.6, ls="--", label="Predicted")
        row_ax[1].set_ylabel("Temperature (°C)")
        row_ax[1].legend(fontsize=8, framealpha=0.6)
        t_rmse = float(np.sqrt(np.mean((t_pred[i] - batch.Y_temperature[i]) ** 2)))
        row_ax[1].set_title(f"Window {i}  |  RMSE = {t_rmse:.4f} °C", fontsize=9)

    axes[-1, 0].set_xlabel("Step in window")
    axes[-1, 1].set_xlabel("Step in window")
    fig.suptitle(f"{title_prefix} — held-out predictions", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_path)


# ── SOC comparison (3 variants) ───────────────────────────────────────────────

def plot_soc_variant_comparison(
    results: Dict[str, Dict],
    out_path: Path,
    cell_id: str = "RW9",
) -> None:
    """Bar chart comparing RMSE / MAE / MAPE for v_only, vta, vta_i."""
    variants = [v for v in ("v_only", "vta", "vta_i") if v in results]
    if not variants:
        return
    metrics = ["rmse", "mae", "mape_pct"]
    labels  = ["RMSE", "MAE", "MAPE (%)"]
    x = np.arange(len(metrics))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    for j, var in enumerate(variants):
        vals = [results[var].get(m, 0) for m in metrics]
        bars = ax.bar(
            x + j * w, vals, w,
            label=SOC_VARIANT_LABELS.get(var, var),
            color=SOC_VARIANT_COLORS[var],
            alpha=0.88,
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=7.5,
            )
    ax.set_xticks(x + w)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric value")
    ax.set_title(f"SOC estimation variants — {cell_id} (Coulomb labels, measured V/T)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_soc_prediction_series(
    time_s: np.ndarray,
    soc_labels: np.ndarray,
    soc_preds: Dict[str, np.ndarray],
    out_path: Path,
    cell_id: str = "RW9",
    max_points: int = 5000,
) -> None:
    """Time-series overlay of Coulomb labels vs each SOC variant prediction."""
    n = min(max_points, len(time_s), len(soc_labels))
    for pred in soc_preds.values():
        n = min(n, len(pred))
    t = time_s[:n]
    t_h = (t.astype(np.float64) - t[0]) / 3600.0

    n_variants = len(soc_preds)
    fig, axes = plt.subplots(n_variants, 1, figsize=(11, 2.6 * n_variants),
                             sharex=True, facecolor=LIGHT_BG)
    if n_variants == 1:
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)

    for ax, (variant, pred) in zip(axes, soc_preds.items()):
        ax.plot(t_h, soc_labels[:n], color=GREY, lw=1.2, label="Coulomb label")
        ax.plot(t_h, pred[:n], color=SOC_VARIANT_COLORS[variant],
                lw=1.5, ls="--", label=SOC_VARIANT_LABELS.get(variant, variant))
        rmse = float(np.sqrt(np.mean((pred[:n] - soc_labels[:n]) ** 2)))
        ax.set_ylabel("SOC")
        ax.set_title(f"{SOC_VARIANT_LABELS.get(variant, variant)}  |  RMSE = {rmse:.4f}", fontsize=9)
        ax.legend(fontsize=8, framealpha=0.6)

    axes[-1].set_xlabel("Time (h)")
    fig.suptitle(f"SOC estimation — {cell_id}", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_path)


# ── Finetune vs scratch (percentage sweep) ───────────────────────────────────

def plot_finetune_vs_scratch_percent(
    rows: List[Dict[str, Any]],
    target: str,
    out_path: Path,
) -> None:
    """RMSE vs % target data: fine-tune from RW9 vs train from scratch."""
    sub = [r for r in rows if r.get("target") == target]
    if not sub:
        return

    fracs = [r["fraction"] * 100 for r in sub]
    ft_v  = [r.get("finetune_voltage_rmse") for r in sub]
    sc_v  = [r.get("scratch_voltage_rmse") for r in sub]
    ft_t  = [r.get("finetune_temp_rmse") for r in sub]
    sc_t  = [r.get("scratch_temp_rmse") for r in sub]

    fig = plt.figure(figsize=(12, 4.5), facecolor=LIGHT_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax_v = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    for ax in (ax_v, ax_t):
        ax.set_facecolor(LIGHT_BG)

    ax_v.plot(fracs, ft_v, "o-", color=FINETUNE_COLOR, lw=1.8,
              ms=5, label=f"Fine-tune from RW9")
    ax_v.plot(fracs, sc_v, "s--", color=SCRATCH_COLOR, lw=1.8,
              ms=5, label="Train from scratch")
    ax_v.set_xlabel("Target data used (%)")
    ax_v.set_ylabel("Held-out voltage RMSE (V)")
    ax_v.set_title("Voltage RMSE", fontweight="bold")
    ax_v.legend(fontsize=9)

    ax_t.plot(fracs, ft_t, "o-", color=FINETUNE_COLOR, lw=1.8, ms=5)
    ax_t.plot(fracs, sc_t, "s--", color=SCRATCH_COLOR, lw=1.8, ms=5)
    ax_t.set_xlabel("Target data used (%)")
    ax_t.set_ylabel("Held-out temperature RMSE (°C)")
    ax_t.set_title("Temperature RMSE", fontweight="bold")

    fig.suptitle(
        f"Transfer learning — {target} (source: RW9)\n"
        f"Fine-tune vs train from scratch",
        fontsize=11, fontweight="bold",
    )
    _savefig(fig, out_path)


def plot_finetune_gain_percent(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Transfer gain (scratch RMSE − finetune RMSE) vs % for all targets."""
    targets = sorted({r["target"] for r in rows})
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    for cell in targets:
        sub = [r for r in rows if r["target"] == cell]
        fracs = [r["fraction"] * 100 for r in sub]
        gains = [r.get("transfer_gain_rmse", 0) for r in sub]
        ax.plot(fracs, gains, "o-", color=CELL_COLORS.get(cell, GREY),
                lw=1.6, ms=5, label=cell)

    ax.axhline(0, color=GREY, lw=0.9, ls=":")
    ax.set_xlabel("Target data used (%)")
    ax.set_ylabel("Transfer gain  (RMSE_scratch − RMSE_finetune)  [V]")
    ax.set_title("Transfer learning gain vs target data fraction", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_finetune_percent(
    rows: List[Dict[str, Any]],
    target: str,
    out_path: Path,
) -> None:
    """Voltage and temperature RMSE vs % target data — two-stage fine-tuning only."""
    sub = [r for r in rows if r.get("target") == target]
    if not sub:
        return

    fracs = [r["fraction"] * 100 for r in sub]
    ft_v  = [r.get("finetune_voltage_rmse") for r in sub]
    ft_t  = [r.get("finetune_temp_rmse") for r in sub]

    fig = plt.figure(figsize=(12, 4.5), facecolor=LIGHT_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax_v = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    for ax in (ax_v, ax_t):
        ax.set_facecolor(LIGHT_BG)

    ax_v.plot(fracs, ft_v, "o-", color=FINETUNE_COLOR, lw=1.8, ms=5,
              label="Two-stage fine-tune from RW9")
    ax_v.set_xlabel("Target data used (%)")
    ax_v.set_ylabel("Held-out voltage RMSE (V)")
    ax_v.set_title("Voltage RMSE", fontweight="bold")
    ax_v.legend(fontsize=9)

    ax_t.plot(fracs, ft_t, "o-", color=GREEN, lw=1.8, ms=5,
              label="Two-stage fine-tune from RW9")
    ax_t.set_xlabel("Target data used (%)")
    ax_t.set_ylabel("Held-out temperature RMSE (°C)")
    ax_t.set_title("Temperature RMSE", fontweight="bold")
    ax_t.legend(fontsize=9)

    fig.suptitle(
        f"Transfer learning — {target} (source: RW9)\n"
        f"Two-stage temperature-aware fine-tuning",
        fontsize=11, fontweight="bold",
    )
    _savefig(fig, out_path)


# ── Hours-based adaptation (Phase 3 equivalent) ─────────────────────────────

def plot_finetune_vs_scratch_hours(
    rows: List[Dict[str, Any]],
    target: str,
    out_path: Path,
) -> None:
    """RMSE vs adaptation hours: fine-tune, scratch, and full-finetune ceiling."""
    sub = [r for r in rows
           if r.get("target") == target and r.get("adaptation_label") != "full"]
    if not sub:
        return

    hours  = [r["adaptation_hours"] for r in sub]
    ft_v   = [r.get("finetune_voltage_rmse") for r in sub]
    sc_v   = [r.get("scratch_voltage_rmse") for r in sub]
    ft_t   = [r.get("finetune_temperature_rmse") for r in sub]
    sc_t   = [r.get("scratch_temperature_rmse") for r in sub]
    r_full = sub[0].get("full_finetune_voltage_rmse")

    fig = plt.figure(figsize=(13, 4.5), facecolor=LIGHT_BG)
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    ax_v = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1])
    for ax in (ax_v, ax_t):
        ax.set_facecolor(LIGHT_BG)

    ax_v.plot(hours, ft_v, "o-",  color=FINETUNE_COLOR, lw=1.8, ms=5, label="Fine-tune from RW9")
    ax_v.plot(hours, sc_v, "s--", color=SCRATCH_COLOR,  lw=1.8, ms=5, label="Train from scratch")
    if r_full is not None and np.isfinite(r_full):
        ax_v.axhline(r_full, color=FULL_COLOR, lw=1.4, ls=":", label="Full fine-tune (ceiling)")
    ax_v.set_xlabel("Adaptation data duration (hours)")
    ax_v.set_ylabel("Held-out voltage RMSE (V)")
    ax_v.set_title("Voltage RMSE", fontweight="bold")
    ax_v.legend(fontsize=9)

    ax_t.plot(hours, ft_t, "o-",  color=FINETUNE_COLOR, lw=1.8, ms=5)
    ax_t.plot(hours, sc_t, "s--", color=SCRATCH_COLOR,  lw=1.8, ms=5)
    ax_t.set_xlabel("Adaptation data duration (hours)")
    ax_t.set_ylabel("Held-out temperature RMSE (°C)")
    ax_t.set_title("Temperature RMSE", fontweight="bold")

    fig.suptitle(
        f"Minimum adaptation data study — {target} (source: RW9)\n"
        "How much target data is needed for effective transfer?",
        fontsize=11, fontweight="bold",
    )
    _savefig(fig, out_path)


def plot_transfer_gain_hours(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Transfer gain vs hours for all target cells on one axes."""
    targets = sorted({r["target"] for r in rows if r.get("adaptation_label") != "full"})
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    for cell in targets:
        sub = [r for r in rows
               if r["target"] == cell and r.get("adaptation_label") != "full"]
        hours = [r["adaptation_hours"] for r in sub]
        gains = [r.get("transfer_gain_rmse", 0) for r in sub]
        ax.plot(hours, gains, "o-", color=CELL_COLORS.get(cell, GREY),
                lw=1.6, ms=5, label=cell)

    ax.axhline(0, color=GREY, lw=0.9, ls=":")
    ax.set_xlabel("Adaptation data duration (hours)")
    ax.set_ylabel("Transfer gain  (RMSE_scratch − RMSE_finetune)  [V]")
    ax.set_title("Transfer gain vs adaptation duration — all targets", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_gap_closed_hours(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Gap-closed fraction (0–1) vs hours.

    gap_fraction = (RMSE_scratch - RMSE_ft) / (RMSE_scratch - RMSE_full)
    1.0 means fine-tune matched full-data training.
    """
    targets = sorted({r["target"] for r in rows if r.get("adaptation_label") != "full"})
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    for thr, ls in ((0.90, ":"), (0.95, "--"), (0.99, "-.")):
        ax.axhline(thr, color=GREY, lw=0.9, ls=ls, alpha=0.7,
                   label=f"{int(thr*100)}% threshold")

    for cell in targets:
        sub = [r for r in rows
               if r["target"] == cell and r.get("adaptation_label") != "full"]
        hours = [r["adaptation_hours"] for r in sub]
        gaps  = [r.get("gap_fraction_finetune", float("nan")) for r in sub]
        ax.plot(hours, gaps, "o-", color=CELL_COLORS.get(cell, GREY),
                lw=1.6, ms=5, label=cell)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Adaptation data duration (hours)")
    ax.set_ylabel("Gap-closed fraction\n(toward full fine-tune RMSE)")
    ax.set_title("Adaptation efficiency — how quickly does transfer converge?", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out_path)


def plot_threshold_bar_chart(rec: Dict[str, Any], out_path: Path) -> None:
    """Grouped bars: minimum hours for 90 / 95 / 98 / 99% per target cell."""
    targets = list(rec.keys())
    thresholds = [("hours_for_90pct", "90%"),
                  ("hours_for_95pct", "95%"),
                  ("hours_for_98pct", "98%"),
                  ("hours_for_99pct", "99%")]
    colors = [ACCENT, ORANGE, GREEN, PURPLE]

    x = np.arange(len(targets))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor=LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)

    for j, (key, label) in enumerate(thresholds):
        vals = [rec[t]["thresholds"].get(key) or np.nan for t in targets]
        bars = ax.bar(x + j * w, vals, w, color=colors[j], alpha=0.88, label=label)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.15,
                    f"{val:.1f}h",
                    ha="center", va="bottom", fontsize=7.5,
                )

    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(targets)
    ax.set_ylabel("Adaptation hours required")
    ax.set_title(
        "Minimum target-battery data for fine-tuning effectiveness\n"
        "(90 / 95 / 98 / 99% gap closed vs full fine-tune, voltage RMSE)",
        fontweight="bold",
    )
    ax.legend(title="Performance threshold")
    fig.tight_layout()
    _savefig(fig, out_path)


# ── Fine-tune training curves (Stage 1 + Stage 2 from JSONL) ─────────────────

def plot_finetune_training_curves(
    log_path: Path,
    out_path: Path,
    stage_label: str = "Stage",
) -> None:
    """Plot train loss + 5 val metrics (RMSE, MSE, MAE, MAPE, R²) from a JSONL log.

    One column per metric.  Call once for Stage 1 log and once for Stage 2 log.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        return

    epochs, train_loss = [], []
    v_rmse, v_mse, v_mae, v_mape, v_r2 = [], [], [], [], []
    t_rmse, t_mse, t_mae, t_mape, t_r2 = [], [], [], [], []

    with log_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            epochs.append(r["epoch"])
            train_loss.append(r.get("train_loss"))
            v_rmse.append(r.get("val_voltage_rmse"))
            v_mse.append(r.get("val_voltage_mse"))
            v_mae.append(r.get("val_voltage_mae"))
            v_mape.append(r.get("mape_v"))
            v_r2.append(r.get("val_voltage_r2"))
            t_rmse.append(r.get("val_temp_rmse"))
            t_mse.append(r.get("val_temp_mse"))
            t_mae.append(r.get("val_temp_mae"))
            t_mape.append(r.get("mape_t"))
            t_r2.append(r.get("val_temp_r2"))

    if not epochs:
        return

    metrics = [
        ("Train loss",   [train_loss],        [""],                  [ACCENT]),
        ("RMSE",         [v_rmse, t_rmse],    ["Voltage (V)", "Temp (°C)"], [ORANGE, GREEN]),
        ("MSE",          [v_mse,  t_mse],     ["Voltage",     "Temp"],      [ORANGE, GREEN]),
        ("MAE",          [v_mae,  t_mae],     ["Voltage (V)", "Temp (°C)"], [ORANGE, GREEN]),
        ("MAPE (%)",     [v_mape, t_mape],    ["Voltage",     "Temp"],      [ORANGE, GREEN]),
        ("R²",           [v_r2,   t_r2],      ["Voltage",     "Temp"],      [ORANGE, GREEN]),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.5 * len(metrics), 3.5),
                             facecolor=LIGHT_BG)
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)

    for ax, (title, series_list, labels, colors) in zip(axes, metrics):
        for series, label, color in zip(series_list, labels, colors):
            clean = [v if v is not None and np.isfinite(v) else np.nan for v in series]
            ax.plot(epochs, clean, color=color, lw=1.5,
                    label=label if label else None)
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontsize=9, fontweight="bold")
        if len(series_list) > 1:
            ax.legend(fontsize=7, framealpha=0.6)

    fig.suptitle(f"{stage_label} — validation metrics per epoch",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    _savefig(fig, out_path)


# ── Actual vs Predicted (scatter + time-series) ───────────────────────────────

def plot_actual_vs_predicted(
    v_pred: np.ndarray,
    v_ref: np.ndarray,
    t_pred: np.ndarray,
    t_ref: np.ndarray,
    out_path: Path,
    *,
    target: str = "",
    fraction: float = 0.0,
    n_series_panels: int = 3,
    max_scatter: int = 8000,
) -> None:
    """Three-section figure: scatter (V), scatter (T), time-series overlay panels.

    Parameters
    ----------
    v_pred / v_ref : flat arrays of all predicted / actual voltage values
    t_pred / t_ref : flat arrays of all predicted / actual temperature values
    n_series_panels: number of representative window overlays to show
    """
    from rw_transfer.metrics import rmse, mape, r2_score

    v_pred = np.asarray(v_pred, dtype=np.float64).ravel()
    v_ref  = np.asarray(v_ref,  dtype=np.float64).ravel()
    t_pred = np.asarray(t_pred, dtype=np.float64).ravel()
    t_ref  = np.asarray(t_ref,  dtype=np.float64).ravel()

    title_tag = f"{target}  |  {fraction:.0%} data" if target else f"{fraction:.0%} data"

    # ── scatter subsample ──────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    sc_idx = rng.choice(len(v_pred), size=min(max_scatter, len(v_pred)), replace=False)

    fig = plt.figure(figsize=(16, 4.5 * (1 + n_series_panels // 2 + 1)),
                     facecolor=LIGHT_BG)
    gs = gridspec.GridSpec(
        2 + n_series_panels, 2,
        figure=fig,
        height_ratios=[1.0, 1.0] + [0.8] * n_series_panels,
        hspace=0.45, wspace=0.35,
    )

    # ── voltage scatter ────────────────────────────────────────────────────────
    ax_vs = fig.add_subplot(gs[0, 0])
    ax_vs.set_facecolor(LIGHT_BG)
    ax_vs.scatter(v_ref[sc_idx], v_pred[sc_idx], s=3, alpha=0.35, color=ACCENT, rasterized=True)
    lo, hi = min(v_ref.min(), v_pred.min()), max(v_ref.max(), v_pred.max())
    ax_vs.plot([lo, hi], [lo, hi], color=GREY, lw=1.2, ls="--", label="Ideal")
    v_r  = rmse(v_pred, v_ref)
    v_mp = mape(v_pred, v_ref)
    v_r2 = r2_score(v_pred, v_ref)
    ax_vs.set_xlabel("Actual Voltage (V)")
    ax_vs.set_ylabel("Predicted Voltage (V)")
    ax_vs.set_title(
        f"Voltage — Actual vs Predicted\n"
        f"RMSE={v_r:.4f} V  MAPE={v_mp:.3f}%  R²={v_r2:.4f}",
        fontsize=9, fontweight="bold",
    )
    ax_vs.legend(fontsize=8)

    # ── temperature scatter ────────────────────────────────────────────────────
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_ts.set_facecolor(LIGHT_BG)
    ax_ts.scatter(t_ref[sc_idx], t_pred[sc_idx], s=3, alpha=0.35, color=ORANGE, rasterized=True)
    lo, hi = min(t_ref.min(), t_pred.min()), max(t_ref.max(), t_pred.max())
    ax_ts.plot([lo, hi], [lo, hi], color=GREY, lw=1.2, ls="--", label="Ideal")
    t_r  = rmse(t_pred, t_ref)
    t_mp = mape(t_pred, t_ref)
    t_r2 = r2_score(t_pred, t_ref)
    ax_ts.set_xlabel("Actual Temperature (°C)")
    ax_ts.set_ylabel("Predicted Temperature (°C)")
    ax_ts.set_title(
        f"Temperature — Actual vs Predicted\n"
        f"RMSE={t_r:.4f} °C  MAPE={t_mp:.3f}%  R²={t_r2:.4f}",
        fontsize=9, fontweight="bold",
    )
    ax_ts.legend(fontsize=8)

    # ── residual histograms ────────────────────────────────────────────────────
    ax_vr = fig.add_subplot(gs[1, 0])
    ax_tr = fig.add_subplot(gs[1, 1])
    for ax in (ax_vr, ax_tr):
        ax.set_facecolor(LIGHT_BG)

    v_res = v_pred - v_ref
    ax_vr.hist(v_res, bins=60, color=ACCENT, alpha=0.75, edgecolor="none")
    ax_vr.axvline(0, color=GREY, lw=1.2, ls="--")
    ax_vr.set_xlabel("Residual (V)")
    ax_vr.set_ylabel("Count")
    ax_vr.set_title(
        f"Voltage residuals  (μ={v_res.mean():.4f}, σ={v_res.std():.4f})",
        fontsize=9,
    )

    t_res = t_pred - t_ref
    ax_tr.hist(t_res, bins=60, color=ORANGE, alpha=0.75, edgecolor="none")
    ax_tr.axvline(0, color=GREY, lw=1.2, ls="--")
    ax_tr.set_xlabel("Residual (°C)")
    ax_tr.set_ylabel("Count")
    ax_tr.set_title(
        f"Temperature residuals  (μ={t_res.mean():.4f}, σ={t_res.std():.4f})",
        fontsize=9,
    )

    # ── time-series overlay panels ─────────────────────────────────────────────
    seq_len = len(v_pred) // max(1, len(v_pred) // 150)
    seq_len = max(10, min(seq_len, 150))
    n_wins  = len(v_pred) // seq_len
    if n_wins >= n_series_panels:
        panel_idxs = np.linspace(0, n_wins - 1, n_series_panels, dtype=int)
        steps = np.arange(seq_len)
        for pi, wi in enumerate(panel_idxs):
            row_i = 2 + pi
            s, e = wi * seq_len, (wi + 1) * seq_len
            ax_v = fig.add_subplot(gs[row_i, 0])
            ax_t = fig.add_subplot(gs[row_i, 1])
            for ax in (ax_v, ax_t):
                ax.set_facecolor(LIGHT_BG)

            ax_v.plot(steps, v_ref[s:e],  color=GREY,   lw=1.3, label="Actual")
            ax_v.plot(steps, v_pred[s:e], color=ACCENT, lw=1.5, ls="--", label="Predicted")
            wv = float(np.sqrt(np.mean((v_pred[s:e] - v_ref[s:e]) ** 2)))
            ax_v.set_title(f"Window {wi}  |  V RMSE={wv:.4f}", fontsize=8)
            ax_v.set_ylabel("Voltage (V)")
            ax_v.legend(fontsize=7, framealpha=0.6)

            ax_t.plot(steps, t_ref[s:e],  color=GREY,   lw=1.3, label="Actual")
            ax_t.plot(steps, t_pred[s:e], color=ORANGE, lw=1.5, ls="--", label="Predicted")
            wt = float(np.sqrt(np.mean((t_pred[s:e] - t_ref[s:e]) ** 2)))
            ax_t.set_title(f"Window {wi}  |  T RMSE={wt:.4f}", fontsize=8)
            ax_t.set_ylabel("Temp (°C)")
            ax_t.legend(fontsize=7, framealpha=0.6)

            if pi == n_series_panels - 1:
                ax_v.set_xlabel("Step in window")
                ax_t.set_xlabel("Step in window")

    fig.suptitle(
        f"Finetuned twin — Actual vs Predicted  [{title_tag}]",
        fontsize=11, fontweight="bold", y=1.01,
    )
    _savefig(fig, out_path)


# ── EDA helpers ───────────────────────────────────────────────────────────────

def plot_cell_overview(series, out_path: Path, max_points: int = 8000) -> None:
    """Voltage / Current / Temperature / Age for one cell."""
    n = min(max_points, len(series.time_s))
    t_h = (series.time_s[:n] - series.time_s[0]) / 3600.0

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True, facecolor=LIGHT_BG)
    for ax in axes:
        ax.set_facecolor(LIGHT_BG)

    axes[0].plot(t_h, series.voltage_v[:n],     color=ACCENT,  lw=0.9)
    axes[0].set_ylabel("Voltage (V)")
    axes[1].plot(t_h, series.current_a[:n],     color=ORANGE,  lw=0.9)
    axes[1].set_ylabel("Current (A)")
    axes[2].plot(t_h, series.temperature_c[:n], color=RED,     lw=0.9)
    axes[2].set_ylabel("Temp (°C)")
    axes[3].plot(t_h, series.age[:n],            color=PURPLE,  lw=0.9)
    axes[3].set_ylabel("Relative age")
    axes[-1].set_xlabel("Time (h)")
    axes[0].set_title(
        f"{series.cell_id} — V / I / T / age  (first {n:,} samples, decimated)",
        fontweight="bold",
    )
    fig.tight_layout()
    _savefig(fig, out_path)
