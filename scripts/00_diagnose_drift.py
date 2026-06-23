#!/usr/bin/env python3
"""
Chained-rollout drift diagnostic for the frozen BDT.

The headline metrics (V RMSE 0.030 V) are teacher-forced over single 150-step
windows. The charging optimizer instead runs the twin OPEN-LOOP: chunks are
chained (each chunk starts from the previous chunk's last *predicted* V/T), so
error compounds. This script quantifies that compounding — it gates every
downstream RL result and fixes the per-horizon uncertainty margins used for
chance-constrained action filtering.

Three evaluations (all against measured data, real current fed to the twin):

1. RW open-loop drift (dt = 1 s, the optimizer's operating regime)
   Random 30-min segments of the stitched RW series across the cell's life.
   -> median / p95 |V_hat - V| and |T_hat - T| vs prediction horizon.
   The p95 curves are saved as conformal-style margins.

2. Per-action conditional error (single chunk, 150 s)
   Windows inside real charge (random walk) steps where the measured current
   is approximately constant at one of the 6 policy setpoints
   {-0.75, -1.5, -2.25, -3.0, -3.75, -4.5} A. -> RMSE per action level.

3. Reference-charge sessions (dt = 10 s — caveat: different sampling period,
   the twin has no dt input) — full CC-CV charge sessions, the closest analog
   to a complete charging session. Reported separately.

Outputs (under --out_dir, default outputs/charging_opt/drift/)
    drift_summary.json
    conformal_margins.npz       (horizon_s, v_q50, v_q95, t_q50, t_q95)
    plots/drift_vs_horizon.png
    plots/per_action_rmse.png

Usage
-----
    venv/bin/python scripts/00_diagnose_drift.py \
        --ckpt outputs/twin_source/20260601_182816/twin_source_RW9.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rw_transfer.config import load_config
from rw_transfer.data.series import load_battery_series
from charging_opt.bdt_rollout import FrozenBDTSimulator
from charging_opt import paths as P
from charging_opt.artifacts import CANONICAL, update_master_registry, write_stage_registry

ACTION_SET = [-0.75, -1.5, -2.25, -3.0, -3.75, -4.5]


# ---------------------------------------------------------------------------
# Segment selection helpers
# ---------------------------------------------------------------------------

def contiguous_1hz_spans(time_s: np.ndarray, tol: float = 0.15) -> list[tuple[int, int]]:
    """
    [start, end) index spans without sampling gaps (dt <= ~1 s).

    The RW stream is nominally 1 Hz but contains occasional sub-second samples;
    those do not break continuity (training windows had the same property).
    Only true gaps (dt > 1 + tol, e.g. step transitions at 10 s) split spans.
    """
    dt = np.diff(time_s)
    ok = (dt > 0) & (dt <= 1.0 + tol)
    spans, start = [], None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            spans.append((start, i + 1))
            start = None
    if start is not None:
        spans.append((start, len(time_s)))
    return spans


def sample_segments(
    spans: list[tuple[int, int]],
    horizon: int,
    n_segments: int,
    rng: np.random.Generator,
) -> list[int]:
    """Segment start indices, spread across all eligible spans."""
    eligible = [(a, b) for a, b in spans if b - a > horizon + 1]
    if not eligible:
        return []
    weights = np.array([b - a - horizon for a, b in eligible], dtype=np.float64)
    weights /= weights.sum()
    counts = rng.multinomial(n_segments, weights)
    starts = []
    for (a, b), c in zip(eligible, counts):
        if c == 0:
            continue
        starts.extend(
            int(s) for s in np.linspace(a, b - horizon - 1, c).astype(int)
        )
    return sorted(starts)


# ---------------------------------------------------------------------------
# Evaluations
# ---------------------------------------------------------------------------

def eval_open_loop_drift(sim, series, horizon: int, n_segments: int, seed: int):
    rng = np.random.default_rng(seed)
    spans = contiguous_1hz_spans(series.time_s)
    starts = sample_segments(spans, horizon, n_segments, rng)
    print(f"  {len(spans)} contiguous 1 Hz spans; evaluating {len(starts)} "
          f"segments of {horizon} s")

    err_v = np.full((len(starts), horizon), np.nan, dtype=np.float32)
    err_t = np.full((len(starts), horizon), np.nan, dtype=np.float32)
    for k, s in enumerate(starts):
        e = s + horizon
        v_hat, t_hat = sim.predict_traj(
            age=float(series.age[s]),
            v0=float(series.voltage_v[s]),
            t0=float(series.temperature_c[s]),
            current_profile=series.current_a[s:e],
        )
        err_v[k] = np.abs(v_hat - series.voltage_v[s:e])
        err_t[k] = np.abs(t_hat - series.temperature_c[s:e])
        if (k + 1) % 50 == 0:
            print(f"    {k + 1}/{len(starts)} segments done")
    return err_v, err_t, starts


def eval_per_action(sim, series, seq_len: int, max_per_action: int, seed: int):
    """Single-chunk RMSE on near-constant-current charge windows per setpoint."""
    rng = np.random.default_rng(seed)
    spans = contiguous_1hz_spans(series.time_s)
    cur = series.current_a
    results = {}
    for action in ACTION_SET:
        cand = []
        for a, b in spans:
            if b - a <= seq_len:
                continue
            for s in range(a, b - seq_len, seq_len):
                w = cur[s: s + seq_len]
                if np.abs(w.mean() - action) < 0.15 and w.std() < 0.05:
                    cand.append(s)
        if len(cand) > max_per_action:
            cand = list(rng.choice(cand, size=max_per_action, replace=False))
        se_v, se_t = [], []
        for s in cand:
            e = s + seq_len
            v_hat, t_hat = sim.predict_traj(
                float(series.age[s]), float(series.voltage_v[s]),
                float(series.temperature_c[s]), cur[s:e],
            )
            se_v.append(np.mean((v_hat - series.voltage_v[s:e]) ** 2))
            se_t.append(np.mean((t_hat - series.temperature_c[s:e]) ** 2))
        results[action] = {
            "n_windows": len(cand),
            "v_rmse": float(np.sqrt(np.mean(se_v))) if se_v else None,
            "t_rmse": float(np.sqrt(np.mean(se_t))) if se_t else None,
        }
        print(f"    I={action:+.2f} A: n={len(cand):4d}  "
              f"V RMSE={results[action]['v_rmse']}  T RMSE={results[action]['t_rmse']}")
    return results


def eval_reference_charges(sim, matlab_dir: str, cell: str):
    """Open-loop over full reference-charge sessions (dt = 10 s, see caveat)."""
    from charging_opt.soc_utils import load_steps_with_age

    steps, step_age = load_steps_with_age(matlab_dir, cell)
    rows = []
    for i, s in enumerate(steps):
        if s.comment.strip().lower() != "reference charge":
            continue
        n = s.voltage_v.size
        if n < 60:
            continue
        v_hat, t_hat = sim.predict_traj(
            float(step_age[i]), float(s.voltage_v[0]),
            float(s.temperature_c[0]), s.current_a,
        )
        rows.append({
            "step": i,
            "age": float(step_age[i]),
            "n": int(n),
            "v_rmse": float(np.sqrt(np.mean((v_hat - s.voltage_v) ** 2))),
            "v_max_abs": float(np.max(np.abs(v_hat - s.voltage_v))),
            "t_rmse": float(np.sqrt(np.mean((t_hat - s.temperature_c) ** 2))),
        })
    return rows


# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="BDT chained-rollout drift diagnostic.")
    p.add_argument("--ckpt", default="outputs/twin_source/20260601_182816/twin_source_RW9.pt")
    p.add_argument("--cell", default="RW9")
    p.add_argument("--config", default=None)
    p.add_argument("--horizon", type=int, default=1800, help="drift horizon (s)")
    p.add_argument("--n_segments", type=int, default=150)
    p.add_argument("--max_per_action", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--artifact_root",
        default=None,
        help="Per-cell drift output root (e.g. outputs/charging_opt/cells/RW10). "
             "Writes stage1_drift/ and plots/stage1_drift/.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    matlab_dir = cfg["data"]["matlab_dir"]
    P.ensure_layout(ROOT)
    if args.artifact_root:
        art = Path(args.artifact_root)
        if not art.is_absolute():
            art = ROOT / art
        models_dir = art / "stage1_drift"
        plots_dir = art / "plots" / "stage1_drift"
    else:
        models_dir = ROOT / P.STAGE1_DRIFT_MODELS
        plots_dir = ROOT / P.STAGE1_DRIFT_PLOTS
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    sim = FrozenBDTSimulator(args.ckpt)
    print(f"Loading {args.cell} stitched series (all steps) ...")
    series = load_battery_series(matlab_dir, args.cell, step_mode="all")
    print(f"  {series.voltage_v.size:,} samples, {series.duration_hours:.0f} h")

    # ── 1. Open-loop drift at 1 Hz ─────────────────────────────────────────────
    print("\n[1/3] Open-loop drift on RW segments (dt = 1 s) ...")
    err_v, err_t, _ = eval_open_loop_drift(
        sim, series, args.horizon, args.n_segments, args.seed
    )
    v_q50 = np.nanmedian(err_v, axis=0)
    v_q95 = np.nanpercentile(err_v, 95, axis=0)
    t_q50 = np.nanmedian(err_t, axis=0)
    t_q95 = np.nanpercentile(err_t, 95, axis=0)
    horizon_s = np.arange(1, args.horizon + 1)
    np.savez(
        models_dir / "conformal_margins.npz",
        horizon_s=horizon_s, v_q50=v_q50, v_q95=v_q95, t_q50=t_q50, t_q95=t_q95,
    )

    checkpoints = [w for w in (150, 300, 600, 900, 1800) if w <= args.horizon]
    drift_at = {
        str(w): {
            "v_q50": float(v_q50[w - 1]), "v_q95": float(v_q95[w - 1]),
            "t_q50": float(t_q50[w - 1]), "t_q95": float(t_q95[w - 1]),
        }
        for w in checkpoints
    }
    print("\n  Drift vs horizon (|error|, median / p95):")
    for w in checkpoints:
        d = drift_at[str(w)]
        print(f"    {w:5d} s : V {d['v_q50']:.4f}/{d['v_q95']:.4f} V   "
              f"T {d['t_q50']:.3f}/{d['t_q95']:.3f} degC")

    # ── 2. Per-action conditional error ───────────────────────────────────────
    print("\n[2/3] Per-action single-chunk RMSE (charge setpoints) ...")
    per_action = eval_per_action(
        sim, series, sim.seq_len, args.max_per_action, args.seed
    )

    # ── 3. Reference-charge sessions ──────────────────────────────────────────
    print("\n[3/3] Reference-charge sessions (dt = 10 s, caveat: dt mismatch) ...")
    ref_rows = eval_reference_charges(sim, matlab_dir, args.cell)
    if ref_rows:
        rv = [r["v_rmse"] for r in ref_rows]
        rt = [r["t_rmse"] for r in ref_rows]
        print(f"  {len(ref_rows)} sessions | V RMSE median={np.median(rv):.4f} "
              f"p95={np.percentile(rv, 95):.4f} | T RMSE median={np.median(rt):.3f}")

    # Distribution shift: reference-charge vs random-walk accuracy gap
    distribution_shift = None
    if ref_rows and drift_at:
        rw_v_q50_150 = drift_at.get("150", {}).get("v_q50", float("nan"))
        ref_v_rmse_median = float(np.median([r["v_rmse"] for r in ref_rows]))
        ratio = ref_v_rmse_median / rw_v_q50_150 if rw_v_q50_150 > 0 else float("nan")
        print(
            f"\n  DISTRIBUTION SHIFT WARNING:\n"
            f"  Random-walk V error (150s median): {rw_v_q50_150:.4f} V\n"
            f"  Reference-charge V RMSE (median): {ref_v_rmse_median:.4f} V\n"
            f"  Ratio: {ratio:.1f}x\n"
            f"  BDT is {ratio:.1f}x less accurate on CC-like profiles "
            f"than on random-walk segments.\n"
            f"  BO results for smooth charging profiles should be "
            f"treated as approximate."
        )
        distribution_shift = {
            "rw_v_rmse_q50_at_150s": rw_v_q50_150,
            "ref_charge_v_rmse_median": ref_v_rmse_median,
            "accuracy_ratio": ratio,
            "warning": (
                f"BDT is {ratio:.1f}x less accurate on CC-like profiles. "
                "Treat BO optimization results as approximate."
            ),
        }

    # ── Plots ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, q50, q95, label, unit in (
        (axes[0], v_q50, v_q95, "Voltage", "V"),
        (axes[1], t_q50, t_q95, "Temperature", "degC"),
    ):
        ax.plot(horizon_s, q50, lw=1.8, label="median |err|")
        ax.plot(horizon_s, q95, lw=1.8, color="tab:red", label="p95 |err| (margin)")
        for w in (150, 300, 600):
            ax.axvline(w, color="gray", ls=":", alpha=0.5)
        ax.set_xlabel("Prediction horizon (s)")
        ax.set_ylabel(f"|{label} error| ({unit})")
        ax.set_title(f"{label}: open-loop chained drift (RW @ 1 Hz)")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "drift_vs_horizon.png", dpi=140)
    plt.close(fig)

    acts = [a for a in ACTION_SET if per_action[a]["v_rmse"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar([f"{a:+.2f}" for a in acts], [per_action[a]["v_rmse"] for a in acts],
                color="tab:blue")
    axes[0].set_ylabel("V RMSE (V)")
    axes[1].bar([f"{a:+.2f}" for a in acts], [per_action[a]["t_rmse"] for a in acts],
                color="tab:orange")
    axes[1].set_ylabel("T RMSE (degC)")
    for ax in axes:
        ax.set_xlabel("Charge setpoint (A)")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Single-chunk (150 s) error conditional on action level")
    fig.tight_layout()
    fig.savefig(plots_dir / "per_action_rmse.png", dpi=140)
    plt.close(fig)
    print(f"\nPlots saved -> {plots_dir}/")

    summary = {
        "checkpoint": str(args.ckpt),
        "cell": args.cell,
        "n_segments": int(err_v.shape[0]),
        "horizon_s": int(args.horizon),
        "drift_at_horizon": drift_at,
        "per_action": {str(k): v for k, v in per_action.items()},
        "reference_charge_sessions": ref_rows,
        "artifacts": {
            "conformal_margins": str(models_dir / "conformal_margins.npz"),
        },
    }
    if distribution_shift is not None:
        summary["distribution_shift"] = distribution_shift
    if args.artifact_root:
        reg_path = models_dir / "registry.json"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        with reg_path.open("w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"Cell drift registry -> {reg_path}")
        print(f"Margins  -> {models_dir / 'conformal_margins.npz'}")
    else:
        write_stage_registry(P.STAGE1_DRIFT, summary, root=ROOT)
        update_master_registry(root=ROOT)
        print(f"Registry -> {CANONICAL['drift_registry']}")
        print(f"Margins  -> {CANONICAL['conformal_margins']}")


if __name__ == "__main__":
    main()
