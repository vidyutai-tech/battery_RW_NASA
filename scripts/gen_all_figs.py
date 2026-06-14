#!/usr/bin/env python3
"""
Publication figure generation for RW9 charging optimization.

Fig 1 — Chebyshev Pareto front
Fig 2 — All 8 profile families (I, V, SoC panels)
Fig 3 — Family ranking (3 metrics)
Fig 4 — Three reference profiles (fast / balanced / lifetime)
Fig 5 — Methodology summary (family optima + BO convergence)
Fig 6 — Physics-grounded degradation (use --with_physics)

Run:
  python scripts/gen_all_figs.py
  python scripts/gen_all_figs.py --with_physics
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig_{os.getenv('USER', 'default')}")

DEFAULT_ENHANCED = ROOT / "outputs/charging_opt_user/hima/stage3_enhanced_20260614"
DEFAULT_EI = ROOT / "outputs/charging_opt_user/hima/stage3_optimization"
DEFAULT_CHEBYSHEV = ROOT / "outputs/charging_opt_user/hima/chebyshev_sweep/chebyshev_sweep_results.json"

C = {
    "cccv": "#1b7837",
    "reduced_cv_cccv": "#4dac26",
    "pulsed": "#2166ac",
    "cc_taper": "#d6604d",
    "adaptive_two_step": "#762a83",
    "adaptive_three_step": "#af8dc3",
    "multi_step_taper": "#e08214",
    "exponential_taper": "#999999",
}
LABELS = {
    "cccv": "CCCV",
    "reduced_cv_cccv": "Reduced-CV CCCV",
    "pulsed": "Pulsed charge/rest",
    "cc_taper": "CC-taper (2-level)",
    "adaptive_two_step": "Adaptive 2-step (SoC)",
    "adaptive_three_step": "Adaptive 3-step (SoC)",
    "multi_step_taper": "Multi-step taper",
    "exponential_taper": "Exponential taper",
}
ORDER = [
    "cccv",
    "reduced_cv_cccv",
    "pulsed",
    "cc_taper",
    "adaptive_two_step",
    "adaptive_three_step",
    "multi_step_taper",
    "exponential_taper",
]
Q_AS = 7560.0
SOC0 = 0.15

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 100,
    }
)


def _t_min(n_samples: int, dt: float = 1.0) -> np.ndarray:
    return np.arange(n_samples) * dt / 60.0


def _soc_from_current(i_arr: np.ndarray, soc0: float = SOC0, q_as: float = Q_AS) -> np.ndarray:
    return soc0 + np.cumsum(i_arr) / q_as


def make_cccv(
    i_cc: float = 1.05,
    v_cv: float = 4.16,
    i_cutoff: float = 0.262,
    soc0: float = SOC0,
    dur_min: float = 104.0,
    q: float = Q_AS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(dur_min * 60)
    t_cv = int(0.74 * n)
    i = np.zeros(n)
    v = np.zeros(n)
    i[:t_cv] = i_cc
    v[:t_cv] = 3.75 + (v_cv - 3.75) * (np.arange(t_cv) / max(t_cv, 1)) ** 0.62
    tc = np.arange(n - t_cv)
    tau = max((n - t_cv) / 3.5, 1.0)
    i[t_cv:] = np.maximum(i_cc * np.exp(-tc / tau), i_cutoff)
    v[t_cv:] = np.clip(v_cv + 0.005 * np.log1p(tc / 800), None, v_cv + 0.008)
    soc = _soc_from_current(i, soc0, q)
    return _t_min(n), i, v, np.clip(soc, 0, 1)


def make_reduced_cv_cccv(
    i_cc: float = 1.063,
    v_cv: float = 4.15,
    i_cutoff: float = 0.462,
    soc0: float = SOC0,
    dur_min: float = 103.0,
    q: float = Q_AS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(dur_min * 60)
    t_cv = int(0.71 * n)
    i = np.zeros(n)
    v = np.zeros(n)
    i[:t_cv] = i_cc
    v[:t_cv] = 3.75 + (v_cv - 3.75) * (np.arange(t_cv) / max(t_cv, 1)) ** 0.62
    tc = np.arange(n - t_cv)
    tau = max((n - t_cv) / 3.2, 1.0)
    i[t_cv:] = np.maximum(i_cc * np.exp(-tc / tau), i_cutoff)
    v[t_cv:] = np.clip(v_cv + 0.004 * np.log1p(tc / 600), None, v_cv + 0.006)
    soc = _soc_from_current(i, soc0, q)
    return _t_min(n), i, v, np.clip(soc, 0, 1)


def make_pulsed(
    i_charge: float = 1.181,
    pulse_on_min: float = 3.53,
    rest_frac: float = 0.08,
    soc0: float = SOC0,
    dur_min: float = 104.0,
    q: float = Q_AS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(dur_min * 60)
    pulse_on_s = max(1, int(pulse_on_min * 60))
    pulse_rest_s = max(1, int(pulse_on_min * 60 * rest_frac))
    period = pulse_on_s + pulse_rest_s
    i = np.zeros(n)
    for k in range(n):
        i[k] = i_charge if (k % period) < pulse_on_s else 0.0
    soc = np.clip(_soc_from_current(i, soc0, q), 0, 1)
    v_base = 3.75 + 0.50 * soc
    v = np.clip(v_base + 0.09 * (i > 0.05).astype(float), 3.75, 4.02)
    return _t_min(n), i, v, soc


def make_cc_taper(
    i_charge: float = 1.246,
    i_floor: float = 0.75,
    soc0: float = SOC0,
    dur_min: float = 98.6,
    q: float = Q_AS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(dur_min * 60)
    taper_pt = int(0.72 * n)
    i = np.zeros(n)
    v = np.zeros(n)
    i[:taper_pt] = i_charge
    v[:taper_pt] = 3.77 + (4.20 - 3.77) * (np.arange(taper_pt) / max(taper_pt, 1)) ** 0.63
    i[taper_pt:] = i_floor
    tc = np.arange(n - taper_pt)
    v[taper_pt:] = np.clip(v[taper_pt - 1] - 0.07 + 0.0003 * tc, 4.10, 4.20)
    soc = _soc_from_current(i, soc0, q)
    return _t_min(n), i, v, np.clip(soc, 0, 1)


def make_adaptive_two_step(
    i1: float = 1.278,
    i2: float = 0.75,
    soc_switch: float = 0.705,
    soc0: float = SOC0,
    dur_min: float = 104.0,
    q: float = Q_AS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(dur_min * 60)
    i = np.zeros(n)
    soc_running = soc0
    for k in range(n):
        i[k] = i2 if soc_running >= soc_switch else i1
        soc_running = float(np.clip(soc_running + i[k] / q, 0, 1))
    soc = np.clip(_soc_from_current(i, soc0, q), 0, 1)
    v = np.clip(3.77 + 0.50 * soc + 0.08 * i, 3.77, 4.20)
    return _t_min(n), i, v, soc


def make_adaptive_three_step(
    i1: float = 1.50,
    i2: float = 1.226,
    i3: float = 0.75,
    soc1: float = 0.189,
    soc2: float = 0.714,
    soc0: float = SOC0,
    dur_min: float = 105.0,
    q: float = Q_AS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(dur_min * 60)
    i = np.zeros(n)
    soc_running = soc0
    for k in range(n):
        if soc_running < soc1:
            i[k] = i1
        elif soc_running < soc2:
            i[k] = i2
        else:
            i[k] = i3
        soc_running = float(np.clip(soc_running + i[k] / q, 0, 1))
    soc = np.clip(_soc_from_current(i, soc0, q), 0, 1)
    v = np.clip(3.77 + 0.50 * soc + 0.08 * i, 3.77, 4.20)
    return _t_min(n), i, v, soc


def make_multi_step_taper(
    i_charge: float = 2.0,
    i_floor: float = 0.884,
    soc0: float = SOC0,
    dur_min: float = 91.8,
    q: float = Q_AS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(dur_min * 60)
    taper1 = int(0.20 * n)
    taper2 = int(0.43 * n)
    taper3 = int(0.65 * n)
    i = np.zeros(n)
    v = np.zeros(n)
    i[:taper1] = i_charge
    i[taper1:taper2] = i_charge - 0.75
    i[taper2:taper3] = i_charge - 1.125
    i[taper3:] = max(i_floor, i_charge - 1.50)
    v[:taper1] = 3.89 + (4.20 - 3.89) * (np.arange(taper1) / max(taper1, 1)) ** 0.55
    for a, b in [(taper1, taper2), (taper2, taper3), (taper3, n)]:
        ln = b - a
        v_start = v[a - 1] - 0.06 if a > 0 else 4.13
        v[a:b] = np.clip(v_start + 0.00012 * np.arange(ln), 4.11, 4.20)
    soc = _soc_from_current(i, soc0, q)
    return _t_min(n), i, v, np.clip(soc, 0, 1)


def make_exponential_taper(
    i0: float = 1.08,
    k_decay: float = 0.32,
    soc0: float = SOC0,
    dur_min: float = 118.2,
    q: float = Q_AS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(dur_min * 60)
    soc_run = np.zeros(n)
    s = soc0
    i = np.zeros(n)
    for idx in range(n):
        s_norm = (s - soc0) / (0.95 - soc0) if (0.95 - soc0) > 0 else 0.0
        i[idx] = i0 * np.exp(-k_decay * s_norm)
        s = min(s + i[idx] / q, 1.0)
        soc_run[idx] = s
    v = np.clip(3.75 + 0.55 * soc_run + 0.06 * i, 3.75, 4.20)
    return _t_min(n), i, v, soc_run


def profile_from_params(
    family_id: str, params: Dict[str, Any], dur_min: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = {k: v for k, v in params.items() if k != "family_id"}
    if family_id == "cccv":
        return make_cccv(dur_min=dur_min, **{k: p[k] for k in ("i_cc", "v_cv", "i_cutoff") if k in p})
    if family_id == "reduced_cv_cccv":
        return make_reduced_cv_cccv(dur_min=dur_min, **{k: p[k] for k in ("i_cc", "v_cv", "i_cutoff") if k in p})
    if family_id == "pulsed":
        return make_pulsed(
            dur_min=dur_min,
            i_charge=p.get("i_charge", 1.18),
            pulse_on_min=p.get("pulse_on_min", 3.5),
            rest_frac=p.get("rest_fraction", 0.08),
        )
    if family_id == "cc_taper":
        return make_cc_taper(
            dur_min=dur_min,
            i_charge=p.get("i_charge", 1.25),
            i_floor=p.get("i_floor", 0.75),
        )
    if family_id == "adaptive_two_step":
        return make_adaptive_two_step(
            dur_min=dur_min,
            i1=p.get("i1", 1.28),
            i2=p.get("i2", 0.75),
            soc_switch=p.get("soc_switch", 0.705),
        )
    if family_id == "adaptive_three_step":
        return make_adaptive_three_step(
            dur_min=dur_min,
            i1=p.get("i1", 1.5),
            i2=p.get("i2", 1.23),
            i3=p.get("i3", 0.75),
            soc1=p.get("soc1", 0.19),
            soc2=p.get("soc2", 0.71),
        )
    if family_id == "multi_step_taper":
        return make_multi_step_taper(
            dur_min=dur_min,
            i_charge=p.get("i_charge", 2.0),
            i_floor=p.get("i_floor", 0.88),
        )
    if family_id == "exponential_taper":
        return make_exponential_taper(
            dur_min=dur_min,
            i0=p.get("i0", 1.08),
            k_decay=p.get("k", 0.32),
        )
    raise KeyError(f"Unknown family_id: {family_id}")


def load_comparison_table(path: Path) -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            fid = row["family_id"]
            meta[fid] = {
                "loss": float(row["loss"]),
                "sei": float(row["sei_per_pct_soc"]),
                "dur": float(row["duration_min"]),
                "feasible": row["feasible"].lower() == "true",
                "params": row["parameters"],
            }
    return meta


def load_best_params(results_json: Path) -> Dict[str, Dict[str, Any]]:
    payload = json.loads(results_json.read_text())
    out: Dict[str, Dict[str, Any]] = {}
    for fid, block in payload["families"].items():
        out[fid] = dict(block["best_params"])
    return out


def load_pulsed_chebyshev_sweep(chebyshev_json: Path) -> List[Tuple[float, float, float]]:
    payload = json.loads(chebyshev_json.read_text())
    sweep: List[Tuple[float, float, float]] = []
    for omega_str, rows in sorted(payload["results_by_omega"].items(), key=lambda x: float(x[0])):
        pulsed = next(r for r in rows if r["family_id"] == "pulsed")
        sweep.append((float(omega_str), float(pulsed["duration_min"]), float(pulsed["sei_per_pct_soc"])))
    return sweep


def load_pulsed_params_at_omega(chebyshev_json: Path, omega: float) -> Dict[str, Any]:
    payload = json.loads(chebyshev_json.read_text())
    key = f"{omega:g}" if f"{omega:g}" in payload["results_by_omega"] else str(omega)
    if key not in payload["results_by_omega"]:
        key = str(float(omega))
    pulsed = next(r for r in payload["results_by_omega"][key] if r["family_id"] == "pulsed")
    return dict(pulsed["params"])


def load_cccv_chebyshev_baseline(chebyshev_json: Path) -> Tuple[float, float]:
    payload = json.loads(chebyshev_json.read_text())
    cccv = next(r for r in payload["results_by_omega"]["0.0"] if r["family_id"] == "cccv")
    return float(cccv["duration_min"]), float(cccv["sei_per_pct_soc"])


def load_convergence_curve(results_json: Path, family_id: str = "cccv") -> np.ndarray:
    payload = json.loads(results_json.read_text())
    hist = payload["families"][family_id]["history"]
    best = np.full(len(hist), np.nan)
    running = np.inf
    for i, entry in enumerate(hist):
        if entry.get("feasible") and entry.get("metrics", {}).get("feasible"):
            running = min(running, float(entry["loss"]))
        if np.isfinite(running):
            best[i] = running
    return best


def hbar(ax, vals, names, cols, xlabel, title, xlim, ref_line=None, val_fmt="{:.1f}"):
    ax.barh(range(len(names)), vals, color=cols, edgecolor="white", linewidth=0.5, height=0.62)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel(xlabel, fontsize=9.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    if ref_line is not None:
        ax.axvline(ref_line, color="#999", ls="--", lw=1, alpha=0.7)
    for i, v in enumerate(vals):
        ax.text(v + (xlim[1] - xlim[0]) * 0.01, i, val_fmt.format(v), va="center", fontsize=8)
    ax.set_xlim(*xlim)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.35)


def build_fig2(out: Path, meta: Dict[str, Dict[str, Any]], profiles: Dict[str, Tuple]) -> None:
    n_fam = len(ORDER)
    fig2 = plt.figure(figsize=(22, 13))
    fig2.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3, n_fam, figure=fig2, hspace=0.18, wspace=0.28, left=0.04, right=0.98, top=0.91, bottom=0.07
    )

    for col, fid in enumerate(ORDER):
        t, i_arr, v_arr, soc_arr = profiles[fid]
        m = meta[fid]
        color = C[fid]
        feasible = m["feasible"]
        alpha = 1.0 if feasible else 0.45

        ax_i = fig2.add_subplot(gs[0, col])
        ax_v = fig2.add_subplot(gs[1, col])
        ax_s = fig2.add_subplot(gs[2, col])

        ax_i.plot(t, i_arr, color=color, lw=1.6, alpha=alpha)
        ax_i.set_ylim(-0.1, max(float(i_arr.max()) * 1.20, 0.5))
        ax_i.set_xlim(0, t[-1])
        status = "FEASIBLE" if feasible else "INFEASIBLE"
        ax_i.set_title(
            f"{LABELS[fid]}\nloss={m['loss']:.1f}  SEI/ΔSoC={m['sei']:.1f}\n"
            f"{m['dur']:.0f} min  [{status}]",
            fontsize=7.5,
            color=color,
            pad=3,
        )
        ax_i.tick_params(labelbottom=False, labelsize=7)
        if col == 0:
            ax_i.set_ylabel("I (A)", fontsize=8)
        ax_i.grid(True)
        if not feasible:
            ax_i.text(
                0.5,
                0.88,
                "✗ Over time limit",
                transform=ax_i.transAxes,
                ha="center",
                fontsize=7,
                color="#cc0000",
                bbox=dict(fc="white", ec="#cc0000", lw=0.6, pad=1.5),
            )

        ax_v.plot(t, v_arr, color="#b2182b", lw=1.4, alpha=alpha)
        ax_v.axhline(4.20, color="#888", ls="--", lw=0.7, alpha=0.6)
        ax_v.set_ylim(max(3.70, float(v_arr.min()) - 0.03), min(4.25, float(v_arr.max()) + 0.04))
        ax_v.set_xlim(0, t[-1])
        ax_v.tick_params(labelbottom=False, labelsize=7)
        if col == 0:
            ax_v.set_ylabel("V (V)", fontsize=8)
        ax_v.grid(True)
        ax_v.text(0.97, 0.94, "4.2 V", transform=ax_v.transAxes, fontsize=6.5, color="#888", ha="right", va="top")

        ax_s.plot(t, soc_arr * 100, color="#333333", lw=1.6, alpha=alpha)
        ax_s.axhline(95, color="#888", ls=":", lw=0.9)
        ax_s.set_ylim(10, 102)
        ax_s.set_xlim(0, t[-1])
        ax_s.set_xlabel("Time (min)", fontsize=8)
        ax_s.tick_params(labelsize=7)
        if col == 0:
            ax_s.set_ylabel("SoC (%)", fontsize=8)
        ax_s.grid(True)
        ax_s.text(0.97, 0.12, "95% target", transform=ax_s.transAxes, fontsize=6.5, color="#888", ha="right")

    for col, fid in enumerate(ORDER):
        ax_s = fig2.axes[col + 2 * n_fam]
        ax_s.text(0.5, -0.38, meta[fid]["params"], transform=ax_s.transAxes, ha="center", fontsize=6.5, color="#555", style="italic")

    for row_idx, row_label in enumerate(["Charging\ncurrent", "Cell\nvoltage", "State of\ncharge"]):
        fig2.text(0.005, 0.88 - row_idx * 0.295, row_label, ha="left", va="center", fontsize=8.5, rotation=90, color="#555")

    fig2.suptitle(
        "All Charging Profile Families — Best BO Result per Family (RW9 Cell)\n"
        "Start: SoC=15%, V=3.711 V, T=24.7 °C  |  Target: SoC≥95%  |  PI Acquisition, 40 evals/family",
        fontsize=11,
        y=0.975,
    )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=C[fid],
            lw=2.5,
            label=f"{LABELS[fid]}  (loss={meta[fid]['loss']:.1f}{'  ✗' if not meta[fid]['feasible'] else ''})",
        )
        for fid in ORDER
    ]
    fig2.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), fontsize=8, framealpha=0.95, edgecolor="#ddd")
    fig2.savefig(out / "fig2_all_families.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def build_fig1(
    out: Path,
    pulsed_sweep: Sequence[Tuple[float, float, float]],
    ref_time: float,
    ref_sei: float,
) -> None:
    fig1, axes1 = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"wspace": 0.38})
    omegas = [p[0] for p in pulsed_sweep]
    durs = [p[1] for p in pulsed_sweep]
    seis = [p[2] for p in pulsed_sweep]
    fast_d, fast_s = durs[-1], seis[-1]

    ax = axes1[0]
    sc = ax.scatter(durs, seis, c=omegas, cmap="RdYlGn_r", vmin=0, vmax=1, s=100, zorder=5, edgecolors="white", linewidths=0.9)
    cb = fig1.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("ω  (0=lifetime → 1=fastest)", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    ax.plot(durs, seis, "--", color="#555", lw=1.4, alpha=0.6, zorder=3, label="Pulsed Pareto front")
    for o, d, s in pulsed_sweep:
        ax.annotate(f"ω={o:.1f}", (d, s), xytext=(5, 3), textcoords="offset points", fontsize=7.5, color="#333")

    ax.scatter([ref_time], [ref_sei], c="#1b7837", s=140, marker="s", edgecolors="white", lw=1.2, zorder=6)
    ax.annotate(
        "CCCV\nlifetime\noptimum",
        (ref_time, ref_sei),
        xytext=(-52, 12),
        textcoords="offset points",
        fontsize=8,
        color="#1b7837",
        arrowprops=dict(arrowstyle="->", color="#1b7837", lw=0.9),
    )
    ax.annotate("", xy=(fast_d, fast_s), xytext=(ref_time, ref_sei), arrowprops=dict(arrowstyle="<->", color="#999", lw=1.3))
    pct_time = (ref_time - fast_d) / ref_time * 100.0
    pct_sei = (fast_s - ref_sei) / ref_sei * 100.0
    ax.text(
        (fast_d + ref_time) / 2,
        (fast_s + ref_sei) / 2 + 2.0,
        f"−{pct_time:.1f}% charge time\n+{pct_sei:.1f}% SEI degradation",
        fontsize=8.5,
        ha="center",
        color="#444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.7),
    )
    ax.set_xlabel("Charge duration (min)")
    ax.set_ylabel("SEI / ΔSoC  (lower = better lifetime)")
    ax.set_title("Directed Pareto Front\nChebyshev Scalarization Sweep (11 ω values × 3 families)", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(True)
    ax.set_xlim(min(durs) - 4, max(durs) + 6)
    ax.set_ylim(min(seis) - 1.5, max(seis) + 2.0)

    ax2 = axes1[1]
    d_arr = np.array(durs)
    s_arr = np.array(seis)
    idx = np.argsort(d_arr)
    delta_sei = s_arr[idx] - ref_sei
    ax2.plot(d_arr[idx], delta_sei, "o-", color=C["pulsed"], lw=2, ms=7, markeredgecolor="white", zorder=4)
    ax2.axhspan(0, 3, alpha=0.08, color="#1b7837")
    ax2.axhspan(3, 8, alpha=0.08, color="#ffa500")
    ax2.axhspan(8, 16, alpha=0.08, color="#e31a1c")
    ax2.text(101, 0.5, "Lifetime zone\n(<2 SEI cost)", fontsize=7.5, color="#1b7837")
    ax2.text(95, 4.5, "Balanced zone", fontsize=7.5, color="#e08214")
    ax2.text(53, 9.5, "Fast zone\n(>8 SEI cost)", fontsize=7.5, color="#e31a1c")
    ax2.axhline(0, color="#999", ls=":", lw=1)
    key = [
        (durs[0], seis[0] - ref_sei, "ω=0.0\n(near-lifetime pulsed)"),
        (durs[5], seis[5] - ref_sei, "ω=0.5\n(balanced)"),
        (fast_d, fast_s - ref_sei, "ω=1.0\n(fastest)"),
    ]
    for x, y, lbl in key:
        ax2.annotate(
            lbl,
            (x, y),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=7.5,
            color="#333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", lw=0.6),
        )
    ax2.set_xlabel("Charge duration (min)")
    ax2.set_ylabel(f"ΔSEI above CCCV baseline ({ref_sei:.1f})")
    ax2.set_title("SEI Degradation Cost vs. CCCV Baseline\nQuantifying the speed-lifetime trade-off", fontsize=11)
    ax2.grid(True)
    ax2.set_xlim(min(durs) - 4, max(durs) + 6)
    ax2.set_ylim(-0.5, max(delta_sei) + 2.0)

    fig1.suptitle("Charging Speed vs. Battery Lifetime — Pareto Analysis  (RW9 Cell, PI-BO)", fontsize=12, y=1.01)
    fig1.savefig(out / "fig1_pareto_front.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)


def build_fig3(out: Path, meta: Dict[str, Dict[str, Any]]) -> None:
    fam_order_rank = [f for f in ORDER if meta[f]["feasible"]]
    names = [LABELS[f] for f in fam_order_rank]
    seis_r = [meta[f]["sei"] for f in fam_order_rank]
    durs_r = [meta[f]["dur"] for f in fam_order_rank]
    loss_r = [meta[f]["loss"] for f in fam_order_rank]
    cols_r = [C[f] for f in fam_order_rank]

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5.2), gridspec_kw={"wspace": 0.45})
    hbar(axes3[0], seis_r, names, cols_r, "SEI / ΔSoC  (lower = better)", "① Degradation Proxy", (65, 74.5), ref_line=68.0)
    axes3[0].text(68.15, -0.6, "CCCV\nbest", fontsize=6.5, color="#555", va="top")
    hbar(axes3[1], durs_r, names, cols_r, "Charge duration (min)", "② Charging Speed", (85, 111), ref_line=105, val_fmt="{:.1f}")
    axes3[1].text(105.2, -0.6, "105 min\nlimit", fontsize=6.5, color="#555", va="top")
    hbar(axes3[2], loss_r, names, cols_r, "Composite loss (lower = better)", "③ Overall Score", (69, 76))

    infeas = [f for f in ORDER if not meta[f]["feasible"]]
    if infeas:
        note = ", ".join(f"{LABELS[f]} (loss={meta[f]['loss']:.0f})" for f in infeas)
        note = f"Excluded infeasible: {note}"
    else:
        note = ""
    for ax in axes3:
        if note:
            ax.text(0.98, 0.02, note, transform=ax.transAxes, ha="right", fontsize=6.5, color="#999")

    legend_handles = [Line2D([0], [0], color=C[f], lw=0, marker="s", ms=10, label=LABELS[f]) for f in fam_order_rank]
    fig3.legend(handles=legend_handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09), fontsize=8, framealpha=0.95, edgecolor="#ddd")
    fig3.suptitle("Multi-Family Optimization Results — RW9 Cell  (PI Acquisition, 40 evals/family)", fontsize=12, y=1.02)
    fig3.savefig(out / "fig3_family_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)


def build_fig4(
    out: Path,
    profiles: Dict[str, Tuple],
    meta: Dict[str, Dict[str, Any]],
    chebyshev_json: Path,
    pulsed_sweep: Sequence[Tuple[float, float, float]],
) -> None:
    fast = pulsed_sweep[-1]
    bal = pulsed_sweep[5]
    fast_params = load_pulsed_params_at_omega(chebyshev_json, 1.0)
    bal_params = load_pulsed_params_at_omega(chebyshev_json, 0.5)
    t_pfast, i_pfast, v_pfast, s_pfast = profile_from_params("pulsed", fast_params, fast[1])
    t_pbal, i_pbal, v_pbal, s_pbal = profile_from_params("pulsed", bal_params, bal[1])
    t_cccv, i_cccv, v_cccv, s_cccv = profiles["cccv"]

    ref_profiles = [
        (
            "Fastest\n(Pulsed, ω=1.0)",
            t_pfast,
            i_pfast,
            v_pfast,
            s_pfast,
            C["pulsed"],
            f"{fast[1]:.1f} min  |  SEI/ΔSoC={fast[2]:.1f}  |  I_peak={i_pfast.max():.2f} A",
        ),
        (
            "Balanced\n(Pulsed, ω=0.5)",
            t_pbal,
            i_pbal,
            v_pbal,
            s_pbal,
            C["adaptive_two_step"],
            f"{bal[1]:.1f} min  |  SEI/ΔSoC={bal[2]:.1f}  |  I_peak={i_pbal.max():.2f} A",
        ),
        (
            "Lifetime\n(CCCV, ω=0.0)",
            t_cccv,
            i_cccv,
            v_cccv,
            s_cccv,
            C["cccv"],
            f"{meta['cccv']['dur']:.1f} min  |  SEI/ΔSoC={meta['cccv']['sei']:.1f}  |  I_peak={i_cccv.max():.2f} A",
        ),
    ]

    fig4, axes4 = plt.subplots(
        3,
        3,
        figsize=(13, 8.5),
        gridspec_kw={"hspace": 0.10, "wspace": 0.30, "left": 0.09, "right": 0.97, "top": 0.88, "bottom": 0.10},
    )
    row_labels = ["Current (A)", "Voltage (V)", "SoC (%)"]
    for col, (name, t, i_a, v_a, soc_a, color, subtitle) in enumerate(ref_profiles):
        for row in range(3):
            ax = axes4[row][col]
            if row == 0:
                ax.plot(t, i_a, color=color, lw=1.9)
                ax.set_ylim(-0.1, float(i_a.max()) * 1.20)
                ax.set_title(f"{name}\n{subtitle}", fontsize=9, color=color, pad=4)
            elif row == 1:
                ax.plot(t, v_a, color="#b2182b", lw=1.7)
                ax.axhline(4.20, color="#888", ls="--", lw=0.7, alpha=0.6)
                ax.set_ylim(max(3.70, float(v_a.min()) - 0.03), min(4.25, float(v_a.max()) + 0.04))
                ax.text(0.97, 0.93, "4.2 V", transform=ax.transAxes, fontsize=7, color="#888", ha="right", va="top")
            else:
                ax.plot(t, soc_a * 100, color="#222", lw=1.9)
                ax.axhline(95, color="#888", ls=":", lw=0.9)
                ax.set_ylim(10, 103)
                ax.set_xlabel("Time (min)", fontsize=9)
                ax.text(0.97, 0.13, "95% target", transform=ax.transAxes, fontsize=7, color="#888", ha="right")
            ax.grid(True)
            ax.set_xlim(0, t[-1])
            if row < 2:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)

    fig4.suptitle(
        "Three Reference Charging Profiles — RW9 Cell\nStart: SoC=15%, V=3.711 V, T=24.7°C  |  Target: SoC≥95%",
        fontsize=12,
        y=0.94,
    )
    fig4.savefig(out / "fig4_reference_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig4)


def build_fig5(
    out: Path,
    meta: Dict[str, Dict[str, Any]],
    pulsed_sweep: Sequence[Tuple[float, float, float]],
    loss_ei: np.ndarray,
    loss_pi: np.ndarray,
    best_pi: float,
) -> None:
    fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"wspace": 0.40})

    ax = axes5[0]
    for fid in ORDER:
        m = meta[fid]
        mk = "s" if not m["feasible"] else "o"
        ms = 90 if not m["feasible"] else 110
        al = 0.4 if not m["feasible"] else 0.92
        ax.scatter(m["dur"], m["sei"], color=C[fid], s=ms, marker=mk, alpha=al, edgecolors="white", lw=1.0, zorder=4)
        ax.annotate(
            LABELS[fid].split("(")[0].strip(),
            (m["dur"], m["sei"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color=C[fid],
        )

    for o, d, s in pulsed_sweep:
        ax.scatter(d, s, c=[[matplotlib.cm.RdYlGn_r(o)]], s=45, marker="^", zorder=3, edgecolors="none", alpha=0.7)
    ax.plot([p[1] for p in pulsed_sweep], [p[2] for p in pulsed_sweep], "--", color="#888", lw=1.1, alpha=0.5, label="Chebyshev sweep (pulsed)")
    ax.scatter([], [], c="gray", marker="^", s=45, label="Sweep points (ω=0→1)")
    ax.set_xlabel("Charge duration (min)")
    ax.set_ylabel("SEI / ΔSoC")
    ax.set_title("All Results: Family Optima + Chebyshev Sweep\nRW9 Cell  (PI acquisition, 40 evals/family)", fontsize=10.5)
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True)

    ax2 = axes5[1]
    n_evals = np.arange(1, len(loss_pi) + 1)
    ax2.plot(n_evals, loss_ei, "o-", color="#d6604d", lw=1.8, ms=4, markeredgecolor="white", label="EI acquisition (original)")
    ax2.plot(n_evals, loss_pi, "s-", color="#2166ac", lw=1.8, ms=4, markeredgecolor="white", label="PI acquisition (enhanced)")
    ax2.axhline(best_pi, color="#1b7837", ls="--", lw=1.2, label=f"Best result (CCCV, PI={best_pi:.2f})")
    mask = np.isfinite(loss_ei) & np.isfinite(loss_pi)
    if mask.any():
        ax2.fill_between(n_evals[mask], loss_pi[mask], loss_ei[mask], alpha=0.12, color="#2166ac")
    ax2.text(
        25,
        np.nanmin(loss_pi) + 0.5,
        "PI converges faster\n(Paper 3, Jiang et al. 2022)",
        fontsize=8,
        color="#2166ac",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", lw=0.6),
    )
    ax2.set_xlabel("BO evaluations")
    ax2.set_ylabel("Best composite loss (running minimum)")
    ax2.set_title("Acquisition Function Comparison\nEI vs PI — CCCV Family", fontsize=10.5)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True)
    ax2.set_xlim(1, len(loss_pi))
    y_vals = np.concatenate([loss_ei[np.isfinite(loss_ei)], loss_pi[np.isfinite(loss_pi)]])
    ax2.set_ylim(float(np.min(y_vals)) - 0.8, float(np.max(y_vals)) + 1.0)

    fig5.suptitle("Methodology Summary — Battery Charging Optimization Pipeline  (RW9 NASA Dataset)", fontsize=12, y=1.02)
    fig5.savefig(out / "fig5_methodology_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/visualization")
    p.add_argument("--enhanced_dir", type=Path, default=DEFAULT_ENHANCED)
    p.add_argument("--ei_dir", type=Path, default=DEFAULT_EI)
    p.add_argument("--chebyshev_json", type=Path, default=DEFAULT_CHEBYSHEV)
    p.add_argument(
        "--with_physics",
        action="store_true",
        help="Also run compare_degradation_models.py for fig6 (needs GPU, ~2 min)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    comparison_csv = args.enhanced_dir / "models/comparison_table.csv"
    results_json = args.enhanced_dir / "models/family_optimization_results.json"
    ei_json = args.ei_dir / "models/family_optimization_results.json"

    meta = load_comparison_table(comparison_csv)
    best_params = load_best_params(results_json)
    profiles = {
        fid: profile_from_params(fid, best_params[fid], meta[fid]["dur"])
        for fid in ORDER
        if fid in best_params and fid in meta
    }
    pulsed_sweep = load_pulsed_chebyshev_sweep(args.chebyshev_json)
    ref_time, ref_sei = load_cccv_chebyshev_baseline(args.chebyshev_json)
    loss_pi = load_convergence_curve(results_json, "cccv")
    loss_ei = load_convergence_curve(ei_json, "cccv")
    best_pi = float(meta["cccv"]["loss"])

    print(f"Loading data from {args.enhanced_dir}")
    print(f"Writing figures to {out}")

    build_fig1(out, pulsed_sweep, ref_time, ref_sei)
    print("  fig1_pareto_front.png")
    build_fig2(out, meta, profiles)
    print("  fig2_all_families.png")
    build_fig3(out, meta)
    print("  fig3_family_ranking.png")
    build_fig4(out, profiles, meta, args.chebyshev_json, pulsed_sweep)
    print("  fig4_reference_profiles.png")
    build_fig5(out, meta, pulsed_sweep, loss_ei, loss_pi, best_pi)
    print("  fig5_methodology_summary.png")

    if args.with_physics:
        import subprocess
        cmd = [
            sys.executable,
            str(ROOT / "scripts/compare_degradation_models.py"),
            "--enhanced_dir", str(args.enhanced_dir),
            "--chebyshev_json", str(args.chebyshev_json),
            "--out_dir", str(out),
        ]
        print("  Running physics degradation comparison (fig6)...")
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        print("  fig6_physics_degradation.png")

    n_figs = 6 if args.with_physics else 5
    print(f"\nAll {n_figs} figures saved to {out}/")


if __name__ == "__main__":
    main()
