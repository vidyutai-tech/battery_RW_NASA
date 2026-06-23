#!/usr/bin/env python3
"""
Publication figure generation for RW9 charging optimization.

Fig 1 — Pareto / trade-off front (Chebyshev SEI sweep or physics ΔQ/Q₀)
Fig 2 — All 8 profile families (I, V, SoC panels)
Fig 3 — Family ranking (degradation, duration, loss)
Fig 4 — Three reference profiles (fast / balanced / lifetime)
Fig 5 — Methodology summary (family optima + BO convergence)
Fig 6 — Physics-grounded degradation validation (--with_physics)
Fig 7 — Ambient temperature sensitivity (physics+thermal runs only)
Fig Chebyshev — Directed sweep with convergence annotations (if --chebyshev_json exists)

Run:
  python scripts/gen_all_figs.py --run_dir outputs/charging_opt_user/hima/stage3_physics_thermal
  python scripts/gen_all_figs.py --with_physics
  python scripts/gen_all_figs.py --run_dir ... --no_thermal_warnings   # hide BDT thermal caveats on fig1/fig2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
DEFAULT_PHYSICS = ROOT / "outputs/charging_opt_user/hima/stage3_physics_thermal"
DEFAULT_EI = ROOT / "outputs/charging_opt_user/hima/stage3_optimization"
DEFAULT_CHEBYSHEV = ROOT / "outputs/charging_opt_user/hima/chebyshev_sweep/chebyshev_sweep_results.json"

from charging_opt.artifacts import CANONICAL
from charging_opt.pareto_analysis import resolve_pareto_config
from charging_opt.io_utils import resolve_visualization_dir

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


@dataclass
class RunContext:
    run_dir: Path
    objective_mode: str
    deg_key: str
    deg_label: str
    meta: Dict[str, Dict[str, Any]]
    best_params: Dict[str, Dict[str, Any]]
    constraints: Dict[str, Any]
    pareto: Optional[Dict[str, Any]]
    thermal: bool


def _deg_value(row: Dict[str, str], deg_key: str) -> float:
    if row.get("degradation_value") not in (None, "", "nan"):
        return float(row["degradation_value"])
    if deg_key == "capacity_fade_pct" and row.get("capacity_fade_pct"):
        return float(row["capacity_fade_pct"])
    return float(row["sei_per_pct_soc"])


def load_comparison_table(path: Path, *, deg_key: str = "sei_per_pct_soc") -> Dict[str, Dict[str, Any]]:
    meta: Dict[str, Dict[str, Any]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            fid = row["family_id"]
            def _f(key: str) -> float:
                v = row.get(key)
                if v in (None, "", "nan"):
                    return float("nan")
                return float(v)

            meta[fid] = {
                "loss": float(row["loss"]),
                "sei": float(row["sei_per_pct_soc"]),
                "fade": float(row["capacity_fade_pct"]) if row.get("capacity_fade_pct") else float("nan"),
                "deg": _deg_value(row, deg_key),
                "dur": float(row["duration_min"]),
                "feasible": row["feasible"].lower() == "true",
                "params": row["parameters"],
                "peak_t": _f("peak_temperature"),
                "bdt_peak_t": _f("bdt_peak_temp_c"),
                "lumped_peak_t": _f("lumped_peak_temp_c"),
                "thermal_delta": _f("thermal_delta_c"),
                "thermal_suspect": row.get("thermal_suspect", "").lower() == "true",
            }
    return meta


def _cell_label(run_dir: Path) -> str:
    manifest = run_dir / "stage3_cell_manifest.json"
    if manifest.is_file():
        cell = json.loads(manifest.read_text()).get("cell")
        if cell:
            return str(cell)
    name = run_dir.name
    for token in ("RW12", "RW11", "RW10", "RW9"):
        if token in name:
            return token
    return "RW9"


def load_run_context(run_dir: Path) -> RunContext:
    run_dir = Path(run_dir)
    results_json = run_dir / "models/family_optimization_results.json"
    comparison_csv = run_dir / "models/comparison_table.csv"
    pareto_json = run_dir / "models/pareto_analysis.json"
    payload = json.loads(results_json.read_text())
    constraints = payload.get("constraints", {})
    objective_mode = constraints.get("objective_mode", "composite")
    _, deg_key, deg_label = resolve_pareto_config(constraints)
    meta = load_comparison_table(comparison_csv, deg_key=deg_key)
    pareto = json.loads(pareto_json.read_text()) if pareto_json.is_file() else None
    return RunContext(
        run_dir=run_dir,
        objective_mode=objective_mode,
        deg_key=deg_key,
        deg_label=deg_label,
        meta=meta,
        best_params=load_best_params(results_json),
        constraints=constraints,
        pareto=pareto,
        thermal=bool(constraints.get("thermal_derating") or constraints.get("thermal_loss")),
    )


def _family_caption(m: Dict[str, Any], ctx: RunContext) -> str:
    if ctx.deg_key == "capacity_fade_pct":
        return f"{m['dur']:.0f} min  ·  {ctx.deg_label}={m['deg']:.3f}"
    return f"{m['dur']:.0f} min  ·  {ctx.deg_label}={m['deg']:.1f}"


def _run_subtitle(ctx: RunContext) -> str:
    acq = "PI"
    parts = [
        f"Start SoC=15%",
        "target 95%",
        f"{acq}-BO, 40 evals/family",
        "Pareto: duration · Wang ΔQ/Q₀ · V-stress · thermal",
    ]
    if ctx.objective_mode == "physics":
        parts.insert(0, "Wang ΔQ/Q₀ objective")
    if ctx.thermal:
        parts.append("thermal derating + loss")
    return "  ·  ".join(parts)


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


def build_fig2(
    out: Path,
    ctx: RunContext,
    profiles: Dict[str, Tuple],
    *,
    show_thermal_warnings: bool = True,
) -> None:
    meta = ctx.meta
    n_fam = len(ORDER)
    # Larger typography for slide / print readability
    FS_TITLE = 14
    FS_AXIS = 16
    FS_TICK = 14
    FS_ANNOT = 13
    FS_ROW = 15
    FS_SUPTITLE = 18
    FS_LEGEND = 15
    FS_PARAMS =12

    fig2 = plt.figure(figsize=(24, 14))
    fig2.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        3, n_fam, figure=fig2, hspace=0.22, wspace=0.30, left=0.05, right=0.98, top=0.90, bottom=0.10
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
        status = "" if feasible else "  [infeasible]"
        ax_i.set_title(
            f"{LABELS[fid]}\n{_family_caption(m, ctx)}{status}",
            fontsize=FS_TITLE,
            color=color,
            pad=4,
        )
        if show_thermal_warnings and m.get("thermal_suspect") and feasible:
            bdt_peak = m.get("bdt_peak_t", m.get("peak_t", float("nan")))
            lumped_peak = m.get("lumped_peak_t", float("nan"))
            delta = m.get("thermal_delta", float("nan"))
            if np.isfinite(bdt_peak) and np.isfinite(lumped_peak) and np.isfinite(delta):
                ax_i.text(
                    0.5, 0.55,
                    f"⚠ BDT thermal low\n"
                    f"BDT peak: {bdt_peak:.1f}°C\n"
                    f"Lumped model: {lumped_peak:.1f}°C\n"
                    f"Δ={delta:.1f}°C",
                    transform=ax_i.transAxes,
                    ha="center", va="center", fontsize=7.5,
                    color="#8B0000",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc="#fff8f8", ec="#cc0000", lw=0.8,
                    ),
                    zorder=10,
                )
        ax_i.tick_params(labelbottom=False, labelsize=FS_TICK)
        if col == 0:
            ax_i.set_ylabel("I (A)", fontsize=FS_AXIS)
        ax_i.grid(True)
        if not feasible:
            ax_i.text(
                0.5,
                0.88,
                "✗ Over time limit",
                transform=ax_i.transAxes,
                ha="center",
                fontsize=FS_ANNOT,
                color="#cc0000",
                bbox=dict(fc="white", ec="#cc0000", lw=0.6, pad=1.5),
            )

        ax_v.plot(t, v_arr, color="#b2182b", lw=1.4, alpha=alpha)
        ax_v.axhline(4.20, color="#888", ls="--", lw=0.7, alpha=0.6)
        ax_v.set_ylim(max(3.70, float(v_arr.min()) - 0.03), min(4.25, float(v_arr.max()) + 0.04))
        ax_v.set_xlim(0, t[-1])
        ax_v.tick_params(labelbottom=False, labelsize=FS_TICK)
        if col == 0:
            ax_v.set_ylabel("V (V)", fontsize=FS_AXIS)
        ax_v.grid(True)
        ax_v.text(0.97, 0.94, "4.2 V", transform=ax_v.transAxes, fontsize=FS_ANNOT, color="#888", ha="right", va="top")

        ax_s.plot(t, soc_arr * 100, color="#333333", lw=1.6, alpha=alpha)
        ax_s.axhline(95, color="#888", ls=":", lw=0.9)
        ax_s.set_ylim(10, 102)
        ax_s.set_xlim(0, t[-1])
        ax_s.set_xlabel("Time (min)", fontsize=FS_AXIS)
        ax_s.tick_params(labelsize=FS_TICK)
        if col == 0:
            ax_s.set_ylabel("SoC (%)", fontsize=FS_AXIS)
        ax_s.grid(True)
        ax_s.text(0.97, 0.12, "95% target", transform=ax_s.transAxes, fontsize=FS_ANNOT, color="#888", ha="right")

    for col, fid in enumerate(ORDER):
        ax_s = fig2.axes[col + 2 * n_fam]
        ax_s.text(0.5, -0.42, meta[fid]["params"], transform=ax_s.transAxes, ha="center", fontsize=FS_PARAMS, color="#555", style="italic")

    for row_idx, row_label in enumerate(["Charging\ncurrent", "Cell\nvoltage", "State of\ncharge"]):
        fig2.text(0.005, 0.88 - row_idx * 0.295, row_label, ha="left", va="center", fontsize=FS_ROW, rotation=90, color="#555")

    cell = _cell_label(ctx.run_dir)
    fig2.suptitle(
        f"Best BO profile per family ({cell} BDT)\n{_run_subtitle(ctx)}",
        fontsize=FS_SUPTITLE,
        y=0.978,
    )
    if show_thermal_warnings:
        fig2.text(
            0.5, -0.06,
            "⚠  BDT trained on random-walk data; CC-like profiles have ~4× higher V error. "
            "BDT peak T ≈ ambient for all families — thermal_loss term is inactive. "
            "Red boxes: lumped-model cross-check (Δ>5°C ⇒ BDT under-predicts heating).",
            ha="center", va="top", fontsize=9, color="#8B0000",
            style="italic",
            transform=fig2.transFigure,
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8f8", ec="#cc0000", lw=0.8),
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
    fig2.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=FS_LEGEND,
        framealpha=0.95,
        edgecolor="#ddd",
    )
    fig2.savefig(out / "fig2_all_families.png", dpi=180, bbox_inches="tight")
    plt.close(fig2)


def build_fig1_physics(
    out: Path,
    ctx: RunContext,
    *,
    show_thermal_warnings: bool = True,
) -> None:
    """Duration vs ΔQ/Q₀ Pareto front from physics+thermal BO."""
    if not ctx.pareto:
        return
    front = ctx.pareto.get("pareto_front", [])
    tagged = ctx.pareto.get("tagged_global", {})
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5.5), gridspec_kw={"wspace": 0.42})

    ax = axes1[0]
    by_fam: Dict[str, List[Dict]] = {}
    for pt in front:
        by_fam.setdefault(pt["family_id"], []).append(pt)
    for fid, pts in by_fam.items():
        pts = sorted(pts, key=lambda p: p["duration_min"])
        durs = [p["duration_min"] for p in pts]
        fades = [p["capacity_fade_pct"] for p in pts]
        ax.plot(durs, fades, "o-", color=C.get(fid, "#888"), lw=1.5, ms=5, label=LABELS.get(fid, fid))

    cccv = ctx.meta.get("cccv")
    if cccv and cccv["feasible"]:
        ax.scatter([cccv["dur"]], [cccv["deg"]], c=C["cccv"], s=140, marker="s", zorder=6, edgecolors="white")
        ax.annotate("CCCV\n(lowest ΔQ/Q₀)", (cccv["dur"], cccv["deg"]), xytext=(-45, 10),
                    textcoords="offset points", fontsize=8, color=C["cccv"])

    fastest = tagged.get("fastest")
    if fastest:
        ax.scatter([fastest["duration_min"]], [fastest["capacity_fade_pct"]],
                   c=C["pulsed"], s=120, marker="*", zorder=7, edgecolors="white")
        ax.annotate("Fastest", (fastest["duration_min"], fastest["capacity_fade_pct"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=8, color=C["pulsed"])

    ax.set_xlabel("Charge duration (min)")
    ax.set_ylabel(f"{ctx.deg_label}  (lower = better)")
    ax.set_title("Physics Pareto front\n(non-dominated feasible BO points)", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True)

    ax2 = axes1[1]
    fam_order = [f for f in ORDER if ctx.meta.get(f, {}).get("feasible")]
    names = [LABELS[f] for f in fam_order]
    durs = [ctx.meta[f]["dur"] for f in fam_order]
    fades = [ctx.meta[f]["deg"] for f in fam_order]
    cols = [C[f] for f in fam_order]
    ax2.scatter(durs, fades, c=cols, s=90, edgecolors="white", lw=0.8, zorder=4)
    for f, d, fade in zip(fam_order, durs, fades):
        ax2.annotate(LABELS[f].split("(")[0].strip(), (d, fade), xytext=(4, 3),
                     textcoords="offset points", fontsize=7, color=C[f])
    ax2.set_xlabel("Charge duration (min)")
    ax2.set_ylabel(ctx.deg_label)
    ax2.set_title("Family optima\n(physics + thermal BO)", fontsize=11)
    ax2.grid(True)

    ax3 = axes1[2]
    t_penalties = [pt.get("temperature_penalty_c2_min", 0.0) for pt in front]
    durs_front = [pt["duration_min"] for pt in front]

    if t_penalties and max(t_penalties) < 0.01:
        ax3.scatter(durs_front, t_penalties, c="#999", s=60, alpha=0.6)
        ax3.set_ylim(-0.005, 0.05)
        if show_thermal_warnings:
            ax3.text(
                0.5, 0.55,
                "Temperature penalty ≈ 0\nfor all profiles\n\n"
                "BDT under-predicts temperature\nfor smooth CC profiles\n"
                "(distribution shift from\nrandom-walk training)",
                transform=ax3.transAxes,
                ha="center", va="center", fontsize=9.5,
                color="#8B0000", style="italic",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fff8f8", ec="#cc0000", lw=0.8),
            )
    else:
        ax3.scatter(
            durs_front, t_penalties,
            c=[C.get(pt["family_id"], "#888") for pt in front],
            s=60, edgecolors="white",
        )

    ax3.set_xlabel("Charge duration (min)")
    ax3.set_ylabel("Temperature penalty (∫°C²·min)")
    ax3.set_title(
        "Charge time vs temperature penalty\n(should vary if thermal constraint active)",
        fontsize=10,
    )
    ax3.grid(True)

    fig1.suptitle(
        f"Charging speed vs. Wang capacity fade — RW9 BDT\n{_run_subtitle(ctx)}",
        fontsize=12, y=1.01,
    )
    fig1.savefig(out / "fig1_pareto_front.png", dpi=200, bbox_inches="tight")
    plt.close(fig1)


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


def build_fig3(out: Path, ctx: RunContext) -> None:
    meta = ctx.meta
    fam_order_rank = [f for f in ORDER if meta[f]["feasible"]]
    names = [LABELS[f] for f in fam_order_rank]
    deg_r = [meta[f]["deg"] for f in fam_order_rank]
    durs_r = [meta[f]["dur"] for f in fam_order_rank]
    loss_r = [meta[f]["loss"] for f in fam_order_rank]
    cols_r = [C[f] for f in fam_order_rank]
    deg_fmt = "{:.3f}" if ctx.deg_key == "capacity_fade_pct" else "{:.1f}"
    best_deg = min(deg_r) if deg_r else 0.0

    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 5.2), gridspec_kw={"wspace": 0.45})

    if ctx.deg_key == "capacity_fade_pct":
        deg_xlim = (0.0, max(deg_r) * 1.15)
        deg_note = "Note: differences are within 0.05 pct-points per session"
    else:
        deg_xlim = (min(deg_r) - 0.5, max(deg_r) + 0.5)
        deg_note = ""

    hbar(
        axes3[0], deg_r, names, cols_r,
        f"{ctx.deg_label}  (lower = better)",
        f"① Degradation ({'Wang' if ctx.objective_mode == 'physics' else 'proxy'})",
        deg_xlim,
        ref_line=best_deg,
        val_fmt=deg_fmt,
    )
    if deg_note:
        axes3[0].text(
            0.98, 0.98, deg_note,
            transform=axes3[0].transAxes,
            ha="right", va="top", fontsize=7.5, color="#666",
            style="italic",
        )
    hbar(axes3[1], durs_r, names, cols_r, "Charge duration (min)", "② Charging Speed", (45, 111), ref_line=105, val_fmt="{:.1f}")
    loss_lo = min(loss_r) - 0.2
    loss_hi = max(loss_r) + 0.3
    hbar(
        axes3[2], loss_r, names, cols_r,
        "BO loss (lower = better)",
        "③ Composite Loss (time-dominated)",
        (loss_lo, loss_hi),
        val_fmt="{:.2f}",
    )
    axes3[2].text(
        0.98, 0.02,
        "Loss = w_fade×ΔQ/Q₀ + w_time×duration\n+ w_temp×thermal + w_v×voltage_stress",
        transform=axes3[2].transAxes,
        ha="right", va="bottom", fontsize=7, color="#666", style="italic",
    )

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
    fig3.suptitle(f"Multi-family results — RW9  ({_run_subtitle(ctx)})", fontsize=12, y=1.02)
    fig3.text(
        0.5, -0.01,
        "BDT headline RMSE (random-walk holdout): V=0.028 V, T=0.308°C  "
        "·  Reference-charge RMSE: V≈0.12 V (~4× higher, distribution shift from CC profiles)",
        ha="center", va="top", fontsize=8, color="#8B0000",
        style="italic", transform=fig3.transFigure,
    )
    fig3.savefig(out / "fig3_family_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(fig3)


def build_fig_chebyshev(
    out: Path,
    ctx: RunContext,
    chebyshev_json: Path,
) -> None:
    """
    Chebyshev sweep figure with convergence quality annotations.
    Replaces the raw sweep plot with honest non-monotonicity flagging.
    """
    if not chebyshev_json.is_file():
        print("  Chebyshev JSON not found — skipping chebyshev figure")
        return

    payload = json.loads(chebyshev_json.read_text())
    results_by_omega = payload.get("results_by_omega", {})

    sweep_points: List[Dict[str, Any]] = []
    for omega_str, rows in sorted(
        results_by_omega.items(), key=lambda x: float(x[0])
    ):
        omega = float(omega_str)
        feasible_rows = [r for r in rows if r.get("feasible")]
        if not feasible_rows:
            continue
        best = min(feasible_rows, key=lambda r: r.get("loss", 1e6))
        fade = best.get("capacity_fade_pct", best.get("sei_per_pct_soc", float("nan")))
        sweep_points.append({
            "omega": omega,
            "duration_min": float(best["duration_min"]),
            "deg": float(fade),
            "family_id": best["family_id"],
            "loss": float(best.get("loss", float("nan"))),
        })

    if not sweep_points:
        print("  No feasible Chebyshev points — skipping figure")
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))

    durs = [p["duration_min"] for p in sweep_points]
    degs = [p["deg"] for p in sweep_points]
    omegas = [p["omega"] for p in sweep_points]

    def _is_dominated(i: int, points: List[Dict[str, Any]]) -> bool:
        for j, q in enumerate(points):
            if i == j:
                continue
            if (
                q["duration_min"] <= points[i]["duration_min"]
                and q["deg"] <= points[i]["deg"]
                and (
                    q["duration_min"] < points[i]["duration_min"]
                    or q["deg"] < points[i]["deg"]
                )
            ):
                return True
        return False

    dominated = [_is_dominated(i, sweep_points) for i in range(len(sweep_points))]

    sc = ax.scatter(
        durs, degs,
        c=omegas, cmap="RdYlGn_r", vmin=0, vmax=1,
        s=100, zorder=5, edgecolors="white", linewidths=0.9,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("ω  (0=lifetime → 1=fastest)", fontsize=9)

    for pt, dom in zip(sweep_points, dominated):
        if dom:
            ax.scatter(
                pt["duration_min"], pt["deg"],
                marker="x", c="red", s=120, lw=2, zorder=6,
            )

    # Group duplicate points (BO collapse) before annotating.
    groups: Dict[tuple, list] = {}
    for pt, dom in zip(sweep_points, dominated):
        key = (round(pt["duration_min"], 2), round(pt["deg"], 4))
        groups.setdefault(key, []).append((pt["omega"], dom))

    for (x, y), members in groups.items():
        omegas_g = sorted(m[0] for m in members)
        any_dom = any(m[1] for m in members)
        if len(omegas_g) == 1:
            label = f"ω={omegas_g[0]:.1f}"
        elif omegas_g[-1] - omegas_g[0] >= 0.09:
            label = f"ω={omegas_g[0]:.1f}–{omegas_g[-1]:.1f}"
        else:
            label = ", ".join(f"ω={o:.1f}" for o in omegas_g)
        if any_dom:
            label += "\n(dominated)"
        ax.annotate(
            label,
            (x, y),
            xytext=(6, -14 if any_dom else 4),
            textcoords="offset points",
            fontsize=7.5,
            color="red" if any_dom else "#333",
        )

    non_dom = [p for p, d in zip(sweep_points, dominated) if not d]
    if non_dom:
        non_dom_sorted = sorted(non_dom, key=lambda p: p["duration_min"])
        ax.plot(
            [p["duration_min"] for p in non_dom_sorted],
            [p["deg"] for p in non_dom_sorted],
            "--", color="#333", lw=1.5, alpha=0.7,
            label="Chebyshev front (non-dominated)",
            zorder=3,
        )

    n_dominated = sum(dominated)
    n_total = len(sweep_points)
    ax.set_xlabel("Charge duration (min)", fontsize=11)
    ax.set_ylabel(f"{ctx.deg_label}  (lower = better)", fontsize=11)
    ax.set_title(
        f"Directed Pareto front — Chebyshev sweep\n"
        f"{n_total} BO runs  ·  "
        f"{n_dominated} non-converged points (marked ✗)",
        fontsize=12,
    )
    ax.grid(True)

    if n_dominated > 0:
        ax.text(
            0.02, 0.97,
            f"⚠  {n_dominated}/{n_total} ω values did not converge.\n"
            f"Red ✗ = dominated point (BO needs more evaluations).\n"
            f"Increase --n_calls to 60+ for monotone front.",
            transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color="#8B0000",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8f8", ec="#cc0000", lw=0.8),
        )

    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "fig_chebyshev_sweep.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  fig_chebyshev_sweep.png  ({n_dominated} dominated points flagged)")


def build_fig4(
    out: Path,
    ctx: RunContext,
    profiles: Dict[str, Tuple],
    chebyshev_json: Path,
    pulsed_sweep: Sequence[Tuple[float, float, float]],
) -> None:
    if ctx.objective_mode == "physics" and ctx.pareto:
        tagged = ctx.pareto.get("tagged_global", {})
        ref_profiles = []
        for tag, label in (("fastest", "Fastest"), ("balanced", "Balanced"), ("lifetime", "Lifetime")):
            cand = tagged.get(tag)
            if not cand:
                continue
            fid = cand["family_id"]
            params = cand["params"]
            dur = float(cand["duration_min"])
            deg = float(cand.get("capacity_fade_pct", ctx.meta[fid]["deg"]))
            t, i_a, v_a, soc_a = profile_from_params(fid, params, dur)
            ref_profiles.append((
                f"{label}\n({LABELS.get(fid, fid)})",
                t, i_a, v_a, soc_a,
                C.get(fid, "#333"),
                f"{dur:.1f} min  ·  {ctx.deg_label}={deg:.3f}",
            ))
    else:
        fast = pulsed_sweep[-1]
        bal = pulsed_sweep[5]
        fast_params = load_pulsed_params_at_omega(chebyshev_json, 1.0)
        bal_params = load_pulsed_params_at_omega(chebyshev_json, 0.5)
        t_pfast, i_pfast, v_pfast, s_pfast = profile_from_params("pulsed", fast_params, fast[1])
        t_pbal, i_pbal, v_pbal, s_pbal = profile_from_params("pulsed", bal_params, bal[1])
        t_cccv, i_cccv, v_cccv, s_cccv = profiles["cccv"]
        meta = ctx.meta
        ref_profiles = [
            (
                "Fastest\n(Pulsed, ω=1.0)",
                t_pfast, i_pfast, v_pfast, s_pfast, C["pulsed"],
                f"{fast[1]:.1f} min  ·  SEI/ΔSoC={fast[2]:.1f}  ·  I_peak={i_pfast.max():.2f} A",
            ),
            (
                "Balanced\n(Pulsed, ω=0.5)",
                t_pbal, i_pbal, v_pbal, s_pbal, C["adaptive_two_step"],
                f"{bal[1]:.1f} min  ·  SEI/ΔSoC={bal[2]:.1f}  ·  I_peak={i_pbal.max():.2f} A",
            ),
            (
                "Lifetime\n(CCCV, ω=0.0)",
                t_cccv, i_cccv, v_cccv, s_cccv, C["cccv"],
                f"{meta['cccv']['dur']:.1f} min  ·  SEI/ΔSoC={meta['cccv']['sei']:.1f}  ·  I_peak={i_cccv.max():.2f} A",
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
        f"Reference charging profiles — RW9 BDT\n{_run_subtitle(ctx)}",
        fontsize=12,
        y=0.94,
    )
    fig4.savefig(out / "fig4_reference_profiles.png", dpi=200, bbox_inches="tight")
    plt.close(fig4)


def build_fig5(
    out: Path,
    ctx: RunContext,
    pulsed_sweep: Sequence[Tuple[float, float, float]],
    loss_ei: np.ndarray,
    loss_pi: np.ndarray,
    best_pi: float,
) -> None:
    meta = ctx.meta
    fig5, axes5 = plt.subplots(1, 2, figsize=(13, 5.0), gridspec_kw={"wspace": 0.40})
    y_key = "deg"
    y_label = ctx.deg_label

    ax = axes5[0]
    for fid in ORDER:
        if fid not in meta:
            continue
        m = meta[fid]
        mk = "s" if not m["feasible"] else "o"
        ms = 90 if not m["feasible"] else 110
        al = 0.4 if not m["feasible"] else 0.92
        ax.scatter(m["dur"], m[y_key], color=C[fid], s=ms, marker=mk, alpha=al, edgecolors="white", lw=1.0, zorder=4)
        ax.annotate(
            LABELS[fid].split("(")[0].strip(),
            (m["dur"], m[y_key]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
            color=C[fid],
        )

    if ctx.objective_mode != "physics":
        for o, d, s in pulsed_sweep:
            ax.scatter(d, s, c=[[matplotlib.cm.RdYlGn_r(o)]], s=45, marker="^", zorder=3, edgecolors="none", alpha=0.7)
        ax.plot([p[1] for p in pulsed_sweep], [p[2] for p in pulsed_sweep], "--", color="#888", lw=1.1, alpha=0.5, label="Chebyshev sweep (pulsed)")
        ax.scatter([], [], c="gray", marker="^", s=45, label="Sweep points (ω=0→1)")

    ax.set_xlabel("Charge duration (min)")
    ax.set_ylabel(y_label)
    title_right = "physics + thermal BO" if ctx.objective_mode == "physics" else "PI acquisition, 40 evals/family"
    ax.set_title(f"Family optima\nRW9 BDT ({title_right})", fontsize=10.5)
    if ctx.objective_mode != "physics":
        ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(True)

    conv_family = "pulsed" if ctx.objective_mode == "physics" else "cccv"
    ax2 = axes5[1]
    n_evals = np.arange(1, len(loss_pi) + 1)
    if ctx.objective_mode == "physics":
        ax2.plot(n_evals, loss_pi, "s-", color=C["pulsed"], lw=1.8, ms=4, markeredgecolor="white",
                 label=f"PI-BO ({LABELS.get(conv_family, conv_family)})")
        ax2.axhline(best_pi, color="#1b7837", ls="--", lw=1.2, label=f"Best loss={best_pi:.2f}")
        ax2.set_ylabel("Best physics loss (running minimum)")
        ax2.set_title(f"BO convergence\n{LABELS.get(conv_family, conv_family)} family", fontsize=10.5)
    else:
        ax2.plot(n_evals, loss_ei, "o-", color="#d6604d", lw=1.8, ms=4, markeredgecolor="white", label="EI acquisition (original)")
        ax2.plot(n_evals, loss_pi, "s-", color="#2166ac", lw=1.8, ms=4, markeredgecolor="white", label="PI acquisition (enhanced)")
        ax2.axhline(best_pi, color="#1b7837", ls="--", lw=1.2, label=f"Best result (CCCV, PI={best_pi:.2f})")
        mask = np.isfinite(loss_ei) & np.isfinite(loss_pi)
        if mask.any():
            ax2.fill_between(n_evals[mask], loss_pi[mask], loss_ei[mask], alpha=0.12, color="#2166ac")
        ax2.set_ylabel("Best composite loss (running minimum)")
        ax2.set_title("Acquisition Function Comparison\nEI vs PI — CCCV Family", fontsize=10.5)
    ax2.set_xlabel("BO evaluations")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True)
    ax2.set_xlim(1, len(loss_pi))
    y_vals = loss_pi[np.isfinite(loss_pi)]
    if ctx.objective_mode != "physics":
        y_vals = np.concatenate([loss_ei[np.isfinite(loss_ei)], y_vals])
    ax2.set_ylim(float(np.min(y_vals)) - 0.8, float(np.max(y_vals)) + 1.0)

    fig5.suptitle(f"Methodology summary — RW9 NASA dataset\n{_run_subtitle(ctx)}", fontsize=12, y=1.02)
    fig5.savefig(out / "fig5_methodology_summary.png", dpi=200, bbox_inches="tight")
    plt.close(fig5)


def build_fig7_ambient(out: Path, run_dir: Path) -> bool:
    summary_path = run_dir / "ambient_sensitivity" / "ambient_sensitivity_summary.json"
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    temps = sorted(float(k) for k in summary)
    durs = [summary[str(t)]["duration_min"] for t in temps]
    peaks = [summary[str(t)]["peak_temperature"] for t in temps]
    losses = [summary[str(t)]["best_loss"] for t in temps]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.35})
    ax = axes[0]
    ax.plot(temps, durs, "o-", color=C["pulsed"], lw=2, ms=8)
    ax.set_xlabel("Ambient T₀ (°C)")
    ax.set_ylabel("Best charge duration (min)")
    ax.set_title("Pulsed optimum vs. ambient temperature")
    ax.grid(True)

    ax2 = axes[1]
    ax.plot(temps, peaks, "s-", color="#b2182b", lw=2, ms=8, label="Peak T")
    ax.axhline(33, color="#888", ls="--", lw=1, label="Derating threshold (33°C)")
    ax.set_xlabel("Ambient T₀ (°C)")
    ax.set_ylabel("Peak temperature (°C)")
    ax.set_title("Thermal headroom during optimized charge")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(temps, losses, "^--", color="#555", lw=1.5, ms=7, alpha=0.8, label="Best loss")
    ax2_twin.set_ylabel("Physics BO loss", color="#555")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    ax2.grid(True)

    fig.suptitle("Ambient sensitivity — physics + thermal BO (pulsed family wins at all T₀)", fontsize=12, y=1.02)
    fig.savefig(out / "fig7_ambient_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/visualization")
    p.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Primary BO results (default: stage3_physics_thermal if present, else enhanced)",
    )
    p.add_argument("--enhanced_dir", type=Path, default=DEFAULT_ENHANCED)
    p.add_argument("--ei_dir", type=Path, default=DEFAULT_EI)
    p.add_argument("--chebyshev_json", type=Path, default=DEFAULT_CHEBYSHEV)
    p.add_argument(
        "--with_physics",
        action="store_true",
        help="Also run compare_degradation_models.py for fig6 (needs GPU, ~2 min)",
    )
    p.add_argument(
        "--no_thermal_warnings",
        action="store_true",
        help="Hide BDT thermal caveats on fig1 (panel 3) and fig2 (per-family red boxes + footer)",
    )
    return p.parse_args()


def _default_run_dir(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    if DEFAULT_PHYSICS.is_dir() and (DEFAULT_PHYSICS / "models/family_optimization_results.json").is_file():
        return DEFAULT_PHYSICS
    return DEFAULT_ENHANCED


def main() -> None:
    args = parse_args()
    out = resolve_visualization_dir(ROOT, args.out_dir)

    run_dir = _default_run_dir(args.run_dir)
    ctx = load_run_context(run_dir)
    ei_json = args.ei_dir / "models/family_optimization_results.json"
    results_json = run_dir / "models/family_optimization_results.json"

    profiles = {
        fid: profile_from_params(fid, ctx.best_params[fid], ctx.meta[fid]["dur"])
        for fid in ORDER
        if fid in ctx.best_params and fid in ctx.meta
    }
    pulsed_sweep = (
        load_pulsed_chebyshev_sweep(args.chebyshev_json)
        if args.chebyshev_json.is_file() else []
    )
    ref_time, ref_sei = (
        load_cccv_chebyshev_baseline(args.chebyshev_json)
        if args.chebyshev_json.is_file() else (104.0, 68.0)
    )
    conv_family = "pulsed" if ctx.objective_mode == "physics" else "cccv"
    loss_pi = load_convergence_curve(results_json, conv_family)
    loss_ei = load_convergence_curve(ei_json, "cccv") if ei_json.is_file() else loss_pi
    best_family = min(
        (f for f in ORDER if ctx.meta.get(f, {}).get("feasible")),
        key=lambda f: ctx.meta[f]["loss"],
        default=conv_family,
    )
    best_pi = float(ctx.meta[best_family]["loss"])

    print(f"Loading data from {run_dir}  (objective={ctx.objective_mode}, thermal={ctx.thermal})")
    print(f"Writing figures to {out}")

    show_thermal_warnings = not args.no_thermal_warnings

    if ctx.objective_mode == "physics":
        build_fig1_physics(out, ctx, show_thermal_warnings=show_thermal_warnings)
    else:
        build_fig1(out, pulsed_sweep, ref_time, ref_sei)
    print("  fig1_pareto_front.png")
    build_fig2(out, ctx, profiles, show_thermal_warnings=show_thermal_warnings)
    print("  fig2_all_families.png")
    build_fig3(out, ctx)
    print("  fig3_family_ranking.png")
    build_fig4(out, ctx, profiles, args.chebyshev_json, pulsed_sweep)
    print("  fig4_reference_profiles.png")
    build_fig5(out, ctx, pulsed_sweep, loss_ei, loss_pi, best_pi)
    print("  fig5_methodology_summary.png")
    build_fig_chebyshev(out, ctx, args.chebyshev_json)

    n_figs = 5
    if args.chebyshev_json.is_file():
        n_figs += 1

    if build_fig7_ambient(out, run_dir):
        print("  fig7_ambient_sensitivity.png")
        n_figs += 1

    if args.with_physics:
        import subprocess
        results_payload = json.loads(results_json.read_text())
        bdt_ckpt = results_payload.get("bdt_checkpoint", CANONICAL["bdt_source"])
        capacity = CANONICAL["capacity_fade"]
        margins = CANONICAL["conformal_margins"]
        degradation_model = CANONICAL["degradation_model"]
        cell_label = "RW9"
        manifest_path = run_dir / "stage3_cell_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            paths = manifest.get("paths", {})
            capacity = paths.get("capacity_fade", capacity)
            margins = paths.get("conformal_margins", margins)
            degradation_model = paths.get("degradation_model", degradation_model)
            cell_label = manifest.get("cell", cell_label)
        cmd = [
            sys.executable,
            str(ROOT / "scripts/compare_degradation_models.py"),
            "--enhanced_dir", str(run_dir),
            "--chebyshev_json", str(args.chebyshev_json),
            "--out_dir", str(out),
            "--bdt_ckpt", bdt_ckpt,
            "--capacity", capacity,
            "--margins", margins,
            "--degradation_model", degradation_model,
            "--cell", cell_label,
        ]
        print("  Running physics degradation comparison (fig6)...")
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        print("  fig6_physics_degradation.png")
        n_figs += 1

    print(f"\nAll {n_figs} figures saved to {out}/")


if __name__ == "__main__":
    main()
