"""Lifetime capacity-fade line plots for different charging policies.

Projects the hybrid calendar+cyclic model forward over repeated *equal-energy*
charge sessions (same constraint as the BO run). This is a **model projection**
anchored to each policy's simulated session (Ah, T, SoC, C-rate, duration) —
not a multi-year lab test under each protocol.

Measured NASA RW9 fade is drawn as a dashed reference on the Ah axis (random-
walk duty, not the same charge protocols).

Usage
-----
    python -m Constrained_BO.lifetime_fade_projection \\
        --bo Constrained_BO/results/ui_runs_1/RW9/gp_bo_results.json \\
        --random Constrained_BO/results/ui_runs_1/RW9/random_search_results.json \\
        --out-dir Constrained_BO/results/grounded_figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from Constrained_BO.bo_degradation_comparison import (
    COLORS,
    DEFAULT_C_RATES,
    _cccv_rate_label,
    _cell_from_meta,
    _eval_optimizer_best,
    _load_json,
    _reward_kwargs,
)
from Constrained_BO.compare_constant_current import _format_profile, _metrics_row
from Constrained_BO.config import Q_RATED_AH
from Constrained_BO.viz import PAPER_DPI, PAPER_LIGHT_BG, apply_paper_style
from Constrained_BO.hybrid_degradation import HybridDegradation, HybridDegradationParameters
from Constrained_BO.objective import evaluate_session
from Constrained_BO.optimize_api import evaluate_cccv_baselines
from Constrained_BO.profiles import get_family, set_profile_bounds
from Constrained_BO.simulator import ChargingSimulator

ROOT = Path(__file__).resolve().parents[1]

# Line styles / colors for lifetime curves
POLICY_STYLE = {
    "CCCV ½C": {"color": "#64748b", "ls": "-", "lw": 2.2},
    "CCCV 1C": {"color": "#2563eb", "ls": "--", "lw": 1.6},
    "CCCV 2C": {"color": "#1e3a8a", "ls": ":", "lw": 1.6},
    "Random": {"color": "#f59e0b", "ls": "-", "lw": 2.4},
    "GP-BO": {"color": "#16a34a", "ls": "-", "lw": 2.6},
    "GP-BO (min Q)": {"color": "#0f766e", "ls": "-", "lw": 2.4},
    # Legacy aliases
    "CC ½C": {"color": "#64748b", "ls": "-", "lw": 2.2},
    "CC 1C": {"color": "#2563eb", "ls": "--", "lw": 1.6},
    "CC 2C": {"color": "#1e3a8a", "ls": ":", "lw": 1.6},
}


def _session_metrics_from_row(row: Dict[str, Any]) -> Dict[str, float]:
    m = row.get("metrics") or {}
    return {
        "ah_throughput": float(m.get("ah_throughput") or 0.0),
        "duration_h": float(m.get("duration_s") or row.get("duration_s") or 0.0) / 3600.0,
        "mean_temperature_c": float(
            m.get("mean_temperature") or row.get("mean_temperature") or 25.0
        ),
        "mean_soc": float(m.get("mean_soc") or 0.5),
        "nominal_c_rate": float(m.get("nominal_c_rate") or 0.0),
        "qloss_total": float(row.get("qloss_total") or m.get("qloss_total") or 0.0),
        "feasible": bool(row.get("feasible")),
        "duration_min": float(row.get("duration_min") or 0.0),
    }


def _collect_policies(
    bo_path: Path,
    random_path: Path,
    *,
    device: str = "cpu",
    include_infeasible_cc: bool = False,
    gpbo_select: str = "min_q",
    c_rates: Sequence[float] = DEFAULT_C_RATES,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Collect policies for lifetime projection.

    gpbo_select:
      - ``min_q`` (default): feasible BO trial with lowest session Q_loss —
        use for capacity-fade figures.
      - ``reward``: optimizer reward/loss winner (may trade fade for speed).
    """
    bo = _load_json(bo_path)
    rnd = _load_json(random_path)
    meta = bo.get("meta") or rnd.get("meta") or {}
    cell = _cell_from_meta(meta)
    if cell.profile_bounds is not None:
        set_profile_bounds(cell.profile_bounds)
    rw = _reward_kwargs(meta)
    simulator = ChargingSimulator.from_cell(cell, device=device)

    c_rates = tuple(float(c) for c in c_rates)
    currents = [float(c) * float(Q_RATED_AH) for c in c_rates]
    cccv_rows = evaluate_cccv_baselines(cell, simulator, currents_a=currents, **rw)

    policies: List[Dict[str, Any]] = []
    for c_rate, row in zip(c_rates, cccv_rows):
        name = _cccv_rate_label(float(c_rate))
        feas = bool(row.get("feasible"))
        if not feas and not include_infeasible_cc:
            continue
        sm = _session_metrics_from_row(row)
        policies.append({
            "name": name,
            "feasible": feas,
            **sm,
            "profile": row.get("profile"),
        })

    rnd_by = "qloss" if gpbo_select == "min_q" else "reward"
    rnd_row = _eval_optimizer_best(
        cell, simulator, rnd,
        method_label="Random", reward_kwargs=rw, by=rnd_by,
    )
    policies.append({
        "name": "Random",
        "feasible": True,
        **_session_metrics_from_row(rnd_row),
        "profile": rnd_row.get("profile"),
    })

    by = "qloss" if gpbo_select == "min_q" else "reward"
    bo_row = _eval_optimizer_best(
        cell, simulator, bo,
        method_label="GP-BO", reward_kwargs=rw, by=by,
    )
    policies.append({
        "name": "GP-BO",
        "feasible": True,
        **_session_metrics_from_row(bo_row),
        "profile": bo_row.get("profile"),
        "gpbo_select": by,
    })

    info = {
        "cell": cell.cell_id,
        "constraint_mode": cell.constraint_mode,
        "energy_fraction": cell.energy_fraction,
        "q_rated_ah": Q_RATED_AH,
        "bdt_ckpt": str(cell.bdt_ckpt),
        "gpbo_select": by,
    }
    return policies, info


def project_fade(
    policies: Sequence[Dict[str, Any]],
    *,
    n_cycles: int = 600,
    soh_anchor_pct: float = 80.0,
    anchor_policy: str = "CCCV ½C",
    anchor_cycle: int = 400,
    hybrid_params: Optional[HybridDegradationParameters] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]], float]:
    """Cumulative hybrid Q → remaining-capacity proxy [%].

    Scale is set so ``anchor_policy`` reaches ``soh_anchor_pct`` at ``anchor_cycle``.
    Ranking across policies is invariant to this affine scale.
    """
    model = HybridDegradation(hybrid_params or HybridDegradationParameters())
    cycles = np.arange(0, n_cycles + 1, dtype=float)

    raw: Dict[str, Dict[str, np.ndarray]] = {}
    for p in policies:
        ah_s = max(float(p["ah_throughput"]), 1e-9)
        t_s = max(float(p["duration_h"]), 1e-12)
        T = float(p["mean_temperature_c"])
        soc = float(p["mean_soc"])
        crate = float(p["nominal_c_rate"])

        ah = cycles * ah_s
        t_h = cycles * t_s
        q_cal = np.array([model.calendar.q_loss(soc, T, float(th)) for th in t_h])
        q_cyc = np.array([model.cyclic.q_loss(float(a), T, crate) for a in ah])
        q_tot = q_cal + q_cyc
        raw[p["name"]] = {
            "cycles": cycles,
            "cum_ah": ah,
            "qloss_calendar": q_cal,
            "qloss_cyclic": q_cyc,
            "qloss_total": q_tot,
            "feasible": np.full(cycles.shape, bool(p["feasible"])),
        }

    if anchor_policy not in raw:
        anchor_policy = next(iter(raw))
    q_anchor = float(raw[anchor_policy]["qloss_total"][int(anchor_cycle)])
    # 100 → soh_anchor over q_anchor
    fade_span = 100.0 - float(soh_anchor_pct)
    scale = fade_span / q_anchor if q_anchor > 0 else 1.0

    for name, d in raw.items():
        rem = 100.0 - scale * d["qloss_total"]
        d["remaining_pct"] = np.clip(rem, 0.0, 100.0)

    return cycles, raw, scale


def _load_measured_cell(cell_id: str) -> Optional[Dict[str, np.ndarray]]:
    path = ROOT / "Constrained_BO/results/grounded_figures/capacity_fade_measured.csv"
    if not path.is_file():
        return None
    refs, pct, ah = [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["cell"] != cell_id.upper():
                continue
            refs.append(float(row["ref_number"]))
            pct.append(float(row["remaining_pct"]))
            ah.append(float(row["cum_throughput_Ah"]))
    if not refs:
        return None
    return {
        "ref_number": np.asarray(refs),
        "remaining_pct": np.asarray(pct),
        "cum_ah": np.asarray(ah),
    }


def plot_lifetime_vs_cycles(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    info: Dict[str, Any],
    *,
    scale: float,
    anchor_cycle: int,
    soh_anchor_pct: float,
    out_path: Path,
) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=PAPER_DPI, facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)

    for p in policies:
        name = p["name"]
        if name not in curves:
            continue
        d = curves[name]
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        ax.plot(
            d["cycles"], d["remaining_pct"],
            color=style["color"], ls=style["ls"], lw=style.get("lw", 2.2),
            label=name,
        )

    ax.axhline(soh_anchor_pct, color="#94a3b8", ls="--", lw=1.2, alpha=0.8)
    ax.axvline(anchor_cycle, color="#94a3b8", ls="--", lw=1.2, alpha=0.5)
    ax.set_xlabel("Equivalent charge cycles (each = one 40% energy session)", fontsize=14)
    ax.set_ylabel("Projected remaining capacity [%]", fontsize=14)
    efrac = info.get("energy_fraction")
    cell = info.get("cell", "?")
    ax.set_title(
        f"{cell}: projected capacity fade under different charging policies\n"
        f"(hybrid calendar+cyclic model · equal-energy sessions"
        f"{f' · {100 * float(efrac):.0f}% energy' if efrac is not None else ''})",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlim(0, float(next(iter(curves.values()))["cycles"][-1]))
    ax.set_ylim(75, 100)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight", facecolor=PAPER_LIGHT_BG)
    plt.close(fig)


def plot_delta_vs_halfc(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    info: Dict[str, Any],
    out_path: Path,
) -> None:
    baseline = "CCCV ½C" if "CCCV ½C" in curves else None
    if baseline is None and "CC ½C" in curves:
        baseline = "CC ½C"
    if baseline is None:
        for p in policies:
            if p["feasible"] and p["name"] in curves:
                baseline = p["name"]
                break
    if baseline is None:
        return
    base = curves[baseline]["remaining_pct"]
    cycles = curves[baseline]["cycles"]

    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.8, 5.4), dpi=PAPER_DPI, facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    ax.axhline(0.0, color="#64748b", lw=1.4)
    for p in policies:
        name = p["name"]
        if name == baseline or not p["feasible"] or name not in curves:
            continue
        d = curves[name]
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        ax.plot(
            cycles, d["remaining_pct"] - base,
            color=style["color"], ls=style["ls"], lw=style.get("lw", 2.2),
            label=name,
        )
    ax.set_xlabel("Equivalent charge cycles", fontsize=14)
    ax.set_ylabel(f"Δ remaining capacity vs {baseline} [%-points]", fontsize=14)
    ax.set_title(
        f"{info.get('cell', '?')}: capacity retention vs {baseline}",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight", facecolor=PAPER_LIGHT_BG)
    plt.close(fig)


def plot_lifetime_vs_ah(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    info: Dict[str, Any],
    measured: Optional[Dict[str, np.ndarray]],
    out_path: Path,
) -> None:
    apply_paper_style()
    fig, (ax, ax_zoom) = plt.subplots(
        1, 2, figsize=(12.2, 5.4), dpi=PAPER_DPI,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        facecolor=PAPER_LIGHT_BG,
    )
    for a in (ax, ax_zoom):
        a.set_facecolor(PAPER_LIGHT_BG)
    for p in policies:
        name = p["name"]
        if (not p["feasible"] and str(name).startswith("CC")) or name not in curves:
            continue
        d = curves[name]
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        lw = style.get("lw", 2.2)
        ax.plot(d["cum_ah"], d["remaining_pct"], color=style["color"], ls=style["ls"], lw=lw, label=name)
        ax_zoom.plot(d["cum_ah"], d["remaining_pct"], color=style["color"], ls=style["ls"], lw=lw, label=name)
    if measured is not None:
        ax.plot(measured["cum_ah"], measured["remaining_pct"], color="#0f172a", ls="--", lw=2.0, alpha=0.75, label="NASA measured")
    ax.set_xlabel("Cumulative ampere-hour throughput [Ah]", fontsize=14)
    ax.set_ylabel("Remaining capacity [%]", fontsize=14)
    ax.set_title(f"{info.get('cell', '')}: fade vs throughput", fontsize=13, fontweight="bold")
    ax.set_ylim(30, 105)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=11, loc="lower left", framealpha=0.95)
    feas = [p for p in policies if p["feasible"] and p["name"] in curves]
    xmax = max(float(curves[p["name"]]["cum_ah"][-1]) for p in feas) if feas else 1.0
    ax_zoom.set_xlabel("Cumulative Ah", fontsize=14)
    ax_zoom.set_ylabel("Projected remaining capacity [%]", fontsize=14)
    ax_zoom.set_title("Policy comparison", fontsize=13, fontweight="bold")
    ax_zoom.set_xlim(0, xmax * 1.02)
    ax_zoom.set_ylim(75, 100)
    ax_zoom.grid(True, linestyle="--", alpha=0.35)
    ax_zoom.tick_params(labelsize=12)
    ax_zoom.legend(fontsize=11, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight", facecolor=PAPER_LIGHT_BG)
    plt.close(fig)


def plot_lifetime_vs_ref_style(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    info: Dict[str, Any],
    measured: Optional[Dict[str, np.ndarray]],
    out_path: Path,
    *,
    n_show: int = 80,
) -> None:
    apply_paper_style()
    q0 = float(Q_RATED_AH)
    fig, ax = plt.subplots(figsize=(9.8, 5.6), dpi=PAPER_DPI, facecolor=PAPER_LIGHT_BG)
    ax.set_facecolor(PAPER_LIGHT_BG)
    ys = []
    for p in policies:
        name = p["name"]
        if not p["feasible"] or name not in curves:
            continue
        d = curves[name]
        n = min(n_show, len(d["cycles"]) - 1)
        x = d["cycles"][: n + 1]
        q_ah = q0 * d["remaining_pct"][: n + 1] / 100.0
        ys.append(float(q_ah[-1]))
        style = POLICY_STYLE.get(name, {"color": "#334155", "ls": "-", "lw": 2.2})
        ax.plot(x, q_ah, color=style["color"], ls=style["ls"], lw=style.get("lw", 2.2), label=name)
    ax.set_xlabel("Equivalent charge cycle index", fontsize=14)
    ax.set_ylabel("Projected capacity [Ah]", fontsize=14)
    ax.set_title(f"{info.get('cell', '?')}: capacity vs cycle", fontsize=14, fontweight="bold")
    if ys:
        ax.set_ylim(min(ys) - 0.05, q0 + 0.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PAPER_DPI, bbox_inches="tight", facecolor=PAPER_LIGHT_BG)
    plt.close(fig)


def save_projection_csv(
    policies: Sequence[Dict[str, Any]],
    curves: Dict[str, Dict[str, np.ndarray]],
    path: Path,
    *,
    stride: int = 10,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "policy", "feasible", "cycle", "cum_ah",
            "qloss_total", "remaining_pct",
            "session_ah", "session_duration_h", "session_mean_T_C", "session_c_rate",
        ])
        meta = {p["name"]: p for p in policies}
        for name, d in curves.items():
            p = meta[name]
            for i in range(0, len(d["cycles"]), stride):
                w.writerow([
                    name, p["feasible"], int(d["cycles"][i]),
                    f"{d['cum_ah'][i]:.4f}", f"{d['qloss_total'][i]:.8f}",
                    f"{d['remaining_pct'][i]:.4f}", f"{p['ah_throughput']:.6f}",
                    f"{p['duration_h']:.6f}", f"{p['mean_temperature_c']:.4f}",
                    f"{p['nominal_c_rate']:.6f}",
                ])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bo", type=Path, default=ROOT / "Constrained_BO/results/ui_runs_1/RW9/gp_bo_results.json")
    ap.add_argument("--random", type=Path, default=ROOT / "Constrained_BO/results/ui_runs_1/RW9/random_search_results.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "Constrained_BO/results/grounded_figures")
    ap.add_argument("--n-cycles", type=int, default=600)
    ap.add_argument("--anchor-cycle", type=int, default=400)
    ap.add_argument("--soh-anchor", type=float, default=80.0)
    ap.add_argument("--include-infeasible-cc", action="store_true")
    ap.add_argument("--gpbo-select", choices=("min_q", "reward"), default="min_q",
                    help="Which GP-BO trial to plot (min_q for fade figures)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    policies, info = _collect_policies(
        args.bo, args.random,
        device=args.device,
        include_infeasible_cc=args.include_infeasible_cc,
        gpbo_select=args.gpbo_select,
    )
    print("Policies:")
    for p in policies:
        print(
            f"  {p['name']:16s} feas={p['feasible']}  "
            f"Ah/sess={p['ah_throughput']:.3f}  T={p['mean_temperature_c']:.2f}°C  "
            f"C={p['nominal_c_rate']:.3f}  t={p['duration_min']:.1f} min  "
            f"Q_sess={p['qloss_total']:.5f}"
        )

    _, curves, scale = project_fade(
        policies,
        n_cycles=args.n_cycles,
        soh_anchor_pct=args.soh_anchor,
        anchor_policy="CCCV ½C",
        anchor_cycle=args.anchor_cycle,
    )
    measured = _load_measured_cell(str(info.get("cell", "RW9")))
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    plot_lifetime_vs_cycles(
        policies, curves, info, scale=scale,
        anchor_cycle=args.anchor_cycle, soh_anchor_pct=args.soh_anchor,
        out_path=out / "fig9_lifetime_fade_vs_cycles.png",
    )
    plot_delta_vs_halfc(policies, curves, info, out / "fig9d_lifetime_delta_vs_halfC.png")
    plot_lifetime_vs_ah(policies, curves, info, measured, out / "fig9b_lifetime_fade_vs_throughput.png")
    plot_lifetime_vs_ref_style(policies, curves, info, measured, out / "fig9c_lifetime_capacity_vs_cycle_index.png")
    save_projection_csv(policies, curves, out / "lifetime_fade_projection.csv")
    meta = {
        **info,
        "n_cycles": args.n_cycles,
        "anchor_cycle": args.anchor_cycle,
        "soh_anchor_pct": args.soh_anchor,
        "scale": scale,
        "note": "GP-BO curve uses min-Q feasible trial when gpbo_select=min_q.",
    }
    (out / "lifetime_fade_projection_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"Wrote → {out / 'fig9_lifetime_fade_vs_cycles.png'}")
    print(f"Wrote → {out / 'fig9c_lifetime_capacity_vs_cycle_index.png'}")


if __name__ == "__main__":
    main()
