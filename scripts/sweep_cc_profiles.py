#!/usr/bin/env python3
"""
Constant-current sweep — no optimization.

Simulates CC-taper profiles at fixed currents and plots:
    * SEI proxy & SEI/ΔSoC vs current
    * Charge duration vs current
    * Optional: terminal voltage vs SoC (BDT check)

Usage
-----
    venv/bin/python scripts/sweep_cc_profiles.py
    venv/bin/python scripts/sweep_cc_profiles.py --max_minutes 150 --plot
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
from charging_opt.artifacts import CANONICAL, update_master_registry
from charging_opt import paths as P
from charging_opt.lifetime_reward import aggregate_lifetime_reward, session_metrics
from charging_opt.profile_simulator import ProfileSimulator, ProfileSpec
from charging_opt.soc_utils import load_ocv_curve
from charging_opt.state_utils import extract_rest_states, pick_start_state


def _profile_label(session: dict) -> str:
    spec = session.get("profile_spec") or {}
    i_cc = spec.get("i_charge", float("nan"))
    i_fl = spec.get("i_floor", float("nan"))
    tapered = any(d.get("ceiling_hit") for d in session.get("decisions", []))
    if abs(i_cc - i_fl) < 1e-6 and not tapered:
        return "constant CC (no taper)"
    if tapered:
        return "CC-taper"
    return "CC"


def run_sweep(args) -> list:
    cfg = load_config(args.config)
    series = load_battery_series(cfg["data"]["matlab_dir"], args.cell, step_mode="all")
    ocv = load_ocv_curve(ROOT / args.ocv_curve)
    start = pick_start_state(extract_rest_states(series, ocv, max_states=3000))

    sim = ProfileSimulator(
        bdt_path=ROOT / args.bdt_ckpt,
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=args.max_minutes,
        soc_target=args.soc_target,
    )

    rows = []
    for i_cc in args.currents:
        spec = ProfileSpec.cc_taper(i_cc, i_floor=0.75)
        session = sim.simulate(start, spec)
        session["profile_spec"] = spec.to_dict()
        _, m = aggregate_lifetime_reward(
            session, soc_target=args.soc_target, max_duration_min=args.max_duration_min,
        )
        rows.append({
            "i_cc_a": i_cc,
            "duration_min": m["duration_min"],
            "delta_soc_pct": m["delta_soc_pct_total"],
            "soc_end": m.get("soc_end"),
            "sei_proxy": m["sei_proxy"],
            "sei_per_pct_soc": m.get("sei_per_pct_soc"),
            "peak_voltage_v": m["peak_voltage"],
            "feasible": m.get("feasible", False),
            "end_reason": m["end_reason"],
            "profile_type": _profile_label(session),
            "loss": m.get("loss"),
            "infeasible_reason": (m.get("components") or {}).get("reason"),
        })
    return start, rows


def print_table(rows: list) -> None:
    print(f"\n{'I_cc':>6} {'time_min':>9} {'dSoC%':>7} {'SEI':>8} {'SEI/%SoC':>9} "
          f"{'V_peak':>7} {'feas':>5}  end / type")
    print("-" * 95)
    for r in rows:
        sei_ps = r["sei_per_pct_soc"]
        sei_ps_s = f"{sei_ps:9.2f}" if sei_ps is not None else f"{'—':>9}"
        print(
            f"{r['i_cc_a']:6.2f} {r['duration_min']:9.1f} {r['delta_soc_pct']:7.1f} "
            f"{r['sei_proxy']:8.0f} {sei_ps_s} {r['peak_voltage_v']:7.3f} "
            f"{str(r['feasible']):>5}  {r['end_reason']} / {r['profile_type']}"
        )


def plot_sweep(rows: list, out_dir: Path, start: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    i = np.array([r["i_cc_a"] for r in rows])
    dur = np.array([r["duration_min"] for r in rows])
    sei = np.array([r["sei_proxy"] for r in rows])
    sei_ps = np.array([
        r["sei_per_pct_soc"] if r["sei_per_pct_soc"] is not None else np.nan
        for r in rows
    ])
    feas = np.array([r["feasible"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.plot(i, sei, "o-", lw=2, color="tab:red", label="SEI proxy (total)")
    ax2 = ax.twinx()
    ax2.plot(i, sei_ps, "s--", lw=1.8, color="tab:blue", label="SEI / ΔSoC")
    ax.set_xlabel("Constant charge current (A)")
    ax.set_ylabel("SEI proxy (total)", color="tab:red")
    ax2.set_ylabel("SEI / ΔSoC (lower = better lifetime)", color="tab:blue")
    ax.set_title("Aging metric vs current")
    ax.grid(alpha=0.3)
    for j, ok in enumerate(feas):
        if not ok:
            ax.axvline(i[j], color="gray", ls=":", alpha=0.25)

    ax = axes[1]
    colors = ["tab:green" if f else "tab:gray" for f in feas]
    ax.bar([f"{x:.2f}" for x in i], dur, color=colors, alpha=0.85)
    ax.set_xlabel("Constant charge current (A)")
    ax.set_ylabel("Duration to stop (min)")
    ax.set_title("Charge time vs current (green = reached SoC target)")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"CC sweep — start SoC={start['soc']:.0%}, V={start['v0']:.2f} V, "
        f"T={start['t0']:.1f} °C",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cc_sweep_sei_and_time.png", dpi=150)
    plt.close(fig)

    # Normalized sensitivity: how much SEI/%SoC improves vs time cost
    feas_rows = [r for r in rows if r["feasible"] and r["sei_per_pct_soc"] is not None]
    if len(feas_rows) >= 2:
        base = min(feas_rows, key=lambda r: r["i_cc_a"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for r in feas_rows:
            d_sei = r["sei_per_pct_soc"] - base["sei_per_pct_soc"]
            d_time = r["duration_min"] - base["duration_min"]
            ax.scatter(d_time, d_sei, s=80, label=f"{r['i_cc_a']:.2f} A")
            ax.annotate(f"{r['i_cc_a']:.1f}A", (d_time, d_sei), xytext=(4, 4),
                        textcoords="offset points", fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"Extra time vs {base['i_cc_a']:.2f} A baseline (min)")
        ax.set_ylabel(f"Δ(SEI/ΔSoC) vs baseline (pct points)")
        ax.set_title("Lifetime gain vs time cost (feasible profiles only)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "cc_sweep_tradeoff.png", dpi=150)
        plt.close(fig)

    print(f"  Plots -> {out_dir}/")


def main() -> None:
    p = argparse.ArgumentParser(description="Constant-current profile sweep.")
    p.add_argument("--config", default=None)
    p.add_argument("--cell", default="RW9")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--ocv_curve", default=CANONICAL["ocv_curve"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument("--out_dir", default=P.STAGE2_PLOTS,
                   help="directory for sweep plots and cc_sweep.json")
    p.add_argument("--currents", type=float, nargs="+",
                   default=[0.75, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0])
    p.add_argument("--max_minutes", type=int, default=150)
    p.add_argument("--max_duration_min", type=float, default=None,
                   help="if set, profiles slower than this are infeasible")
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--plot", action="store_true", default=True)
    args = p.parse_args()

    start, rows = run_sweep(args)
    print(f"Start: SoC={start['soc']:.2f}  V={start['v0']:.3f}  T={start['t0']:.1f}°C")
    print(f"SoC target={args.soc_target:.0%}  sim_horizon={args.max_minutes} min")
    if args.max_duration_min is not None:
        print(f"Time constraint: duration <= {args.max_duration_min:.0f} min")
    print_table(rows)

    out_dir = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"start_state": start, "rows": rows, "soc_target": args.soc_target,
               "max_minutes": args.max_minutes}
    with (out_dir / "cc_sweep.json").open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    if args.plot:
        plot_sweep(rows, out_dir, start)

    feas = [r for r in rows if r["feasible"]]
    if feas:
        span = max(r["sei_per_pct_soc"] for r in feas) - min(r["sei_per_pct_soc"] for r in feas)
        print(f"\nFeasible SEI/ΔSoC span: {span:.2f} pct-points "
              f"({100*span/min(r['sei_per_pct_soc'] for r in feas):.1f}% relative)")
        if span < 3.0:
            print("  → Metric is WEAKLY sensitive to current among feasible profiles.")
            print("    Optimizer will prefer minimum current unless a time constraint is added.")

    update_master_registry(root=ROOT)


if __name__ == "__main__":
    main()
