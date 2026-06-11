#!/usr/bin/env python3
"""
Diagnose the lifetime BO objective — compare constant CC-taper baselines.

Prints loss components (SEI/ΔSoC, feasibility, shortfall) for fixed currents
so you can verify scaling before trusting Bayesian optimization.

Usage
-----
    venv/bin/python scripts/diagnose_bo_objective.py
    venv/bin/python scripts/diagnose_bo_objective.py --currents 1.2 2.0 3.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rw_transfer.config import load_config
from rw_transfer.data.series import load_battery_series
from charging_opt.artifacts import CANONICAL
from charging_opt.lifetime_reward import aggregate_lifetime_reward
from charging_opt.profile_simulator import ProfileSimulator, ProfileSpec
from charging_opt.soc_utils import load_ocv_curve
from charging_opt.state_utils import extract_rest_states, pick_start_state


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose lifetime BO objective scaling.")
    p.add_argument("--config", default=None)
    p.add_argument("--cell", default="RW9")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--currents", type=float, nargs="+", default=[0.75, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5])
    p.add_argument("--max_minutes", type=int, default=90)
    p.add_argument("--soc_target", type=float, default=0.95)
    args = p.parse_args()

    cfg = load_config(args.config)
    series = load_battery_series(cfg["data"]["matlab_dir"], args.cell, step_mode="all")
    ocv = load_ocv_curve(ROOT / CANONICAL["ocv_curve"])
    start = pick_start_state(extract_rest_states(series, ocv, max_states=3000))

    sim = ProfileSimulator(
        bdt_path=ROOT / args.bdt_ckpt,
        capacity_path=ROOT / CANONICAL["capacity_fade"],
        margins_path=ROOT / CANONICAL["conformal_margins"],
        max_minutes=args.max_minutes,
        soc_target=args.soc_target,
    )

    print(f"Start: SoC={start['soc']:.2f}  V={start['v0']:.3f}  T={start['t0']:.1f}°C")
    print(f"SoC target={args.soc_target:.0%}  time budget={args.max_minutes} min\n")
    print(f"{'I_cc':>6} {'feas':>5} {'loss':>8} {'SEI/%SoC':>9} {'dSoC%':>7} "
          f"{'shortfall':>10} {'dur_min':>8} {'end':>22}")
    print("-" * 88)

    rows = []
    for i_cc in args.currents:
        spec = ProfileSpec.cc_taper(i_cc, i_floor=0.75)
        session = sim.simulate(start, spec)
        _, m = aggregate_lifetime_reward(session, soc_target=args.soc_target)
        c = m.get("components") or {}
        rows.append((i_cc, m))
        print(
            f"{i_cc:6.2f} {str(m.get('feasible', False)):>5} {m.get('loss', float('nan')):8.2f} "
            f"{m.get('sei_per_pct_soc') or float('nan'):9.2f} "
            f"{m.get('delta_soc_pct_total', float('nan')):7.1f} "
            f"{c.get('soc_shortfall_pct', float('nan')):10.2f} "
            f"{m.get('duration_min', float('nan')):8.1f} "
            f"{m.get('end_reason', ''):>22}"
        )

    feas = [r for r in rows if r[1].get("feasible")]
    if feas:
        best = min(feas, key=lambda r: r[1]["loss"])
        print(f"\nBest FEASIBLE constant CC: {best[0]:.2f} A  "
              f"(loss={best[1]['loss']:.2f}, SEI/%SoC={best[1]['sei_per_pct_soc']:.2f})")
    else:
        print("\nNo constant CC profile reached SoC target within time budget.")
        print("Increase --max_minutes or check BDT / capacity calibration.")


if __name__ == "__main__":
    main()
