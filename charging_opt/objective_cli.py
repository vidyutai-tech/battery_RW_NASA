"""Shared CLI helpers for lifetime objective configuration (Priority 2)."""

from __future__ import annotations

import argparse

from charging_opt.lifetime_reward import LifetimeWeights, ObjectiveMode


def add_objective_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--objective",
        choices=("composite", "legacy"),
        default="composite",
        help="composite = Priority-2 multi-term loss; legacy = SEI/ΔSoC + small tie-breakers",
    )
    p.add_argument("--w_sei", type=float, default=1.0, help="weight on SEI/ΔSoC term")
    p.add_argument("--w_time", type=float, default=0.02, help="weight on duration (min)")
    p.add_argument(
        "--w_temp", type=float, default=0.05,
        help="weight on ∫max(0,T−35°C)² dt (°C²·min)",
    )
    p.add_argument(
        "--w_vstress", type=float, default=0.08,
        help="weight on ∫max(0,V−4.0)² dt (V²·min)",
    )
    p.add_argument("--v_ref_stress", type=float, default=4.0, help="V reference for stress integral")
    p.add_argument("--t_comfort_c", type=float, default=35.0, help="T comfort for penalty integral")


def objective_from_args(args) -> tuple[LifetimeWeights, ObjectiveMode, dict]:
    if args.objective == "legacy":
        weights = LifetimeWeights.legacy()
    else:
        weights = LifetimeWeights(
            sei=float(args.w_sei),
            time=float(args.w_time),
            temperature=float(args.w_temp),
            voltage_stress=float(args.w_vstress),
        )
    refs = {
        "v_ref_stress": float(args.v_ref_stress),
        "t_comfort_c": float(args.t_comfort_c),
    }
    return weights, args.objective, refs
