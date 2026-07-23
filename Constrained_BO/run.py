#!/usr/bin/env python3
"""Run constrained charging-profile optimization (GP-BO default; random-search baseline)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".mplconfig"))

from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np

from Constrained_BO.bayesian_optimizer import optimize_all_families_gp_bo
from Constrained_BO.config import SOC_START, finetune_frac_for, get_cell_config, ENERGY_FRACTION_BY_CELL
from Constrained_BO.objective import energy_required_j, evaluate_session, full_capacity_joules
from Constrained_BO.ocv import ocv_curve_path
from Constrained_BO.profiles import DEFAULT_FAMILIES, ProfileParams, get_family, set_profile_bounds
from Constrained_BO.simulator import ChargingSimulator
from Constrained_BO.viz import plot_best_profiles


def _optimize_family_random(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family_id: str,
    *,
    n_random: int = 80,
    seed: int = 42,
    reward_mode: str = "hybrid_qloss",
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    z: float = 0.55,
) -> Dict[str, Any]:
    family = get_family(family_id)
    rng = np.random.default_rng(seed)

    candidates: List[ProfileParams] = list(family.seed_params())
    while len(candidates) < n_random:
        candidates.append(family.sample_random(rng))

    history = []
    best_loss = float("inf")
    best_params: Optional[ProfileParams] = None
    best_metrics: Optional[Dict] = None
    best_session: Optional[Dict] = None
    best_feasible = False

    for params in candidates:
        session = simulator.simulate(initial_state, params, family=family)
        loss, metrics = evaluate_session(
            session,
            reward_mode=reward_mode,  # type: ignore[arg-type]
            w_time=w_time,
            w_temperature=w_temperature,
            w_soc=w_soc,
            w_qloss=w_qloss,
            z=z,
        )
        feasible = bool(metrics["feasible"])
        history.append({
            "family_id": family_id,
            "params": params.to_dict(),
            "loss": loss,
            "feasible": feasible,
            "metrics": metrics,
            "end_reason": metrics["end_reason"],
        })

        # Same rule as GP-BO: prefer feasible, then minimum hybrid loss.
        take = False
        if best_params is None:
            take = True
        elif feasible and not best_feasible:
            take = True
        elif feasible == best_feasible and loss < best_loss:
            take = True
        if take:
            best_loss = loss
            best_params = params
            best_metrics = metrics
            best_session = session
            best_feasible = feasible

    return {
        "family_id": family_id,
        "family_label": family.label,
        "best_params": best_params.to_dict() if best_params else None,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "best_session": best_session,
        "history": history,
        "n_evaluated": len(history),
        "method": "random_search",
    }


def _optimize_all_families_random(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    *,
    families: Optional[List[str]] = None,
    n_random: int = 80,
    seed: int = 42,
    reward_mode: str = "hybrid_qloss",
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    z: float = 0.55,
) -> Dict[str, Any]:
    families = families or DEFAULT_FAMILIES
    results = {}
    for i, fid in enumerate(families):
        results[fid] = _optimize_family_random(
            simulator,
            initial_state,
            fid,
            n_random=n_random,
            seed=seed + i * 1000,
            reward_mode=reward_mode,
            w_time=w_time,
            w_temperature=w_temperature,
            w_soc=w_soc,
            w_qloss=w_qloss,
            z=z,
        )
    return results


def _is_writable_out_dir(d: Path) -> bool:
    """Directory must exist (or be creatable) and output files must be writable."""
    import os

    if d.exists():
        if not os.access(d, os.W_OK):
            return False
        for name in ("constrained_bo_results.json", "best_profiles.png"):
            f = d / name
            if f.exists() and not os.access(f, os.W_OK):
                return False
        return True

    parent = d.parent
    return parent.exists() and os.access(parent, os.W_OK)


def _resolve_out_dir(cell_id: str, out_dir: Path | None) -> Path:
    """Pick a writable output directory with fallbacks."""
    import getpass

    if out_dir is not None:
        return Path(out_dir)

    base = Path(__file__).resolve().parent / "results"
    user = getpass.getuser()
    candidates = [
        base / cell_id,
        base / user / cell_id,
        base / "_run" / user / cell_id,
    ]

    for i, candidate in enumerate(candidates):
        if _is_writable_out_dir(candidate):
            if i > 0:
                print(
                    f"Warning: {candidates[0]} is not writable; "
                    f"writing to {candidate} instead.\n"
                    f"  To reuse {candidates[0]}, run: "
                    f"sudo chown -R $USER {candidates[0]}"
                )
            return candidate

    # Last resort: create _run path (new files, no root-owned conflicts).
    last = candidates[-1]
    print(
        f"Warning: no writable results directory found; creating {last}\n"
        f"  Fix ownership with: sudo chown -R $USER {base / user}"
    )
    return last


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not os.access(path, os.W_OK):
        raise PermissionError(
            f"Cannot write {path} (permission denied). "
            f"Re-run without --out-dir to auto-fallback, or run: "
            f"sudo chown -R $USER {path.parent}"
        )
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _strip_sessions(results: Dict[str, Dict]) -> Dict[str, Dict]:
    """Remove heavy trajectory arrays from JSON export."""
    out = {}
    for fid, res in results.items():
        entry = {k: v for k, v in res.items() if k != "best_session"}
        out[fid] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Constrained BO — GP Bayesian optimization (hybrid Q_loss default)",
    )
    parser.add_argument("--cell", default="RW9", help="Cell ID (RW9–RW12, LFP)")
    parser.add_argument("--cells", nargs="+", default=None, help="Run multiple cells")
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument(
        "--method",
        choices=("gp_bo", "random_search"),
        default="gp_bo",
        help="gp_bo = Gaussian-process BO (default); random_search = baseline",
    )
    parser.add_argument(
        "--n-calls",
        type=int,
        default=40,
        help="GP-BO evaluations per family (including seeds)",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=10,
        help="GP-BO warm-start budget: family seeds + extra random points",
    )
    parser.add_argument(
        "--acq-func",
        choices=("PI", "EI", "LCB"),
        default="PI",
        help="Acquisition function for gp_minimize (default PI)",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=80,
        help="Random samples per family (random_search only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refit-ocv", action="store_true", help="Re-fit OCV curve before run")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: Constrained_BO/results/<cell>)",
    )
    parser.add_argument(
        "--reward-mode",
        choices=("hybrid_qloss", "legacy_temp_time"),
        default="hybrid_qloss",
        help="hybrid_qloss = w1*dSoC - w2*Qloss - w3*t^z; "
             "legacy_temp_time = previous temp+time reward",
    )
    parser.add_argument(
        "--w-soc", type=float, default=1.0,
        help="Weight on ΔSoC (hybrid mode)",
    )
    parser.add_argument(
        "--w-qloss", type=float, default=1.0,
        help="Weight on Q_calendar + Q_cyclic (hybrid mode)",
    )
    parser.add_argument(
        "--w-time", type=float, default=0.1,
        help="Weight on t^z (hybrid) or legacy time reward (legacy mode)",
    )
    parser.add_argument(
        "--w-temperature", type=float, default=1.0,
        help="Weight on temperature reward (legacy_temp_time only)",
    )
    parser.add_argument(
        "--z", type=float, default=0.55,
        help="Power-law exponent for time penalty / calendar z (hybrid)",
    )
    parser.add_argument(
        "--soc-target",
        type=float,
        default=None,
        help="Absolute SoC target (classic mode, default 0.95)",
    )
    parser.add_argument(
        "--soc-delta",
        type=float,
        default=None,
        help="SoC increase from start; enables energy mode (alias for --energy-fraction)",
    )
    parser.add_argument(
        "--energy-fraction",
        type=float,
        default=None,
        help="Deliver this fraction of full pack energy in J (e.g. 0.40 = 40%% of Q×V_nom)",
    )
    parser.add_argument(
        "--max-duration-min",
        type=float,
        default=None,
        help="Simulation horizon in minutes (default 150)",
    )
    parser.add_argument(
        "--v-nom",
        type=float,
        default=None,
        help="Override nominal voltage for E_full = Q_rated × V_nom "
        "(default: OCV at 50% SoC from cell NASA data)",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=None,
        metavar="SEC",
        help="Fixed BDT re-anchor interval in seconds (default: auto from drift error)",
    )
    parser.add_argument(
        "--no-auto-decision-interval",
        action="store_true",
        help="Use 30 s re-anchor interval instead of auto-selecting from drift error",
    )
    args = parser.parse_args()

    cells = args.cells or [args.cell.upper()]
    for cell_id in cells:
        cell_id = cell_id.upper()
        energy_fraction = args.energy_fraction
        if energy_fraction is None and args.soc_delta is None:
            energy_fraction = ENERGY_FRACTION_BY_CELL.get(cell_id)
        cell = get_cell_config(cell_id, refit_ocv=args.refit_ocv)
        cell = cell.with_run_overrides(
            soc_target=args.soc_target,
            soc_delta=args.soc_delta,
            energy_fraction=energy_fraction,
            max_duration_min=args.max_duration_min,
            v_nom=args.v_nom,
            decision_interval_s=args.decision_interval,
            auto_decision_interval=(
                not args.no_auto_decision_interval and args.decision_interval is None
            ),
        )
        if cell.profile_bounds is not None:
            set_profile_bounds(cell.profile_bounds)
        out_dir = _resolve_out_dir(cell_id, args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {cell_id} ===")
        print(f"BDT: {cell.bdt_ckpt}")
        print(f"Start state: {cell.start_state}")
        print(f"V_nom: {cell.v_nom:.4f} V", end="")
        if cell_id == "LFP":
            print(" (OCV at 50% SoC from LFP 1C Reference discharge)")
        else:
            print(f" (OCV at 50% SoC from {ocv_curve_path(cell_id).name})")
        print(f"Constraint mode: {cell.constraint_mode}")
        if cell.constraint_mode == "energy":
            e_full = full_capacity_joules(cell.q_rated_as, cell.v_nom)
            e_req = energy_required_j(cell.q_rated_as, cell.energy_fraction, cell.v_nom)
            print(
                f"Energy target: {cell.energy_fraction:.0%} of {e_full:.0f} J "
                f"→ {e_req:.0f} J required  (SoC stop ≈ {cell.soc_target:.0%})"
            )
        else:
            print(f"SoC target: {cell.soc_target:.0%}")
        print(f"Max duration: {cell.max_duration_min} min")
        if cell.profile_bounds is not None:
            b = cell.profile_bounds
            print(
                f"Profile bounds: I=[{b.i_min_a}, {b.i_max_a}] A  "
                f"V_cv=[{b.v_cv_min_v}, {b.v_cv_max_v}] V  "
                f"SoC switch=[{b.soc_switch_min}, {b.soc_switch_max}]"
            )

        simulator = ChargingSimulator.from_cell(cell, device=args.device)
        dt_info = simulator.decision_interval_info
        print(
            f"Decision interval: {simulator.decision_interval_s} s "
            f"(method={dt_info.get('method', dt_info.get('source', '?'))})"
        )
        if dt_info.get("scores"):
            print(f"  calibration scores (V RMSE + 0.01·T RMSE): {dt_info['scores']}")
        # Legacy mode historically used w_time=1.0; keep that if user did not override.
        w_time = float(args.w_time)
        if args.reward_mode == "legacy_temp_time" and abs(w_time - 0.1) < 1e-15:
            w_time = 1.0

        print(f"Method: {args.method}")
        print(f"Reward mode: {args.reward_mode}")
        if args.reward_mode == "hybrid_qloss":
            print(
                f"  R = {args.w_soc}*dSoC - {args.w_qloss}*Q_loss - "
                f"{w_time}*t_h^{args.z}"
            )

        if args.method == "gp_bo":
            results = optimize_all_families_gp_bo(
                simulator,
                cell.start_state,
                families=args.families,
                n_calls=args.n_calls,
                n_initial_points=args.n_initial,
                seed=args.seed,
                reward_mode=args.reward_mode,  # type: ignore[arg-type]
                w_time=w_time,
                w_temperature=args.w_temperature,
                w_soc=args.w_soc,
                w_qloss=args.w_qloss,
                z=args.z,
                acq_func=args.acq_func,
            )
        else:
            results = _optimize_all_families_random(
                simulator,
                cell.start_state,
                families=args.families,
                n_random=args.n_random,
                seed=args.seed,
                reward_mode=args.reward_mode,
                w_time=w_time,
                w_temperature=args.w_temperature,
                w_soc=args.w_soc,
                w_qloss=args.w_qloss,
                z=args.z,
            )

        for fid, res in results.items():
            m = res["best_metrics"]
            status = "OK" if m["feasible"] else "INFEASIBLE"
            energy_note = ""
            if m.get("constraint_mode") == "energy":
                energy_note = (
                    f"  E={m['energy_delivered_j']:.0f}/"
                    f"{m['energy_required_j']:.0f}J"
                )
            qloss_note = ""
            if m.get("reward_mode") == "hybrid_qloss":
                qloss_note = (
                    f"  Q={m.get('qloss_total', 0):.4g} "
                    f"(cal={m.get('qloss_calendar', 0):.3g},"
                    f"cyc={m.get('qloss_cyclic', 0):.3g})"
                )
            print(
                f"  {res['family_label']:22s}  loss={m['loss']:7.2f}  "
                f"reward={m['total_reward']:.3f}  "
                f"time={m['duration_min']:.1f} min{energy_note}{qloss_note}  [{status}]"
            )

        meta: Dict[str, Any] = {
            "cell": cell_id,
            "bdt_ckpt": str(cell.bdt_ckpt),
            "finetune_fraction": finetune_frac_for(cell_id) if cell_id not in ("RW9",) else None,
            "soc_start": cell.start_state.get("soc", SOC_START),
            "ocv_curve": str(ocv_curve_path(cell_id)),
            "start_state": cell.start_state,
            "soc_target": cell.soc_target,
            "max_duration_min": cell.max_duration_min,
            "constraint_mode": cell.constraint_mode,
            "v_nom": cell.v_nom,
            "method": args.method,
            "reward_mode": args.reward_mode,
            "reward_weights": {
                "w_soc": args.w_soc,
                "w_qloss": args.w_qloss,
                "w_time": w_time,
                "w_temperature": args.w_temperature,
                "z": args.z,
            },
            "families": args.families,
            "decision_interval_s": simulator.decision_interval_s,
            "decision_interval_selection": simulator.decision_interval_info,
            "seed": args.seed,
        }
        if args.method == "gp_bo":
            meta["n_calls"] = args.n_calls
            meta["n_initial"] = args.n_initial
            meta["acq_func"] = args.acq_func
        else:
            meta["n_random"] = args.n_random
        if cell.profile_bounds is not None:
            meta["profile_bounds"] = cell.profile_bounds.to_dict()
        if cell.constraint_mode == "energy":
            meta["energy_fraction"] = cell.energy_fraction
            meta["energy_full_j"] = full_capacity_joules(cell.q_rated_as, cell.v_nom)
            meta["energy_required_j"] = energy_required_j(
                cell.q_rated_as, cell.energy_fraction, cell.v_nom,
            )

        payload: Dict[str, Any] = {
            "meta": meta,
            "families": _strip_sessions(results),
        }

        json_path = out_dir / "constrained_bo_results.json"
        _write_json(json_path, payload)
        print(f"Wrote {json_path}")

        if args.method == "gp_bo":
            title = (
                f"GP-BO ({args.acq_func}), n_calls={args.n_calls}, "
                f"max {cell.max_duration_min:.0f} min, {args.reward_mode}"
            )
        else:
            title = (
                f"random search, n={args.n_random}, "
                f"max {cell.max_duration_min:.0f} min, {args.reward_mode}"
            )
        if cell.constraint_mode == "energy":
            title += f", energy ≥ {cell.energy_fraction:.0%} of pack"
        fig = plot_best_profiles(
            results,
            cell_id=cell_id,
            soc_target=cell.soc_target,
            soc_start=cell.start_state["soc"],
            out_path=out_dir / "best_profiles.png",
            title_suffix=title,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"Wrote {out_dir / 'best_profiles.png'}")


if __name__ == "__main__":
    main()
