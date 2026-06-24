#!/usr/bin/env python3
"""Run constrained charging-profile optimization (random-search baseline)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import numpy as np

from Constrained_BO.config import SOC_START, get_cell_config
from Constrained_BO.objective import energy_required_j, evaluate_session, full_capacity_joules
from Constrained_BO.ocv import ocv_curve_path
from Constrained_BO.profiles import DEFAULT_FAMILIES, ProfileParams, get_family, set_profile_bounds
from Constrained_BO.simulator import ChargingSimulator
from Constrained_BO.viz import plot_best_profiles


def _optimize_family(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family_id: str,
    *,
    n_random: int = 80,
    seed: int = 42,
    w_time: float = 1.0,
    w_temperature: float = 1.0,
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

    for params in candidates:
        session = simulator.simulate(initial_state, params, family=family)
        loss, metrics = evaluate_session(
            session, w_time=w_time, w_temperature=w_temperature,
        )
        history.append({
            "family_id": family_id,
            "params": params.to_dict(),
            "loss": loss,
            "feasible": metrics["feasible"],
            "metrics": metrics,
            "end_reason": metrics["end_reason"],
        })

        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_metrics = metrics
            best_session = session

    return {
        "family_id": family_id,
        "family_label": family.label,
        "best_params": best_params.to_dict() if best_params else None,
        "best_loss": best_loss,
        "best_metrics": best_metrics,
        "best_session": best_session,
        "history": history,
        "n_evaluated": len(history),
    }


def _optimize_all_families(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    *,
    families: Optional[List[str]] = None,
    n_random: int = 80,
    seed: int = 42,
    w_time: float = 1.0,
    w_temperature: float = 1.0,
) -> Dict[str, Any]:
    families = families or DEFAULT_FAMILIES
    results = {}
    for i, fid in enumerate(families):
        results[fid] = _optimize_family(
            simulator,
            initial_state,
            fid,
            n_random=n_random,
            seed=seed + i * 1000,
            w_time=w_time,
            w_temperature=w_temperature,
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
    """Pick a writable output directory (fallback to results/<user>/<cell>)."""
    import getpass

    if out_dir is not None:
        return Path(out_dir)

    base = Path(__file__).resolve().parent / "results"
    primary = base / cell_id
    if _is_writable_out_dir(primary):
        return primary

    user = getpass.getuser()
    fallback = base / user / cell_id
    print(
        f"Warning: {primary} has root-owned or read-only outputs; "
        f"writing to {fallback} instead.\n"
        f"  To reuse {primary}, run: sudo chown -R $USER {primary}"
    )
    return fallback


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
    parser = argparse.ArgumentParser(description="Constrained BO — random-search baseline")
    parser.add_argument("--cell", default="RW9", help="Cell ID (RW9, RW10, RW11, RW12)")
    parser.add_argument("--cells", nargs="+", default=None, help="Run multiple cells")
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--n-random", type=int, default=80, help="Random samples per family")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refit-ocv", action="store_true", help="Re-fit OCV curve before run")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: Constrained_BO/results/<cell>)",
    )
    parser.add_argument("--w-time", type=float, default=1.0)
    parser.add_argument("--w-temperature", type=float, default=1.0)
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
        cell = get_cell_config(cell_id, refit_ocv=args.refit_ocv)
        cell = cell.with_run_overrides(
            soc_target=args.soc_target,
            soc_delta=args.soc_delta,
            energy_fraction=args.energy_fraction,
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
        print(f"V_nom: {cell.v_nom:.4f} V (OCV at 50% SoC from {ocv_curve_path(cell_id).name})")
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
        results = _optimize_all_families(
            simulator,
            cell.start_state,
            families=args.families,
            n_random=args.n_random,
            seed=args.seed,
            w_time=args.w_time,
            w_temperature=args.w_temperature,
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
            print(
                f"  {res['family_label']:22s}  loss={m['loss']:7.2f}  "
                f"reward={m['total_reward']:.3f}  "
                f"time={m['duration_min']:.1f} min{energy_note}  [{status}]"
            )

        meta: Dict[str, Any] = {
            "cell": cell_id,
            "bdt_ckpt": str(cell.bdt_ckpt),
            "finetune_fraction": "0.20" if cell_id != "RW9" else None,
            "soc_start": cell.start_state.get("soc", SOC_START),
            "ocv_curve": str(ocv_curve_path(cell_id)),
            "start_state": cell.start_state,
            "soc_target": cell.soc_target,
            "max_duration_min": cell.max_duration_min,
            "constraint_mode": cell.constraint_mode,
            "v_nom": cell.v_nom,
            "n_random": args.n_random,
            "method": "random_search",
            "reward_weights": {
                "w_time": args.w_time,
                "w_temperature": args.w_temperature,
            },
            "families": args.families,
            "decision_interval_s": simulator.decision_interval_s,
            "decision_interval_selection": simulator.decision_interval_info,
        }
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

        title = f"random search, n={args.n_random}, max {cell.max_duration_min:.0f} min"
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
