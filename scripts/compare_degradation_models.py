#!/usr/bin/env python3
"""
Compare SEI proxy vs Wang physics model across BO results; generate Fig 6.

Reads family optima + Chebyshev sweep, re-scores each profile through the BDT,
and writes:
  - outputs/visualization/degradation_model_comparison.json
  - outputs/visualization/fig6_physics_degradation.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig_{os.getenv('USER', 'default')}")

from charging_opt.artifacts import CANONICAL, resolve_bdt_ckpt
from charging_opt.charging_profile_family import FAMILY_LABELS, ProfileParams, get_family
from charging_opt.lifetime_reward import aggregate_lifetime_reward
from charging_opt.physics_degradation import get_degradation_model, physics_aware_loss
from charging_opt.profile_simulator import ProfileSimulator

DEFAULT_ENHANCED = ROOT / "outputs/charging_opt_user/hima/stage3_enhanced_20260614"
DEFAULT_CHEBYSHEV = ROOT / "outputs/charging_opt_user/hima/chebyshev_sweep/chebyshev_sweep_results.json"


def _load_family_best(results_json: Path) -> List[Dict[str, Any]]:
    payload = json.loads(results_json.read_text())
    rows = []
    for fid, block in payload["families"].items():
        rows.append({
            "label": block.get("family_label", FAMILY_LABELS.get(fid, fid)),
            "family_id": fid,
            "params": block["best_params"],
            "reported_loss": block.get("best_loss"),
            "reported_sei": block.get("best_metrics", {}).get("sei_per_pct_soc"),
            "reported_dur": block.get("best_metrics", {}).get("duration_min"),
            "feasible": block.get("best_metrics", {}).get("feasible", True),
        })
    return rows


def _load_chebyshev_pulsed(chebyshev_json: Path) -> List[Dict[str, Any]]:
    payload = json.loads(chebyshev_json.read_text())
    rows = []
    for omega_str, entries in sorted(payload["results_by_omega"].items(), key=lambda x: float(x[0])):
        pulsed = next(r for r in entries if r["family_id"] == "pulsed")
        rows.append({
            "label": f"Pulsed ω={float(omega_str):.1f}",
            "family_id": "pulsed",
            "params": pulsed["params"],
            "feasible": pulsed.get("feasible", True),
        })
    return rows


def _score_profile(
    sim: ProfileSimulator,
    start: Dict[str, float],
    row: Dict[str, Any],
    *,
    soc_target: float,
    max_duration_min: float,
) -> Dict[str, Any]:
    params = ProfileParams.from_dict(row["params"])
    family = get_family(params.family_id)
    session = sim.simulate_params(start, params, family=family)
    _, metrics = aggregate_lifetime_reward(
        session,
        soc_target=soc_target,
        max_duration_min=max_duration_min,
    )
    phys = metrics.get("physics_degradation", {})
    physics_loss = None
    if metrics.get("feasible") and phys:
        physics_loss, _ = physics_aware_loss(
            phys,
            duration_min=float(metrics.get("duration_min", 0)),
            voltage_stress_v2_min=float(metrics.get("voltage_stress_v2_min", 0)),
            temperature_penalty_c2_min=float(metrics.get("temperature_penalty_c2_min", 0)),
        )
    return {
        "label": row["label"],
        "family_id": row["family_id"],
        "feasible": bool(metrics.get("feasible")),
        "loss": float(metrics.get("loss", np.nan)),
        "sei_per_pct_soc": metrics.get("sei_per_pct_soc"),
        "duration_min": metrics.get("duration_min"),
        "capacity_fade_pct": metrics.get("capacity_fade_pct"),
        "equiv_cycles_to_eol": metrics.get("equiv_cycles_to_eol"),
        "physics_loss": physics_loss,
        "model_source": metrics.get("physics_model_source"),
    }


def _spearman(x: List[float], y: List[float]) -> Dict[str, float]:
    if len(x) < 3:
        return {"rho": float("nan"), "p": float("nan")}
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p)}


def plot_fig6(profiles: List[Dict[str, Any]], out_png: Path, model_info: Dict) -> None:
    feas = [p for p in profiles if p.get("feasible") and p.get("sei_per_pct_soc") is not None
            and p.get("capacity_fade_pct") is not None and p.get("equiv_cycles_to_eol") is not None]
    if len(feas) < 3:
        print("WARNING: fewer than 3 feasible profiles — skipping Fig 6.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"wspace": 0.38})
    b1 = model_info.get("b1_eff_at_25c")
    gamma = model_info.get("gamma", 0.55)
    formula = f"$\\Delta Q = B_1(C) \\cdot e^{{-E_a/RT}} \\cdot Ah^{{{gamma:.2f}}}$"
    if b1 is not None:
        formula += f"  [$B_{{1,eff}}$={b1:.2e} @ 25°C, RW9 calibrated]"

    # Panel 1: grouped bars for key profiles
    key_labels = [
        "CCCV", "Reduced-CV CCCV", "Pulsed charge/rest", "CC-taper (2-level)",
        "Adaptive 2-step (SoC)", "Pulsed ω=0.5", "Pulsed ω=1.0",
    ]
    key = [p for p in feas if p["label"] in key_labels or any(k in p["label"] for k in ("ω=0.5", "ω=1.0", "CCCV"))]
    if len(key) < 4:
        key = feas[: min(8, len(feas))]
    x = np.arange(len(key))
    sei_vals = [p["sei_per_pct_soc"] for p in key]
    fade_vals = [p["capacity_fade_pct"] * 10 for p in key]
    cycles = [p["equiv_cycles_to_eol"] for p in key]

    ax = axes[0]
    w = 0.35
    ax.bar(x - w / 2, sei_vals, w, label="SEI/ΔSoC (proxy)", color="#2166ac")
    ax.bar(x + w / 2, fade_vals, w, label="Wang fade ×10 (%)", color="#d6604d", hatch="//")
    ax2 = ax.twinx()
    ax2.plot(x, cycles, "o--", color="#1b7837", label="Sessions to 20% fade")
    ax.set_xticks(x)
    ax.set_xticklabels([p["label"].replace(" ", "\n") for p in key], fontsize=7)
    ax.set_ylabel("SEI/ΔSoC or scaled fade")
    ax2.set_ylabel("Est. sessions to 20% loss")
    ax.set_title("Degradation proxy vs.\nphysics-calibrated Wang model", fontsize=10)
    ax.legend(fontsize=7, loc="upper left")
    ax2.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: duration vs cycles
    ax = axes[1]
    durs = [p["duration_min"] for p in feas]
    cyc = [p["equiv_cycles_to_eol"] for p in feas]
    seis = [p["sei_per_pct_soc"] for p in feas]
    sc = ax.scatter(durs, cyc, c=seis, cmap="RdYlGn_r", s=60, edgecolors="white", linewidths=0.8)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("SEI/ΔSoC (proxy)")
    for p in feas:
        if p["label"] in ("CCCV", "Pulsed ω=1.0", "Pulsed ω=0.5", "Multi-step taper (voltage)"):
            ax.annotate(p["label"].split("(")[0].strip(), (p["duration_min"], p["equiv_cycles_to_eol"]),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("Charge duration (min)")
    ax.set_ylabel("Est. sessions to 20% capacity loss")
    ax.set_title("Speed vs. physics-based cycle life\n" + formula, fontsize=9)
    ax.grid(True, alpha=0.35)
    ax.text(0.03, 0.03, "← Faster charge", transform=ax.transAxes, fontsize=8, color="#666")
    ax.text(0.03, 0.93, "Better cycle life ↑", transform=ax.transAxes, fontsize=8, color="#666")

    # Panel 3: rank correlation
    ax = axes[2]
    order = sorted(feas, key=lambda p: p["sei_per_pct_soc"])
    names = [p["label"] for p in order]
    sei_o = [p["sei_per_pct_soc"] for p in order]
    cyc_o = [p["equiv_cycles_to_eol"] for p in order]
    xo = np.arange(len(order))
    ax.plot(xo, sei_o, "o-", color="#b2182b", lw=1.8, label="SEI/ΔSoC (proxy)")
    ax2 = ax.twinx()
    ax2.plot(xo, cyc_o, "s--", color="#2166ac", lw=1.8, label="Sessions to 20% fade (Wang)")
    ax.set_xticks(xo)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=6, rotation=0)
    ax.set_ylabel("SEI / ΔSoC", color="#b2182b")
    ax2.set_ylabel("Sessions to EOL", color="#2166ac")
    rho = _spearman(sei_o, [-c for c in cyc_o])
    ax.set_title(
        f"Rank order: SEI proxy vs Wang model\n"
        f"Spearman ρ={rho['rho']:.2f}  (p={rho['p']:.3g})",
        fontsize=10,
    )
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Physics-Grounded Degradation Analysis — RW9 Cell\n"
        "Wang capacity-fade model calibrated from measured reference discharges",
        fontsize=12, y=1.02,
    )
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enhanced_dir", type=Path, default=DEFAULT_ENHANCED)
    p.add_argument("--chebyshev_json", type=Path, default=DEFAULT_CHEBYSHEV)
    p.add_argument("--out_dir", type=Path, default=ROOT / "outputs/visualization")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--margins", default=CANONICAL["conformal_margins"])
    p.add_argument("--max_duration_min", type=float, default=105.0)
    p.add_argument("--soc_target", type=float, default=0.95)
    args = p.parse_args()

    results_json = args.enhanced_dir / "models/family_optimization_results.json"
    payload = json.loads(results_json.read_text())
    start = dict(payload["initial_state"])
    constraints = payload.get("constraints", {})

    model = get_degradation_model(reload=True)
    bdt_path = resolve_bdt_ckpt(args.bdt_ckpt, root=ROOT)
    sim = ProfileSimulator(
        bdt_path=bdt_path,
        capacity_path=ROOT / args.capacity,
        margins_path=ROOT / args.margins,
        max_minutes=150,
        soc_target=args.soc_target,
    )

    rows = _load_family_best(results_json)
    if args.chebyshev_json.is_file():
        rows.extend(_load_chebyshev_pulsed(args.chebyshev_json))

    profiles = [
        _score_profile(
            sim, start, row,
            soc_target=constraints.get("soc_target", args.soc_target),
            max_duration_min=constraints.get("max_duration_min", args.max_duration_min),
        )
        for row in rows
    ]
    feas = [p for p in profiles if p.get("feasible")]

    sei_vals = [p["sei_per_pct_soc"] for p in feas]
    fade_vals = [p["capacity_fade_pct"] for p in feas]
    cyc_vals = [p["equiv_cycles_to_eol"] for p in feas]
    dur_vals = [p["duration_min"] for p in feas]

    family_feas = [p for p in feas if "ω=" not in p["label"]]
    best_sei = min(family_feas, key=lambda p: p["sei_per_pct_soc"]) if family_feas else None
    best_fade = min(feas, key=lambda p: p["capacity_fade_pct"]) if feas else None
    best_cyc = max(feas, key=lambda p: p["equiv_cycles_to_eol"]) if feas else None

    out = {
        "n_profiles": len(profiles),
        "n_feasible": len(feas),
        "model_calibration": model.calibration_info,
        "spearman": {
            "sei_vs_capacity_fade_pct": _spearman(sei_vals, fade_vals),
            "sei_vs_equiv_cycles_to_eol": _spearman(sei_vals, cyc_vals),
            "duration_vs_equiv_cycles": _spearman(dur_vals, cyc_vals),
        },
        "global_best": {
            "by_sei": {"label": best_sei["label"], "sei": best_sei["sei_per_pct_soc"]} if best_sei else None,
            "by_capacity_fade": {"label": best_fade["label"], "fade_pct": best_fade["capacity_fade_pct"]} if best_fade else None,
            "by_equiv_cycles": {"label": best_cyc["label"], "cycles": best_cyc["equiv_cycles_to_eol"]} if best_cyc else None,
        },
        "family_best_by_sei": best_sei["family_id"] if best_sei else None,
        "family_best_by_physics": min(family_feas, key=lambda p: p["capacity_fade_pct"])["family_id"] if family_feas else None,
        "profiles": profiles,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "degradation_model_comparison.json"
    json_path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    plot_fig6(profiles, args.out_dir / "fig6_physics_degradation.png", model.calibration_info)

    sp = out["spearman"]["sei_vs_equiv_cycles_to_eol"]
    print(f"Saved {json_path}")
    print(f"Saved {args.out_dir / 'fig6_physics_degradation.png'}")
    print(f"Feasible profiles: {len(feas)}/{len(profiles)}")
    print(f"Spearman(SEI, equiv_cycles): ρ={sp['rho']:.3f}  p={sp['p']:.4g}")
    if best_sei and out["family_best_by_physics"]:
        print(f"Best by SEI: {out['family_best_by_sei']}  |  Best by physics: {out['family_best_by_physics']}")


if __name__ == "__main__":
    main()
