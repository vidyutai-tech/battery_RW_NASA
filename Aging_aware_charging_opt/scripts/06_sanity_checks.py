#!/usr/bin/env python
"""STAGE 6 — sanity checks on profiles, simulator physics, and reward.

Must pass before any optimization. Uses the frozen degradation model and the
frozen CCCV 1C anchors. Does not tune anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.config import OptimizationSpec, provenance, stage_dir, write_json
from aacopt.degradation import session_degradation
from aacopt.evaluate import (
    cccv_params, evaluate_params, load_anchors, load_degradation_model, make_simulator,
)
from aacopt.profiles import (
    PulsedFamily, ThreeStepFamily, TwoStepFamily, bind_search, get_family,
)
from aacopt.reward import energy_delivered_j

STAGE = "06_sanity_checks"


def _assert(checks: list, name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "passed": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default="RW9")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    bind_search()
    opt = OptimizationSpec.load()
    from aacopt.config import RewardSpec
    reward = RewardSpec.load()
    model = load_degradation_model()
    device = args.device or opt.device
    cell = args.cell.upper()
    sim, state = make_simulator(cell, device=device)
    anchors = load_anchors(cell)
    checks: list = []

    # ── CCCV shape ──────────────────────────────────────────────────────────
    params = cccv_params(1.0)
    _, metrics, session = evaluate_params(
        sim, state, params, model=model, spec=reward, anchors=anchors,
    )
    i = np.asarray(session["current_a"])
    v = np.asarray(session["voltage_v"])
    t = np.asarray(session["temperature_c"])
    soc = np.asarray(session["soc"])
    i_cc = -params.values["i_cc"]
    cc_mask = np.abs(i - i_cc) < 0.05
    cv_mask = (~cc_mask) & (i < -0.01)
    _assert(checks, "cccv_has_cc_leg", bool(cc_mask.any()),
            f"CC samples={int(cc_mask.sum())} / {i.size}")
    _assert(checks, "cccv_has_cv_or_energy_stop",
            bool(cv_mask.any()) or session["end_reason"] in ("energy target", "SoC full"),
            f"CV samples={int(cv_mask.sum())} end={session['end_reason']}")
    _assert(checks, "cccv_voltage_ceiling",
            float(v.max()) <= opt.session.v_max + 0.02,
            f"V_peak={float(v.max()):.4f} V_max={opt.session.v_max}")
    _assert(checks, "cccv_soc_monotone",
            bool(np.all(np.diff(soc) >= -1e-9)),
            f"ΔSoC={float(soc[-1]-soc[0]):.3f}")
    _assert(checks, "cccv_1c_feasible", bool(metrics["feasible"]),
            f"e_norm={metrics['e_norm']:.3f} shortfall={metrics['energy_shortfall']:.3f}")
    _assert(checks, "reward_positive", float(metrics["reward"]) > 0.0,
            f"R={metrics['reward']:.3f}  ΔSOC={metrics['delta_soc']:.3f}  "
            f"Q={metrics['q_total']:.3e}")

    e = energy_delivered_j(v, i, session["time_s"])
    q_as = sim.q_rated_as
    ah = float(np.sum(-i)) / 3600.0
    _assert(checks, "energy_positive", e > 0.0, f"E={e:.1f} J")
    _assert(checks, "coulomb_matches_dsoc",
            abs(ah - (soc[-1] - soc[0]) * (q_as / 3600.0)) < 0.02,
            f"Ah={ah:.4f} ΔSOC·Q={((soc[-1]-soc[0])*q_as/3600):.4f}")

    # ── 2-step staircase ────────────────────────────────────────────────────
    p2 = TwoStepFamily.from_dict({"i1": 3.0, "i2": 1.5, "soc_switch": 0.40})
    _, m2, s2 = evaluate_params(sim, state, p2, model=model, spec=reward, anchors=anchors)
    i2 = np.asarray(s2["current_a"])
    soc2 = np.asarray(s2["soc"])
    # Commanded |I| should drop after the switch SoC (allowing CV taper).
    before = i2[soc2 < p2.values["soc_switch"]]
    after = i2[soc2 >= p2.values["soc_switch"]]
    if before.size == 0 or after.size == 0:
        drop = False
        detail = "missing stage samples"
    else:
        drop = float(np.median(np.abs(after))) <= float(np.median(np.abs(before))) + 0.05
        detail = f"|I|_med before={np.median(np.abs(before)):.2f} after={np.median(np.abs(after)):.2f}"
    _assert(checks, "two_step_staircase", drop, detail)
    _assert(checks, "two_step_soc_monotone", bool(np.all(np.diff(soc2) >= -1e-9)),
            f"ΔSoC={float(soc2[-1]-soc2[0]):.3f}")

    # ── 3-step ──────────────────────────────────────────────────────────────
    p3 = ThreeStepFamily.from_dict(
        {"i1": 4.0, "i2": 2.5, "i3": 1.25, "soc1": 0.35, "soc2": 0.50},
    )
    _, _, s3 = evaluate_params(sim, state, p3, model=model, spec=reward, anchors=anchors)
    soc3 = np.asarray(s3["soc"])
    _assert(checks, "three_step_soc_monotone", bool(np.all(np.diff(soc3) >= -1e-9)),
            f"ΔSoC={float(soc3[-1]-soc3[0]):.3f}")
    i3 = np.asarray(s3["current_a"])
    _assert(checks, "three_step_charge_current", bool((i3 < -0.05).any()),
            f"min I={float(i3.min()):.2f} A")

    # ── pulsed on/rest ──────────────────────────────────────────────────────
    pp = PulsedFamily.from_dict(
        {"i_charge": 2.2, "pulse_on_min": 5.0, "rest_fraction": 1.0, "i_floor": 0.75},
    )
    _, _, sp = evaluate_params(sim, state, pp, model=model, spec=reward, anchors=anchors)
    ip = np.asarray(sp["current_a"])
    rest = np.abs(ip) < 0.02
    charge = ip < -0.2
    _assert(checks, "pulsed_has_rest", bool(rest.any()), f"rest samples={int(rest.sum())}")
    _assert(checks, "pulsed_has_charge", bool(charge.any()), f"charge samples={int(charge.sum())}")

    # ── degradation physics (no BDT) ────────────────────────────────────────
    n = 3600
    soc_path = np.linspace(0.20, 0.60, n)
    t_c = np.full(n, 25.0)
    q_nom = model.p.q_nominal_ah

    def qcyc(c_rate: float) -> float:
        i_chg = np.full(n, -c_rate * q_nom)
        d = session_degradation(model, current_a=i_chg, temperature_c=t_c, soc=soc_path, dt_s=1.0)
        return float(d["q_cyclic"])

    q_lo, q_hi = qcyc(0.5), qcyc(2.0)
    _assert(checks, "crate_increases_qcyc", q_hi > q_lo,
            f"Q_cyc(0.5C)={q_lo:.3e}  Q_cyc(2C)={q_hi:.3e}")

    i_1c = np.full(n, -1.0 * q_nom)
    q_cool = session_degradation(model, current_a=i_1c, temperature_c=np.full(n, 15.0),
                                 soc=soc_path, dt_s=1.0)
    q_hot = session_degradation(model, current_a=i_1c, temperature_c=np.full(n, 45.0),
                                soc=soc_path, dt_s=1.0)
    _assert(checks, "temp_increases_qcyc", q_hot["q_cyclic"] > q_cool["q_cyclic"],
            f"15°C={q_cool['q_cyclic']:.3e}  45°C={q_hot['q_cyclic']:.3e}")
    _assert(checks, "temp_increases_qcal", q_hot["q_calendar"] > q_cool["q_calendar"],
            f"15°C={q_cool['q_calendar']:.3e}  45°C={q_hot['q_calendar']:.3e}")

    rest_low = session_degradation(
        model, current_a=np.zeros(n), temperature_c=t_c, soc=np.full(n, 0.2), dt_s=1.0,
    )
    rest_high = session_degradation(
        model, current_a=np.zeros(n), temperature_c=t_c, soc=np.full(n, 0.9), dt_s=1.0,
    )
    _assert(checks, "high_soc_rest_increases_qcal",
            rest_high["q_calendar"] > rest_low["q_calendar"],
            f"SOC0.2={rest_low['q_calendar']:.3e}  SOC0.9={rest_high['q_calendar']:.3e}")

    _assert(checks, "current_within_search_bounds",
            float(np.abs(i).max()) <= opt.space.i_max_a + 1e-6,
            f"|I|_max={float(np.abs(i).max()):.3f} A")

    passed = all(c["passed"] for c in checks)
    payload = {
        "cell": cell,
        "n_checks": len(checks),
        "n_passed": sum(1 for c in checks if c["passed"]),
        "passed": bool(passed),
        "checks": checks,
        "cccv_1c_metrics": {k: v for k, v in metrics.items() if k not in ("weights", "anchors")},
        "provenance": provenance(
            STAGE, configs=["paths", "degradation_fitted", "reward", "optimization"],
        ),
    }
    out = stage_dir(STAGE) / "sanity_checks.json"
    write_json(out, payload)
    print(f"\nWrote {out}")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
