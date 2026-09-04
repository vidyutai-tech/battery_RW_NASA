"""Paper reward (Eq. 10) and energy integrals.

    R = w_soc * ΔSOC − w_loss * Q_loss − w_time * t_h^{z_time}

Q_loss uses the calibrated hybrid model in the paper's session-mean closed form.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from aacopt.config import RewardSpec
from aacopt.degradation import HybridDegradationModel


def full_capacity_joules(q_rated_as: float, v_nom: float) -> float:
    return float(q_rated_as) * float(v_nom)


def energy_required_j(q_rated_as: float, energy_fraction: float, v_nom: float) -> float:
    return float(energy_fraction) * full_capacity_joules(q_rated_as, v_nom)


def energy_delivered_j(voltage_v, current_a, time_s) -> float:
    v = np.asarray(voltage_v, dtype=np.float64)
    i = np.asarray(current_a, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    if v.size == 0:
        return 0.0
    power_w = -v * i
    if t.size <= 1:
        dt = 1.0 if t.size == 0 else float(t[0] if t[0] > 0 else 1.0)
        return float(max(0.0, power_w[0] * dt))
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(max(0.0, trapz(power_w, t)))


def score_session(
    session: Dict[str, Any],
    *,
    model: HybridDegradationModel,
    spec: RewardSpec,
    anchors: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    del anchors
    i = np.asarray(session["current_a"], dtype=np.float64)
    v = np.asarray(session["voltage_v"], dtype=np.float64)
    t_c = np.asarray(session["temperature_c"], dtype=np.float64)
    soc = np.asarray(session["soc"], dtype=np.float64)
    time_s = np.asarray(session["time_s"], dtype=np.float64)
    duration_s = float(i.size)
    duration_min = duration_s / 60.0

    e_del = energy_delivered_j(v, i, time_s)
    e_req = float(session.get("energy_required_j", 0.0)) or energy_required_j(
        float(session.get("q_rated_as", 2.2 * 3600.0)),
        float(session.get("energy_fraction", 0.40)),
        float(session.get("v_nom", 3.7)),
    )
    e_norm = float(e_del / max(e_req, 1e-12))

    deg = model.session_closed_form(
        current_a=i, temperature_c=t_c, soc=soc, dt_s=1.0,
    )
    q_total = float(deg["q_total"])
    delta_soc = float(deg["delta_soc"])
    t_h = float(deg["duration_h"])
    time_raw = (t_h ** spec.z_time) if t_h > 0.0 else 0.0

    soc_term = spec.w_soc * delta_soc
    q_term = spec.w_loss * q_total
    time_term = spec.w_time * time_raw
    reward = float(soc_term - q_term - time_term)
    if not np.isfinite(reward):
        reward = -1e6

    shortfall = max(0.0, 1.0 - e_norm)
    peak_v = float(v.max()) if v.size else 0.0
    v_over = max(0.0, peak_v - float(session.get("v_max", 4.2)))
    penalty = (spec.energy_shortfall_penalty_scale * shortfall
               + spec.voltage_ceiling_penalty_scale * v_over)
    feasible = (e_del >= e_req - 1e-3) and v_over <= 1e-3
    loss = (
        -float(reward)
        + spec.duration_loss_weight * duration_min
        + penalty
    )
    return {
        "reward": float(reward),
        "loss": float(loss),
        "feasible": bool(feasible),
        "soc_reward": float(soc_term),
        "qloss_penalty": float(q_term),
        "time_penalty": float(time_term),
        "time_penalty_raw": float(time_raw),
        "e_delivered_j": e_del,
        "e_required_j": e_req,
        "e_norm": float(e_norm),
        "delta_soc": delta_soc,
        "soc_end": float(soc[-1]) if soc.size else 0.0,
        "q_calendar": float(deg["q_calendar"]),
        "q_cyclic": float(deg["q_cyclic"]),
        "q_total": q_total,
        "qloss_total": q_total,
        "ah_throughput": float(deg["ah_throughput"]),
        "nominal_c_rate": float(deg["nominal_c_rate"]),
        "mean_soc": float(deg["mean_soc"]),
        "duration_s": duration_s,
        "duration_h": t_h,
        "duration_min": duration_min,
        "peak_v": peak_v,
        "peak_t": float(t_c.max()) if t_c.size else 0.0,
        "mean_t": float(deg["mean_temperature_c"]),
        "mean_charge_c_rate": float(deg["nominal_c_rate"]),
        "ah_charge": float(deg["ah_throughput"]),
        "end_reason": session.get("end_reason"),
        "energy_shortfall": float(shortfall),
        "penalty": float(penalty),
        "weights": spec.to_dict(),
    }
