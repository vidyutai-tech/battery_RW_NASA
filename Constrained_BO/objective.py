"""Rewards, energy (∫ V·I dt), and session objective evaluation."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

TEMP_PLATEAU = 1.5
TEMP_FLOOR = -2.2
TEMP_LOW_C = 15.0
TEMP_HIGH_C = 35.0
TEMP_MAX_C = 50.0

TIME_MAX_REWARD = 1.5
TIME_DECAY_PER_S = 0.01
TIME_ZERO_AT_S = 150.0

V_NOM_FALLBACK = 3.7

SOC_PENALTY_SCALE = 300.0
ENERGY_PENALTY_SCALE = 300.0
VOLTAGE_PENALTY_SCALE = 100.0


def full_capacity_joules(q_rated_as: float, v_nom: float = V_NOM_FALLBACK) -> float:
    """Full-pack energy (J): ``q_rated_as`` in A·s (= q_rated_ah × 3600) times ``v_nom``."""
    return float(q_rated_as) * float(v_nom)


def energy_required_j(
    q_rated_as: float,
    energy_fraction: float,
    v_nom: float = V_NOM_FALLBACK,
) -> float:
    """Energy constraint: fraction of full capacity (e.g. 0.40 → 40% of E_full)."""
    return float(energy_fraction) * full_capacity_joules(q_rated_as, v_nom)


def energy_delivered_j(
    voltage_v: np.ndarray,
    current_a: np.ndarray,
    time_s: np.ndarray,
) -> float:
    """Integrate V·I dt from BDT arrays (negative charge current → energy in = -V·I)."""
    v = np.asarray(voltage_v, dtype=np.float64)
    i = np.asarray(current_a, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    if v.size == 0:
        return 0.0
    power_w = -v * i
    if t.size <= 1:
        dt = 1.0 if t.size == 0 else float(t[0] if t[0] > 0 else 1.0)
        return float(max(0.0, power_w[0] * dt))
    return float(max(0.0, np.trapz(power_w, t)))


def temperature_reward(t_c: float) -> float:
    """Plateau 1.5 in [15, 35] °C; linear penalty outside (unbounded, no floor at 0)."""
    t = float(t_c)
    if TEMP_LOW_C <= t <= TEMP_HIGH_C:
        return TEMP_PLATEAU
    slope = (TEMP_PLATEAU - TEMP_FLOOR) / TEMP_LOW_C
    if t < TEMP_LOW_C:
        return TEMP_PLATEAU - slope * (TEMP_LOW_C - t)
    return TEMP_PLATEAU - slope * (t - TEMP_HIGH_C)


def time_reward(t_sec: float) -> float:
    """Linear decay: 1.5 at 0 s → 0 at 150 s; continues negative for longer charges."""
    return TIME_MAX_REWARD - TIME_DECAY_PER_S * float(t_sec)


def mean_temperature_reward(temperature_c: np.ndarray) -> float:
    if temperature_c.size == 0:
        return 0.0
    return float(np.mean([temperature_reward(t) for t in temperature_c]))


def aggregate_reward(
    temperature_c: np.ndarray,
    duration_s: float,
    *,
    w_time: float = 1.0,
    w_temperature: float = 1.0,
) -> dict:
    tr = mean_temperature_reward(temperature_c)
    tim = time_reward(duration_s)
    total = w_time * tim + w_temperature * tr
    return {
        "temperature_reward": tr,
        "time_reward": tim,
        "total_reward": total,
        "reward_weights": {"w_time": w_time, "w_temperature": w_temperature},
    }


def evaluate_session(
    session: Dict,
    *,
    w_time: float = 1.0,
    w_temperature: float = 1.0,
) -> Tuple[float, Dict]:
    duration_s = float(session["current_a"].size)
    duration_min = duration_s / 60.0
    soc_end = float(session["soc"][-1]) if session["soc"].size else 0.0
    soc_start = float(session["initial_state"].get("soc", soc_end))
    soc_target = float(session["soc_target"])
    peak_v = float(np.max(session["voltage_v"])) if session["voltage_v"].size else 0.0
    peak_t = float(np.max(session["temperature_c"])) if session["temperature_c"].size else 0.0
    mean_t = float(np.mean(session["temperature_c"])) if session["temperature_c"].size else 0.0

    q_rated = float(session["q_rated_as"])
    v_nom = float(session.get("v_nom", V_NOM_FALLBACK))
    constraint_mode = session.get("constraint_mode", "soc")
    energy_full_j = float(session.get("energy_full_j", full_capacity_joules(q_rated, v_nom)))
    energy_delivered = energy_delivered_j(
        session["voltage_v"], session["current_a"], session["time_s"],
    )
    energy_required = float(session.get("energy_required_j", 0.0))
    if constraint_mode == "energy" and energy_required <= 0.0:
        frac = float(session.get("energy_fraction", 0.0))
        energy_required = energy_required_j(q_rated, frac, v_nom)

    rewards = aggregate_reward(
        session["temperature_c"],
        duration_s,
        w_time=w_time,
        w_temperature=w_temperature,
    )

    if constraint_mode == "energy":
        energy_shortfall_j = max(0.0, energy_required - energy_delivered)
        feasible = energy_delivered >= energy_required - 1e-3
    else:
        energy_shortfall_j = 0.0
        feasible = (
            session["end_reason"] == "SoC target"
            and soc_end >= soc_target - 1e-4
        )

    loss = -rewards["total_reward"]
    loss += 1e-3 * duration_min
    if not feasible:
        if constraint_mode == "energy":
            rel_shortfall = energy_shortfall_j / max(energy_required, 1e-6)
            loss += ENERGY_PENALTY_SCALE * rel_shortfall
        else:
            loss += SOC_PENALTY_SCALE * max(0.0, soc_target - soc_end)
    if peak_v > 4.2 + 1e-4:
        loss += VOLTAGE_PENALTY_SCALE * (peak_v - 4.2)

    metrics = {
        "feasible": feasible,
        "loss": float(loss),
        "constraint_mode": constraint_mode,
        "total_reward": rewards["total_reward"],
        "time_reward": rewards["time_reward"],
        "temperature_reward": rewards["temperature_reward"],
        "duration_min": duration_min,
        "duration_s": duration_s,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "soc_target": soc_target,
        "soc_delta": soc_end - soc_start,
        "end_reason": session["end_reason"],
        "peak_voltage": peak_v,
        "peak_temperature": peak_t,
        "mean_temperature": mean_t,
        "energy_delivered_j": energy_delivered,
        "energy_required_j": energy_required,
        "energy_full_j": energy_full_j,
        "energy_fraction": session.get("energy_fraction"),
        "energy_shortfall_j": energy_shortfall_j,
        "reward_weights": rewards["reward_weights"],
    }
    return loss, metrics
