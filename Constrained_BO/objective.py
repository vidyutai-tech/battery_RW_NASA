"""Rewards, energy (∫ V·I dt), and session objective evaluation."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import numpy as np

from Constrained_BO.hybrid_degradation import (
    HybridDegradationParameters,
    compute_session_reward,
    session_throughput_metrics,
)

TEMP_PLATEAU = 1.5
TEMP_FLOOR = -2.2
TEMP_LOW_C = 20.0  # optimal band lower bound (°C)
TEMP_HIGH_C = 30.0  # optimal band upper bound (°C)
TEMP_MAX_C = 50.0

TIME_MAX_REWARD = 1.5
TIME_DECAY_PER_S = 0.01
TIME_ZERO_AT_S = 150.0

V_NOM_FALLBACK = 3.7

SOC_PENALTY_SCALE = 300.0
ENERGY_PENALTY_SCALE = 300.0
VOLTAGE_PENALTY_SCALE = 100.0
QLOSS_CAP_PENALTY_SCALE = 80.0
DEFAULT_DURATION_LOSS_WEIGHT = 1e-3

RewardMode = Literal["legacy_temp_time", "hybrid_qloss"]
DEFAULT_REWARD_MODE: RewardMode = "hybrid_qloss"

# Terminology note (Phase 1 hybrid degradation model): qloss_calendar / qloss_cyclic
# / qloss_total are a **Relative Capacity-Loss Index**, not a calibrated "Capacity
# Fade (%)". The underlying coefficients (hybrid_degradation.py) are literature
# values for other cells/chemistries and have not been fit to NASA RW9-RW12 or LFP
# aging data. Use these values only to *rank* charging strategies against each other
# under the same model; do not report them as an absolute percent capacity loss.
QLOSS_TERMINOLOGY = "relative_capacity_loss_index"
QLOSS_TERMINOLOGY_NOTE = (
    "qloss_calendar/qloss_cyclic/qloss_total are a dimensionless Relative "
    "Capacity-Loss Index (ranking signal), not a calibrated Capacity Fade (%). "
    "See README 'Hybrid degradation methodology' for details."
)

# Units for every metric exported in evaluate_session()/aggregate_reward() output.
# Kept alongside the code (not just docs) so results JSON can embed this table via
# run.py's meta, and so it stays in sync with the fields actually produced.
RESULT_METRIC_UNITS: Dict[str, str] = {
    "loss": "dimensionless (scalar BO objective)",
    "total_reward": "dimensionless",
    "time_reward": "dimensionless",
    "temperature_reward": "dimensionless",
    "soc_reward": "dimensionless",
    "qloss_calendar": "relative capacity-loss index (dimensionless, NOT % fade)",
    "qloss_cyclic": "relative capacity-loss index (dimensionless, NOT % fade)",
    "qloss_total": "relative capacity-loss index (dimensionless, NOT % fade)",
    "qloss_penalty": "dimensionless",
    "time_penalty": "dimensionless",
    "ah_throughput": "Ah (ampere-hours)",
    "nominal_c_rate": "C-rate (dimensionless, multiples of 1C)",
    "max_c_rate": "C-rate (dimensionless, multiples of 1C)",
    "mean_soc": "fraction [0, 1]",
    "efc": "equivalent full cycles (dimensionless)",
    "duration_min": "minutes",
    "duration_s": "seconds",
    "soc_start": "fraction [0, 1]",
    "soc_end": "fraction [0, 1]",
    "soc_target": "fraction [0, 1]",
    "soc_delta": "fraction [0, 1]",
    "peak_voltage": "V",
    "peak_temperature": "degC",
    "mean_temperature": "degC",
    "dT_peak_charge": "degC",
    "dT_dt_max_charge": "degC/s",
    "stem_charge_per_kj": "degC/kJ",
    "rest_cooling_mean": "degC (mean T_start_rest - T_end_rest across rest intervals)",
    "energy_delivered_j": "J (joules)",
    "energy_required_j": "J (joules)",
    "energy_full_j": "J (joules)",
    "energy_fraction": "fraction [0, 1]",
    "energy_shortfall_j": "J (joules)",
    "voltage_margin_v": "V (v_max - peak_voltage; negative = ceiling exceeded)",
    "energy_margin_j": "J (energy_delivered_j - energy_required_j; energy mode)",
    "soc_margin": "fraction [0, 1] (soc_end - soc_target; SoC mode)",
}


def reward_kwargs_from_meta(meta: Dict) -> Dict:
    """Build evaluate_session reward kwargs from results JSON meta."""
    rw = meta.get("reward_weights", {}) or {}
    mode = meta.get("reward_mode", rw.get("reward_mode", DEFAULT_REWARD_MODE))
    return {
        "reward_mode": mode,
        "w_soc": float(rw.get("w_soc", 1.0)),
        "w_qloss": float(rw.get("w_qloss", 1.0)),
        "w_time": float(rw.get("w_time", 0.1 if mode == "hybrid_qloss" else 1.0)),
        "w_temperature": float(rw.get("w_temperature", 1.0)),
        "z": float(rw.get("z", rw.get("z_time", 0.55))),
    }


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
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(max(0.0, trapz(power_w, t)))


def temperature_reward(t_c: float) -> float:
    """Plateau TEMP_PLATEAU in [TEMP_LOW_C, TEMP_HIGH_C]; linear penalty outside."""
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


def make_hybrid_params(
    *,
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    w_time: float = 0.1,
    z: float = 0.55,
) -> HybridDegradationParameters:
    return HybridDegradationParameters(
        w_soc=float(w_soc),
        w_qloss=float(w_qloss),
        w_time=float(w_time),
        z_time=float(z),
        z_cal=float(z),
    )


def aggregate_reward(
    session: Dict,
    duration_s: float,
    *,
    reward_mode: RewardMode = DEFAULT_REWARD_MODE,
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    z: float = 0.55,
    hybrid_params: Optional[HybridDegradationParameters] = None,
) -> dict:
    from Constrained_BO.bdt_thermal import bdt_thermal_metrics

    thermal = bdt_thermal_metrics(session)
    # Physical throughput/SoC descriptors (Ah, C-rate, EFC, mean SoC) are properties
    # of the session trajectory, independent of which reward_mode is scored — so
    # they are always computed and reported, even under legacy_temp_time.
    phys = session_throughput_metrics(session)

    if reward_mode == "legacy_temp_time":
        tr = mean_temperature_reward(session["temperature_c"])
        tim = time_reward(duration_s)
        total = float(w_time) * tim + float(w_temperature) * tr
        return {
            "reward_mode": reward_mode,
            "temperature_reward": tr,
            "time_reward": tim,
            "soc_reward": 0.0,
            "qloss_calendar": 0.0,
            "qloss_cyclic": 0.0,
            "qloss_total": 0.0,
            "qloss_penalty": 0.0,
            "time_penalty": 0.0,
            "total_reward": total,
            "thermal_metrics": thermal,
            "ah_throughput": phys["ah_throughput"],
            "nominal_c_rate": phys["nominal_c_rate"],
            "max_c_rate": phys["max_c_rate"],
            "mean_soc": phys["mean_soc"],
            "efc": phys["efc"],
            "reward_weights": {
                "w_time": float(w_time),
                "w_temperature": float(w_temperature),
            },
        }

    params = hybrid_params or make_hybrid_params(
        w_soc=w_soc, w_qloss=w_qloss, w_time=w_time, z=z,
    )
    hybrid = compute_session_reward(session, params=params)
    # Legacy keys kept for compare scripts that still read them.
    return {
        "reward_mode": reward_mode,
        "temperature_reward": 0.0,
        "time_reward": -hybrid["time_penalty"],
        "soc_reward": hybrid["soc_reward"],
        "qloss_calendar": hybrid["qloss_calendar"],
        "qloss_cyclic": hybrid["qloss_cyclic"],
        "qloss_total": hybrid["qloss_total"],
        "qloss_penalty": hybrid["qloss_penalty"],
        "time_penalty": hybrid["time_penalty"],
        "mean_soc": phys["mean_soc"],
        "max_c_rate": phys["max_c_rate"],
        "efc": phys["efc"],
        "ah_throughput": hybrid["ah_throughput"],
        "nominal_c_rate": hybrid["nominal_c_rate"],
        "total_reward": hybrid["total_reward"],
        "thermal_metrics": thermal,
        "reward_weights": hybrid["reward_weights"],
        "degradation_params": hybrid.get("degradation_params"),
        "hybrid_metrics": hybrid,
    }


def evaluate_session(
    session: Dict,
    *,
    reward_mode: RewardMode = DEFAULT_REWARD_MODE,
    w_time: float = 0.1,
    w_temperature: float = 1.0,
    w_soc: float = 1.0,
    w_qloss: float = 1.0,
    z: float = 0.55,
    hybrid_params: Optional[HybridDegradationParameters] = None,
    qloss_cap: Optional[float] = None,
    qloss_cap_scale: float = QLOSS_CAP_PENALTY_SCALE,
    duration_loss_weight: float = DEFAULT_DURATION_LOSS_WEIGHT,
) -> Tuple[float, Dict]:
    """Evaluate one simulated session.

    Optional ``qloss_cap`` adds a soft penalty when session Q_loss exceeds the
    cap (typical use: Random-search best Q so GP-BO must match or beat it).
    """
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
        session,
        duration_s,
        reward_mode=reward_mode,
        w_time=w_time,
        w_temperature=w_temperature,
        w_soc=w_soc,
        w_qloss=w_qloss,
        z=z,
        hybrid_params=hybrid_params,
    )
    thermal = rewards.get("thermal_metrics", {})

    if constraint_mode == "energy":
        energy_shortfall_j = max(0.0, energy_required - energy_delivered)
        feasible = energy_delivered >= energy_required - 1e-3
    else:
        energy_shortfall_j = 0.0
        feasible = (
            session["end_reason"] == "SoC target"
            and soc_end >= soc_target - 1e-4
        )

    q_tot = float(rewards.get("qloss_total", 0.0) or 0.0)
    qloss_cap_penalty = 0.0
    if qloss_cap is not None and np.isfinite(float(qloss_cap)):
        excess = max(0.0, q_tot - float(qloss_cap))
        if excess > 0.0:
            # Relative excess so the penalty scale is comparable across cells.
            rel = excess / max(float(qloss_cap), 1e-9)
            qloss_cap_penalty = float(qloss_cap_scale) * rel

    loss = -rewards["total_reward"]
    loss += float(duration_loss_weight) * duration_min
    loss += qloss_cap_penalty
    if not feasible:
        if constraint_mode == "energy":
            rel_shortfall = energy_shortfall_j / max(energy_required, 1e-6)
            loss += ENERGY_PENALTY_SCALE * rel_shortfall
        else:
            loss += SOC_PENALTY_SCALE * max(0.0, soc_target - soc_end)
    v_ceiling = float(session.get("v_max", 4.2))
    if peak_v > v_ceiling + 1e-4:
        loss += VOLTAGE_PENALTY_SCALE * (peak_v - v_ceiling)

    # Constraint margins: positive = headroom to the limit, negative = violated.
    constraint_margins: Dict[str, float] = {
        "voltage_margin_v": v_ceiling - peak_v,
    }
    if constraint_mode == "energy":
        constraint_margins["energy_margin_j"] = energy_delivered - energy_required
    else:
        constraint_margins["soc_margin"] = soc_end - soc_target
    if qloss_cap is not None and np.isfinite(float(qloss_cap)):
        constraint_margins["qloss_margin"] = float(qloss_cap) - q_tot

    metrics = {
        "feasible": feasible,
        "loss": float(loss),
        "constraint_mode": constraint_mode,
        "reward_mode": rewards.get("reward_mode", reward_mode),
        "total_reward": rewards["total_reward"],
        "time_reward": rewards["time_reward"],
        "temperature_reward": rewards["temperature_reward"],
        "soc_reward": rewards.get("soc_reward", 0.0),
        "qloss_calendar": rewards.get("qloss_calendar", 0.0),
        "qloss_cyclic": rewards.get("qloss_cyclic", 0.0),
        "qloss_total": rewards.get("qloss_total", 0.0),
        "qloss_penalty": rewards.get("qloss_penalty", 0.0),
        "qloss_cap": None if qloss_cap is None else float(qloss_cap),
        "qloss_cap_penalty": float(qloss_cap_penalty),
        "time_penalty": rewards.get("time_penalty", 0.0),
        "ah_throughput": rewards.get("ah_throughput"),
        "nominal_c_rate": rewards.get("nominal_c_rate"),
        "max_c_rate": rewards.get("max_c_rate"),
        "mean_soc": rewards.get("mean_soc"),
        "efc": rewards.get("efc"),
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
        "dT_peak_charge": thermal.get("dT_peak_charge"),
        "dT_dt_max_charge": thermal.get("dT_dt_max_charge"),
        "stem_charge_per_kj": thermal.get("stem_charge_per_kj"),
        "rest_cooling_mean": thermal.get("rest_cooling_mean"),
        "energy_delivered_j": energy_delivered,
        "energy_required_j": energy_required,
        "energy_full_j": energy_full_j,
        "energy_fraction": session.get("energy_fraction"),
        "energy_shortfall_j": energy_shortfall_j,
        "constraint_margins": constraint_margins,
        "reward_weights": rewards["reward_weights"],
        "degradation_params": rewards.get("degradation_params"),
        "duration_loss_weight": float(duration_loss_weight),
        "qloss_terminology": QLOSS_TERMINOLOGY_NOTE,
    }
    return loss, metrics
