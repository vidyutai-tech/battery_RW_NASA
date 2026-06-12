"""
Stage 2 — lifetime objective for Bayesian optimization.

Feasibility first (SoC target, time limit, no hard T violation), then among
feasible profiles minimize a **composite degradation objective** (Priority 2):

    Loss = w_sei * SEI/ΔSoC
         + w_time * duration_min
         + w_temp * ∫ max(0, T − T_comfort)² dt
         + w_vstress * ∫ max(0, V − V_ref)² dt

Integrals are computed on the BDT-predicted trajectories (1 Hz samples) and
reported in ``°C²·min`` and ``V²·min`` for readability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np

from charging_opt.reward import (
    DEFAULT_K_ARRHENIUS,
    EVAL_CONSTRAINT_PENALTY,
    compute_sei_proxy,
)

INFEASIBLE_BASE = 200.0
SHORTFALL_SLOPE = 5.0
TIME_EXCESS_SLOPE = 2.0

V_REF_STRESS = 4.0
T_COMFORT_C = 35.0
DT_S = 1.0

ObjectiveMode = Literal["composite", "legacy"]


@dataclass(frozen=True)
class LifetimeWeights:
    """
    Composite objective weights (feasible profiles only).

    Defaults keep SEI/ΔSoC dominant (~70) while voltage stress and time
    provide meaningful tie-breakers on RW9 at room temperature.
    """

    sei: float = 1.0
    time: float = 0.02
    temperature: float = 0.05
    voltage_stress: float = 0.08

    @classmethod
    def legacy(cls) -> LifetimeWeights:
        """Pre–Priority-2 tie-break weights (SEI/ΔSoC + linear peak T + time)."""
        return cls(sei=1.0, time=0.02, temperature=0.5, voltage_stress=0.0)

    def to_dict(self) -> Dict[str, float]:
        return {
            "w_sei": self.sei,
            "w_time": self.time,
            "w_temperature": self.temperature,
            "w_voltage_stress": self.voltage_stress,
        }


def compute_voltage_stress(voltage_v: np.ndarray, *, v_ref: float = V_REF_STRESS) -> Dict[str, float]:
    """Integral of max(0, V − v_ref)² over the session (V²·s and V²·min)."""
    v = np.asarray(voltage_v, dtype=np.float64)
    if v.size == 0:
        return {"voltage_stress_vs": 0.0, "voltage_stress_v2_min": 0.0}
    excess = np.maximum(v - float(v_ref), 0.0)
    integral_vs = float(np.sum(np.square(excess)) * DT_S)
    return {
        "voltage_stress_vs": integral_vs,
        "voltage_stress_v2_min": integral_vs / 60.0,
    }


def compute_temperature_penalty(
    temperature_c: np.ndarray,
    *,
    t_comfort_c: float = T_COMFORT_C,
) -> Dict[str, float]:
    """Integral of max(0, T − t_comfort)² over the session (°C²·s and °C²·min)."""
    t = np.asarray(temperature_c, dtype=np.float64)
    if t.size == 0:
        return {"temperature_penalty_cs": 0.0, "temperature_penalty_c2_min": 0.0}
    excess = np.maximum(t - float(t_comfort_c), 0.0)
    integral_cs = float(np.sum(np.square(excess)) * DT_S)
    return {
        "temperature_penalty_cs": integral_cs,
        "temperature_penalty_c2_min": integral_cs / 60.0,
        "peak_temp_excess_c": float(np.max(excess)),
    }


def session_metrics(
    session: Dict,
    k: float = DEFAULT_K_ARRHENIUS,
    *,
    v_ref_stress: float = V_REF_STRESS,
    t_comfort_c: float = T_COMFORT_C,
) -> Dict:
    """Extract parameters of interest from a simulated session."""
    if session["current_a"].size == 0:
        return {"empty": True}
    soc = session["soc"]
    i = session["current_a"]
    t = session["temperature_c"]
    v = session["voltage_v"]
    soc0 = session["initial_state"]["soc"]
    d_soc_total = float(soc[-1] - soc0) * 100.0
    sei_total = compute_sei_proxy(i, t, DT_S, k=k)
    v_stress = compute_voltage_stress(v, v_ref=v_ref_stress)
    t_pen = compute_temperature_penalty(t, t_comfort_c=t_comfort_c)
    return {
        "duration_min": float(i.size / 60.0),
        "duration_s": float(i.size),
        "end_reason": session["end_reason"],
        "soc_end": float(soc[-1]),
        "delta_soc_pct_total": d_soc_total,
        "sei_proxy": float(sei_total),
        "sei_per_pct_soc": float(sei_total / d_soc_total) if d_soc_total > 1e-6 else None,
        "peak_voltage": float(np.max(v)),
        "peak_temperature": float(np.max(t)),
        "violated": session["end_reason"] == "temperature violation",
        **v_stress,
        **t_pen,
    }


def composite_loss(
    m: Dict,
    weights: LifetimeWeights,
    *,
    objective_mode: ObjectiveMode = "composite",
) -> Tuple[float, Dict[str, float]]:
    """Build scalar loss and component breakdown from session metrics."""
    sei_term = float(m["sei_per_pct_soc"])
    time_term = float(m["duration_min"])
    temp_c2_min = float(m.get("temperature_penalty_c2_min", 0.0))
    vstress_v2_min = float(m.get("voltage_stress_v2_min", 0.0))
    temp_excess_peak = float(m.get("peak_temp_excess_c", 0.0))

    if objective_mode == "legacy":
        loss = (
            weights.sei * sei_term
            + weights.temperature * temp_excess_peak
            + weights.time * time_term
        )
        components = {
            "objective_mode": "legacy",
            "sei_term": weights.sei * sei_term,
            "time_term": weights.time * time_term,
            "temperature_term": weights.temperature * temp_excess_peak,
            "voltage_stress_term": 0.0,
        }
    else:
        loss = (
            weights.sei * sei_term
            + weights.time * time_term
            + weights.temperature * temp_c2_min
            + weights.voltage_stress * vstress_v2_min
        )
        components = {
            "objective_mode": "composite",
            "sei_term": weights.sei * sei_term,
            "time_term": weights.time * time_term,
            "temperature_term": weights.temperature * temp_c2_min,
            "voltage_stress_term": weights.voltage_stress * vstress_v2_min,
        }
    components["loss"] = float(loss)
    return float(loss), components


def aggregate_lifetime_reward(
    session: Dict,
    *,
    soc_target: float = 0.95,
    soc_tol: float = 0.005,
    max_duration_min: Optional[float] = None,
    t_comfort_c: float = T_COMFORT_C,
    v_ref_stress: float = V_REF_STRESS,
    weights: LifetimeWeights = LifetimeWeights(),
    objective_mode: ObjectiveMode = "composite",
    k: float = DEFAULT_K_ARRHENIUS,
) -> Tuple[float, Dict]:
    """
    Return (reward, info). BO minimizes ``info['loss']``; reward = -loss.
    """
    m = session_metrics(
        session, k=k, v_ref_stress=v_ref_stress, t_comfort_c=t_comfort_c,
    )
    if m.get("empty"):
        loss = EVAL_CONSTRAINT_PENALTY
        m.update(feasible=False, loss=loss, reward=-loss)
        return -loss, m

    soc0 = session["initial_state"]["soc"]
    required_pct = (soc_target - soc0) * 100.0
    shortfall_pct = max(0.0, required_pct - m["delta_soc_pct_total"])
    reached_target = m["soc_end"] >= soc_target - soc_tol

    if m["violated"]:
        loss = EVAL_CONSTRAINT_PENALTY
        m.update(
            feasible=False,
            loss=loss,
            reward=-loss,
            components={"violation": "temperature"},
        )
        return -loss, m

    sei_term = m["sei_per_pct_soc"]
    time_term = m["duration_min"]
    time_excess = (
        max(0.0, time_term - float(max_duration_min))
        if max_duration_min is not None
        else 0.0
    )

    if not reached_target:
        loss = INFEASIBLE_BASE + SHORTFALL_SLOPE * shortfall_pct
        if sei_term is not None and np.isfinite(sei_term):
            loss += 0.1 * sei_term
        m.update(
            feasible=False,
            loss=float(loss),
            reward=-float(loss),
            components={
                "infeasible": True,
                "reason": "soc_shortfall",
                "soc_shortfall_pct": shortfall_pct,
                "time_excess_min": time_excess,
                "sei_per_pct_soc": sei_term,
                "duration_min": time_term,
            },
        )
        return -float(loss), m

    if time_excess > 0.0:
        loss = INFEASIBLE_BASE + TIME_EXCESS_SLOPE * time_excess
        if sei_term is not None and np.isfinite(sei_term):
            loss += 0.1 * sei_term
        m.update(
            feasible=False,
            loss=float(loss),
            reward=-float(loss),
            components={
                "infeasible": True,
                "reason": "time_excess",
                "soc_shortfall_pct": 0.0,
                "time_excess_min": time_excess,
                "max_duration_min": float(max_duration_min),
                "sei_per_pct_soc": sei_term,
                "duration_min": time_term,
            },
        )
        return -float(loss), m

    if sei_term is None or not np.isfinite(sei_term):
        loss = INFEASIBLE_BASE
        m.update(feasible=False, loss=loss, reward=-loss)
        return -loss, m

    loss, comp = composite_loss(m, weights, objective_mode=objective_mode)
    m.update(
        feasible=True,
        loss=float(loss),
        reward=-float(loss),
        components={
            "infeasible": False,
            "reason": "ok",
            "soc_shortfall_pct": 0.0,
            "time_excess_min": 0.0,
            "max_duration_min": max_duration_min,
            "soc_target": soc_target,
            "v_ref_stress": v_ref_stress,
            "t_comfort_c": t_comfort_c,
            "weights": weights.to_dict(),
            "sei_per_pct_soc": sei_term,
            "duration_min": time_term,
            "temperature_penalty_c2_min": m.get("temperature_penalty_c2_min"),
            "voltage_stress_v2_min": m.get("voltage_stress_v2_min"),
            **comp,
        },
    )
    return -float(loss), m
