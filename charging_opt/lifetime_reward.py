"""
Stage 2 — lifetime objective for Bayesian optimization.

Design (avoids boundary collapse to minimum current):

* **Feasibility first.** A profile must reach ``soc_target`` and, when set,
  finish within ``max_duration_min``. Infeasible profiles receive a structured
  penalty so BO never prefers "full but too slow" over "full and on time".

* **Primary objective (feasible only).** Minimize ``SEI / ΔSoC`` (degradation
  intensity). This is on the order of 60–80 for RW9, directly interpretable.

* **Secondary tie-breakers** (small, normalized): peak temperature excess,
  session duration. These break ties between feasible profiles without
  dominating the objective.

Reward returned to the optimizer is negated loss: higher is better, but
``components['loss']`` is what BO minimizes and is human-readable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from charging_opt.reward import (
    DEFAULT_K_ARRHENIUS,
    EVAL_CONSTRAINT_PENALTY,
    compute_sei_proxy,
)

# Profiles that miss the SoC target are heavily penalised but still ranked by
# how close they came (so BO can move toward feasibility).
INFEASIBLE_BASE = 200.0
SHORTFALL_SLOPE = 5.0       # extra loss per pct-point of missed SoC
TIME_EXCESS_SLOPE = 2.0      # extra loss per minute over max_duration_min


@dataclass(frozen=True)
class LifetimeWeights:
    """Secondary tie-break weights (primary = SEI/ΔSoC among feasible profiles)."""

    temperature: float = 0.5   # per °C above comfort band
    time: float = 0.02         # per minute of session (small)


def session_metrics(session: Dict, k: float = DEFAULT_K_ARRHENIUS) -> Dict:
    """Extract parameters of interest from a simulated session."""
    if session["current_a"].size == 0:
        return {"empty": True}
    soc = session["soc"]
    i = session["current_a"]
    t = session["temperature_c"]
    v = session["voltage_v"]
    soc0 = session["initial_state"]["soc"]
    d_soc_total = float(soc[-1] - soc0) * 100.0
    sei_total = compute_sei_proxy(i, t, 1.0, k=k)
    return {
        "duration_min": float(i.size / 60.0),
        "end_reason": session["end_reason"],
        "soc_end": float(soc[-1]),
        "delta_soc_pct_total": d_soc_total,
        "sei_proxy": float(sei_total),
        "sei_per_pct_soc": float(sei_total / d_soc_total) if d_soc_total > 1e-6 else None,
        "peak_voltage": float(np.max(v)),
        "peak_temperature": float(np.max(t)),
        "violated": session["end_reason"] == "temperature violation",
    }


def aggregate_lifetime_reward(
    session: Dict,
    *,
    soc_target: float = 0.95,
    soc_tol: float = 0.005,
    max_duration_min: Optional[float] = None,
    t_comfort_c: float = 35.0,
    weights: LifetimeWeights = LifetimeWeights(),
    k: float = DEFAULT_K_ARRHENIUS,
) -> Tuple[float, Dict]:
    """
    Return (reward, info). BO minimizes ``info['loss']``; reward = -loss.

    Feasible: reached ``soc_target`` (within ``soc_tol``), optional
    ``duration <= max_duration_min``, no temperature violation.
    Loss ≈ ``sei_per_pct_soc`` (+ tiny tie-breakers).
    """
    m = session_metrics(session, k=k)
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
    temp_excess = max(0.0, m["peak_temperature"] - t_comfort_c)
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
                "temp_excess_c": temp_excess,
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
                "temp_excess_c": temp_excess,
                "duration_min": time_term,
            },
        )
        return -float(loss), m

    if sei_term is None or not np.isfinite(sei_term):
        loss = INFEASIBLE_BASE
        m.update(feasible=False, loss=loss, reward=-loss)
        return -loss, m

    loss = (
        sei_term
        + weights.temperature * temp_excess
        + weights.time * time_term
    )
    m.update(
        feasible=True,
        loss=float(loss),
        reward=-float(loss),
        components={
            "infeasible": False,
            "reason": "ok",
            "sei_per_pct_soc": sei_term,
            "soc_shortfall_pct": 0.0,
            "time_excess_min": 0.0,
            "max_duration_min": max_duration_min,
            "temp_excess_c": temp_excess,
            "duration_min": time_term,
            "temp_penalty": weights.temperature * temp_excess,
            "time_penalty": weights.time * time_term,
        },
    )
    return -float(loss), m
