"""
Stage 2 — lifetime objective for Bayesian optimization.

ENHANCEMENTS vs original:
  - chebyshev_loss(): Chebyshev scalarization (Wang & Jiang 2023, Paper 2)
    enables directed Pareto front construction by sweeping omega values.
    Unlike linear scalarization, this can recover ALL Pareto points including
    those on non-convex parts of the frontier (which your plots show exist).
  - ObjectiveMode now includes "chebyshev" as a valid mode (used when
    chebyshev_omega is passed to FamilyBayesianOptimizer).
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

ObjectiveMode = Literal["composite", "legacy", "chebyshev", "physics"]


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


def compute_voltage_stress(
    voltage_v: np.ndarray, *, v_ref: float = V_REF_STRESS
) -> Dict[str, float]:
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
    elif objective_mode == "physics":
        raise ValueError("physics loss is built in aggregate_lifetime_reward()")
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


# ---------------------------------------------------------------------------
# NEW: Chebyshev scalarization (Enhancement 4)
# ---------------------------------------------------------------------------

# Utopia / nadir values calibrated from your existing Pareto front results:
#   Fastest: duration=50.5 min, SEI=81.5
#   Lifetime: duration=104.5 min, SEI=68.0
_UTOPIA_SEI = 68.0    # best (minimum) SEI on your Pareto front
_UTOPIA_TIME = 50.5   # fastest duration on your Pareto front (min)
_NADIR_SEI = 82.0     # worst SEI among feasible profiles
_NADIR_TIME = 105.0   # slowest allowed duration (your max_duration_min)

# Physics Chebyshev scales (RW9 physics+thermal BO, pulsed family)
_UTOPIA_FADE_PCT = 1.007
_NADIR_FADE_PCT = 1.08


def chebyshev_degradation_config(
    objective_mode: str = "composite",
) -> tuple[str, float, float, float, float]:
    """Return (degradation_key, utopia_time, utopia_deg, nadir_time, nadir_deg)."""
    if objective_mode == "physics":
        return (
            "capacity_fade_pct",
            _UTOPIA_TIME,
            _UTOPIA_FADE_PCT,
            _NADIR_TIME,
            _NADIR_FADE_PCT,
        )
    return (
        "sei_per_pct_soc",
        _UTOPIA_TIME,
        _UTOPIA_SEI,
        _NADIR_TIME,
        _NADIR_SEI,
    )


def chebyshev_loss(
    metrics: Dict,
    *,
    omega: float = 0.5,
    utopia_sei: float = _UTOPIA_SEI,
    utopia_time: float = _UTOPIA_TIME,
    nadir_sei: float = _NADIR_SEI,
    nadir_time: float = _NADIR_TIME,
    degradation_key: str = "sei_per_pct_soc",
    utopia_degradation: Optional[float] = None,
    nadir_degradation: Optional[float] = None,
) -> float:
    """
    Chebyshev scalarization for directed Pareto front construction.

    Wang & Jiang (2023, J. Power Sources), Eq. 12:

        g = max(omega * |SEI - SEI*| / span_SEI,
                (1-omega) * |t - t*| / span_t)

    Unlike linear scalarization, Chebyshev can recover ALL Pareto points
    including those on non-convex parts of the frontier.

    Usage:
        omega=0.0 → minimize SEI only  (→ Lifetime profile)
        omega=1.0 → minimize time only (→ Fastest profile)
        omega=0.5 → balanced           (→ Balanced / knee profile)

    Sweep omegas ∈ {0.0, 0.1, 0.2, ..., 1.0} across BO runs to get
    uniformly distributed Pareto points, rather than mining a single
    run's history (which clusters around the single-objective optimum).

    Args:
        metrics: output of aggregate_lifetime_reward()
        omega: weight on time vs SEI trade-off [0, 1]
        utopia_sei, utopia_time: best achievable values (from prior runs)
        nadir_sei, nadir_time: worst feasible values (defines scale)

    Returns:
        scalar loss (lower = better). Returns 1e6 for infeasible metrics.
    """
    if not metrics.get("feasible", False):
        return 1e6  # let the constraint penalty structure handle infeasibility

    deg = metrics.get(degradation_key)
    if deg is None and degradation_key != "sei_per_pct_soc":
        deg = metrics.get("sei_per_pct_soc")
    t = metrics.get("duration_min")
    if deg is None or t is None or not np.isfinite(deg) or not np.isfinite(t):
        return 1e6

    utopia_deg = utopia_degradation if utopia_degradation is not None else utopia_sei
    nadir_deg = nadir_degradation if nadir_degradation is not None else nadir_sei
    span_deg = max(nadir_deg - utopia_deg, 1e-6)
    span_t = max(nadir_time - utopia_time, 1e-6)

    deg_term = abs(deg - utopia_deg) / span_deg
    t_term = abs(t - utopia_time) / span_t

    return float(max(omega * t_term, (1.0 - omega) * deg_term))


# ---------------------------------------------------------------------------
# Main aggregate function (unchanged logic, chebyshev handled upstream)
# ---------------------------------------------------------------------------

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

    Note: when objective_mode="chebyshev", the loss returned here is still
    the composite loss. The chebyshev override is applied AFTER this function
    returns, in FamilyBayesianOptimizer._evaluate() via chebyshev_loss().
    This keeps the metrics dict consistent for downstream reporting.
    """
    # Chebyshev is a post-processing override; physics uses its own loss builder.
    if objective_mode == "chebyshev":
        internal_mode: ObjectiveMode = "composite"
    elif objective_mode == "physics":
        internal_mode = "composite"
    else:
        internal_mode = objective_mode

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

    from charging_opt.physics_degradation import (
        compute_physics_degradation,
        physics_aware_loss,
    )

    age = float(session.get("initial_state", {}).get("age", 0.0))
    phys = compute_physics_degradation(
        session,
        q0_as=session.get("q_as"),
        use_paper1_sei=True,
        current_q_loss_pct=max(age * 40.0, 0.0),
    )
    m.update(
        capacity_fade_pct=phys["capacity_fade_pct"],
        capacity_fade_frac=phys["capacity_fade_frac"],
        equiv_cycles_to_eol=phys["equiv_cycles_to_eol"],
        ah_throughput=phys["ah_throughput"],
        nominal_c_rate=phys["nominal_c_rate"],
        physics_model_source=phys["model_source"],
        physics_degradation=phys,
    )
    if "sei_fade_pct_per_cycle_paper1" in phys:
        m["sei_fade_pct_per_cycle_paper1"] = phys["sei_fade_pct_per_cycle_paper1"]

    if objective_mode == "physics":
        loss, comp = physics_aware_loss(
            phys,
            duration_min=time_term,
            w_time=weights.time,
            w_vstress=weights.voltage_stress,
            w_temp=weights.temperature,
            voltage_stress_v2_min=float(m.get("voltage_stress_v2_min", 0.0)),
            temperature_penalty_c2_min=float(m.get("temperature_penalty_c2_min", 0.0)),
        )
    else:
        loss, comp = composite_loss(m, weights, objective_mode=internal_mode)

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
            "capacity_fade_pct": phys["capacity_fade_pct"],
            "equiv_cycles_to_eol": phys["equiv_cycles_to_eol"],
            **comp,
        },
    )
    return -float(loss), m
