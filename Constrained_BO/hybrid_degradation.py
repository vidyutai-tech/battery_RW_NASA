"""
Hybrid battery degradation: Q_loss = Q_calendar + Q_cyclic.

Calendar (Eq. 5) — Arrhenius, SoC- and time-dependent:
    Q_calendar = A_cal * exp(B_cal * SOC_mean)
               * exp(-(E_a,cal + C_cal * SOC_mean) / (R * T_mean,K))
               * t_cal^z_cal

Cyclic (Eq. 8) — throughput-based; C-rate-dependent B, Ea, z from Table 7:
    Q_cyclic = B_cyc * exp(-E_a,cyc / (R * T_mean,K)) * Ah^z_cyc

Inputs (SOC, T, I, t, Ah) come from the BDT electro-thermal trajectory.
Q_loss is a relative capacity-loss index for ranking profiles, not calibrated
percent fade for a specific cell.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

R_UNIVERSAL = 8.314  # J mol^-1 K^-1 (paper)
SECONDS_PER_HOUR = 3600.0
CHARGE_I_EPS_A = 0.01

# Table 7 C-rate grid for B_cyc(I), E_a,cyc(I), z_cyc(I).
TABLE7_C_RATES = np.array([0.5, 2.0, 6.0, 10.0], dtype=np.float64)
TABLE7_B = np.array([30330.0, 19300.0, 12000.0, 11500.0], dtype=np.float64)
TABLE7_EA = np.array([31500.0, 31000.0, 29500.0, 28000.0], dtype=np.float64)
TABLE7_Z = np.array([0.552, 0.554, 0.560, 0.560], dtype=np.float64)


def _interp_crate(c_rate: float, y: np.ndarray) -> float:
    """Linear interpolate vs Table-7 C-rates; clamp outside the grid."""
    return float(np.interp(abs(float(c_rate)), TABLE7_C_RATES, y))


@dataclass
class HybridDegradationParameters:
    """Calendar / cyclic coefficients and BO reward weights."""

    R: float = R_UNIVERSAL

    # Calendar Eq. (5)
    A_cal: float = 1e-3
    B_cal: float = 2.5
    C_cal: float = 3500.0
    Ea_cal: float = 32_000.0  # J/mol
    z_cal: float = 0.55

    # Reward: R = w_soc * ΔSoC - w_qloss * Q_loss - w_time * t_h^z_time
    w_soc: float = 1.0
    w_qloss: float = 1.0
    w_time: float = 0.1
    z_time: float = 0.55

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CalendarDegradation:
    """Eq. (5) calendar-aging contribution."""

    params: HybridDegradationParameters = field(default_factory=HybridDegradationParameters)

    def q_loss(
        self,
        soc: float,
        temperature_c: float,
        t_hours: float,
    ) -> float:
        """Q_calendar at mean SoC, mean T, elapsed t_cal [h]."""
        p = self.params
        soc = float(np.clip(soc, 0.0, 1.0))
        t_h = max(float(t_hours), 0.0)
        t_k = max(float(temperature_c) + 273.15, 200.0)

        arrh = np.exp(-(p.Ea_cal + p.C_cal * soc) / (p.R * t_k))
        return float(max(0.0, p.A_cal * np.exp(p.B_cal * soc) * arrh * (t_h ** p.z_cal)))

    def incremental(
        self,
        soc: float,
        temperature_c: float,
        t_hours_prev: float,
        t_hours_next: float,
    ) -> float:
        q1 = self.q_loss(soc, temperature_c, t_hours_prev)
        q2 = self.q_loss(soc, temperature_c, t_hours_next)
        return float(max(0.0, q2 - q1))


@dataclass
class CyclicDegradation:
    """Eq. (8) cyclic-aging contribution with C-rate-dependent coefficients."""

    params: HybridDegradationParameters = field(default_factory=HybridDegradationParameters)

    def coeffs_at_crate(self, c_rate: float) -> Tuple[float, float, float]:
        """Return (B_cyc, E_a,cyc, z_cyc) interpolated at the given C-rate."""
        return (
            _interp_crate(c_rate, TABLE7_B),
            _interp_crate(c_rate, TABLE7_EA),
            _interp_crate(c_rate, TABLE7_Z),
        )

    def q_loss(
        self,
        ah_throughput: float,
        temperature_c: float,
        c_rate: float,
    ) -> float:
        """Q_cyclic at cumulative Ah, mean T, and C-rate."""
        p = self.params
        ah = max(float(ah_throughput), 0.0)
        t_k = max(float(temperature_c) + 273.15, 200.0)
        b, ea, z = self.coeffs_at_crate(c_rate)
        fade = b * np.exp(-ea / (p.R * t_k)) * (ah ** z if ah > 0.0 else 0.0)
        if not np.isfinite(fade):
            return 0.0
        return float(max(0.0, fade))

    def incremental(
        self,
        ah_prev: float,
        ah_next: float,
        temperature_c: float,
        c_rate: float,
    ) -> float:
        q1 = self.q_loss(ah_prev, temperature_c, c_rate)
        q2 = self.q_loss(ah_next, temperature_c, c_rate)
        return float(max(0.0, q2 - q1))


@dataclass
class HybridDegradation:
    """Q_loss = Q_calendar + Q_cyclic."""

    params: HybridDegradationParameters = field(default_factory=HybridDegradationParameters)
    calendar: CalendarDegradation = field(init=False)
    cyclic: CyclicDegradation = field(init=False)

    def __post_init__(self) -> None:
        self.calendar = CalendarDegradation(self.params)
        self.cyclic = CyclicDegradation(self.params)


def _session_arrays(session: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    i = np.asarray(session["current_a"], dtype=np.float64)
    t_c = np.asarray(session["temperature_c"], dtype=np.float64)
    soc = np.asarray(session["soc"], dtype=np.float64)
    q_as = float(session.get("q_rated_as", session.get("q_as", 7560.0)))
    return i, t_c, soc, q_as


def session_throughput_metrics(session: Dict) -> Dict[str, float]:
    """Ah, C-rate, EFC, mean SoC/T, duration from a BDT session dict.

    - duration / mean_soc / mean_temperature: full session (calendar ages during rest).
    - ah_throughput / C-rate: charge-only samples (I < -CHARGE_I_EPS_A).
    """
    i, t_c, soc, q_as = _session_arrays(session)
    n = int(i.size)
    duration_s = float(n)
    duration_h = duration_s / SECONDS_PER_HOUR
    q_ah = q_as / SECONDS_PER_HOUR if q_as > 0 else 1.0

    charge_mask = i < -CHARGE_I_EPS_A
    abs_i = np.abs(i)
    ah = float(np.sum(abs_i[charge_mask]) / SECONDS_PER_HOUR) if charge_mask.any() else 0.0

    if charge_mask.any():
        mean_i = float(abs_i[charge_mask].mean())
        max_i = float(abs_i[charge_mask].max())
    else:
        mean_i = 0.0
        max_i = 0.0

    mean_t = float(t_c.mean()) if t_c.size else 25.0
    mean_soc = float(soc.mean()) if soc.size else 0.5

    c_rate = mean_i / q_ah if q_ah > 0 else 0.0
    max_c_rate = max_i / q_ah if q_ah > 0 else 0.0
    soc_start = float(session.get("initial_state", {}).get("soc", soc[0] if soc.size else 0.0))
    soc_end = float(soc[-1]) if soc.size else soc_start
    # One EFC = full charge + full discharge (2 × rated Ah).
    efc = ah / (2.0 * q_ah) if q_ah > 0 else 0.0
    return {
        "duration_s": duration_s,
        "duration_h": duration_h,
        "ah_throughput": ah,
        "mean_charge_current_a": mean_i,
        "max_charge_current_a": max_i,
        "nominal_c_rate": c_rate,
        "max_c_rate": max_c_rate,
        "mean_temperature_c": mean_t,
        "mean_soc": mean_soc,
        "soc_start": soc_start,
        "soc_end": soc_end,
        "delta_soc": soc_end - soc_start,
        "q_rated_ah": q_ah,
        "efc": efc,
    }


def compute_session_degradation(
    session: Dict,
    *,
    params: Optional[HybridDegradationParameters] = None,
    model: Optional[HybridDegradation] = None,
) -> Dict[str, float]:
    """Session-level Q_calendar + Q_cyclic (closed forms on mean SoC / T / Ah)."""
    params = params or HybridDegradationParameters()
    model = model or HybridDegradation(params)
    m = session_throughput_metrics(session)

    q_cal = model.calendar.q_loss(m["mean_soc"], m["mean_temperature_c"], m["duration_h"])
    q_cyc = model.cyclic.q_loss(m["ah_throughput"], m["mean_temperature_c"], m["nominal_c_rate"])
    b, ea, z = model.cyclic.coeffs_at_crate(m["nominal_c_rate"])

    out = {
        **m,
        "qloss_calendar": q_cal,
        "qloss_cyclic": q_cyc,
        "qloss_total": q_cal + q_cyc,
        "cyclic_B": b,
        "cyclic_Ea": ea,
        "cyclic_z": z,
    }
    for k, v in list(out.items()):
        if isinstance(v, float) and not np.isfinite(v):
            out[k] = 0.0
    return out


def compute_step_degradation(
    session: Dict,
    *,
    params: Optional[HybridDegradationParameters] = None,
    model: Optional[HybridDegradation] = None,
) -> Dict[str, np.ndarray]:
    """
    Per-sample incremental ΔQ via cumulative differences:

        ΔQ_cal[k] = Q_cal(t_k) - Q_cal(t_{k-1})
        ΔQ_cyc[k] = Q_cyc(Ah_k) - Q_cyc(Ah_{k-1})

    Calendar advances every sample; cyclic Ah only on charge samples.
    Scalar qloss_* totals are the session closed-form values (BO ranking).
    """
    params = params or HybridDegradationParameters()
    model = model or HybridDegradation(params)
    i, t_c, soc, q_as = _session_arrays(session)
    n = int(i.size)
    q_ah = q_as / SECONDS_PER_HOUR if q_as > 0 else 1.0

    d_q_cal = np.zeros(n, dtype=np.float64)
    d_q_cyc = np.zeros(n, dtype=np.float64)
    ah_cum = 0.0
    for k in range(n):
        dt_h = 1.0 / SECONDS_PER_HOUR
        t_prev = k * dt_h
        t_next = (k + 1) * dt_h
        soc_k = float(soc[k]) if soc.size else 0.5
        t_k = float(t_c[k]) if t_c.size else 25.0
        d_q_cal[k] = model.calendar.incremental(soc_k, t_k, t_prev, t_next)

        i_k = float(i[k])
        charging = i_k < -CHARGE_I_EPS_A
        di_ah = (abs(i_k) / SECONDS_PER_HOUR) if charging else 0.0
        ah_prev = ah_cum
        ah_cum = ah_prev + di_ah
        c_rate = (abs(i_k) / q_ah) if (charging and q_ah > 0) else 0.0
        d_q_cyc[k] = model.cyclic.incremental(ah_prev, ah_cum, t_k, c_rate)

    session_deg = compute_session_degradation(session, params=params, model=model)

    return {
        "delta_qloss_calendar": d_q_cal,
        "delta_qloss_cyclic": d_q_cyc,
        "delta_qloss_total": d_q_cal + d_q_cyc,
        "qloss_calendar": session_deg["qloss_calendar"],
        "qloss_cyclic": session_deg["qloss_cyclic"],
        "qloss_total": session_deg["qloss_total"],
        "qloss_calendar_step_sum": float(np.sum(d_q_cal)),
        "qloss_cyclic_step_sum": float(np.sum(d_q_cyc)),
        "qloss_total_step_sum": float(np.sum(d_q_cal) + np.sum(d_q_cyc)),
        "ah_throughput": session_deg["ah_throughput"],
    }


def compute_session_reward(
    session: Dict,
    *,
    params: Optional[HybridDegradationParameters] = None,
    model: Optional[HybridDegradation] = None,
) -> Dict[str, float]:
    """R = w_soc * ΔSoC - w_qloss * Q_loss - w_time * t_h^z_time."""
    params = params or HybridDegradationParameters()
    deg = compute_session_degradation(session, params=params, model=model)
    t_h = deg["duration_h"]
    time_pen = (t_h ** params.z_time) if t_h > 0.0 else 0.0
    soc_term = params.w_soc * deg["delta_soc"]
    qloss_term = params.w_qloss * deg["qloss_total"]
    time_term = params.w_time * time_pen
    total = soc_term - qloss_term - time_term
    if not np.isfinite(total):
        total = -1e6

    return {
        **deg,
        "soc_reward": float(soc_term),
        "qloss_penalty": float(qloss_term),
        "time_penalty": float(time_term),
        "time_penalty_raw": float(time_pen),
        "total_reward": float(total),
        "reward_weights": {
            "w_soc": params.w_soc,
            "w_qloss": params.w_qloss,
            "w_time": params.w_time,
            "z_time": params.z_time,
        },
        "degradation_params": {
            "A_cal": params.A_cal,
            "B_cal": params.B_cal,
            "C_cal": params.C_cal,
            "Ea_cal": params.Ea_cal,
            "z_cal": params.z_cal,
        },
    }


def compute_step_reward(
    session: Dict,
    *,
    params: Optional[HybridDegradationParameters] = None,
    model: Optional[HybridDegradation] = None,
) -> Dict[str, np.ndarray]:
    """Per-step r_t = w_soc * ΔSoC_t - w_qloss * ΔQ_t - w_time * Δ(t^z)."""
    params = params or HybridDegradationParameters()
    model = model or HybridDegradation(params)
    i, t_c, soc, q_as = _session_arrays(session)
    n = int(i.size)
    step = compute_step_degradation(session, params=params, model=model)

    r = np.zeros(n, dtype=np.float64)
    soc_r = np.zeros(n, dtype=np.float64)
    q_pen = np.zeros(n, dtype=np.float64)
    t_pen = np.zeros(n, dtype=np.float64)
    soc0 = float(session.get("initial_state", {}).get("soc", soc[0] if soc.size else 0.0))
    prev_soc = soc0
    for k in range(n):
        soc_k = float(soc[k]) if soc.size else prev_soc
        d_soc = soc_k - prev_soc
        prev_soc = soc_k
        t_prev_h = k / SECONDS_PER_HOUR
        t_next_h = (k + 1) / SECONDS_PER_HOUR
        d_tz = (t_next_h ** params.z_time) - (t_prev_h ** params.z_time)
        soc_r[k] = params.w_soc * d_soc
        q_pen[k] = params.w_qloss * float(step["delta_qloss_total"][k])
        t_pen[k] = params.w_time * max(0.0, d_tz)
        r[k] = soc_r[k] - q_pen[k] - t_pen[k]

    return {
        "r_t": r,
        "soc_reward_t": soc_r,
        "qloss_penalty_t": q_pen,
        "time_penalty_t": t_pen,
        "total_reward": float(np.sum(r)),
        **step,
    }


# Aliases
compute_session_loss = compute_session_reward
compute_step_loss = compute_step_reward
