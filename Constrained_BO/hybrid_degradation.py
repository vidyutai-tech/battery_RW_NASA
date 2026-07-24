"""
Hybrid calendar + cyclic degradation models for Constrained_BO rewards.

Terminology note (IMPORTANT): none of the coefficients below have been calibrated
against NASA RW9-RW12 (or LFP) aging data. They are semi-empirical fits taken from
other cells/chemistries in the cited literature. Consequently every Q_loss produced
by this module is a **Relative Capacity-Loss Index** — a dimensionless, physically
motivated stress score useful for *ranking* charging strategies against one another
under the *same* model — and must NOT be reported or interpreted as an absolute
"Capacity Fade (%)" of any specific cell. See README "Hybrid degradation
methodology" section for the full caveat.

Reference: Sinha, S. S., Lehman, B., and Bharadwaj, P. (2026). "Life Extension of
Lithium-Ion Battery: Degradation Comprehension, Modeling, Characterization, and
Mitigation Strategies." IEEE Open Journal of Power Electronics, vol. 7.
DOI: 10.1109/OJPEL.2025.3639205.

Default cyclic model: Eq. (7) / Table 7 — interpolate B(I), Ea(I), z(I) vs C-rate,
then
    Q_cyclic = B(I) * exp(-Ea / (R * T_K)) * Ah^z

Default calendar model: Eq. (2)
    Q_calendar = A * exp(B * SoC) * exp(-(Ea + C * SoC) / (R * T_K)) * t_h^z

Optional calendar models: Eq. (3), Eq. (5) (selectable via
``HybridDegradationParameters.calendar_model``; alternatives for sensitivity
analysis only — never summed with Eq. (2)).

Optional cyclic model: Eq. (8) (continuous current-dependent form, gated behind
``use_alpha_current``; off by default because its α coefficient is not
chemistry-calibrated here — see README limitations).

Arrhenius uses absolute temperature in Kelvin (physically consistent). Time for
power-law aging is in **hours**. Ah is absolute ampere-hour throughput.
SoC is fractional [0, 1]. See ``RESULT_METRIC_UNITS`` in objective.py for the
full units table of every exported metric.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Literal, Optional, Tuple

import numpy as np

R_UNIVERSAL = 8.314462618  # J/mol/K
T_REF_K = 298.15
SECONDS_PER_HOUR = 3600.0
CHARGE_I_EPS_A = 0.01

CalendarModel = Literal["eq2", "eq3", "eq5"]

# Table 7, Sinha et al. (2026), "Life Extension of Lithium-Ion Battery: Degradation
# Comprehension, Modeling, Characterization, and Mitigation Strategies," IEEE Open
# Journal of Power Electronics, vol. 7 (citing [112]). Verbatim per-C-rate fits of
# Eq. (7): Q_loss = B * exp(-Ea / (R*T)) * Ah^z.
#
# NOTE (known literature non-monotonicity): the published B, Ea pairs trade off such
# that, at fixed Ah and T, Q_cyclic(C/2) > Q_cyclic(2C) — i.e. the interpolated curve
# dips between C/2 and 2C before rising monotonically from 2C through 10C. This is a
# property of the source curve fits (not a bug in this implementation); see
# Constrained_BO/tests/test_hybrid_degradation.py for the monotonic region tested.
TABLE7_C_RATES = np.array([0.5, 2.0, 6.0, 10.0], dtype=np.float64)
TABLE7_B = np.array([30330.0, 19300.0, 12000.0, 11500.0], dtype=np.float64)
TABLE7_EA = np.array([31500.0, 31000.0, 29500.0, 28000.0], dtype=np.float64)
TABLE7_Z = np.array([0.552, 0.554, 0.560, 0.560], dtype=np.float64)


def _interp_crate(c_rate: float, y: np.ndarray) -> float:
    """Linear interpolate vs Table-7 C-rates; clamp outside the grid."""
    c = abs(float(c_rate))
    return float(np.interp(c, TABLE7_C_RATES, y))



@dataclass
class HybridDegradationParameters:
    """Tunable hybrid degradation + reward weights."""

    # Universal
    R: float = R_UNIVERSAL
    T_ref_K: float = T_REF_K

    # Calendar Eq (2) stubs (recalibrate as needed)
    A: float = 1e-3
    B: float = 2.5
    C: float = 3500.0
    Ea_cal: float = 32_000.0  # J/mol
    z_cal: float = 0.55
    calendar_model: CalendarModel = "eq2"

    # Optional calendar Eq (5)
    eq5_alpha: float = 2.14e4
    eq5_Ea: float = 36_360.0  # J/mol (36.36 kJ/mol)
    eq5_z: float = 0.789

    # Optional cyclic α|I| form (off by default; Table 7 is primary)
    use_alpha_current: bool = False
    Ea_cyc_base: float = 31_500.0
    alpha_I: float = 0.0
    B_fallback: float = 21_681.0
    z_cyc_fallback: float = 0.55

    # Reward weights: R = w1*ΔSoC - w2*Q_total - w3*t_h^z
    w_soc: float = 1.0
    w_qloss: float = 1.0
    w_time: float = 0.1
    z_time: float = 0.55

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["calendar_model"] = str(self.calendar_model)
        return d


@dataclass
class CalendarDegradation:
    params: HybridDegradationParameters = field(default_factory=HybridDegradationParameters)

    def q_loss(
        self,
        soc: float,
        temperature_c: float,
        t_hours: float,
    ) -> float:
        """Cumulative calendar fade at elapsed ``t_hours``."""
        p = self.params
        soc = float(np.clip(soc, 0.0, 1.0))
        t_h = max(float(t_hours), 0.0)
        t_k = float(temperature_c) + 273.15
        t_k = max(t_k, 200.0)  # numerical floor

        if p.calendar_model == "eq3":
            # Empirical form; T in °C (coeff scale).
            term_soc = 0.019 * (soc ** 0.823) + 0.5195
            term_t = 3.258e-9 * (float(temperature_c) ** 5.087) + 0.295
            return float(max(0.0, term_soc * term_t * (t_h ** 0.8)))

        if p.calendar_model == "eq5":
            arrh = np.exp(-p.eq5_Ea / (p.R * t_k))
            return float(max(0.0, p.eq5_alpha * arrh * (t_h ** p.eq5_z)))

        # Eq (2)
        arrh = np.exp(-(p.Ea_cal + p.C * soc) / (p.R * t_k))
        return float(max(0.0, p.A * np.exp(p.B * soc) * arrh * (t_h ** p.z_cal)))

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
    params: HybridDegradationParameters = field(default_factory=HybridDegradationParameters)

    def coeffs_at_crate(self, c_rate: float) -> Tuple[float, float, float]:
        c = abs(float(c_rate))
        p = self.params
        if p.use_alpha_current:
            b = _interp_crate(c, TABLE7_B)
            ea = p.Ea_cyc_base + p.alpha_I * c
            z = p.z_cyc_fallback
            return b, ea, z
        return (
            _interp_crate(c, TABLE7_B),
            _interp_crate(c, TABLE7_EA),
            _interp_crate(c, TABLE7_Z),
        )

    def q_loss(
        self,
        ah_throughput: float,
        temperature_c: float,
        c_rate: float,
    ) -> float:
        """Cumulative cyclic fade at cumulative ``ah_throughput``."""
        p = self.params
        ah = max(float(ah_throughput), 0.0)
        t_k = max(float(temperature_c) + 273.15, 200.0)
        b, ea, z = self.coeffs_at_crate(c_rate)
        arrh = np.exp(-ea / (p.R * t_k))
        fade = b * arrh * (ah ** z if ah > 0.0 else 0.0)
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
    """Ah, mean/max charge C-rate, EFC, mean SoC/T, duration hours from a session dict.

    Units: duration_s [s], duration_h [h], ah_throughput [Ah],
    mean_charge_current_a [A], nominal_c_rate / max_c_rate [dimensionless, 1/h
    convention i.e. multiples of the rated 1C current], mean_temperature_c [degC],
    mean_soc / soc_start / soc_end / delta_soc [fraction, 0-1], q_rated_ah [Ah],
    efc [equivalent full cycles, dimensionless].
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
        mean_t = float(t_c[charge_mask].mean()) if t_c.size else 25.0
        mean_soc = float(soc[charge_mask].mean()) if soc.size else 0.5
    else:
        mean_i = 0.0
        max_i = 0.0
        mean_t = float(t_c.mean()) if t_c.size else 25.0
        mean_soc = float(soc.mean()) if soc.size else 0.5

    c_rate = mean_i / q_ah if q_ah > 0 else 0.0
    max_c_rate = max_i / q_ah if q_ah > 0 else 0.0
    soc_start = float(session.get("initial_state", {}).get("soc", soc[0] if soc.size else 0.0))
    soc_end = float(soc[-1]) if soc.size else soc_start
    # EFC convention: one Equivalent Full Cycle = a full charge + full discharge
    # (2 x rated Ah of throughput). A charge-only session therefore contributes at
    # most 0.5 EFC when it moves the cell across its entire rated capacity.
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
    """Session-level Q_calendar + Q_cyclic (cumulative closed forms)."""
    params = params or HybridDegradationParameters()
    model = model or HybridDegradation(params)
    m = session_throughput_metrics(session)

    q_cal = model.calendar.q_loss(m["mean_soc"], m["mean_temperature_c"], m["duration_h"])
    q_cyc = model.cyclic.q_loss(m["ah_throughput"], m["mean_temperature_c"], m["nominal_c_rate"])
    q_tot = q_cal + q_cyc
    b, ea, z = model.cyclic.coeffs_at_crate(m["nominal_c_rate"])

    out = {
        **m,
        "qloss_calendar": q_cal,
        "qloss_cyclic": q_cyc,
        "qloss_total": q_tot,
        "cyclic_B": b,
        "cyclic_Ea": ea,
        "cyclic_z": z,
        "calendar_model": str(params.calendar_model),
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
    Per-sample incremental ΔQ using cumulative differences:

        ΔQ_cal[k] = Q_cal(t_k) - Q_cal(t_{k-1})
        ΔQ_cyc[k] = Q_cyc(Ah_k) - Q_cyc(Ah_{k-1})

    with local SoC / T / C-rate at step k (avoids summing t^z independently).
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

        di_ah = abs(float(i[k])) / SECONDS_PER_HOUR
        ah_prev = ah_cum
        ah_cum = ah_prev + di_ah
        c_rate = (abs(float(i[k])) / q_ah) if q_ah > 0 else 0.0
        d_q_cyc[k] = model.cyclic.incremental(ah_prev, ah_cum, t_k, c_rate)

    return {
        "delta_qloss_calendar": d_q_cal,
        "delta_qloss_cyclic": d_q_cyc,
        "delta_qloss_total": d_q_cal + d_q_cyc,
        "qloss_calendar": float(np.sum(d_q_cal)),
        "qloss_cyclic": float(np.sum(d_q_cyc)),
        "qloss_total": float(np.sum(d_q_cal) + np.sum(d_q_cyc)),
    }


def compute_session_reward(
    session: Dict,
    *,
    params: Optional[HybridDegradationParameters] = None,
    model: Optional[HybridDegradation] = None,
) -> Dict[str, float]:
    """
    R = w1 * ΔSoC - w2 * Q_loss_total - w3 * t_h^z
    """
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
            "A": params.A,
            "B": params.B,
            "C": params.C,
            "Ea_cal": params.Ea_cal,
            "z_cal": params.z_cal,
            "calendar_model": params.calendar_model,
        },
    }


def compute_step_reward(
    session: Dict,
    *,
    params: Optional[HybridDegradationParameters] = None,
    model: Optional[HybridDegradation] = None,
) -> Dict[str, np.ndarray]:
    """
    Per-step r_t = w1 * ΔSoC_t - w2 * ΔQ_t - w3 * Δ(t^z).

    Time term uses cumulative power-law increments: t_{k}^z - t_{k-1}^z.
    """
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


# Aliases matching the requested API names
compute_session_loss = compute_session_reward
compute_step_loss = compute_step_reward
