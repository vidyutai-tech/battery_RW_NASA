"""BDT-derived thermal metrics and temperature reward from predicted trajectories.

All quantities are computed from the closed-loop BDT rollout ``(T(t), I(t))`` at 1 Hz.
They intentionally use **charge-phase** statistics so long idle rest at low temperature
does not inflate the temperature score (unlike a naive session mean).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

TEMP_PLATEAU = 1.5
TEMP_FLOOR = -2.2

CHARGE_I_EPS_A = 0.02

# Reward mapping: penalise BDT-predicted thermal stress (tune to °C / °C/s / °·s/kJ scales).
PEAK_RISE_WEIGHT = 0.50       # per °C above t0 during charge
HEATING_RATE_WEIGHT = 3.50    # per °C/s max step rise during charge (transient stress)
STEM_WEIGHT = 0.003           # per (°C·s/kJ) charge-phase exposure per delivered J
REST_COOLING_WEIGHT = 0.30    # bonus per °C recovered across a rest interval

# Component blend (sums to 1). Heating rate weighted highest: pulse design primarily
# limits transient BDT-predicted dT/dt, not session-mean temperature.
W_PEAK = 0.20
W_RATE = 0.65
W_STEM = 0.10
W_COOL = 0.05


def _charge_mask(current_a: np.ndarray) -> np.ndarray:
    return np.asarray(current_a, dtype=np.float64) < -CHARGE_I_EPS_A


def _rest_mask(current_a: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(current_a, dtype=np.float64)) <= CHARGE_I_EPS_A


def _energy_delivered_j(voltage_v, current_a, time_s) -> float:
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


def bdt_thermal_metrics(session: Dict) -> Dict[str, float]:
    """
    Extract scientifically interpretable thermal quantities from a BDT session.

    Returns
    -------
    dT_peak
        ``max(T) - t0`` over the full rollout (worst-case rise above start anchor).
    dT_peak_charge
        ``max(T | charging) - t0`` — peak rise during applied charge current only.
    dT_dt_max_charge
        Maximum single-step ``ΔT/Δt`` (°C/s at 1 Hz) while charging; captures
        aggressive transient heating from the BDT.
    stem_charge_per_kj
        Specific Thermal Exposure during charge: ``∫ max(0, T-t0) dt / E`` over
        charge intervals only, in °C·s per kJ delivered.
    rest_cooling_mean
        Mean temperature drop ``T_start_rest - T_end_rest`` across rest intervals
        (°C); positive when the BDT predicts cooling during zero-current rest.
  charge_fraction
        Fraction of session seconds with ``|I| > eps`` (duty cycle).
    """
    T = np.asarray(session["temperature_c"], dtype=np.float64)
    I = np.asarray(session["current_a"], dtype=np.float64)
    t = np.asarray(session["time_s"], dtype=np.float64)
    t0 = float(session["initial_state"].get("t0", float(T[0]) if T.size else 25.0))

    if T.size == 0:
        return {
            "t0_c": t0,
            "dT_peak": 0.0,
            "dT_peak_charge": 0.0,
            "dT_dt_max_charge": 0.0,
            "stem_charge_per_kj": 0.0,
            "rest_cooling_mean": 0.0,
            "charge_fraction": 0.0,
            "energy_delivered_j": 0.0,
        }

    charge = _charge_mask(I)
    rest = _rest_mask(I)
    energy_j = _energy_delivered_j(session["voltage_v"], I, t)

    dT_peak = float(T.max() - t0)
    dT_peak_charge = float(T[charge].max() - t0) if charge.any() else dT_peak

    dT = np.diff(T)
    charge_step = charge[:-1] & charge[1:]
    if charge_step.any():
        dT_dt_max_charge = float(np.max(dT[charge_step]))
    elif charge[:-1].any():
        dT_dt_max_charge = float(np.max(dT[charge[:-1]]))
    else:
        dT_dt_max_charge = 0.0

    if charge.sum() > 1:
        t_ch, T_ch = t[charge], T[charge]
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
        stem_charge = float(trapz(np.maximum(0.0, T_ch - t0), t_ch))
    else:
        stem_charge = float(np.maximum(0.0, T[charge] - t0).sum()) if charge.any() else 0.0
    stem_charge_per_kj = stem_charge / max(energy_j, 1.0) * 1000.0

    rest_coolings: list[float] = []
    in_rest = False
    t_enter = T[0]
    for i in range(len(I)):
        if rest[i] and not in_rest:
            in_rest = True
            t_enter = T[i - 1] if i > 0 else T[i]
        elif not rest[i] and in_rest:
            in_rest = False
            rest_coolings.append(float(t_enter - T[i - 1]))
    rest_cooling_mean = float(np.mean(rest_coolings)) if rest_coolings else 0.0

    return {
        "t0_c": t0,
        "dT_peak": dT_peak,
        "dT_peak_charge": dT_peak_charge,
        "dT_dt_max_charge": dT_dt_max_charge,
        "stem_charge_per_kj": stem_charge_per_kj,
        "rest_cooling_mean": rest_cooling_mean,
        "charge_fraction": float(charge.mean()),
        "energy_delivered_j": energy_j,
    }


def _clip_reward(r: float) -> float:
    return float(max(TEMP_FLOOR, min(TEMP_PLATEAU, r)))


def temperature_reward_from_bdt(session: Dict) -> Tuple[float, Dict[str, float]]:
    """
    Map BDT thermal metrics to a scalar reward in ``[-2.2, 1.5]``.

    Components (equally weighted):
      - penalise charge-phase peak rise above ``t0``;
      - penalise max BDT heating rate during charge;
      - penalise energy-normalised charge thermal exposure (STEM);
      - reward inter-pulse cooling during rest intervals.
    """
    m = bdt_thermal_metrics(session)

    r_peak = _clip_reward(
        TEMP_PLATEAU - PEAK_RISE_WEIGHT * max(0.0, m["dT_peak_charge"]),
    )
    r_rate = _clip_reward(
        TEMP_PLATEAU - HEATING_RATE_WEIGHT * max(0.0, m["dT_dt_max_charge"]),
    )
    r_stem = _clip_reward(
        TEMP_PLATEAU - STEM_WEIGHT * m["stem_charge_per_kj"],
    )
    r_cool = _clip_reward(
        TEMP_PLATEAU + REST_COOLING_WEIGHT * max(0.0, m["rest_cooling_mean"]),
    )

    total = float(
        W_PEAK * r_peak + W_RATE * r_rate + W_STEM * r_stem + W_COOL * r_cool
    )
    components = {
        "r_peak_charge": r_peak,
        "r_heating_rate": r_rate,
        "r_stem_charge": r_stem,
        "r_rest_cooling": r_cool,
        **m,
    }
    return total, components
