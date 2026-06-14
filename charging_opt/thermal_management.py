"""
Thermal analysis utilities for the RW9 charging optimizer.

Level 1 — BDT temperature-aware current derating (safe inside BO loop).
Level 2 — Ambient T0 sensitivity sweeps (real BDT input).
Level 3 — Lumped thermal model for standalone design exploration only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ThermalDeratingController:
    """Reduce charging current when BDT-predicted temperature exceeds threshold."""

    t_comfort_c: float = 33.0
    t_max_c: float = 40.0
    i_min_a: float = 0.25
    derate_mode: str = "linear"

    def derate(self, target_i: float, current_temp_c: float) -> float:
        magnitude = abs(float(target_i))
        sign = -1.0 if target_i < 0 else 1.0
        if current_temp_c <= self.t_comfort_c:
            return float(target_i)
        if current_temp_c >= self.t_max_c:
            return sign * self.i_min_a
        if self.derate_mode == "linear":
            frac = (current_temp_c - self.t_comfort_c) / (self.t_max_c - self.t_comfort_c)
            derated = magnitude * (1.0 - frac) + self.i_min_a * frac
        else:
            derated = self.i_min_a if current_temp_c > 37.0 else magnitude * 0.50
        return sign * max(derated, self.i_min_a)

    def temperature_loss(
        self,
        temperature_c: np.ndarray,
        *,
        w_comfort: float = 0.5,
        w_hard: float = 5.0,
    ) -> float:
        t = np.asarray(temperature_c, dtype=np.float64)
        soft_excess = np.maximum(t - self.t_comfort_c, 0.0)
        hard_excess = np.maximum(t - self.t_max_c, 0.0)
        return float(w_comfort * np.mean(soft_excess) + w_hard * np.mean(hard_excess))

    def feasibility_check(self, temperature_c: np.ndarray) -> Dict:
        t = np.asarray(temperature_c, dtype=np.float64)
        return {
            "peak_temperature_c": float(t.max()),
            "mean_temperature_c": float(t.mean()),
            "pct_above_comfort": float(np.mean(t > self.t_comfort_c) * 100),
            "pct_above_hard_limit": float(np.mean(t > self.t_max_c) * 100),
            "temperature_feasible": bool(t.max() <= self.t_max_c),
            "temperature_comfortable": bool(t.max() <= self.t_comfort_c),
        }


def ambient_sensitivity_states(
    base_state: Dict[str, float],
    ambient_temps_c: List[float],
) -> List[Dict[str, float]]:
    out = []
    for t_amb in ambient_temps_c:
        s = dict(base_state)
        s["t0"] = float(t_amb)
        delta_t = t_amb - base_state.get("t0", 25.0)
        s["v0"] = float(base_state.get("v0", 3.711) + delta_t * 0.0005)
        out.append(s)
    return out


def compare_ambient_results(results_by_temp: Dict[float, Dict]) -> Dict:
    summary = {}
    for t_amb, results in sorted(results_by_temp.items()):
        best_fid = min(
            (fid for fid, r in results.items() if r.best_metrics.get("feasible")),
            key=lambda fid: results[fid].best_loss,
            default=None,
        )
        if best_fid is None:
            continue
        best = results[best_fid]
        m = best.best_metrics
        summary[t_amb] = {
            "best_family": best_fid,
            "best_loss": best.best_loss,
            "sei_per_pct_soc": m.get("sei_per_pct_soc"),
            "duration_min": m.get("duration_min"),
            "peak_temperature": m.get("peak_temperature"),
            "best_params": best.best_params.to_dict(),
        }
    return summary


@dataclass
class LumpedThermalModel:
    """Standalone lumped thermal model — not for use inside BDT optimization."""

    mass_g: float = 47.0
    cp_j_per_g_k: float = 0.88
    r_internal_ohm: float = 0.09
    h_conv_w_per_m2k: float = 10.0
    a_surf_m2: float = 0.0048
    t_ambient_c: float = 25.0

    @property
    def thermal_mass_j_per_k(self) -> float:
        return self.mass_g * self.cp_j_per_g_k

    def step(
        self,
        temp_c: float,
        current_a: float,
        dt_s: float = 1.0,
        h_cool_w_per_m2k: float = 0.0,
        a_cool_m2: float = 0.002,
        t_cool_c: float = 20.0,
    ) -> float:
        q_gen = current_a ** 2 * self.r_internal_ohm
        q_nat = self.h_conv_w_per_m2k * self.a_surf_m2 * (temp_c - self.t_ambient_c)
        q_cool = h_cool_w_per_m2k * a_cool_m2 * (temp_c - t_cool_c)
        dT = (q_gen - q_nat - q_cool) / self.thermal_mass_j_per_k * dt_s
        return float(temp_c + dT)

    def simulate(
        self,
        current_profile_a: np.ndarray,
        t0_c: float = 25.0,
        h_cool_w_per_m2k: float = 0.0,
        t_cool_c: float = 20.0,
        dt_s: float = 1.0,
    ) -> np.ndarray:
        n = len(current_profile_a)
        T = np.zeros(n)
        T[0] = t0_c
        for k in range(1, n):
            T[k] = self.step(
                T[k - 1], abs(current_profile_a[k - 1]), dt_s,
                h_cool_w_per_m2k, 0.002, t_cool_c,
            )
        return T

    def max_feasible_c_rate(
        self,
        t_limit_c: float = 40.0,
        t0_c: float = 25.0,
        duration_min: float = 90.0,
        h_cool_w_per_m2k: float = 0.0,
        q0_ah: float = 2.1,
        c_rate_grid: Optional[np.ndarray] = None,
    ) -> float:
        if c_rate_grid is None:
            c_rate_grid = np.arange(0.5, 4.1, 0.05)
        n = int(duration_min * 60)
        for c in c_rate_grid[::-1]:
            i = np.full(n, c * q0_ah)
            T = self.simulate(i, t0_c=t0_c, h_cool_w_per_m2k=h_cool_w_per_m2k)
            if T.max() <= t_limit_c:
                return float(c)
        return float(c_rate_grid[0])

    def cooling_required_for_c_rate(
        self,
        c_rate: float,
        t_limit_c: float = 40.0,
        t0_c: float = 25.0,
        duration_min: float = 90.0,
        q0_ah: float = 2.1,
        h_cool_grid: Optional[np.ndarray] = None,
    ) -> float:
        if h_cool_grid is None:
            h_cool_grid = np.arange(0, 501, 5, dtype=float)
        n = int(duration_min * 60)
        i = np.full(n, c_rate * q0_ah)
        for h in h_cool_grid:
            T = self.simulate(i, t0_c=t0_c, h_cool_w_per_m2k=h)
            if T.max() <= t_limit_c:
                return float(h)
        return float(h_cool_grid[-1])


def standalone_thermal_analysis(
    current_profiles: Dict[str, np.ndarray],
    thermal_model: Optional[LumpedThermalModel] = None,
    t0_c: float = 24.7,
    t_limit_c: float = 40.0,
) -> Dict[str, Dict]:
    if thermal_model is None:
        thermal_model = LumpedThermalModel(t_ambient_c=t0_c)

    results = {}
    for name, i_arr in current_profiles.items():
        i_arr = np.asarray(i_arr, dtype=np.float64)
        T_no_cool = thermal_model.simulate(i_arr, t0_c=t0_c)
        T_with_cool = thermal_model.simulate(i_arr, t0_c=t0_c, h_cool_w_per_m2k=100.0)
        q0_ah = 2.1
        h_needed = thermal_model.cooling_required_for_c_rate(
            c_rate=float(np.abs(i_arr).mean() / q0_ah) if i_arr.size else 1.0,
            t_limit_c=t_limit_c,
            t0_c=t0_c,
            duration_min=len(i_arr) / 60.0,
            q0_ah=q0_ah,
        )
        results[name] = {
            "peak_temp_no_cooling_c": float(T_no_cool.max()),
            "peak_temp_with_cooling_c": float(T_with_cool.max()),
            "h_cool_needed_w_per_m2k": h_needed,
            "feasible_without_cooling": bool(T_no_cool.max() <= t_limit_c),
            "feasible_with_liquid_cool": bool(T_with_cool.max() <= t_limit_c),
            "note": (
                "Lumped thermal model (standalone analysis). "
                "Not used in BDT optimization loop."
            ),
        }
    return results
