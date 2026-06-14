"""
Physics-grounded battery degradation models for charging optimization.

1. WangCapacityFade — empirical power-law fade (Wang et al. 2011), calibrated
   from NASA RW9 measured capacity fade when available.
2. sei_growth_paper1 — simplified SEI growth (Padisala et al. 2025, Paper 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

from charging_opt.artifacts import CANONICAL

R_UNIVERSAL = 8.314
GAMMA = 0.55
T_REF_K = 298.15

PAPER4_B1 = {
    0.5: 31_630,
    2.0: 21_681,
    6.0: 12_934,
    10.0: 15_512,
}

_SEI_COEFFS = {
    10: {"a": 0.01058, "b": 4.577, "c": 0.03375},
    20: {"a": 0.01236, "b": 3.587, "c": 0.03640},
    30: {"a": 0.01398, "b": 2.938, "c": 0.03766},
    40: {"a": 0.01566, "b": 2.441, "c": 0.03806},
}

_DEGRADATION_MODEL: Optional["WangCapacityFade"] = None


def ea1_from_crate(c_rate: float) -> float:
    """Positive activation energy (J/mol); decreases mildly with C-rate."""
    return max(18_000.0, 38_000.0 - 400.0 * float(abs(c_rate)))


@dataclass
class WangCapacityFade:
    """ΔQ(Ah, T, I) = B1(I) · exp(−Ea1(I) / R·T) · Ah^γ"""

    _b1_interp: object = field(repr=False, default=None)
    calibrated_from_data: bool = False
    calibration_info: Dict = field(default_factory=dict)

    @classmethod
    def from_paper4(cls) -> "WangCapacityFade":
        c_rates = np.array(sorted(PAPER4_B1.keys()), dtype=float)
        b1_vals = np.array([PAPER4_B1[c] for c in c_rates], dtype=float)
        interp = interp1d(
            c_rates, b1_vals, kind="linear",
            bounds_error=False, fill_value=(b1_vals[0], b1_vals[-1]),
        )
        return cls(
            _b1_interp=interp,
            calibrated_from_data=False,
            calibration_info={"source": "paper4_abbasi2024"},
        )

    @classmethod
    def from_saved(cls, path: str | Path) -> "WangCapacityFade":
        path = Path(path)
        data = np.load(path)
        c_rates = np.asarray(data["c_rates"], dtype=float)
        b1_vals = np.asarray(data["b1_vals"], dtype=float)
        interp = interp1d(
            c_rates, b1_vals, kind="linear",
            bounds_error=False, fill_value=(b1_vals[0], b1_vals[-1]),
        )
        info = {"source": str(data["source"][0]) if "source" in data else "saved_npz"}
        for key in ("b1_eff_at_25c", "gamma", "r2", "n_points", "q0_ah"):
            if key in data:
                val = data[key]
                info[key] = float(val[0]) if np.ndim(val) else float(val)
        calibrated = bool(data["calibrated"][0]) if "calibrated" in data else True
        return cls(_b1_interp=interp, calibrated_from_data=calibrated, calibration_info=info)

    @classmethod
    def from_rw9(
        cls,
        capacity_fade_path: str | Path | None = None,
    ) -> "WangCapacityFade":
        if capacity_fade_path is None:
            capacity_fade_path = CANONICAL["capacity_fade"]
        path = Path(capacity_fade_path)
        if not path.is_file():
            print(f"  [WangCapacityFade] {path} not found — using Paper 4 coefficients.")
            return cls.from_paper4()

        data = np.load(path)
        age = np.asarray(data["age"], dtype=float)
        q = np.asarray(data["q_full_as"], dtype=float)

        q0 = float(np.interp(0.0, age, q))
        delta_q_frac = np.maximum((q0 - q) / q0, 1e-6)
        q0_ah = q0 / 3600.0
        ah_throughput = age * 1750.0 * q0_ah

        valid = (ah_throughput > 0) & np.isfinite(delta_q_frac)
        ah_v = ah_throughput[valid]
        dq_v = delta_q_frac[valid]

        def model(ah: np.ndarray, b1_eff: float) -> np.ndarray:
            return b1_eff * np.power(ah, GAMMA)

        try:
            popt, _ = curve_fit(
                model, ah_v, dq_v, p0=[1e-4], bounds=(0.0, 1e-1), maxfev=5000,
            )
            b1_eff = float(popt[0])
            residuals = dq_v - model(ah_v, b1_eff)
            r2 = 1.0 - np.sum(residuals ** 2) / np.sum((dq_v - dq_v.mean()) ** 2)
            info = {
                "source": "rw9_calibrated",
                "b1_eff_at_25c": b1_eff,
                "gamma": GAMMA,
                "r2": float(r2),
                "n_points": int(valid.sum()),
                "q0_ah": q0_ah,
            }
            print(f"  [WangCapacityFade] Calibrated from RW9: B1_eff={b1_eff:.4e}, R²={r2:.3f}")
        except RuntimeError:
            print("  [WangCapacityFade] Curve fit failed — using Paper 4 coefficients.")
            return cls.from_paper4()

        c_rates = np.array(sorted(PAPER4_B1.keys()), dtype=float)
        b1_paper4 = np.array([PAPER4_B1[c] for c in c_rates], dtype=float)
        b1_scaled = b1_eff * b1_paper4 / PAPER4_B1[2.0]
        interp = interp1d(
            c_rates, b1_scaled, kind="linear",
            bounds_error=False, fill_value=(b1_scaled[0], b1_scaled[-1]),
        )
        return cls(_b1_interp=interp, calibrated_from_data=True, calibration_info=info)

    def b1(self, c_rate: float) -> float:
        if self._b1_interp is None:
            return float(PAPER4_B1.get(2.0, 21_681))
        return float(self._b1_interp(abs(c_rate)))

    def capacity_fade_fraction(
        self,
        ah_throughput: float,
        mean_temperature_c: float,
        c_rate: float,
    ) -> float:
        b1 = self.b1(abs(c_rate))
        fade = b1 * (max(ah_throughput, 1e-9) ** GAMMA)
        if abs(mean_temperature_c - 25.0) > 0.25:
            ea1 = ea1_from_crate(c_rate)
            t_k = mean_temperature_c + 273.15
            ref = np.exp(-ea1 / (R_UNIVERSAL * T_REF_K))
            arrh = np.exp(-ea1 / (R_UNIVERSAL * t_k)) / max(ref, 1e-12)
            fade *= arrh
        return float(np.clip(fade, 0.0, 0.5))

    def equivalent_cycles_to_eol(
        self,
        ah_per_session: float,
        mean_temperature_c: float,
        c_rate: float,
        eol_threshold: float = 0.20,
    ) -> float:
        fade_per_session = self.capacity_fade_fraction(
            ah_per_session, mean_temperature_c, c_rate,
        )
        if fade_per_session <= 0:
            return float("inf")
        return eol_threshold / fade_per_session


def get_degradation_model(
    *,
    capacity_fade_path: str | Path | None = None,
    degradation_model_path: str | Path | None = None,
    reload: bool = False,
) -> WangCapacityFade:
    global _DEGRADATION_MODEL
    if _DEGRADATION_MODEL is not None and not reload:
        return _DEGRADATION_MODEL

    if degradation_model_path is None:
        degradation_model_path = CANONICAL["degradation_model"]
    saved = Path(degradation_model_path)
    if saved.is_file():
        _DEGRADATION_MODEL = WangCapacityFade.from_saved(saved)
    else:
        _DEGRADATION_MODEL = WangCapacityFade.from_rw9(capacity_fade_path)
    return _DEGRADATION_MODEL


def sei_growth_paper1(q_loss_pct: float, i_c: float) -> float:
    q_loss_pct = float(np.clip(q_loss_pct, 10, 40))
    levels = np.array([10, 20, 30, 40], dtype=float)
    i_c = float(abs(i_c))
    a_vals = np.array([_SEI_COEFFS[int(l)]["a"] for l in levels])
    b_vals = np.array([_SEI_COEFFS[int(l)]["b"] for l in levels])
    c_vals = np.array([_SEI_COEFFS[int(l)]["c"] for l in levels])
    a = float(np.interp(q_loss_pct, levels, a_vals))
    b = float(np.interp(q_loss_pct, levels, b_vals))
    c = float(np.interp(q_loss_pct, levels, c_vals))
    return float(a * (i_c ** b) + c)


def compute_physics_degradation(
    session: Dict,
    *,
    model: Optional[WangCapacityFade] = None,
    q0_as: Optional[float] = None,
    use_paper1_sei: bool = False,
    current_q_loss_pct: float = 0.0,
) -> Dict:
    if model is None:
        model = get_degradation_model()

    i_arr = np.asarray(session["current_a"], dtype=np.float64)
    t_arr = np.asarray(session["temperature_c"], dtype=np.float64)

    charge_current = np.abs(i_arr)
    charging_mask = charge_current > 0.01
    ah_throughput = float(np.sum(charge_current) / 3600.0)

    if charging_mask.any():
        mean_temp = float(t_arr[charging_mask].mean())
        mean_i = float(charge_current[charging_mask].mean())
    else:
        mean_temp = float(t_arr.mean()) if t_arr.size else 25.0
        mean_i = 0.0

    q0 = float(q0_as) if q0_as is not None else float(session.get("q_as", 7560.0))
    q0_ah = q0 / 3600.0
    nominal_c_rate = mean_i / q0_ah if q0_ah > 0 else 1.0

    fade_frac = model.capacity_fade_fraction(ah_throughput, mean_temp, nominal_c_rate)
    equiv_cyc = model.equivalent_cycles_to_eol(ah_throughput, mean_temp, nominal_c_rate)

    result = {
        "ah_throughput": ah_throughput,
        "mean_temperature_c": mean_temp,
        "nominal_c_rate": nominal_c_rate,
        "capacity_fade_frac": fade_frac,
        "capacity_fade_pct": fade_frac * 100.0,
        "equiv_cycles_to_eol": equiv_cyc,
        "model_source": model.calibration_info.get("source", "unknown"),
    }

    if use_paper1_sei:
        result["sei_fade_pct_per_cycle_paper1"] = sei_growth_paper1(
            q_loss_pct=max(current_q_loss_pct, 10.0),
            i_c=nominal_c_rate,
        )
    return result


def physics_aware_loss(
    physics_metrics: Dict,
    *,
    w_fade: float = 50.0,
    w_time: float = 0.02,
    w_vstress: float = 0.08,
    w_temp: float = 0.05,
    voltage_stress_v2_min: float = 0.0,
    temperature_penalty_c2_min: float = 0.0,
    duration_min: float = 0.0,
) -> Tuple[float, Dict]:
    fade_term = w_fade * physics_metrics["capacity_fade_frac"]
    time_term = w_time * duration_min
    v_term = w_vstress * voltage_stress_v2_min
    t_term = w_temp * temperature_penalty_c2_min
    loss = fade_term + time_term + v_term + t_term
    components = {
        "objective_mode": "physics_degradation",
        "fade_term": fade_term,
        "time_term": time_term,
        "voltage_stress_term": v_term,
        "temperature_term": t_term,
        "capacity_fade_pct": physics_metrics["capacity_fade_pct"],
        "equiv_cycles_to_eol": physics_metrics.get("equiv_cycles_to_eol"),
        "model_source": physics_metrics.get("model_source"),
    }
    return float(loss), components


def calibrate_and_save(
    capacity_fade_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> WangCapacityFade:
    if capacity_fade_path is None:
        capacity_fade_path = CANONICAL["capacity_fade"]
    if out_path is None:
        out_path = CANONICAL["degradation_model"]

    model = WangCapacityFade.from_rw9(capacity_fade_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    c_test = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
    b1_vals = np.array([model.b1(c) for c in c_test])
    extra = {
        k: np.array([v])
        for k, v in model.calibration_info.items()
        if isinstance(v, (int, float, bool, str)) and k != "gamma"
    }
    np.savez(
        out,
        c_rates=c_test,
        b1_vals=b1_vals,
        gamma=np.array([GAMMA]),
        calibrated=np.array([model.calibrated_from_data]),
        **extra,
    )
    print(f"Degradation model saved -> {out}")
    print(f"  Calibrated from data: {model.calibrated_from_data}")
    print(f"  Source: {model.calibration_info.get('source')}")
    if "r2" in model.calibration_info:
        print(f"  R² = {model.calibration_info['r2']:.4f}")

    global _DEGRADATION_MODEL
    _DEGRADATION_MODEL = model
    return model


if __name__ == "__main__":
    print("=== Physics Degradation Model Self-Test ===\n")
    model = get_degradation_model(reload=True)
    for c in [0.5, 1.0, 1.5, 2.0, 3.0]:
        fade = model.capacity_fade_fraction(2.0, 25.0, c)
        cyc = model.equivalent_cycles_to_eol(2.0, 25.0, c)
        print(f"  C={c:.1f}C  fade/session={fade * 100:.4f}%  EOL cycles≈{cyc:.0f}")
