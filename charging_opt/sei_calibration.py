"""
Calibrate Arrhenius k for the SEI proxy from measured capacity fade (RW9).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from charging_opt.reward import DEFAULT_K_ARRHENIUS


def calibrate_arrhenius_k(
    capacity_table: dict,
    *,
    nominal_temperature_c: float = 25.0,
    k_bounds: Tuple[float, float] = (0.01, 0.15),
) -> Tuple[float, Dict]:
    """
    Fit k so a simple SEI proxy tracks normalized capacity vs relative age.

    Returns (k_opt, fit_info). Falls back to DEFAULT_K_ARRHENIUS if fit fails.

    WARNING: This calibration is non-functional. The residual function's
    exponent is always zero because it computes k*(T_nom - T_nom). The SEI
    proxy k is not calibrated from data. Use the Wang ΔQ/Q₀ model
    (physics_degradation.py) as the primary degradation metric instead.
    """
    age = np.asarray(capacity_table["age"], dtype=np.float64)
    q_full = np.asarray(capacity_table["q_full_as"], dtype=np.float64)

    valid = np.isfinite(q_full) & np.isfinite(age) & (q_full > 0)
    if valid.sum() < 3:
        return DEFAULT_K_ARRHENIUS, {
            "k_opt": DEFAULT_K_ARRHENIUS,
            "default_k": DEFAULT_K_ARRHENIUS,
            "fit_residual": None,
            "n_samples": int(valid.sum()),
            "note": "insufficient capacity fade samples",
        }

    age_v = age[valid]
    q_norm = q_full[valid] / q_full[valid][0]
    log_q = np.log(np.clip(q_norm, 1e-3, 1.0))

    def residual(k: float) -> float:
        sei_proxy = age_v * np.exp(k * (nominal_temperature_c - nominal_temperature_c))
        A = sei_proxy.reshape(-1, 1)
        coef = np.linalg.lstsq(A, log_q, rcond=None)[0][0]
        preds = coef * sei_proxy
        return float(np.mean((log_q - preds) ** 2))

    result = minimize_scalar(residual, bounds=k_bounds, method="bounded")
    k_opt = float(result.x)

    return k_opt, {
        "k_opt": k_opt,
        "default_k": DEFAULT_K_ARRHENIUS,
        "fit_residual": float(result.fun),
        "n_samples": int(valid.sum()),
    }
