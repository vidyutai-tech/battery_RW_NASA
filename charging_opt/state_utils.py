"""Real battery states for charging-profile optimization."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from charging_opt.soc_utils import soc_from_ocv


def extract_rest_states(
    series,
    ocv_spline,
    max_states: int = 2000,
    v_range=(3.25, 4.15),
    i_max_abs: float = 0.02,
) -> List[Dict[str, float]]:
    """
    Charging-session start states from REAL rest samples.

    At rest the terminal voltage ~ OCV, so ``soc = ocv_to_soc(v)`` is unbiased.
    """
    comments = np.array([("rest" in str(c).lower()) for c in series.comment])
    mask = (
        comments
        & (np.abs(series.current_a) <= i_max_abs)
        & (series.voltage_v >= v_range[0])
        & (series.voltage_v <= v_range[1])
        & np.isfinite(series.temperature_c)
    )
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError("No rest samples found for state extraction")
    if idx.size > max_states:
        idx = idx[np.linspace(0, idx.size - 1, max_states).astype(int)]
    states = []
    for j in idx:
        v = float(series.voltage_v[j])
        states.append({
            "v0": v,
            "t0": float(series.temperature_c[j]),
            "age": float(series.age[j]),
            "soc": float(soc_from_ocv(ocv_spline, v)),
            "prev_i": 0.0,
        })
    return states


def pick_start_state(
    states: List[Dict[str, float]],
    *,
    age_max: float = 0.25,
    soc_range=(0.08, 0.35),
    t_range=(20.0, 32.0),
) -> Dict[str, float]:
    """Representative mid-life, low-SoC start for showcase optimization."""

    def score(s):
        return (
            s["age"] * 2.0
            + abs(s["soc"] - 0.15) * 3.0
            + max(0.0, abs(s["t0"] - 24.0) - 5.0) * 0.15
        )

    cand = [
        s for s in states
        if s["age"] < age_max
        and soc_range[0] <= s["soc"] <= soc_range[1]
        and t_range[0] <= s["t0"] <= t_range[1]
    ]
    if not cand:
        cand = [s for s in states if s["soc"] < 0.35]
    return dict(min(cand, key=score))
