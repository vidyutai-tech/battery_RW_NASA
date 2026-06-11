"""
Reward function for charging-profile optimization.

    r = Delta_SoC(pct points) - lambda * SEI_proxy        (per simulated segment)

Design decisions (differ deliberately from a naive OCV-inversion reward):

* **Delta-SoC by coulomb counting, not OCV inversion of loaded voltage.**
  During charge the terminal voltage sits above OCV by the IR drop
  (~0.1-0.5 V at the policy's setpoints), so inverting V(t) through the OCV
  curve would systematically over-credit aggressive currents. Inside a
  simulated rollout the current is the action and is known exactly:
      Delta_SoC = integral(-I dt) / Q(age)        (NASA RW: I < 0 = charge)
  with the age-dependent capacity Q(age) from the reference discharges
  (RW9 fades 2.2 -> 0.85 Ah, so a fixed Q would mis-scale by up to ~2.6x).

* **Reaching the voltage ceiling is NOT a violation.** It is the natural end
  of the CC phase; the simulator truncates the trajectory there (and the
  truncated trajectory is what this function scores). The *chance-constrained*
  ceiling (v_max minus the p95 open-loop drift margin) is enforced by the
  caller through the truncation threshold.

* **Temperature is the punished constraint.** If T_hat + t_margin exceeds
  t_max anywhere, the segment gets ``violation_penalty`` and the episode ends.
  The penalty is moderate (default -10, same order as a few steps of Delta-SoC
  gain) rather than -100: a smoother target landscape for Q-learning. A hard
  -100 disqualification is still applied at *evaluation* time
  (``EVAL_CONSTRAINT_PENALTY``), never inside Q-targets.

* **SEI proxy** = integral |I| * exp(k (T - T_ref)) dt — Arrhenius-weighted
  charge throughput. k = 0.05 / K corresponds to Ea ~ 12 kJ/mol near 298 K;
  flag as approximate unless calibrated against the capacity-fade table.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

DEFAULT_LAMBDA = 0.002        # SEI penalty weight
DEFAULT_K_ARRHENIUS = 0.05    # 1/K
DEFAULT_T_REF_C = 25.0        # reference temperature (Celsius)
DEFAULT_T_MAX = 40.0          # hard temperature ceiling (Celsius)
DEFAULT_V_MAX = 4.2           # voltage ceiling (V)
TRAIN_VIOLATION_PENALTY = -10.0   # used inside Q-targets
EVAL_CONSTRAINT_PENALTY = -100.0  # hard disqualifier for evaluation/ranking


def compute_sei_proxy(
    current_a: np.ndarray,
    temperature_c: np.ndarray,
    dt_seconds: float = 1.0,
    k: float = DEFAULT_K_ARRHENIUS,
    t_ref_c: float = DEFAULT_T_REF_C,
) -> float:
    """
    Arrhenius-weighted charge throughput (arbitrary units, comparable across
    profiles). Temperature differences in Celsius equal differences in Kelvin,
    so the exponent uses Celsius directly.
    """
    arr = np.exp(k * (np.asarray(temperature_c, dtype=np.float64) - t_ref_c))
    return float(np.sum(np.abs(current_a) * dt_seconds * arr))


def compute_reward(
    v_pred: np.ndarray,
    t_pred: np.ndarray,
    current_profile: np.ndarray,
    q_as: float,
    dt_seconds: float = 1.0,
    *,
    lam: float = DEFAULT_LAMBDA,
    k: float = DEFAULT_K_ARRHENIUS,
    v_max: float = DEFAULT_V_MAX,
    t_max: float = DEFAULT_T_MAX,
    v_margin: float = 0.0,
    t_margin: float = 0.0,
    violation_penalty: float = TRAIN_VIOLATION_PENALTY,
) -> Tuple[float, Dict]:
    """
    Score one simulated segment (already truncated at the voltage ceiling).

    Args:
        v_pred, t_pred:   BDT-predicted trajectories (truncated by the caller).
        current_profile:  applied current, same length as the trajectories.
        q_as:             cell capacity at this age (Ampere-seconds).
        v_margin/t_margin: p95 open-loop drift margins at this horizon
                          (chance-constrained tightening).

    Returns (reward, info). ``info['violated']`` signals episode termination
    with penalty; ceiling truncation is NOT a violation.
    """
    v_pred = np.asarray(v_pred, dtype=np.float64)
    t_pred = np.asarray(t_pred, dtype=np.float64)
    current_profile = np.asarray(current_profile, dtype=np.float64)
    if v_pred.size == 0:
        return 0.0, {"violated": False, "violation_type": None,
                     "delta_soc_pct": 0.0, "sei_proxy": 0.0, "reward": 0.0,
                     "peak_voltage": None, "peak_temperature": None}

    delta_soc_pct = float(np.sum(-current_profile * dt_seconds)) / float(q_as) * 100.0
    sei = compute_sei_proxy(current_profile, t_pred, dt_seconds, k=k)
    info: Dict = {
        "delta_soc_pct": delta_soc_pct,
        "sei_proxy": sei,
        "sei_penalty": lam * sei,
        "peak_voltage": float(np.max(v_pred)),
        "peak_temperature": float(np.max(t_pred)),
        "violated": False,
        "violation_type": None,
    }

    # Voltage is enforced by the simulator truncating at the margin-tightened
    # ceiling (v_max - v_margin), with the crossing sample kept. This check is
    # defensive only: it fires when a caller forgot to truncate.
    if np.any(v_pred > v_max):
        info.update(violated=True, violation_type="voltage")
        info["reward"] = violation_penalty
        return violation_penalty, info

    if np.any(t_pred + t_margin > t_max):
        info.update(violated=True, violation_type="temperature")
        info["reward"] = violation_penalty
        return violation_penalty, info

    reward = delta_soc_pct - lam * sei
    info["reward"] = reward
    return reward, info
