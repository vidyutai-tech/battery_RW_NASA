"""
Stage 1 — simulate parametric charging profiles through the frozen BDT.

Profiles are defined by a small vector of interpretable parameters (CC current,
optional pulsed rest, taper floor). The simulator re-anchors on predicted V/T
every ``decision_interval`` seconds so open-loop drift stays within the
conformal margin used at training time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from charging_opt.bdt_rollout import FrozenBDTSimulator
from charging_opt.charging_profile_family import (
    CcTaperLegacyFamily,
    ChargingProfileFamily,
    ProfileParams,
    SimulationContext,
    get_family,
)
from charging_opt.reward import (
    DEFAULT_K_ARRHENIUS,
    DEFAULT_T_MAX,
    DEFAULT_V_MAX,
    compute_sei_proxy,
)
from charging_opt.artifacts import CANONICAL
from charging_opt.soc_utils import load_capacity_curve

TAPER_STEP_A = 0.75
MIN_REST_MIN = 5.0  # rests shorter than this are treated as continuous CC


@dataclass(frozen=True)
class ProfileSpec:
    """
    Parametric charging template (CC-taper with optional pulsed rest).

    * ``i_charge`` — primary CC magnitude (A, positive; applied as negative I).
    * ``pulse_on_min`` — charge burst length before optional rest (minutes).
    * ``pulse_rest_min`` — rest at 0 A between bursts; values below
      ``MIN_REST_MIN`` are snapped to 0 (continuous charging).
    * ``i_floor`` — minimum taper current when the voltage ceiling is hit;
      must be at least one taper step below ``i_charge`` when taper is used.
    """

    i_charge: float
    pulse_on_min: float
    pulse_rest_min: float
    i_floor: float

    @classmethod
    def from_vector(cls, x: List[float] | np.ndarray) -> ProfileSpec:
        i_charge, pulse_on_min, pulse_rest_min, i_floor = [float(v) for v in x]
        i_charge = float(np.clip(i_charge, 0.75, 4.5))
        if pulse_rest_min < MIN_REST_MIN:
            pulse_rest_min = 0.0
        i_floor_max = i_charge - TAPER_STEP_A
        if i_floor_max < 0.75:
            i_floor = 0.75
        else:
            i_floor = float(np.clip(i_floor, 0.75, i_floor_max))
        return cls(
            i_charge=i_charge,
            pulse_on_min=max(5.0, pulse_on_min),
            pulse_rest_min=pulse_rest_min,
            i_floor=i_floor,
        )

    @classmethod
    def cc_taper(cls, i_charge: float, i_floor: float = 0.75) -> ProfileSpec:
        """Continuous CC-taper (no pulsed rest) — useful for diagnostics."""
        i_charge = float(np.clip(i_charge, 0.75, 4.5))
        i_floor_max = i_charge - TAPER_STEP_A
        if i_floor_max < 0.75:
            i_floor_eff = 0.75
        else:
            i_floor_eff = float(np.clip(i_floor, 0.75, i_floor_max))
        return cls(
            i_charge=i_charge, pulse_on_min=30.0, pulse_rest_min=0.0, i_floor=i_floor_eff,
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> ProfileSpec:
        return cls(
            i_charge=float(d["i_charge"]),
            pulse_on_min=float(d["pulse_on_min"]),
            pulse_rest_min=float(d["pulse_rest_min"]),
            i_floor=float(d["i_floor"]),
        )


class ProfileSimulator:
    """Closed-loop BDT rollout for one parametric profile."""

    def __init__(
        self,
        bdt_path: str | Path,
        capacity_path: str | Path = CANONICAL["capacity_fade"],
        margins_path: Optional[str | Path] = CANONICAL["conformal_margins"],
        device: str = "auto",
        decision_interval: int = 30,
        max_minutes: int = 90,
        soc_target: float = 0.95,
        v_max: float = DEFAULT_V_MAX,
        t_max: float = DEFAULT_T_MAX,
        k_arrhenius: float = DEFAULT_K_ARRHENIUS,
    ):
        self.sim = FrozenBDTSimulator(bdt_path, device=device)
        self.q_of_age = load_capacity_curve(capacity_path)
        self.decision_interval = int(decision_interval)
        self.max_minutes = max_minutes
        self.soc_target = soc_target
        self.v_max = v_max
        self.t_max = t_max
        self.k = k_arrhenius

        self.v_margin, self.t_margin = 0.0, 0.0
        if margins_path is not None and Path(margins_path).exists():
            m = np.load(margins_path)
            h = min(self.decision_interval, int(m["horizon_s"][-1])) - 1
            self.v_margin = float(m["v_q95"][h])
            self.t_margin = float(m["t_q95"][h])
            print(
                f"Profile simulator: {self.decision_interval}s steps, "
                f"ceilings V<={v_max - self.v_margin:.3f} V, "
                f"T<={t_max - self.t_margin:.2f} degC"
            )

    def simulate(self, initial_state: Dict[str, float], spec: ProfileSpec) -> Dict:
        """Legacy entry point — wraps :class:`ProfileSpec` as cc_taper_legacy."""
        params = CcTaperLegacyFamily.from_vector(
            [
                spec.i_charge,
                spec.pulse_on_min,
                spec.pulse_rest_min,
                spec.i_floor,
            ],
            allow_pulsed=spec.pulse_rest_min >= MIN_REST_MIN,
        )
        return self.simulate_params(initial_state, params)

    def simulate_params(
        self,
        initial_state: Dict[str, float],
        params: ProfileParams,
        *,
        family: Optional[ChargingProfileFamily] = None,
    ) -> Dict:
        """Closed-loop rollout for any registered profile family."""
        family = family or get_family(params.family_id)
        state = dict(initial_state)
        state.setdefault("prev_i", 0.0)
        q_as = float(self.q_of_age(state["age"]))
        v_ceiling_global = self.v_max - self.v_margin
        n_decisions = self.max_minutes * 60 // self.decision_interval

        ctx = family.init_context(params)
        i_all, v_all, t_all = [], [], []
        decisions: List[Dict] = []
        end_reason = "time budget"

        for _ in range(int(n_decisions)):
            target_i = family.target_current(state, ctx, params)
            step_ceiling = family.cv_ceiling(params, v_ceiling_global, ctx)

            next_state, v_traj, t_traj, ceiling_hit = self.sim.single_step(
                state, target_i, n_steps=self.decision_interval, v_ceiling=step_ceiling,
            )
            profile = np.full(v_traj.size, target_i, dtype=np.float64)
            delta_soc = float(np.sum(-profile)) / q_as
            sei = compute_sei_proxy(profile, t_traj, 1.0, k=self.k)
            next_state = dict(next_state)
            next_state["soc"] = float(np.clip(state["soc"] + delta_soc, 0.0, 1.0))

            if target_i != 0.0 and not ctx.in_rest:
                ctx.charge_elapsed += v_traj.size

            violated = bool(np.any(t_traj + self.t_margin > self.t_max))
            decisions.append({
                "t_start_s": len(i_all),
                "duration_s": int(v_traj.size),
                "action_a": target_i,
                "soc_before": state["soc"],
                "soc_after": next_state["soc"],
                "v_end": next_state["v0"],
                "t_end": next_state["t0"],
                "delta_soc_pct": delta_soc * 100.0,
                "sei_penalty": sei,
                "ceiling_hit": bool(ceiling_hit),
                "phase": ctx.phase,
            })
            i_all.extend(profile.tolist())
            v_all.extend(v_traj.tolist())
            t_all.extend(t_traj.tolist())
            state = next_state

            ctx, early = family.after_step(
                state, ctx, params,
                ceiling_hit=ceiling_hit,
                v_traj=v_traj,
                global_ceiling=v_ceiling_global,
            )
            if early:
                end_reason = early
                break

            if violated:
                end_reason = "temperature violation"
                break
            if state["soc"] >= self.soc_target:
                end_reason = "SoC target"
                break

            family_end = family.end_check(
                state, ctx, params,
                ceiling_hit=ceiling_hit,
                step_samples=v_traj.size,
                target_i=target_i,
            )
            if family_end:
                end_reason = family_end
                break

        i_arr = np.asarray(i_all, dtype=np.float64)
        session = {
            "initial_state": dict(initial_state),
            "profile_spec": params.to_dict(),
            "family_id": params.family_id,
            "time_s": np.arange(i_arr.size, dtype=np.float64),
            "current_a": i_arr,
            "voltage_v": np.asarray(v_all),
            "temperature_c": np.asarray(t_all),
            "soc": initial_state["soc"] + np.cumsum(-i_arr) / q_as,
            "decisions": decisions,
            "end_reason": end_reason,
            "q_as": q_as,
        }
        return session

    @staticmethod
    def merged_segments(session: Dict) -> List[Dict]:
        """Collapse consecutive equal-current steps into (I, duration) segments."""
        segs: List[Dict] = []
        for d in session["decisions"]:
            if segs and abs(segs[-1]["current_a"] - d["action_a"]) < 1e-9:
                segs[-1]["duration_s"] += d["duration_s"]
            else:
                segs.append({"current_a": d["action_a"], "duration_s": d["duration_s"]})
        return segs
