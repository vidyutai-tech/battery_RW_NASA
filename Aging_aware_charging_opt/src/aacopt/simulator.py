"""Closed-loop charging rollout on the frozen BDT."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from aacopt.bdt import FrozenBDT
from aacopt.config import SessionSpec
from aacopt.profiles import ProfileFamily, ProfileParams, get_family
from aacopt.reward import energy_delivered_j, energy_required_j, full_capacity_joules


class ChargingSimulator:
    def __init__(
        self,
        bdt: FrozenBDT,
        *,
        q_rated_as: float,
        session: SessionSpec,
        v_nom: float,
    ):
        self.bdt = bdt
        self.q_rated_as = float(q_rated_as)
        self.session = session
        self.v_nom = float(v_nom)
        self.energy_full_j = full_capacity_joules(self.q_rated_as, self.v_nom)
        self.energy_required_j = energy_required_j(
            self.q_rated_as, session.energy_fraction, self.v_nom,
        )

    def simulate(self, initial_state: Dict[str, float], params: ProfileParams,
                 *, family: Optional[ProfileFamily] = None) -> Dict:
        family = family or get_family(params.family_id)
        sess = self.session
        state = dict(initial_state)
        state.setdefault("prev_i", 0.0)
        state.setdefault("age", 0.0)
        state["soc"] = float(state.get("soc", sess.soc_start))
        ctx = family.init_context(params)
        n_decisions = int(sess.max_duration_min * 60 // sess.decision_interval_s)
        v_ceiling = sess.v_max
        i_all: List[float] = []
        v_all: List[float] = []
        t_all: List[float] = []
        end_reason = "time budget"
        energy_mode = sess.constraint_mode == "energy" and self.energy_required_j > 0.0

        for _ in range(n_decisions):
            target_i = family.target_current(state, ctx, params)
            if (energy_mode and getattr(ctx, "phase", None) == "cv"
                    and params.family_id == "cccv" and target_i != 0.0):
                i_cc = float(params.values.get("i_cc", abs(target_i)))
                i_cut = float(params.values.get("i_cutoff", 0.01))
                i_hold = max(i_cut, 0.1, 0.1 * i_cc)
                if abs(target_i) + 1e-9 < i_hold:
                    target_i = -float(i_hold)
                    ctx.i_level = float(i_hold)

            step_ceiling = family.cv_ceiling(params, v_ceiling, ctx)
            next_state, v_traj, t_traj, ceiling_hit = self.bdt.single_step(
                state, target_i, n_steps=sess.decision_interval_s,
                v_ceiling=step_ceiling, switch_pad=sess.bdt_switch_pad,
            )
            n = int(v_traj.size)
            if (getattr(ctx, "phase", None) == "cv" and target_i != 0.0
                    and n > 0 and n < sess.decision_interval_s):
                need = int(sess.decision_interval_s) - n
                v_hold = float(min(float(v_traj[-1]), step_ceiling))
                t_hold = float(t_traj[-1])
                v_traj = np.concatenate([v_traj, np.full(need, v_hold, dtype=v_traj.dtype)])
                t_traj = np.concatenate([t_traj, np.full(need, t_hold, dtype=t_traj.dtype)])
                next_state = dict(next_state)
                next_state["v0"] = v_hold
                next_state["t0"] = t_hold
                next_state["prev_i"] = float(target_i)
                n = int(v_traj.size)
                ceiling_hit = True

            profile = np.full(n, target_i, dtype=np.float64)
            next_state = dict(next_state)
            next_state["soc"] = float(np.clip(
                state["soc"] + float(np.sum(-profile)) / self.q_rated_as, 0.0, 1.0,
            ))
            if target_i != 0.0 and not ctx.in_rest:
                ctx.charge_elapsed += n
            i_all.extend(profile.tolist())
            v_all.extend(v_traj.tolist())
            t_all.extend(t_traj.tolist())
            state = next_state

            if energy_mode and n > 0:
                i_arr_tmp = np.asarray(i_all, dtype=np.float64)
                v_arr_tmp = np.asarray(v_all, dtype=np.float64)
                t_arr_tmp = np.arange(i_arr_tmp.size, dtype=np.float64)
                e_del = energy_delivered_j(v_arr_tmp, i_arr_tmp, t_arr_tmp)
                if e_del >= self.energy_required_j - 1e-3:
                    lo = max(1, i_arr_tmp.size - n)
                    hi = i_arr_tmp.size
                    cut = hi
                    while lo < hi:
                        mid = (lo + hi) // 2
                        e_mid = energy_delivered_j(
                            v_arr_tmp[:mid], i_arr_tmp[:mid], t_arr_tmp[:mid],
                        )
                        if e_mid >= self.energy_required_j - 1e-3:
                            cut, hi = mid, mid
                        else:
                            lo = mid + 1
                    i_all, v_all, t_all = i_all[:cut], v_all[:cut], t_all[:cut]
                    i_cut = np.asarray(i_all, dtype=np.float64)
                    state["soc"] = float(np.clip(
                        float(initial_state.get("soc", 0.0))
                        + float(np.sum(-i_cut)) / self.q_rated_as, 0.0, 1.0,
                    ))
                    if v_all:
                        state["v0"] = float(v_all[-1])
                        state["t0"] = float(t_all[-1])
                    end_reason = "energy target"
                    break

            ctx, early = family.after_step(
                state, ctx, params, ceiling_hit=ceiling_hit,
                v_traj=v_traj, global_ceiling=v_ceiling,
            )
            if early:
                end_reason = early
                break
            if energy_mode:
                if state["soc"] >= 1.0 - 1e-6:
                    end_reason = "SoC full"
                    break
            elif state["soc"] >= sess.soc_start + sess.energy_fraction:
                end_reason = "SoC target"
                break
            family_end = family.end_check(
                state, ctx, params, ceiling_hit=ceiling_hit,
                step_samples=n, target_i=target_i,
            )
            if family_end:
                if energy_mode and family_end == "CV cutoff current":
                    family_end = None
            if family_end:
                end_reason = family_end
                break

        i_arr = np.asarray(i_all, dtype=np.float64)
        soc_traj = np.clip(
            initial_state["soc"] + np.cumsum(-i_arr) / self.q_rated_as, 0.0, 1.0,
        ) if i_arr.size else np.asarray([initial_state["soc"]], dtype=np.float64)
        return {
            "initial_state": dict(initial_state),
            "profile_params": params.to_dict(),
            "family_id": params.family_id,
            "time_s": np.arange(i_arr.size, dtype=np.float64),
            "current_a": i_arr,
            "voltage_v": np.asarray(v_all, dtype=np.float64),
            "temperature_c": np.asarray(t_all, dtype=np.float64),
            "soc": soc_traj,
            "end_reason": end_reason,
            "q_rated_as": self.q_rated_as,
            "constraint_mode": sess.constraint_mode,
            "energy_fraction": sess.energy_fraction,
            "energy_required_j": self.energy_required_j,
            "energy_full_j": self.energy_full_j,
            "v_nom": self.v_nom,
            "v_max": sess.v_max,
            "decision_interval_s": sess.decision_interval_s,
        }
