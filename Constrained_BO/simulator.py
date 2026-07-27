"""Frozen BDT wrapper and closed-loop charging rollout."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from rw_transfer.training.twin_trainer import TwinTrainer

from Constrained_BO.config import CellConfig, DEFAULT_DECISION_INTERVAL_S
from Constrained_BO.decision_interval import select_decision_interval_s
from Constrained_BO.objective import (
    V_NOM_FALLBACK,
    energy_delivered_j,
    energy_required_j,
    full_capacity_joules,
)
from Constrained_BO.profiles import ProfileFamily, ProfileParams, SimulationContext, get_family

V_CEILING = 4.2


class FrozenBDT:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        *,
        current_scale: float = 1.0,
    ):
        trainer = TwinTrainer.load(Path(checkpoint), device=device)
        self.model = trainer.model
        self.device = trainer.device
        self.seq_len = self.model.seq_len
        self.current_scale = float(current_scale)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _to_bdt_current(self, current_a: np.ndarray) -> np.ndarray:
        return np.asarray(current_a, dtype=np.float32) * self.current_scale

    def predict_traj(
        self,
        age: float,
        v0: float,
        t0: float,
        current_profile: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        profile = self._to_bdt_current(current_profile)
        return self.model.predict(
            relative_age=float(age),
            v0=float(v0),
            t0=float(t0),
            current_profile=profile,
        )

    def single_step(
        self,
        state: Dict[str, float],
        action_a: float,
        n_steps: int,
        v_ceiling: float = V_CEILING,
        switch_pad: int = 5,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, bool]:
        """NASA convention: negative current = charge."""
        prev_i = float(state.get("prev_i", 0.0))
        pad = int(switch_pad) if abs(prev_i - float(action_a)) > 1e-9 else 0
        profile = np.concatenate([
            np.full(pad, prev_i, dtype=np.float32),
            np.full(int(n_steps), float(action_a), dtype=np.float32),
        ])
        # predict_traj applies current_scale once — do not pre-scale here
        # (double-scaling would cancel LFP's -1 sign flip and any other ≠1 scale).
        v_pred, t_pred = self.predict_traj(
            state["age"], state["v0"], state["t0"], profile,
        )
        v_traj, t_traj = v_pred[pad:], t_pred[pad:]

        over = np.flatnonzero(v_traj >= v_ceiling)
        terminated = bool(over.size)
        if terminated:
            cut = int(over[0]) + 1
            v_traj, t_traj = v_traj[:cut], t_traj[:cut]

        next_state = {
            "v0": float(v_traj[-1]) if v_traj.size else state["v0"],
            "t0": float(t_traj[-1]) if t_traj.size else state["t0"],
            "age": state["age"],
            "prev_i": float(action_a) if v_traj.size else prev_i,
        }
        return next_state, v_traj, t_traj, terminated


class ChargingSimulator:
    def __init__(
        self,
        bdt: FrozenBDT,
        *,
        q_rated_as: float,
        soc_target: float = 0.95,
        max_duration_min: float = 150.0,
        decision_interval_s: int = DEFAULT_DECISION_INTERVAL_S,
        v_max: float = V_CEILING,
        constraint_mode: str = "soc",
        energy_fraction: Optional[float] = None,
        v_nom: float = V_NOM_FALLBACK,
    ):
        self.bdt = bdt
        self.q_rated_as = float(q_rated_as)
        self.soc_target = float(soc_target)
        self.max_duration_min = float(max_duration_min)
        self.decision_interval_s = int(decision_interval_s)
        self.decision_interval_info: Dict = {}
        self.v_max = float(v_max)
        self.constraint_mode = constraint_mode
        self.energy_fraction = energy_fraction
        self.v_nom = float(v_nom)
        self.energy_full_j = full_capacity_joules(self.q_rated_as, self.v_nom)
        self.energy_required_j = (
            energy_required_j(self.q_rated_as, energy_fraction, self.v_nom)
            if energy_fraction is not None
            else 0.0
        )

    @classmethod
    def from_cell(cls, cell: CellConfig, device: str = "auto") -> ChargingSimulator:
        bdt = FrozenBDT(
            cell.bdt_ckpt,
            device=device,
            current_scale=cell.bdt_current_scale,
        )
        interval_info: Dict = {"source": "default", "selected_s": DEFAULT_DECISION_INTERVAL_S}
        interval_s = cell.decision_interval_s
        if interval_s is None and cell.auto_decision_interval:
            interval_s, interval_info = select_decision_interval_s(
                bdt,
                cell.cell_id,
                cell.start_state,
                candidates=cell.decision_interval_candidates,
            )
        elif interval_s is None:
            interval_s = DEFAULT_DECISION_INTERVAL_S
            interval_info = {
                "method": "default",
                "source": "default",
                "selected_s": int(interval_s),
            }
        else:
            interval_info = {
                "method": "fixed",
                "source": "fixed",
                "selected_s": int(interval_s),
            }

        sim = cls(
            bdt,
            q_rated_as=cell.q_rated_as,
            soc_target=cell.soc_target,
            max_duration_min=cell.max_duration_min,
            constraint_mode=cell.constraint_mode,
            energy_fraction=cell.energy_fraction,
            v_nom=cell.v_nom,
            decision_interval_s=interval_s,
            v_max=cell.v_max,
        )
        sim.decision_interval_info = interval_info
        return sim

    def simulate(
        self,
        initial_state: Dict[str, float],
        params: ProfileParams,
        *,
        family: Optional[ProfileFamily] = None,
    ) -> Dict:
        family = family or get_family(params.family_id)
        state = dict(initial_state)
        state.setdefault("prev_i", 0.0)
        state["soc"] = float(state.get("soc", 0.15))

        ctx = family.init_context(params)
        v_ceiling = self.v_max
        n_decisions = int(self.max_duration_min * 60 // self.decision_interval_s)

        i_all: List[float] = []
        v_all: List[float] = []
        t_all: List[float] = []
        end_reason = "time budget"
        energy_mode = self.constraint_mode == "energy" and self.energy_required_j > 0.0

        for _ in range(n_decisions):
            target_i = family.target_current(state, ctx, params)
            # Energy-constrained CCCV: do not crawl at i_cutoff before the
            # energy target — hold a minimum CV current (≥0.1·I_cc, ≥0.1 A)
            # until the energy check stops the session.
            if (
                energy_mode
                and getattr(ctx, "phase", None) == "cv"
                and params.family_id == "cccv"
                and target_i != 0.0
            ):
                i_cc = float(params.values.get("i_cc", abs(target_i)))
                i_cut = float(params.values.get("i_cutoff", 0.01))
                i_hold = max(i_cut, 0.1, 0.1 * i_cc)
                if abs(target_i) + 1e-9 < i_hold:
                    target_i = -float(i_hold)
                    ctx.i_level = float(i_hold)

            step_ceiling = family.cv_ceiling(params, v_ceiling, ctx)

            next_state, v_traj, t_traj, ceiling_hit = self.bdt.single_step(
                state,
                target_i,
                n_steps=self.decision_interval_s,
                v_ceiling=step_ceiling,
            )
            n = int(v_traj.size)

            # CV hold: once in constant-voltage phase, a commanded current that
            # still kisses the ceiling would otherwise truncate each BDT step to
            # ~1 sample, tapering to i_cutoff before the energy target is met.
            # Extend the remainder of the decision interval at V clamped to the
            # CV ceiling so CC→CV baselines can finish the energy window.
            if (
                getattr(ctx, "phase", None) == "cv"
                and target_i != 0.0
                and n > 0
                and n < self.decision_interval_s
            ):
                need = int(self.decision_interval_s) - n
                v_hold = float(min(float(v_traj[-1]), step_ceiling))
                t_hold = float(t_traj[-1])
                v_traj = np.concatenate([v_traj, np.full(need, v_hold, dtype=v_traj.dtype)])
                t_traj = np.concatenate([t_traj, np.full(need, t_hold, dtype=t_traj.dtype)])
                next_state = dict(next_state)
                next_state["v0"] = v_hold
                next_state["t0"] = t_hold
                next_state["prev_i"] = float(target_i)
                n = int(v_traj.size)
                # Still at the voltage limit — CV taper should continue next step.
                ceiling_hit = True

            profile = np.full(n, target_i, dtype=np.float64)
            delta_soc = float(np.sum(-profile)) / self.q_rated_as
            next_state = dict(next_state)
            next_state["soc"] = float(np.clip(state["soc"] + delta_soc, 0.0, 1.0))

            if target_i != 0.0 and not ctx.in_rest:
                ctx.charge_elapsed += n

            i_all.extend(profile.tolist())
            v_all.extend(v_traj.tolist())
            t_all.extend(t_traj.tolist())
            state = next_state

            # Energy mode: stop when ∫ V·I dt reaches the required joules.
            # Truncate the last decision chunk so we do not overshoot the target.
            if energy_mode and n > 0:
                i_arr_tmp = np.asarray(i_all, dtype=np.float64)
                v_arr_tmp = np.asarray(v_all, dtype=np.float64)
                t_arr_tmp = np.arange(i_arr_tmp.size, dtype=np.float64)
                e_del = energy_delivered_j(v_arr_tmp, i_arr_tmp, t_arr_tmp)
                if e_del >= self.energy_required_j - 1e-3:
                    # Binary-search earliest cut in the last chunk that meets target
                    # using the same integrator as evaluate_session.
                    lo = max(1, i_arr_tmp.size - n)
                    hi = i_arr_tmp.size
                    cut = hi
                    while lo < hi:
                        mid = (lo + hi) // 2
                        e_mid = energy_delivered_j(
                            v_arr_tmp[:mid], i_arr_tmp[:mid], t_arr_tmp[:mid],
                        )
                        if e_mid >= self.energy_required_j - 1e-3:
                            cut = mid
                            hi = mid
                        else:
                            lo = mid + 1
                    i_all = i_all[:cut]
                    v_all = v_all[:cut]
                    t_all = t_all[:cut]
                    i_cut = np.asarray(i_all, dtype=np.float64)
                    state["soc"] = float(np.clip(
                        float(initial_state.get("soc", 0.0))
                        + float(np.sum(-i_cut)) / self.q_rated_as,
                        0.0, 1.0,
                    ))
                    if v_all:
                        state["v0"] = float(v_all[-1])
                        state["t0"] = float(t_all[-1])
                    end_reason = "energy target"
                    break

            ctx, early = family.after_step(
                state, ctx, params,
                ceiling_hit=ceiling_hit,
                v_traj=v_traj,
                global_ceiling=v_ceiling,
            )
            if early:
                end_reason = early
                break

            # SoC stop only in SoC-constraint mode. In energy mode the energy
            # check above is authoritative; SoC=1.0 remains a hard safety stop.
            if energy_mode:
                if state["soc"] >= 1.0 - 1e-6:
                    end_reason = "SoC full"
                    break
            elif state["soc"] >= self.soc_target:
                end_reason = "SoC target"
                break

            family_end = family.end_check(
                state, ctx, params,
                ceiling_hit=ceiling_hit,
                step_samples=n,
                target_i=target_i,
            )
            # Energy-constrained sessions: CV cutoff is not a failure — keep
            # charging until the energy target (or time budget / SoC full).
            if family_end:
                if energy_mode and family_end == "CV cutoff current":
                    family_end = None
            if family_end:
                end_reason = family_end
                break

        i_arr = np.asarray(i_all, dtype=np.float64)
        soc_traj = np.clip(
            initial_state["soc"] + np.cumsum(-i_arr) / self.q_rated_as,
            0.0, 1.0,
        )
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
            "soc_target": self.soc_target,
            "constraint_mode": self.constraint_mode,
            "energy_fraction": self.energy_fraction,
            "energy_required_j": self.energy_required_j,
            "energy_full_j": self.energy_full_j,
            "v_nom": self.v_nom,
            "v_max": self.v_max,
            "decision_interval_s": self.decision_interval_s,
        }
