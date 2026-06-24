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
    energy_required_j,
    full_capacity_joules,
)
from Constrained_BO.profiles import ProfileFamily, ProfileParams, SimulationContext, get_family

V_CEILING = 4.2


class FrozenBDT:
    def __init__(self, checkpoint: str | Path, device: str = "auto"):
        trainer = TwinTrainer.load(Path(checkpoint), device=device)
        self.model = trainer.model
        self.device = trainer.device
        self.seq_len = self.model.seq_len
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def predict_traj(
        self,
        age: float,
        v0: float,
        t0: float,
        current_profile: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict(
            relative_age=float(age),
            v0=float(v0),
            t0=float(t0),
            current_profile=np.asarray(current_profile, dtype=np.float32),
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
        bdt = FrozenBDT(cell.bdt_ckpt, device=device)
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

        for _ in range(n_decisions):
            target_i = family.target_current(state, ctx, params)
            step_ceiling = family.cv_ceiling(params, v_ceiling, ctx)

            next_state, v_traj, t_traj, ceiling_hit = self.bdt.single_step(
                state,
                target_i,
                n_steps=self.decision_interval_s,
                v_ceiling=step_ceiling,
            )
            n = int(v_traj.size)
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

            ctx, early = family.after_step(
                state, ctx, params,
                ceiling_hit=ceiling_hit,
                v_traj=v_traj,
                global_ceiling=v_ceiling,
            )
            if early:
                end_reason = early
                break

            if state["soc"] >= self.soc_target:
                end_reason = "SoC target"
                break

            family_end = family.end_check(
                state, ctx, params,
                ceiling_hit=ceiling_hit,
                step_samples=n,
                target_i=target_i,
            )
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
            "decision_interval_s": self.decision_interval_s,
        }
