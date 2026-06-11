"""
Frozen BDT used as a read-only simulator for charging-profile optimization.

Wraps the existing ``BatteryDigitalTwin`` via ``TwinTrainer.load`` (the
checkpoint stores ``model_state`` plus architecture hyperparameters, so the
trainer reconstructs the exact source/fine-tuned model). Weights are
structurally frozen — ``requires_grad=False`` on every parameter.

Conventions (NASA RW):
    * current I < 0 means charging;
    * the model operates at the data's native 1 Hz sampling (dt = 1 s) for RW
      operational steps — profiles passed in are interpreted sample-per-second;
    * profiles longer than ``seq_len`` (150) are predicted with the model's
      chained chunking (each chunk re-anchored on the previous chunk's last
      predicted V/T), which is exactly how closed-loop drift accumulates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from rw_transfer.training.twin_trainer import TwinTrainer

DEFAULT_V_CEILING = 4.2


class FrozenBDTSimulator:
    """Read-only physics oracle: (state, current profile) -> V/T trajectories."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        trainer = TwinTrainer.load(self.checkpoint_path, device=device)
        self.model = trainer.model
        self.device = trainer.device
        self.seq_len = self.model.seq_len
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        n = sum(p.numel() for p in self.model.parameters())
        print(f"Frozen BDT loaded: {self.checkpoint_path.name} "
              f"({n:,} params, device={self.device}, seq_len={self.seq_len})")

    # -- raw trajectory prediction (no termination logic) ----------------------

    def predict_traj(
        self,
        age: float,
        v0: float,
        t0: float,
        current_profile: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chained chunked prediction; returns (V_hat, T_hat) of profile length."""
        return self.model.predict(
            relative_age=float(age),
            v0=float(v0),
            t0=float(t0),
            current_profile=np.asarray(current_profile, dtype=np.float32),
        )

    # -- rollout with voltage-ceiling termination -------------------------------

    def rollout(
        self,
        state: Dict[str, float],
        current_profile: np.ndarray,
        v_ceiling: float = DEFAULT_V_CEILING,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Roll the twin forward from ``state = {'v0', 't0', 'age'}``.

        The trajectory is truncated at the first sample where V_hat >= v_ceiling
        (the charger would switch to CV / stop there), and the truncation is
        applied BEFORE any reward computation downstream.

        Returns (v_traj, t_traj, terminated_early).
        """
        v_pred, t_pred = self.predict_traj(
            state["age"], state["v0"], state["t0"], current_profile
        )
        over = np.flatnonzero(v_pred >= v_ceiling)
        if over.size:
            cut = int(over[0]) + 1  # keep the crossing sample
            return v_pred[:cut], t_pred[:cut], True
        return v_pred, t_pred, False

    def single_step(
        self,
        state: Dict[str, float],
        action_a: float,
        n_steps: int,
        v_ceiling: float = DEFAULT_V_CEILING,
        switch_pad: int = 5,
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, bool]:
        """
        Apply one constant current setpoint for ``n_steps`` seconds.

        Current-switch handling: ``state['v0']`` is the loaded voltage under
        the PREVIOUS current (``state['prev_i']``, 0 for rest). The model only
        sees the step change through its delta-I feature when the switch
        happens INSIDE the chunk — a chunk of purely constant current cannot
        predict the instantaneous IR relaxation/jump at the setpoint change.
        So the chunk is padded with ``switch_pad`` samples of the previous
        current and the pad is discarded from the returned trajectories.
        Without this, a state that touched the voltage ceiling stays pinned
        there even after tapering to a lower current.

        Returns (next_state, v_traj, t_traj, terminated_early). Age is constant
        within a session (a charging session is negligible vs cell lifetime).
        ``next_state['prev_i']`` is set for the next decision.
        """
        prev_i = float(state.get("prev_i", 0.0))
        pad = int(switch_pad) if abs(prev_i - float(action_a)) > 1e-9 else 0
        profile = np.concatenate([
            np.full(pad, prev_i, dtype=np.float32),
            np.full(int(n_steps), float(action_a), dtype=np.float32),
        ])
        v_pred, t_pred = self.predict_traj(
            state["age"], state["v0"], state["t0"], profile
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
