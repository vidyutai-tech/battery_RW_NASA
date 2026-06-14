"""
Constrained Thompson Sampling BO (cTS) — separate objective and feasibility GPs.

Alternative to penalty-encoded constraints in gp_minimize. Two surrogate models:
one for scalar loss (among feasible runs), one for feasibility (0/1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


@dataclass
class ConstrainedTSResult:
    best_x: List[float]
    best_loss: float
    best_feasible: bool
    history: List[Dict]
    n_calls: int


class ConstrainedTSOptimizer:
    """
    Thompson-sampling BO with separate objective and constraint surrogates.

    At each step after warm-up:
      1. Sample candidate points uniformly in the search box
      2. Draw one posterior sample from each GP for all candidates
      3. Prefer candidates whose constraint sample predicts feasibility
      4. Among those, pick minimum objective sample
    """

    def __init__(
        self,
        search_space_bounds: List[Tuple[float, float]],
        objective_fn: Callable[[List[float]], Tuple[float, bool]],
        *,
        n_candidates: int = 1000,
        random_state: int = 42,
    ):
        self.bounds = np.asarray(search_space_bounds, dtype=np.float64)
        self.objective_fn = objective_fn
        self.n_candidates = int(n_candidates)
        self.rng = np.random.default_rng(random_state)

        kernel = Matern(nu=2.5)
        self.gp_obj = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
        )
        self.gp_con = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=3,
        )

        self.X: List[List[float]] = []
        self.y_obj: List[float] = []
        self.y_con: List[float] = []

    def _sample_candidates(self) -> np.ndarray:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return self.rng.uniform(lo, hi, size=(self.n_candidates, len(lo)))

    def _record(self, x: List[float], loss: float, feasible: bool) -> Dict:
        self.X.append(list(x))
        self.y_obj.append(float(loss if feasible else 200.0))
        self.y_con.append(0.0 if feasible else 1.0)
        entry = {"x": list(x), "loss": float(loss), "feasible": bool(feasible)}
        return entry

    def _next_point(self) -> np.ndarray:
        if len(self.X) < 3:
            lo, hi = self.bounds[:, 0], self.bounds[:, 1]
            return self.rng.uniform(lo, hi)

        X = np.asarray(self.X, dtype=np.float64)
        self.gp_obj.fit(X, np.asarray(self.y_obj, dtype=np.float64))
        self.gp_con.fit(X, np.asarray(self.y_con, dtype=np.float64))

        candidates = self._sample_candidates()
        obj_samples = self.gp_obj.sample_y(
            candidates, n_samples=1,
            random_state=int(self.rng.integers(1_000_000)),
        ).ravel()
        con_samples = self.gp_con.sample_y(
            candidates, n_samples=1,
            random_state=int(self.rng.integers(1_000_000)),
        ).ravel()

        feasible_mask = con_samples < 0.5
        if feasible_mask.any():
            feas_idx = np.flatnonzero(feasible_mask)
            best_feas = feas_idx[np.argmin(obj_samples[feas_idx])]
            return candidates[best_feas]
        return candidates[np.argmin(con_samples)]

    def optimize(
        self,
        n_calls: int = 40,
        x0: Optional[List[List[float]]] = None,
    ) -> ConstrainedTSResult:
        history: List[Dict] = []

        if x0:
            for x in x0:
                loss, feasible = self.objective_fn(x)
                history.append(self._record(x, loss, feasible))

        remaining = max(0, n_calls - len(history))
        for _ in range(remaining):
            x_next = self._next_point()
            loss, feasible = self.objective_fn(list(x_next))
            history.append(self._record(list(x_next), loss, feasible))

        feas = [h for h in history if h["feasible"]]
        if feas:
            best = min(feas, key=lambda h: h["loss"])
        else:
            best = min(history, key=lambda h: h["loss"])

        return ConstrainedTSResult(
            best_x=best["x"],
            best_loss=float(best["loss"]),
            best_feasible=bool(best["feasible"]),
            history=history,
            n_calls=len(history),
        )
