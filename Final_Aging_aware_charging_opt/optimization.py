"""Charging optimization: profile families, degradation, reward, and GP-BO.

Random search uses the same families, budget, and objective so the comparison
figures stay aligned with the previous study. GP-BO uses the updated settings
in ``config.py`` (equalized weights, seed 42, training-current cap).
"""

from __future__ import annotations

import argparse
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
from skopt import gp_minimize

from config import (
    DEGRADATION, GP_BO_SEED, OptimizationSpec, Paths, RESULTS_DIR, REWARD_SCALE,
    RewardSpec, SearchSpace, SessionSpec, file_hash, family_result_name,
    provenance, read_json, result_path, set_results_dir, write_json,
)
from twin import FrozenBDT

"""Parametric charging-profile families (CCCV, 2-step, 3-step, pulsed)."""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np


_space: Optional[SearchSpace] = None
_session: Optional[SessionSpec] = None


def bind_search(spec: Optional[OptimizationSpec] = None) -> None:
    global _space, _session
    spec = spec or OptimizationSpec.load()
    _space = spec.space
    _session = spec.session


def space() -> SearchSpace:
    if _space is None:
        bind_search()
    return _space


def session() -> SessionSpec:
    if _session is None:
        bind_search()
    return _session


def _soc_start() -> float:
    return float(session().soc_start)


@dataclass
class ProfileParams:
    family_id: str
    values: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"family_id": self.family_id, **self.values}


@dataclass
class SimulationContext:
    phase: str = "cc"
    i_level: float = 0.0
    charge_elapsed: float = 0.0
    rest_elapsed: float = 0.0
    in_rest: bool = False
    extra: Dict[str, float] = field(default_factory=dict)


class ProfileFamily(ABC):
    family_id: ClassVar[str]
    label: ClassVar[str]

    @classmethod
    @abstractmethod
    def param_bounds(cls) -> Dict[str, Tuple[float, float]]:
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        ...

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        vals = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in cls.param_bounds().items()}
        return cls.from_dict(vals)

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        return []

    @classmethod
    def param_names(cls) -> List[str]:
        return list(cls.param_bounds().keys())

    @classmethod
    def search_space(cls):
        from skopt.space import Real
        return [Real(float(lo), float(hi), name=name) for name, (lo, hi) in cls.param_bounds().items()]

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        names = cls.param_names()
        if len(x) != len(names):
            raise ValueError(f"{cls.family_id}: expected {len(names)} params {names}, got {len(x)}")
        return cls.from_dict({k: float(v) for k, v in zip(names, x)})

    @classmethod
    def to_vector(cls, params: ProfileParams) -> List[float]:
        return [float(params.values[k]) for k in cls.param_names()]

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [cls.to_vector(p) for p in cls.seed_params()]

    def init_context(self, params: ProfileParams) -> SimulationContext:
        return SimulationContext(phase="cc", i_level=self._bulk_current(params))

    @abstractmethod
    def _bulk_current(self, params: ProfileParams) -> float:
        ...

    @abstractmethod
    def target_current(self, state: Dict[str, float], ctx: SimulationContext, params: ProfileParams) -> float:
        ...

    def cv_ceiling(self, params, global_ceiling, ctx) -> float:
        return global_ceiling

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        return ctx, None

    def end_check(self, state, ctx, params, *, ceiling_hit, step_samples, target_i):
        return None


def _two_step_soc_bounds() -> Tuple[float, float]:
    b = space()
    soc_hi = b.soc_switch_max
    lo = min(_soc_start() + b.min_stage_dsoc, soc_hi - 1e-3)
    hi = max(lo + 1e-3, soc_hi)
    return lo, hi


def _three_step_soc_bounds():
    b = space()
    soc_hi = b.soc_switch_max
    start = _soc_start()
    soc1_lo = min(start + b.min_stage1_dsoc, soc_hi - b.min_stage_gap_dsoc - 1e-3)
    soc1_hi = max(soc1_lo + 1e-3, min(0.55, soc_hi - b.min_stage_gap_dsoc))
    soc2_lo = soc1_lo + b.min_stage_gap_dsoc
    soc2_hi = max(soc2_lo + 1e-3, soc_hi)
    return (soc1_lo, soc1_hi), (soc2_lo, soc2_hi)


class CCCVFamily(ProfileFamily):
    family_id = "cccv"
    label = "CCCV"

    @classmethod
    def param_bounds(cls) -> Dict[str, Tuple[float, float]]:
        b = space()
        i_lo, i_hi = b.i_bounds()
        v_lo, v_hi = b.v_cv_bounds()
        return {
            "i_cc": (i_lo, i_hi),
            "v_cv": (v_lo, v_hi),
            "i_cutoff": (0.01, min(0.50, i_hi - b.cv_step_a)),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        b = space()
        i_lo, i_hi = b.i_bounds()
        v_lo, v_hi = b.v_cv_bounds()
        i_cc = float(np.clip(values["i_cc"], i_lo, i_hi))
        v_cv = float(np.clip(values["v_cv"], v_lo, v_hi))
        i_cutoff = float(np.clip(values["i_cutoff"], 0.01, min(0.50, i_cc - b.cv_step_a)))
        return ProfileParams(family_id=cls.family_id, values={"i_cc": i_cc, "v_cv": v_cv, "i_cutoff": i_cutoff})

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        b = space()
        i_lo, i_hi = b.i_bounds()
        v_lo, v_hi = b.v_cv_bounds()
        ref = {"i_cc": 2.0, "v_cv": 4.2, "i_cutoff": 0.05}
        seeds = [ref]
        for i_cc in (i_lo, (i_lo + i_hi) / 2, i_hi):
            seeds.append({"i_cc": i_cc, "v_cv": v_hi, "i_cutoff": ref["i_cutoff"]})
        for v_cv in (v_lo, (v_lo + v_hi) / 2, v_hi):
            seeds.append({"i_cc": ref["i_cc"], "v_cv": v_cv, "i_cutoff": ref["i_cutoff"]})
        unique = {tuple(sorted(s.items())) for s in seeds}
        return [cls.from_dict(dict(t)) for t in unique]

    def _bulk_current(self, params):
        return params.values["i_cc"]

    def cv_ceiling(self, params, global_ceiling, ctx):
        if ctx.phase == "cv":
            return min(global_ceiling, params.values["v_cv"])
        return global_ceiling

    def target_current(self, state, ctx, params):
        if ctx.phase == "cc":
            return -params.values["i_cc"]
        return -max(ctx.i_level, params.values["i_cutoff"])

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        b = space()
        v_cv = params.values["v_cv"]
        if ctx.phase == "cc":
            if ceiling_hit or (v_traj.size and float(np.max(v_traj)) >= v_cv - 1e-4):
                ctx.phase = "cv"
                ctx.i_level = params.values["i_cc"]
            return ctx, None
        if ctx.phase == "cv" and ceiling_hit:
            ctx.i_level = max(params.values["i_cutoff"], ctx.i_level - b.cv_step_a)
        return ctx, None

    def end_check(self, state, ctx, params, *, ceiling_hit, step_samples, target_i):
        if ctx.phase == "cv" and target_i != 0.0:
            if ctx.i_level <= params.values["i_cutoff"] + 1e-6 and ceiling_hit and step_samples <= 1:
                return "CV cutoff current"
        return None


class TwoStepFamily(ProfileFamily):
    family_id = "two_step"
    label = "2-step (SoC)"

    @classmethod
    def param_bounds(cls) -> Dict[str, Tuple[float, float]]:
        b = space()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = _two_step_soc_bounds()
        return {
            "i1": (min(i_lo + b.min_step_di_a, i_hi), i_hi),
            "i2": (i_lo, i_hi),
            "soc_switch": (soc_lo, soc_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        b = space()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = _two_step_soc_bounds()
        i1_raw, i2_raw = float(values["i1"]), float(values["i2"])
        if abs(i1_raw - i2_raw) < 1e-9:
            i1 = float(np.clip(i1_raw, i_lo, i_hi))
            i2 = i1
            soc_sw = float(np.clip(values.get("soc_switch", soc_lo), soc_lo, soc_hi))
        else:
            i1 = float(np.clip(i1_raw, i_lo + b.min_step_di_a, i_hi))
            i2 = float(np.clip(i2_raw, i_lo, max(i_lo, i1 - b.min_step_di_a)))
            soc_sw = float(np.clip(values["soc_switch"], soc_lo, soc_hi))
        return ProfileParams(family_id=cls.family_id, values={"i1": i1, "i2": i2, "soc_switch": soc_sw})

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        b = space()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = _two_step_soc_bounds()
        i1 = float(rng.uniform(i_lo + b.min_step_di_a, i_hi))
        i2 = float(rng.uniform(i_lo, max(i_lo, i1 - b.min_step_di_a)))
        return cls.from_dict({"i1": i1, "i2": i2, "soc_switch": float(rng.uniform(soc_lo, soc_hi))})

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        b = space()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = _two_step_soc_bounds()
        mid = 0.5 * (soc_lo + soc_hi)
        seeds = [
            {"i1": min(3.0, i_hi), "i2": max(i_lo, 1.5), "soc_switch": mid},
            {"i1": min(2.0, i_hi), "i2": max(i_lo, 1.0), "soc_switch": mid},
            {"i1": min(4.0, i_hi), "i2": max(i_lo, 2.0), "soc_switch": soc_lo},
            {"i1": i_hi, "i2": max(i_lo, 0.5 * (i_lo + i_hi)), "soc_switch": soc_hi},
        ]
        return [cls.from_dict(dict(t)) for t in {tuple(sorted(s.items())) for s in seeds}]

    def _bulk_current(self, params):
        return params.values["i1"]

    def _commanded(self, state, params):
        return params.values["i1"] if float(state["soc"]) < params.values["soc_switch"] else params.values["i2"]

    def init_context(self, params):
        ctx = super().init_context(params)
        ctx.i_level = self._commanded({"soc": _soc_start()}, params)
        return ctx

    def target_current(self, state, ctx, params):
        ctx.i_level = self._commanded(state, params)
        return -ctx.i_level

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        b = space()
        if ceiling_hit:
            ctx.i_level = max(b.min_charge_a, ctx.i_level - b.cv_step_a)
        return ctx, None


class ThreeStepFamily(ProfileFamily):
    family_id = "three_step"
    label = "3-step (SoC)"

    @classmethod
    def param_bounds(cls) -> Dict[str, Tuple[float, float]]:
        b = space()
        i_lo, i_hi = b.i_bounds()
        (soc1_lo, soc1_hi), (soc2_lo, soc2_hi) = _three_step_soc_bounds()
        return {
            "i1": (min(i_lo + 2.0 * b.min_step_di_a, i_hi), i_hi),
            "i2": (i_lo, i_hi),
            "i3": (i_lo, i_hi),
            "soc1": (soc1_lo, soc1_hi),
            "soc2": (soc2_lo, soc2_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        b = space()
        i_lo, i_hi = b.i_bounds()
        (soc1_lo, soc1_hi), (soc2_lo, soc2_hi) = _three_step_soc_bounds()
        i1_raw, i2_raw, i3_raw = float(values["i1"]), float(values["i2"]), float(values["i3"])
        want_flat = abs(i1_raw - i2_raw) < 1e-9 and abs(i2_raw - i3_raw) < 1e-9
        if want_flat:
            i1 = i2 = i3 = float(np.clip(i1_raw, i_lo, i_hi))
        else:
            i1 = float(np.clip(i1_raw, i_lo + 2.0 * b.min_step_di_a, i_hi))
            i2 = float(np.clip(i2_raw, i_lo + b.min_step_di_a, max(i_lo + b.min_step_di_a, i1 - b.min_step_di_a)))
            i3 = float(np.clip(i3_raw, i_lo, max(i_lo, i2 - b.min_step_di_a)))
        soc1 = float(np.clip(values["soc1"], soc1_lo, soc1_hi))
        soc2 = float(np.clip(values["soc2"], max(soc2_lo, soc1 + b.min_stage_gap_dsoc), soc2_hi))
        return ProfileParams(family_id=cls.family_id, values={"i1": i1, "i2": i2, "i3": i3, "soc1": soc1, "soc2": soc2})

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        b = space()
        i_lo, i_hi = b.i_bounds()
        (soc1_lo, soc1_hi), (soc2_lo, soc2_hi) = _three_step_soc_bounds()
        i1 = float(rng.uniform(i_lo + 2.0 * b.min_step_di_a, i_hi))
        i2 = float(rng.uniform(i_lo + b.min_step_di_a, max(i_lo + b.min_step_di_a, i1 - b.min_step_di_a)))
        i3 = float(rng.uniform(i_lo, max(i_lo, i2 - b.min_step_di_a)))
        soc1 = float(rng.uniform(soc1_lo, soc1_hi))
        soc2_floor = max(soc2_lo, soc1 + b.min_stage_gap_dsoc)
        soc2 = float(rng.uniform(soc2_floor, max(soc2_floor + 1e-6, soc2_hi)))
        return cls.from_dict({"i1": i1, "i2": i2, "i3": i3, "soc1": soc1, "soc2": soc2})

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        b = space()
        i_lo, i_hi = b.i_bounds()
        (soc1_lo, soc1_hi), (_, soc2_hi) = _three_step_soc_bounds()
        mid1 = 0.5 * (soc1_lo + soc1_hi)
        seeds = [
            {"i1": min(3.0, i_hi), "i2": min(2.0, i_hi), "i3": max(i_lo, 1.0),
             "soc1": mid1, "soc2": mid1 + b.min_stage_gap_dsoc},
            {"i1": min(4.0, i_hi), "i2": min(2.5, i_hi), "i3": max(i_lo, 1.25),
             "soc1": soc1_lo, "soc2": soc1_lo + b.min_stage_gap_dsoc},
            {"i1": i_hi, "i2": max(i_lo + b.min_step_di_a, 0.6 * i_hi),
             "i3": i_lo, "soc1": mid1, "soc2": min(soc2_hi, mid1 + 0.20)},
        ]
        return [cls.from_dict(dict(t)) for t in {tuple(sorted(s.items())) for s in seeds}]

    def _bulk_current(self, params):
        return params.values["i1"]

    def _commanded(self, state, params):
        soc, v = float(state["soc"]), params.values
        if soc < v["soc1"]:
            return v["i1"]
        if soc < v["soc2"]:
            return v["i2"]
        return v["i3"]

    def init_context(self, params):
        ctx = super().init_context(params)
        ctx.i_level = self._commanded({"soc": _soc_start()}, params)
        return ctx

    def target_current(self, state, ctx, params):
        ctx.i_level = self._commanded(state, params)
        return -ctx.i_level

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        b = space()
        if ceiling_hit:
            ctx.i_level = max(b.min_charge_a, ctx.i_level - b.cv_step_a)
        return ctx, None


class PulsedFamily(ProfileFamily):
    family_id = "pulsed"
    label = "Pulsed charge/rest"

    @classmethod
    def param_bounds(cls) -> Dict[str, Tuple[float, float]]:
        b = space()
        i_lo, i_hi = b.i_bounds()
        return {
            "i_charge": (i_lo, i_hi),
            "pulse_on_min": (b.pulse_on_min_min, b.pulse_on_max_min),
            "rest_fraction": (b.rest_fraction_min, b.rest_fraction_max),
            "i_floor": (i_lo, i_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        b = space()
        i_lo, i_hi = b.i_bounds()
        i_cc = float(np.clip(values["i_charge"], i_lo, i_hi))
        pulse_on = float(np.clip(values["pulse_on_min"], b.pulse_on_min_min, b.pulse_on_max_min))
        rest_frac = float(np.clip(values["rest_fraction"], b.rest_fraction_min, b.rest_fraction_max))
        i_floor = float(np.clip(values["i_floor"], i_lo, min(i_cc - 0.05, i_hi)))
        return ProfileParams(family_id=cls.family_id, values={
            "i_charge": i_cc, "pulse_on_min": pulse_on,
            "pulse_rest_min": max(0.5, pulse_on * rest_frac),
            "rest_fraction": rest_frac, "i_floor": i_floor,
        })

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        b = space()
        i_lo, i_hi = b.i_bounds()
        i_charge = float(rng.uniform(i_lo, i_hi))
        return cls.from_dict({
            "i_charge": i_charge,
            "pulse_on_min": float(rng.uniform(b.pulse_on_min_min, b.pulse_on_max_min)),
            "rest_fraction": float(rng.uniform(b.rest_fraction_min, b.rest_fraction_max)),
            "i_floor": float(rng.uniform(i_lo, max(i_lo, i_charge - 0.05))),
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        b = space()
        i_lo, i_hi = b.i_bounds()
        pul = {"i_charge": 1.0, "pulse_on_min": 10.0, "rest_fraction": 2.0, "i_floor": 0.75}
        seeds = [pul, {**pul, "i_charge": i_hi}, {**pul, "i_charge": 0.5 * (i_lo + i_hi)}]
        return [cls.from_dict(dict(t)) for t in {tuple(sorted(s.items())) for s in seeds}]

    def _bulk_current(self, params):
        return params.values["i_charge"]

    def init_context(self, params):
        ctx = SimulationContext(phase="pulsed", i_level=params.values["i_charge"])
        ctx.extra["pulse_on_s"] = params.values["pulse_on_min"] * 60.0
        ctx.extra["pulse_rest_s"] = params.values["pulse_rest_min"] * 60.0
        ctx.extra["i_floor"] = params.values["i_floor"]
        return ctx

    def target_current(self, state, ctx, params):
        if ctx.in_rest:
            return 0.0
        if ctx.extra["pulse_rest_s"] > 0.0 and ctx.charge_elapsed >= ctx.extra["pulse_on_s"]:
            ctx.in_rest = True
            ctx.rest_elapsed = 0.0
            return 0.0
        return -ctx.i_level

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        b = space()
        i_floor = ctx.extra["i_floor"]
        if ctx.in_rest:
            ctx.rest_elapsed += v_traj.size
            if ctx.rest_elapsed >= ctx.extra["pulse_rest_s"]:
                ctx.in_rest = False
                ctx.charge_elapsed = 0.0
        elif ceiling_hit and not ctx.in_rest and ctx.i_level > i_floor + 1e-9:
            ctx.i_level = max(i_floor, ctx.i_level - b.cv_step_a)
            ctx.charge_elapsed = 0.0
        return ctx, None

    def end_check(self, state, ctx, params, *, ceiling_hit, step_samples, target_i):
        if (ceiling_hit and ctx.i_level <= ctx.extra["i_floor"] + 1e-9
                and target_i != 0.0 and step_samples <= 1):
            return "V ceiling @ min current"
        return None


FAMILY_REGISTRY: Dict[str, Type[ProfileFamily]] = {
    CCCVFamily.family_id: CCCVFamily,
    TwoStepFamily.family_id: TwoStepFamily,
    ThreeStepFamily.family_id: ThreeStepFamily,
    PulsedFamily.family_id: PulsedFamily,
}


def get_family(family_id: str) -> ProfileFamily:
    if family_id not in FAMILY_REGISTRY:
        raise KeyError(f"Unknown family {family_id!r}")
    return FAMILY_REGISTRY[family_id]()

"""Hybrid calendar + cyclic degradation model.

Equations are unchanged from the study methodology; the coefficients are
re-derived in this project (``scripts/03_fit_degradation.py``) and loaded from
``configs/degradation_fitted.yaml``. This module hard-codes no fitted
coefficient.

Model
-----
::

    Q_cal(SOC, T, t_h) = A_cal * exp(B_cal*SOC)
                              * exp(-(Ea_cal + C_cal*SOC) / (R*T_K))
                              * t_h ** z_cal

    Q_cyc(Ah, T, C)    = B_cyc(C) * exp(-Ea_cyc(C) / (R*T_K)) * Ah ** z_cyc(C)

    Q_total            = Q_cal + Q_cyc

The cyclic coefficients live on a C-rate grid (default ``{0.5, 1.0, 2.0}``).
Throughput at an arbitrary C-rate is split between the two neighbouring grid
nodes with linear ("tent") weights, clamped outside the grid, and each node
carries its own accumulator. At a grid node this is exactly the single-rate
equation above; across a duty cycle spanning several C-rates it is the natural
mixture, ``Q_cyc = sum_b Q_cyc(Ah_b, T, C_b)``. The weighting is continuous in
C, so the objective surface the optimizer sees has no bin-boundary steps.

Accumulation under time-varying stress
--------------------------------------
A sub-linear power law cannot be advanced by summing increments evaluated at
different stresses. Equivalent-time (state-shift) integration is used:

    t_eq = (Q/k)**(1/z)          Q <- k * (t_eq + delta) ** z

Substituting ``u = Q**(1/z)`` turns this recursion into a plain sum, which is
both the exact solution and the reason the fit is cheap:

    u_m = u_{m-1} + k_m**(1/z) * delta_m        Q_m = u_m ** z

so ``Q_M = ( sum_m k_m**(1/z) * delta_m ) ** z``. Properties: exact reduction
to the closed form under constant stress, continuity when the stress changes,
monotone non-decreasing, and independent of the order in which stresses are
applied.
"""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

R_UNIVERSAL = 8.314          # J mol^-1 K^-1
KELVIN_OFFSET = 273.15
MIN_KELVIN = 200.0
Z_FLOOR = 1e-6


@dataclass(frozen=True)
class DegradationParameters:
    """Calibrated physical coefficients. No reward weights live here."""

    A_cal: float
    B_cal: float
    C_cal: float
    Ea_cal: float
    z_cal: float
    c_rate_bins: Tuple[float, ...]
    B_cyc: Tuple[float, ...]
    Ea_cyc: Tuple[float, ...]
    z_cyc: Tuple[float, ...]
    R: float = R_UNIVERSAL
    q_nominal_ah: float = 2.2
    provenance: str = "unspecified"

    def __post_init__(self) -> None:
        n = len(self.c_rate_bins)
        for name in ("B_cyc", "Ea_cyc", "z_cyc"):
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"{name} has {len(getattr(self, name))} entries, expected {n}"
                )
        if list(self.c_rate_bins) != sorted(self.c_rate_bins):
            raise ValueError("c_rate_bins must be ascending")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "A_cal": float(self.A_cal),
            "B_cal": float(self.B_cal),
            "C_cal": float(self.C_cal),
            "Ea_cal": float(self.Ea_cal),
            "z_cal": float(self.z_cal),
            "c_rate_bins": [float(x) for x in self.c_rate_bins],
            "B_cyc": [float(x) for x in self.B_cyc],
            "Ea_cyc": [float(x) for x in self.Ea_cyc],
            "z_cyc": [float(x) for x in self.z_cyc],
            "R": float(self.R),
            "q_nominal_ah": float(self.q_nominal_ah),
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DegradationParameters":
        return cls(
            A_cal=float(d["A_cal"]),
            B_cal=float(d["B_cal"]),
            C_cal=float(d["C_cal"]),
            Ea_cal=float(d["Ea_cal"]),
            z_cal=float(d["z_cal"]),
            c_rate_bins=tuple(float(x) for x in d["c_rate_bins"]),
            B_cyc=tuple(float(x) for x in d["B_cyc"]),
            Ea_cyc=tuple(float(x) for x in d["Ea_cyc"]),
            z_cyc=tuple(float(x) for x in d["z_cyc"]),
            R=float(d.get("R", R_UNIVERSAL)),
            q_nominal_ah=float(d.get("q_nominal_ah", 2.2)),
            provenance=str(d.get("provenance", "unspecified")),
        )

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DegradationParameters":
        """Deployed pooled fit stored in ``config.DEGRADATION``."""
        del path
        return cls.from_dict(DEGRADATION)


def calendar_from_anchors(
    anchors: Dict[str, Any], *, R: float = R_UNIVERSAL, scale: float = 1.0,
) -> Dict[str, float]:
    """Derive the calendar coefficients from interpretable literature anchors.

    NASA RW cannot identify the calendar branch (no storage arm; ~9 % rest, all
    at low SOC; elapsed time collinear with throughput at r ~ 0.99), so the
    branch is fixed from published calendar-aging results instead of fitted —
    see the ``calendar_branch`` block of ``configs/degradation.yaml`` for the
    disclosure and citations.

    The anchors are checkable physical statements rather than coefficients:

    ``z_cal``
        Power-law exponent (0.5 = SEI-diffusion-limited sqrt-of-time).
    ``Ea_cal_j_per_mol``
        Apparent activation energy of calendar fade.
    ``soc_acceleration_ratio``
        Rate ratio between SOC = 1 and SOC = 0; ``B_cal = ln(ratio)``.
    ``fade_fraction_at_anchor`` at (``anchor_time_h``, ``anchor_soc``,
    ``anchor_temperature_k``)
        Pins the remaining scale degree of freedom, so ``A_cal`` is *solved
        for* rather than asserted.

    ``scale`` multiplies the anchored fade level and exists for the declared
    sensitivity sweep — it is the knob the downstream robustness check turns,
    and ``scale = 0`` gives the pure-cyclic limit.
    """
    z_cal = float(anchors["z_cal"])
    ea_cal = float(anchors["Ea_cal_j_per_mol"])
    c_cal = float(anchors["C_cal"])
    ratio = float(anchors["soc_acceleration_ratio"])
    if ratio <= 0.0:
        raise ValueError("soc_acceleration_ratio must be positive")
    b_cal = float(np.log(ratio))

    y_anchor = float(anchors["fade_fraction_at_anchor"]) * float(scale)
    t_h = float(anchors["anchor_time_h"])
    soc = float(anchors["anchor_soc"])
    t_k = float(anchors["anchor_temperature_k"])
    if t_h <= 0.0:
        raise ValueError("anchor_time_h must be positive")

    shape = (
        np.exp(b_cal * soc)
        * np.exp(-(ea_cal + c_cal * soc) / (R * t_k))
        * t_h ** z_cal
    )
    a_cal = 0.0 if y_anchor <= 0.0 else float(y_anchor / shape)
    return {
        "A_cal": a_cal,
        "B_cal": b_cal,
        "C_cal": c_cal,
        "Ea_cal": ea_cal,
        "z_cal": z_cal,
    }


def kelvin(t_c) -> np.ndarray:
    return np.maximum(np.asarray(t_c, dtype=np.float64) + KELVIN_OFFSET, MIN_KELVIN)


def tent_weights(c_rate, bins: Sequence[float]) -> np.ndarray:
    """Linear split of throughput between neighbouring C-rate grid nodes.

    Returns shape ``(..., n_bins)`` rows summing to 1. Continuous in ``c_rate``;
    clamped to the end node outside the grid.
    """
    c = np.abs(np.asarray(c_rate, dtype=np.float64))
    g = np.asarray(bins, dtype=np.float64)
    n = g.size
    w = np.zeros(c.shape + (n,), dtype=np.float64)
    if n == 1:
        w[..., 0] = 1.0
        return w
    cc = np.clip(c, g[0], g[-1])
    hi = np.clip(np.searchsorted(g, cc, side="left"), 1, n - 1)
    lo = hi - 1
    span = g[hi] - g[lo]
    frac = np.where(span > 0, (cc - g[lo]) / span, 0.0)
    idx = np.indices(c.shape)
    w[(*idx, lo)] = 1.0 - frac
    w[(*idx, hi)] += frac
    return w


def bin_throughput(
    c_rate, ah, bins: Sequence[float],
) -> np.ndarray:
    """Sum ``ah`` into C-rate bins using tent weights. Returns ``(n_bins,)``."""
    c = np.asarray(c_rate, dtype=np.float64).ravel()
    a = np.asarray(ah, dtype=np.float64).ravel()
    if c.size == 0:
        return np.zeros(len(bins))
    return (tent_weights(c, bins) * a[:, None]).sum(axis=0)


class HybridDegradationModel:
    """Evaluate the calibrated model: closed form, cumulative, or per-session."""

    def __init__(self, params: DegradationParameters):
        self.p = params
        self._bins = np.asarray(params.c_rate_bins, dtype=np.float64)
        self._B = np.asarray(params.B_cyc, dtype=np.float64)
        self._Ea = np.asarray(params.Ea_cyc, dtype=np.float64)
        self._z = np.maximum(np.asarray(params.z_cyc, dtype=np.float64), Z_FLOOR)
        self._z_cal = max(float(params.z_cal), Z_FLOOR)

    # ── stress-dependent rate coefficients ──────────────────────────────────

    def calendar_k(self, soc, t_c) -> np.ndarray:
        """k in Q_cal = k * t_h**z_cal."""
        p = self.p
        s = np.clip(np.asarray(soc, dtype=np.float64), 0.0, 1.0)
        return (
            p.A_cal
            * np.exp(p.B_cal * s)
            * np.exp(-(p.Ea_cal + p.C_cal * s) / (p.R * kelvin(t_c)))
        )

    def cyclic_k_bins(self, t_c) -> np.ndarray:
        """k per grid node in Q_cyc,b = k_b * Ah_b**z_b. Shape ``(..., n_bins)``."""
        tk = kelvin(t_c)
        return self._B * np.exp(-self._Ea / (self.p.R * np.atleast_1d(tk)[..., None]))

    def cyclic_coeffs_at(self, c_rate) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Coefficients linearly interpolated to an arbitrary C-rate (reporting)."""
        c = np.abs(np.asarray(c_rate, dtype=np.float64))
        return (
            np.interp(c, self._bins, self._B),
            np.interp(c, self._bins, self._Ea),
            np.interp(c, self._bins, self._z),
        )

    # ── closed forms (constant stress) ──────────────────────────────────────

    def q_calendar(self, soc, t_c, t_hours) -> np.ndarray:
        t_h = np.maximum(np.asarray(t_hours, dtype=np.float64), 0.0)
        return self.calendar_k(soc, t_c) * np.power(t_h, self._z_cal)

    def q_cyclic_binned(self, ah_bins, t_c) -> np.ndarray:
        """Q_cyc from per-bin throughput. ``ah_bins`` shape ``(..., n_bins)``."""
        a = np.maximum(np.asarray(ah_bins, dtype=np.float64), 0.0)
        k = self.cyclic_k_bins(t_c)
        return np.sum(k * np.power(a, self._z), axis=-1)

    def q_cyclic(self, ah, t_c, c_rate) -> np.ndarray:
        """Q_cyc for throughput ``ah`` all delivered at a single C-rate."""
        a = np.atleast_1d(np.asarray(ah, dtype=np.float64))
        w = tent_weights(np.atleast_1d(np.asarray(c_rate, dtype=np.float64)), self._bins)
        out = self.q_cyclic_binned(w * a[..., None], t_c)
        return out if np.ndim(ah) else float(np.asarray(out).ravel()[0])

    def session_closed_form(
        self,
        *,
        current_a: np.ndarray,
        temperature_c: np.ndarray,
        soc: np.ndarray,
        dt_s: float = 1.0,
        rest_current_threshold_a: float = 0.01,
    ) -> Dict[str, float]:
        """Paper session-mean closed form (Eqs. calendar + cyclic).

        Calendar uses full-session mean SOC / T and elapsed hours.
        Cyclic uses charge-only Ah and mean charge C-rate, with the calibrated
        C-rate grid (tent interpolation of B_cyc, Ea_cyc, z_cyc).
        """
        i = np.asarray(current_a, dtype=np.float64)
        t_c = np.asarray(temperature_c, dtype=np.float64)
        s = np.asarray(soc, dtype=np.float64)
        n = int(i.size)
        duration_h = n * float(dt_s) / 3600.0
        charging = i < -abs(float(rest_current_threshold_a))
        q_ah = float(self.p.q_nominal_ah)
        if charging.any():
            ah = float(np.sum(np.abs(i[charging])) * float(dt_s) / 3600.0)
            c_rate = float(np.mean(np.abs(i[charging])) / max(q_ah, 1e-12))
        else:
            ah, c_rate = 0.0, 0.0
        mean_soc = float(s.mean()) if s.size else 0.0
        mean_t = float(t_c.mean()) if t_c.size else 25.0
        q_cal = float(np.asarray(self.q_calendar(mean_soc, mean_t, duration_h)).ravel()[0])
        q_cyc = float(self.q_cyclic(ah, mean_t, c_rate))
        return {
            "duration_h": duration_h,
            "mean_soc": mean_soc,
            "mean_temperature_c": mean_t,
            "ah_throughput": ah,
            "nominal_c_rate": c_rate,
            "delta_soc": float(s[-1] - s[0]) if s.size else 0.0,
            "q_calendar": q_cal,
            "q_cyclic": q_cyc,
            "q_total": q_cal + q_cyc,
        }


"""Paper reward (Eq. 10) and energy integrals.

    R = w_soc * ΔSOC − w_loss * Q_loss − w_time * t_h^{z_time}

Q_loss uses the calibrated hybrid model in the paper's session-mean closed form.
"""


from typing import Any, Dict, Optional

import numpy as np


def full_capacity_joules(q_rated_as: float, v_nom: float) -> float:
    return float(q_rated_as) * float(v_nom)


def energy_required_j(q_rated_as: float, energy_fraction: float, v_nom: float) -> float:
    return float(energy_fraction) * full_capacity_joules(q_rated_as, v_nom)


def energy_delivered_j(voltage_v, current_a, time_s) -> float:
    v = np.asarray(voltage_v, dtype=np.float64)
    i = np.asarray(current_a, dtype=np.float64)
    t = np.asarray(time_s, dtype=np.float64)
    if v.size == 0:
        return 0.0
    power_w = -v * i
    if t.size <= 1:
        dt = 1.0 if t.size == 0 else float(t[0] if t[0] > 0 else 1.0)
        return float(max(0.0, power_w[0] * dt))
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(max(0.0, trapz(power_w, t)))


def score_session(
    session: Dict[str, Any],
    *,
    model: HybridDegradationModel,
    spec: RewardSpec,
    anchors: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    del anchors
    i = np.asarray(session["current_a"], dtype=np.float64)
    v = np.asarray(session["voltage_v"], dtype=np.float64)
    t_c = np.asarray(session["temperature_c"], dtype=np.float64)
    soc = np.asarray(session["soc"], dtype=np.float64)
    time_s = np.asarray(session["time_s"], dtype=np.float64)
    duration_s = float(i.size)
    duration_min = duration_s / 60.0

    e_del = energy_delivered_j(v, i, time_s)
    e_req = float(session.get("energy_required_j", 0.0)) or energy_required_j(
        float(session.get("q_rated_as", 2.2 * 3600.0)),
        float(session.get("energy_fraction", 0.40)),
        float(session.get("v_nom", 3.7)),
    )
    e_norm = float(e_del / max(e_req, 1e-12))

    deg = model.session_closed_form(
        current_a=i, temperature_c=t_c, soc=soc, dt_s=1.0,
    )
    q_total = float(deg["q_total"])
    delta_soc = float(deg["delta_soc"])
    t_h = float(deg["duration_h"])
    time_raw = (t_h ** spec.z_time) if t_h > 0.0 else 0.0

    soc_term = spec.w_soc * delta_soc
    q_term = spec.w_loss * q_total
    time_term = spec.w_time * time_raw
    # reward = REWARD_SCALE * float(soc_term - q_term - time_term)
    reward = 1000.0 * float(soc_term - q_term - time_term)
    if not np.isfinite(reward):
        reward = -1e6 * 1000.0

    shortfall = max(0.0, 1.0 - e_norm)
    peak_v = float(v.max()) if v.size else 0.0
    v_over = max(0.0, peak_v - float(session.get("v_max", 4.2)))
    # Penalties stay on the same scale as the ×1000 reward, so a charge that
    # misses the energy target cannot outrank one that finishes.
    penalty = 1000.0 * (
        spec.energy_shortfall_penalty_scale * shortfall
        + spec.voltage_ceiling_penalty_scale * v_over
    )
    feasible = (e_del >= e_req - 1e-3) and v_over <= 1e-3
    loss = (
        -float(reward)
        + spec.duration_loss_weight * duration_min
        + penalty
    )
    return {
        "reward": float(reward),
        "loss": float(loss),
        "feasible": bool(feasible),
        "soc_reward": float(soc_term),
        "qloss_penalty": float(q_term),
        "time_penalty": float(time_term),
        "time_penalty_raw": float(time_raw),
        "e_delivered_j": e_del,
        "e_required_j": e_req,
        "e_norm": float(e_norm),
        "delta_soc": delta_soc,
        "soc_end": float(soc[-1]) if soc.size else 0.0,
        "q_calendar": float(deg["q_calendar"]),
        "q_cyclic": float(deg["q_cyclic"]),
        "q_total": q_total,
        "qloss_total": q_total,
        "ah_throughput": float(deg["ah_throughput"]),
        "nominal_c_rate": float(deg["nominal_c_rate"]),
        "mean_soc": float(deg["mean_soc"]),
        "duration_s": duration_s,
        "duration_h": t_h,
        "duration_min": duration_min,
        "peak_v": peak_v,
        "peak_t": float(t_c.max()) if t_c.size else 0.0,
        "mean_t": float(deg["mean_temperature_c"]),
        "mean_charge_c_rate": float(deg["nominal_c_rate"]),
        "ah_charge": float(deg["ah_throughput"]),
        "end_reason": session.get("end_reason"),
        "energy_shortfall": float(shortfall),
        "penalty": float(penalty),
        "weights": spec.to_dict(),
    }

"""Closed-loop charging rollout on the frozen BDT."""


from typing import Dict, List, Optional

import numpy as np



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

"""Session evaluation helpers: start state, simulate + score."""


from typing import Dict, Optional, Tuple



def load_ocv(cell: str) -> OcvCurve:
    from twin import load_ocv as _load_ocv
    return _load_ocv(cell)


def start_state_and_vnom(cell: str, session: SessionSpec) -> Tuple[Dict[str, float], float]:
    ocv = load_ocv(cell)
    v0 = float(ocv.voltage_from_soc(session.soc_start))
    v_nom = float(ocv.nominal_voltage()) if ocv.nominal_voltage() > 0 else session.v_nom_fallback
    return {
        "soc": float(session.soc_start),
        "v0": v0,
        "t0": float(session.ambient_t0_c),
        "age": 0.0,
        "prev_i": 0.0,
    }, v_nom


def load_degradation_model() -> HybridDegradationModel:
    return HybridDegradationModel(DegradationParameters.load())


def load_anchors(cell: str) -> Dict[str, float]:
    del cell
    return {}


def evaluate_params(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    params: ProfileParams,
    *,
    model: HybridDegradationModel,
    spec: RewardSpec,
    anchors: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict, Dict]:
    family = get_family(params.family_id)
    session = simulator.simulate(initial_state, params, family=family)
    metrics = score_session(session, model=model, spec=spec, anchors=anchors)
    return float(metrics["loss"]), metrics, session


def make_simulator(cell: str, device: str = "auto", session: Optional[SessionSpec] = None) -> Tuple[ChargingSimulator, Dict[str, float]]:
    bind_search()
    spec = OptimizationSpec.load()
    sess = spec.session if session is None else session
    paths = Paths.load()
    state, v_nom = start_state_and_vnom(cell, sess)
    q_ah = float(DEGRADATION["q_nominal_ah"])
    bdt = FrozenBDT(paths.checkpoint(cell), device=device)
    sim = ChargingSimulator(bdt, q_rated_as=q_ah * 3600.0, session=sess, v_nom=v_nom)
    return sim, state


def cccv_params(c_rate: float, *, v_cv: float = 4.20, i_cutoff: float = 0.05) -> ProfileParams:
    q_ah = float(DEGRADATION["q_nominal_ah"])
    bind_search()
    return CCCVFamily.from_dict({
        "i_cc": float(c_rate) * q_ah,
        "v_cv": float(v_cv),
        "i_cutoff": float(i_cutoff),
    })


def jsonable_session(session: Dict) -> Dict:
    out = {}
    for k, v in session.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def jsonable_family_result(result: Dict, *, include_session: bool = True) -> Dict:
    skip = set() if include_session else {"best_session"}
    out = {}
    for k, v in result.items():
        if k in skip:
            continue
        if k == "best_session" and v is not None:
            out[k] = jsonable_session(v)
        else:
            out[k] = v
    return out

"""GP-BO over one profile family (skopt), same objective as random search."""


from typing import Any, Dict, List, Sequence

import numpy as np
from skopt import gp_minimize


DUPLICATE_JITTER_SCALE = 1e-4


def ensure_twin_eval(simulator: ChargingSimulator) -> None:
    """Put the digital twin in inference mode for every GP-BO evaluation.

    The degradation ``model`` argument is not a torch module. Dropout lives
    on ``simulator.bdt.model``.
    """
    twin = getattr(getattr(simulator, "bdt", None), "model", None)
    if twin is not None and hasattr(twin, "eval"):
        twin.eval()


def _vector_key(x: Sequence[float], ndigits: int = 6) -> tuple:
    return tuple(round(float(v), ndigits) for v in x)


def optimize_family_gp_bo(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family_id: str,
    *,
    model,
    spec,
    anchors: Dict[str, float],
    n_calls: int,
    n_initial_points: int,
    seed: int,
    acq_func: str = "EI",
    acq_xi: float = 0.01,
    noise: float = 1e-6,
) -> Dict[str, Any]:
    ensure_twin_eval(simulator)
    family = get_family(family_id)
    cls = type(family)
    history: List[Dict[str, Any]] = []
    seen = set()
    rng = np.random.default_rng(int(seed))
    space = cls.search_space()
    lo = [float(d.low) for d in space]
    hi = [float(d.high) for d in space]

    def _jitter(x: List[float]) -> List[float]:
        key = _vector_key(x)
        if key not in seen:
            return list(x)
        out = list(x)
        for _ in range(8):
            for i, (a, b) in enumerate(zip(lo, hi)):
                span = max(b - a, 1e-9)
                out[i] = float(np.clip(out[i] + rng.normal(0.0, DUPLICATE_JITTER_SCALE * span), a, b))
            if _vector_key(out) not in seen:
                return out
        return out

    def objective(x: List[float]) -> float:
        ensure_twin_eval(simulator)
        x = _jitter(list(x))
        seen.add(_vector_key(x))
        params = cls.from_vector(x)
        loss, metrics, _ = evaluate_params(
            simulator, initial_state, params, model=model, spec=spec, anchors=anchors,
        )
        history.append({
            "family_id": family_id,
            "params": params.to_dict(),
            "loss": float(loss),
            "reward": float(metrics["reward"]),
            "feasible": bool(metrics["feasible"]),
            "metrics": {k: v for k, v in metrics.items() if k not in ("weights", "anchors")},
        })
        return float(loss)

    seeds = list(cls.seed_points())
    n_seed = len(seeds)
    n_random_extra = max(0, int(n_initial_points) - n_seed)
    n_calls_eff = max(int(n_calls), n_seed + n_random_extra)
    gp_kwargs: Dict[str, Any] = {
        "func": objective,
        "dimensions": space,
        "n_calls": n_calls_eff,
        "n_initial_points": n_random_extra,
        "x0": seeds if seeds else None,
        "random_state": int(seed),
        "acq_func": acq_func,
        "noise": float(noise),
    }
    if acq_func == "EI":
        gp_kwargs["xi"] = float(acq_xi)
    skopt_result = gp_minimize(**gp_kwargs)

    feasible = [h for h in history if h["feasible"]]
    pool = feasible if feasible else history
    best = min(pool, key=lambda h: h["loss"])
    best_params = cls.from_dict(best["params"])
    ensure_twin_eval(simulator)
    _, best_metrics, best_session = evaluate_params(
        simulator, initial_state, best_params, model=model, spec=spec, anchors=anchors,
    )
    return {
        "family_id": family_id,
        "family_label": family.label,
        "method": "gp_bo",
        "best_params": best_params.to_dict(),
        "best_loss": float(best["loss"]),
        "best_reward": float(best_metrics["reward"]),
        "best_metrics": best_metrics,
        "best_session": best_session,
        "history": history,
        "n_evaluated": len(history),
        "n_calls": n_calls_eff,
        "n_seed_points": n_seed,
        "n_initial_points": int(n_initial_points),
        "acq_func": acq_func,
        "seed": int(seed),
        "skopt_fun": float(skopt_result.fun) if skopt_result is not None else None,
        "warm_start_from_random_search": False,
    }

"""Random search over one profile family, identical budget to GP-BO."""


from typing import Any, Dict, List

import numpy as np



def optimize_family_random(
    simulator: ChargingSimulator,
    initial_state: Dict[str, float],
    family_id: str,
    *,
    model,
    spec,
    anchors: Dict[str, float],
    n_calls: int,
    seed: int,
) -> Dict[str, Any]:
    family = get_family(family_id)
    cls = type(family)
    rng = np.random.default_rng(int(seed))
    history: List[Dict[str, Any]] = []
    seeds = list(cls.seed_params())
    n_seed = len(seeds)

    def _run(params) -> None:
        loss, metrics, _ = evaluate_params(
            simulator, initial_state, params, model=model, spec=spec, anchors=anchors,
        )
        history.append({
            "family_id": family_id,
            "params": params.to_dict(),
            "loss": float(loss),
            "reward": float(metrics["reward"]),
            "feasible": bool(metrics["feasible"]),
            "metrics": {k: v for k, v in metrics.items() if k not in ("weights", "anchors")},
        })

    for p in seeds:
        _run(p)
        if len(history) >= n_calls:
            break
    while len(history) < n_calls:
        _run(cls.sample_random(rng))

    feasible = [h for h in history if h["feasible"]]
    pool = feasible if feasible else history
    best = min(pool, key=lambda h: h["loss"])
    best_params = cls.from_dict(best["params"])
    _, best_metrics, best_session = evaluate_params(
        simulator, initial_state, best_params, model=model, spec=spec, anchors=anchors,
    )
    return {
        "family_id": family_id,
        "family_label": family.label,
        "method": "random_search",
        "best_params": best_params.to_dict(),
        "best_loss": float(best["loss"]),
        "best_reward": float(best_metrics["reward"]),
        "best_metrics": best_metrics,
        "best_session": best_session,
        "history": history,
        "n_evaluated": len(history),
        "n_seed_points": n_seed,
        "seed": int(seed),
    }


def evaluate_baselines(simulator, initial_state, *, model, spec, anchors):
    opt = OptimizationSpec.load()
    rows = []
    for b in opt.baselines:
        params = cccv_params(float(b["c_rate"]), v_cv=float(b["v_cv"]), i_cutoff=float(b["i_cutoff"]))
        loss, metrics, session = evaluate_params(
            simulator, initial_state, params, model=model, spec=spec, anchors=anchors,
        )
        rows.append({
            "name": b["name"],
            "c_rate": float(b["c_rate"]),
            "params": params.to_dict(),
            "loss": float(loss),
            "metrics": {k: v for k, v in metrics.items() if k not in ("weights",)},
            "end_reason": session.get("end_reason"),
            "duration_s": int(session["current_a"].size),
        })
    return rows


REQUIREMENTS = (
    {"name": "e30_t20", "energy_fraction": 0.30, "max_duration_min": 20.0},
    {"name": "e40_t30", "energy_fraction": 0.40, "max_duration_min": 30.0},
    {"name": "e50_t30", "energy_fraction": 0.50, "max_duration_min": 30.0},
    {"name": "e50_t20", "energy_fraction": 0.50, "max_duration_min": 20.0},
    {"name": "e60_t30", "energy_fraction": 0.60, "max_duration_min": 30.0},
)


def _shift_reward_fields(blob: Dict[str, Any], delta: float) -> None:
    """Add ``delta`` to reward and subtract it from loss. Ranking is unchanged."""
    if not isinstance(blob, dict):
        return
    if isinstance(blob.get("reward"), (int, float)) and "reward_raw" not in blob:
        blob["reward_raw"] = float(blob["reward"])
        blob["reward"] = float(blob["reward"]) + delta
    if isinstance(blob.get("loss"), (int, float)) and "loss_raw" not in blob:
        blob["loss_raw"] = float(blob["loss"])
        blob["loss"] = float(blob["loss"]) - delta
    metrics = blob.get("metrics")
    if isinstance(metrics, dict):
        _shift_reward_fields(metrics, delta)
    best_metrics = blob.get("best_metrics")
    if isinstance(best_metrics, dict):
        _shift_reward_fields(best_metrics, delta)
    if "best_reward" in blob and "best_reward_raw" not in blob:
        blob["best_reward_raw"] = float(blob["best_reward"])
        blob["best_reward"] = float(blob["best_reward"]) + delta
    if "best_loss" in blob and "best_loss_raw" not in blob:
        blob["best_loss_raw"] = float(blob["best_loss"])
        blob["best_loss"] = float(blob["best_loss"]) - delta
    for h in blob.get("history") or []:
        _shift_reward_fields(h, delta)


def zero_rewards_to_halfc(baselines: List[Dict[str, Any]], *blobs: Dict[str, Any]) -> float:
    """Set CCCV 0.5C reward to 0. Better profiles become positive.

    The weights make each term about 1 on a 0.5C charge, so the raw sum is
    about −1. Subtracting that raw 0.5C reward is a constant shift: GP-BO
    still selects the same profile.
    """
    half = next(b for b in baselines if b.get("name") == "CCCV 0.5C")
    metrics = half["metrics"]
    raw = float(metrics.get("reward_raw", metrics["reward"]))
    delta = -raw
    if half.get("reward_zero") == "CCCV 0.5C":
        return delta
    for row in baselines:
        _shift_reward_fields(row, delta)
        row["reward_zero"] = "CCCV 0.5C"
    for blob in blobs:
        if blob.get("reward_zero") == "CCCV 0.5C":
            continue
        _shift_reward_fields(blob, delta)
        blob["reward_zero"] = "CCCV 0.5C"
        blob["reward_shift_added"] = delta
        for fam in (blob.get("best_by_family") or {}).values():
            _shift_reward_fields(fam, delta)
        for row in blob.get("baselines") or []:
            if row.get("reward_zero") != "CCCV 0.5C":
                _shift_reward_fields(row, delta)
                row["reward_zero"] = "CCCV 0.5C"
    return delta


def shift_saved_rewards() -> None:
    """Rewrite the saved 40% search so the reported reward is vs 0.5C CCCV."""
    opt = OptimizationSpec.load()
    for method in ("gp_bo", "random_search"):
        for cell in Paths.load().cells:
            base_path = result_path(f"{method}_{cell}_baselines.json")
            if not base_path.is_file():
                continue
            baselines = read_json(base_path)
            if baselines and baselines[0].get("reward_zero") == "CCCV 0.5C":
                print(f"  already shifted {method} {cell}", flush=True)
                continue
            families = {}
            for fid in opt.families:
                path = result_path(family_result_name(method, cell, fid))
                if path.is_file():
                    families[fid] = read_json(path)
            summary_path = result_path(f"{method}_{cell}_summary.json")
            summary = read_json(summary_path) if summary_path.is_file() else None
            blobs = list(families.values())
            if summary is not None:
                blobs.append(summary)
            delta = zero_rewards_to_halfc(baselines, *blobs)
            write_json(base_path, baselines)
            for fid, blob in families.items():
                write_json(result_path(family_result_name(method, cell, fid)), blob)
            if summary is not None:
                write_json(summary_path, summary)
            print(f"  shifted {method} {cell}  by {delta:+.4f}", flush=True)


def run_cell(
    cell: str,
    method: str,
    *,
    device: str = "auto",
    families=None,
    resume: bool = True,
    n_evals=None,
    session: Optional[SessionSpec] = None,
    name_prefix: str = "",
) -> dict:
    if method not in ("random_search", "gp_bo"):
        raise ValueError(method)
    bind_search()
    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    paths = Paths.load()
    model = load_degradation_model()
    anchors = load_anchors(cell)
    simulator, state = make_simulator(cell, device=device, session=session)
    fams = list(families or opt.families)
    n_calls = int(n_evals if n_evals is not None else opt.n_evals_per_family)
    family_results = {}
    t0 = time.perf_counter()

    def _out(kind: str) -> Path:
        stem = f"{name_prefix}_{method}_{cell.upper()}_{kind}.json" if name_prefix else ""
        if name_prefix:
            return result_path(f"requirements/{stem}")
        if kind in opt.families:
            return result_path(family_result_name(method, cell, kind))
        return result_path(f"{method}_{cell.upper()}_{kind}.json")

    for fid in fams:
        fam_path = _out(fid)
        if resume and fam_path.is_file():
            print(f"  [{cell} {method}] skip existing {fid}", flush=True)
            family_results[fid] = read_json(fam_path)
            continue
        seed = opt.seed_for(cell, fid, method)
        print(f"  [{cell} {method}] {fid}  n={n_calls}  seed={seed}", flush=True)
        ft = time.perf_counter()
        if method == "random_search":
            result = optimize_family_random(
                simulator, state, fid, model=model, spec=reward, anchors=anchors,
                n_calls=n_calls, seed=seed,
            )
        else:
            n_init = min(int(opt.gp_bo["n_initial_points"]), n_calls)
            result = optimize_family_gp_bo(
                simulator, state, fid, model=model, spec=reward, anchors=anchors,
                n_calls=n_calls,
                n_initial_points=n_init,
                seed=seed,
                acq_func=str(opt.gp_bo.get("acq_func", "EI")),
                acq_xi=float(opt.gp_bo.get("acq_xi", 0.01)),
                noise=float(opt.gp_bo.get("noise", 1e-6)),
            )
        result["wall_clock_s"] = float(time.perf_counter() - ft)
        payload = jsonable_family_result(result, include_session=True)
        write_json(fam_path, payload)
        family_results[fid] = payload
        mtr = result["best_metrics"]
        print(
            f"    best R={mtr['reward']:.4f}  loss={result['best_loss']:.4f}  "
            f"feas={mtr['feasible']}  t={mtr['duration_min']:.2f} min  "
            f"Q={mtr['q_total']:.3e}  dSOC={mtr['delta_soc']:.3f}  [{mtr['end_reason']}]",
            flush=True,
        )

    baselines = evaluate_baselines(simulator, state, model=model, spec=reward, anchors=anchors)
    write_json(_out("baselines"), baselines)
    feasible_any = any(
        (family_results[f].get("best_metrics") or {}).get("feasible") for f in fams if f in family_results
    )
    summary = {
        "cell": cell.upper(),
        "method": method,
        "families": fams,
        "n_evals_per_family": n_calls,
        "reward_scale": REWARD_SCALE,
        "gp_bo_seed": GP_BO_SEED,
        "device": str(simulator.bdt.device),
        "checkpoint": str(paths.checkpoint(cell)),
        "checkpoint_sha256": file_hash(paths.checkpoint(cell)),
        "start_state": state,
        "v_nom": simulator.v_nom,
        "reward": reward.to_dict(),
        "baselines": baselines,
        "best_by_family": {
            fid: {
                "best_params": family_results[fid]["best_params"],
                "best_loss": family_results[fid]["best_loss"],
                "best_reward": family_results[fid]["best_reward"],
                "feasible": family_results[fid]["best_metrics"]["feasible"],
                "q_total": family_results[fid]["best_metrics"]["q_total"],
                "duration_min": family_results[fid]["best_metrics"].get(
                    "duration_min", family_results[fid]["best_metrics"]["duration_h"] * 60.0
                ),
                "peak_t": family_results[fid]["best_metrics"].get("peak_t"),
                "end_reason": family_results[fid]["best_metrics"]["end_reason"],
            }
            for fid in fams if fid in family_results
        },
        "any_feasible": bool(feasible_any),
        "wall_clock_s": float(time.perf_counter() - t0),
        "energy_fraction": float(simulator.session.energy_fraction),
        "max_duration_min": float(simulator.session.max_duration_min),
        "provenance": provenance(
            method, inputs=[paths.checkpoint(cell)],
        ),
    }
    zero_rewards_to_halfc(baselines, summary, *family_results.values())
    write_json(_out("baselines"), baselines)
    for fid, blob in family_results.items():
        write_json(_out(fid), blob)
    write_json(_out("summary"), summary)
    return summary


def run_requirements(*, device: str = "auto", cells=None, n_evals: int = 40) -> List[Dict[str, Any]]:
    """GP-BO under several energy targets and hard time limits.

    A profile is feasible only if it delivers the energy before the time limit.
    The reported reward is still the advantage over that case's own 0.5C CCCV.
    """
    opt = OptimizationSpec.load()
    paths = Paths.load()
    cells = [c.upper() for c in (cells or paths.cells)]
    rows: List[Dict[str, Any]] = []
    for spec in REQUIREMENTS:
        session = replace(
            opt.session,
            energy_fraction=float(spec["energy_fraction"]),
            max_duration_min=float(spec["max_duration_min"]),
        )
        print(
            f"\n######## {spec['name']}  energy={session.energy_fraction:.0%}  "
            f"limit={session.max_duration_min:.0f} min ########",
            flush=True,
        )
        for cell in cells:
            print(f"\n======== requirements {spec['name']}  {cell} ========", flush=True)
            summary = run_cell(
                cell, "gp_bo", device=device, session=session,
                name_prefix=str(spec["name"]), resume=True, n_evals=n_evals,
            )
            winner = None
            for fid, blob in summary["best_by_family"].items():
                if not blob.get("feasible"):
                    continue
                if winner is None or blob["best_loss"] < winner["best_loss"]:
                    winner = {"family": fid, **blob}
            rows.append({
                "case": spec["name"],
                "cell": cell,
                "energy_fraction": session.energy_fraction,
                "max_duration_min": session.max_duration_min,
                "winner_family": None if winner is None else winner["family"],
                "reward": None if winner is None else winner["best_reward"],
                "duration_min": None if winner is None else winner["duration_min"],
                "q_total": None if winner is None else winner["q_total"],
                "feasible": winner is not None,
                "best_params": None if winner is None else winner["best_params"],
                "by_family": {
                    fid: {
                        "feasible": blob["feasible"],
                        "reward": blob["best_reward"],
                        "duration_min": blob["duration_min"],
                        "loss": blob["best_loss"],
                    }
                    for fid, blob in summary["best_by_family"].items()
                },
            })
            fam = rows[-1]["winner_family"] or "none"
            rew = rows[-1]["reward"]
            print(f"  WINNER {cell} {spec['name']}: {fam}  R={rew}", flush=True)
    write_json(result_path("requirements/summary.json"), rows)
    _plot_requirement_winners(rows)
    return rows


def _plot_requirement_winners(rows: List[Dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = []
    for row in rows:
        if row["case"] not in cases:
            cases.append(row["case"])
    cells = []
    for row in rows:
        if row["cell"] not in cells:
            cells.append(row["cell"])
    labels = {
        "cccv": "CCCV", "two_step": "Two-step", "three_step": "Three-step",
        "pulsed": "Pulsed",
    }
    colors = {"cccv": "#2563eb", "two_step": "#16a34a", "three_step": "#ea580c", "pulsed": "#7c3aed"}
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(cases))
    width = 0.18
    for i, cell in enumerate(cells):
        heights = []
        cols = []
        for case in cases:
            hit = next(r for r in rows if r["case"] == case and r["cell"] == cell)
            fam = hit["winner_family"]
            heights.append(0.0 if hit["reward"] is None else float(hit["reward"]))
            cols.append("#94a3b8" if fam is None else colors.get(fam, "#111827"))
        offset = (i - (len(cells) - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width=width, color=cols, edgecolor="white")
        for bar, case in zip(bars, cases):
            hit = next(r for r in rows if r["case"] == case and r["cell"] == cell)
            fam = labels.get(hit["winner_family"], "infeasible")
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                fam, ha="center", va="bottom", fontsize=8, rotation=90,
            )
    ax.axhline(0.0, color="#111827", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{int(100 * next(r['energy_fraction'] for r in rows if r['case']==c))}% / "
        f"{int(next(r['max_duration_min'] for r in rows if r['case']==c))} min"
        for c in cases
    ])
    ax.set_ylabel("Reward vs 0.5C CCCV")
    ax.set_title("Winning family for each energy target and time limit")
    fig.tight_layout()
    path = result_path("requirements/family_by_requirement.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


TOP6_CASES = (
    {"name": "e30_t20", "energy_fraction": 0.30, "max_duration_min": 20.0, "label": "30% in 20 min"},
    {"name": "e40_t30", "energy_fraction": 0.40, "max_duration_min": 30.0, "label": "40% in 30 min"},
    {"name": "e50_t30", "energy_fraction": 0.50, "max_duration_min": 30.0, "label": "50% in 30 min"},
    {"name": "e60_t45", "energy_fraction": 0.60, "max_duration_min": 45.0, "label": "60% in 45 min"},
)


def _param_text(params: Dict[str, Any]) -> str:
    fid = str(params.get("family_id", ""))
    if fid == "cccv":
        return f"I={params['i_cc']:.2f} A, Vcv={params['v_cv']:.2f} V, cutoff={params['i_cutoff']:.2f} A"
    if fid == "two_step":
        return (
            f"I1={params['i1']:.2f} A, I2={params['i2']:.2f} A, "
            f"switch SOC={params['soc_switch']:.2f}"
        )
    if fid == "three_step":
        return (
            f"I1={params['i1']:.2f} A, I2={params['i2']:.2f} A, I3={params['i3']:.2f} A, "
            f"SOC={params['soc1']:.2f}/{params['soc2']:.2f}"
        )
    if fid == "pulsed":
        return (
            f"I={params['i_charge']:.2f} A, on={params['pulse_on_min']:.1f} min, "
            f"rest={params['pulse_rest_min']:.1f} min, floor={params.get('i_floor', 0):.2f} A"
        )
    return ", ".join(f"{k}={v}" for k, v in params.items() if k != "family_id")


def _top6_for_cell(case_name: str, cell: str, k: int = 6) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for fid in OptimizationSpec.load().families:
        path = result_path(f"requirements/{case_name}_gp_bo_{cell}_{fid}.json")
        if not path.is_file():
            raise FileNotFoundError(path)
        blob = read_json(path)
        for idx, h in enumerate(blob.get("history") or []):
            if not h.get("feasible"):
                continue
            pool.append({
                "family": fid,
                "eval_index": idx + 1,
                "loss": float(h["loss"]),
                "reward": float(h["reward"]),
                "params": h["params"],
                "metrics": h["metrics"],
            })
    pool.sort(key=lambda r: r["loss"])
    return pool[:k]


def _ensure_case(spec: Dict[str, Any], *, device: str, cells: List[str], n_evals: int) -> None:
    opt = OptimizationSpec.load()
    missing = False
    for cell in cells:
        for fid in opt.families:
            if not result_path(f"requirements/{spec['name']}_gp_bo_{cell}_{fid}.json").is_file():
                missing = True
    if not missing:
        return
    session = replace(
        opt.session,
        energy_fraction=float(spec["energy_fraction"]),
        max_duration_min=float(spec["max_duration_min"]),
    )
    print(f"\n######## search {spec['label']} ########", flush=True)
    for cell in cells:
        run_cell(
            cell, "gp_bo", device=device, session=session,
            name_prefix=str(spec["name"]), resume=True, n_evals=n_evals,
        )


def write_top6(*, device: str = "auto", n_evals: int = 40) -> None:
    """Top-6 feasible profiles for the requested energy/time cases."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = list(Paths.load().cells)
    for spec in TOP6_CASES:
        _ensure_case(spec, device=device, cells=cells, n_evals=n_evals)

    labels = {"cccv": "CCCV", "two_step": "Two-step", "three_step": "Three-step", "pulsed": "Pulsed"}
    colors = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#0891b2", "#be123c"]
    all_rows: List[Dict[str, Any]] = []
    md = [
        "# Top 6 profiles by energy target and time limit",
        "",
        "Ranked by GP-BO loss among feasible evaluations. "
        "Reward is the advantage over that case's own 0.5C CCCV charge.",
        "",
    ]
    opt = OptimizationSpec.load()
    for spec in TOP6_CASES:
        session = replace(
            opt.session,
            energy_fraction=float(spec["energy_fraction"]),
            max_duration_min=float(spec["max_duration_min"]),
        )
        md.append(f"## {spec['label']}")
        md.append("")
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=False)
        for ax, cell in zip(axes.ravel(), cells):
            top = _top6_for_cell(spec["name"], cell)
            sim, state = make_simulator(cell, device=device, session=session)
            md.append(f"### {cell}")
            md.append("")
            md.append("| Rank | Family | Reward | Time (min) | Q_loss | Peak V | Profile |")
            md.append("|---:|---|---:|---:|---:|---:|---|")
            for rank, row in enumerate(top, 1):
                m = row["metrics"]
                text = _param_text(row["params"])
                md.append(
                    f"| {rank} | {labels.get(row['family'], row['family'])} | "
                    f"{row['reward']:.3f} | {m['duration_min']:.1f} | {m['q_total']:.3e} | "
                    f"{m['peak_v']:.3f} | {text} |"
                )
                params = type(get_family(row["family"])).from_dict(
                    {k: v for k, v in row["params"].items() if k != "family_id"}
                )
                traced = sim.simulate(state, params)
                t_min = np.asarray(traced["time_s"], dtype=np.float64) / 60.0
                i_chg = -np.asarray(traced["current_a"], dtype=np.float64)
                ax.plot(t_min, i_chg, color=colors[rank - 1], lw=1.8, label=f"#{rank} {labels.get(row['family'], row['family'])}")
                all_rows.append({
                    "case": spec["name"],
                    "label": spec["label"],
                    "energy_fraction": spec["energy_fraction"],
                    "max_duration_min": spec["max_duration_min"],
                    "cell": cell,
                    "rank": rank,
                    "family": row["family"],
                    "reward": row["reward"],
                    "loss": row["loss"],
                    "duration_min": m["duration_min"],
                    "q_total": m["q_total"],
                    "peak_v": m["peak_v"],
                    "peak_t": m["peak_t"],
                    "delta_soc": m["delta_soc"],
                    "profile": text,
                    "params": row["params"],
                })
            if not top:
                ax.set_title(f"{cell}: no feasible profile")
            else:
                ax.set_title(cell)
                ax.legend(fontsize=8, loc="best")
            ax.set_xlabel("Time (min)")
            ax.set_ylabel("Charge current (A)")
            md.append("")
        fig.suptitle(spec["label"])
        fig.tight_layout()
        fig_path = result_path(f"requirements/top6_{spec['name']}.png")
        fig.savefig(fig_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fig_path}", flush=True)

    table_path = result_path("requirements/top6_summary.md")
    table_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_json(result_path("requirements/top6_summary.json"), all_rows)
    print(f"wrote {table_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="GP-BO and random-search charging optimization")
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["random_search", "gp_bo"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-evals", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--shift-reward", action="store_true",
                    help="Rewrite saved rewards as advantage over CCCV 0.5C")
    ap.add_argument("--requirements", action="store_true",
                    help="GP-BO on the energy/time-limit cases")
    ap.add_argument("--top6", action="store_true",
                    help="Top-6 table and current figures for the four energy/time cases")
    ap.add_argument("--results-dir", default=None,
                    help="Write here instead of results/. A relative path is inside the package.")
    args = ap.parse_args()
    if args.results_dir:
        dest = set_results_dir(args.results_dir)
        print(f"results dir {dest}", flush=True)
    opt = OptimizationSpec.load()
    paths = Paths.load()
    cells = [c.upper() for c in (args.cells or paths.cells)]
    device = args.device or opt.device
    if args.shift_reward:
        shift_saved_rewards()
        return 0
    if args.requirements:
        n_req = int(args.n_evals if args.n_evals is not None else 40)
        run_requirements(device=device, cells=cells, n_evals=n_req)
        return 0
    if args.top6:
        n_req = int(args.n_evals if args.n_evals is not None else 40)
        write_top6(device=device, n_evals=n_req)
        return 0
    ok = True
    for method in args.methods:
        for cell in cells:
            print(f"\n======== {method}  {cell} ========", flush=True)
            summary = run_cell(
                cell, method, device=device, families=args.families,
                resume=not args.no_resume, n_evals=args.n_evals,
            )
            if not summary["any_feasible"]:
                ok = False
                print(f"FAIL {cell} {method}: no feasible point")
    print(f"wrote {RESULTS_DIR}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
