"""Parametric charging profile families for constrained optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np

from Constrained_BO.profile_catalog import ProfileBounds

CV_STEP_A = 0.25
MIN_CHARGE_A = 0.05

_active_bounds: Optional[ProfileBounds] = None


def set_profile_bounds(bounds: ProfileBounds) -> None:
    global _active_bounds
    _active_bounds = bounds


def set_profile_catalog(bounds: ProfileBounds) -> None:
    """Backward-compatible alias."""
    set_profile_bounds(bounds)


def active_bounds() -> ProfileBounds:
    if _active_bounds is None:
        return ProfileBounds.defaults("RW9")
    return _active_bounds


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
        """Inclusive (low, high) bounds for random search."""

    @classmethod
    @abstractmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        ...

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        vals = {
            k: float(rng.uniform(lo, hi))
            for k, (lo, hi) in cls.param_bounds().items()
        }
        return cls.from_dict(vals)

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        return []

    def init_context(self, params: ProfileParams) -> SimulationContext:
        return SimulationContext(phase="cc", i_level=self._bulk_current(params))

    @abstractmethod
    def _bulk_current(self, params: ProfileParams) -> float:
        ...

    @abstractmethod
    def target_current(
        self,
        state: Dict[str, float],
        ctx: SimulationContext,
        params: ProfileParams,
    ) -> float:
        """Applied current (A): negative = charge, 0 = rest."""

    def cv_ceiling(
        self,
        params: ProfileParams,
        global_ceiling: float,
        ctx: SimulationContext,
    ) -> float:
        return global_ceiling

    def after_step(
        self,
        state: Dict[str, float],
        ctx: SimulationContext,
        params: ProfileParams,
        *,
        ceiling_hit: bool,
        v_traj: np.ndarray,
        global_ceiling: float,
    ) -> Tuple[SimulationContext, Optional[str]]:
        return ctx, None

    def end_check(
        self,
        state: Dict[str, float],
        ctx: SimulationContext,
        params: ProfileParams,
        *,
        ceiling_hit: bool,
        step_samples: int,
        target_i: float,
    ) -> Optional[str]:
        return None


class CCCVFamily(ProfileFamily):
    family_id = "cccv"
    label = "CCCV"

    @classmethod
    def param_bounds(cls) -> Dict[str, Tuple[float, float]]:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        v_lo, v_hi = b.v_cv_bounds()
        return {
            "i_cc": (i_lo, i_hi),
            "v_cv": (v_lo, v_hi),
            "i_cutoff": (0.01, min(0.50, i_hi - CV_STEP_A)),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        v_lo, v_hi = b.v_cv_bounds()
        i_cc = float(np.clip(values["i_cc"], i_lo, i_hi))
        v_cv = float(np.clip(values["v_cv"], v_lo, v_hi))
        i_cutoff = float(np.clip(
            values["i_cutoff"],
            0.01,
            min(0.50, i_cc - CV_STEP_A),
        ))
        return ProfileParams(family_id=cls.family_id, values={
            "i_cc": i_cc, "v_cv": v_cv, "i_cutoff": i_cutoff,
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        b = active_bounds()
        ref = dict(b.seed_cccv)
        i_lo, i_hi = b.i_bounds()
        v_lo, v_hi = b.v_cv_bounds()
        seeds = [ref]
        for i_cc in (i_lo, (i_lo + i_hi) / 2, i_hi):
            seeds.append({"i_cc": i_cc, "v_cv": v_hi, "i_cutoff": ref["i_cutoff"]})
        for v_cv in (v_lo, (v_lo + v_hi) / 2, v_hi):
            seeds.append({"i_cc": ref["i_cc"], "v_cv": v_cv, "i_cutoff": ref["i_cutoff"]})
        unique = {tuple(sorted(s.items())) for s in seeds}
        return [cls.from_dict(dict(t)) for t in unique]

    def _bulk_current(self, params: ProfileParams) -> float:
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
        v_cv = params.values["v_cv"]
        if ctx.phase == "cc":
            if ceiling_hit or (v_traj.size and float(np.max(v_traj)) >= v_cv - 1e-4):
                ctx.phase = "cv"
                ctx.i_level = params.values["i_cc"]
            return ctx, None
        if ctx.phase == "cv" and ceiling_hit:
            ctx.i_level = max(params.values["i_cutoff"], ctx.i_level - CV_STEP_A)
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
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = b.soc_bounds()
        return {
            "i1": (i_lo, i_hi),
            "i2": (i_lo, i_hi),
            "soc_switch": (soc_lo, soc_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = b.soc_bounds()
        i1 = float(np.clip(values["i1"], i_lo, i_hi))
        i2 = float(np.clip(values["i2"], i_lo, min(i1, i_hi)))
        soc_sw = float(np.clip(values["soc_switch"], soc_lo, soc_hi))
        return ProfileParams(family_id=cls.family_id, values={
            "i1": i1, "i2": i2, "soc_switch": soc_sw,
        })

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = b.soc_bounds()
        i1 = float(rng.uniform(i_lo, i_hi))
        i2 = float(rng.uniform(i_lo, i1))
        return cls.from_dict({
            "i1": i1,
            "i2": i2,
            "soc_switch": float(rng.uniform(soc_lo, soc_hi)),
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = b.soc_bounds()
        ref_i = b.seed_cccv["i_cc"]
        seeds = []
        for soc in (soc_lo, (soc_lo + soc_hi) / 2, soc_hi):
            for i1 in (i_hi, ref_i, (i_lo + i_hi) / 2):
                i2 = max(i_lo, i1 * 0.5)
                seeds.append({"i1": i1, "i2": i2, "soc_switch": soc})
        unique = {tuple(sorted(s.items())) for s in seeds}
        return [cls.from_dict(dict(t)) for t in unique]

    def _bulk_current(self, params):
        return params.values["i1"]

    def _commanded(self, state, params):
        if float(state["soc"]) < params.values["soc_switch"]:
            return params.values["i1"]
        return params.values["i2"]

    def init_context(self, params):
        ctx = super().init_context(params)
        ctx.i_level = self._commanded({"soc": 0.0}, params)
        return ctx

    def target_current(self, state, ctx, params):
        ctx.i_level = self._commanded(state, params)
        return -ctx.i_level

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        if ceiling_hit:
            ctx.i_level = max(MIN_CHARGE_A, ctx.i_level - CV_STEP_A)
        return ctx, None


class ThreeStepFamily(ProfileFamily):
    family_id = "three_step"
    label = "3-step (SoC)"

    @classmethod
    def param_bounds(cls) -> Dict[str, Tuple[float, float]]:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = b.soc_bounds()
        return {
            "i1": (i_lo, i_hi),
            "i2": (i_lo, i_hi),
            "i3": (i_lo, i_hi),
            "soc1": (soc_lo, min(0.55, soc_hi)),
            "soc2": (max(0.35, soc_lo), soc_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = b.soc_bounds()
        i1 = float(np.clip(values["i1"], i_lo, i_hi))
        i2 = float(np.clip(values["i2"], i_lo, min(i1, i_hi)))
        i3 = float(np.clip(values["i3"], i_lo, min(i2, i_hi)))
        soc1 = float(np.clip(values["soc1"], soc_lo, min(0.55, soc_hi)))
        soc2 = float(np.clip(values["soc2"], max(0.35, soc1 + 0.05), soc_hi))
        return ProfileParams(family_id=cls.family_id, values={
            "i1": i1, "i2": i2, "i3": i3, "soc1": soc1, "soc2": soc2,
        })

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = b.soc_bounds()
        i1 = float(rng.uniform(i_lo, i_hi))
        i2 = float(rng.uniform(i_lo, i1))
        i3 = float(rng.uniform(i_lo, i2))
        soc1 = float(rng.uniform(soc_lo, min(0.55, soc_hi)))
        soc2 = float(rng.uniform(max(soc1 + 0.05, 0.35), soc_hi))
        return cls.from_dict({
            "i1": i1, "i2": i2, "i3": i3, "soc1": soc1, "soc2": soc2,
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        soc_lo, soc_hi = b.soc_bounds()
        mid = (i_lo + i_hi) / 2
        seeds = [
            {"i1": i_hi, "i2": mid, "i3": i_lo, "soc1": soc_lo, "soc2": soc_hi},
            {"i1": mid, "i2": (i_lo + mid) / 2, "i3": i_lo,
             "soc1": (soc_lo + soc_hi) / 2, "soc2": soc_hi},
        ]
        unique = {tuple(sorted(s.items())) for s in seeds}
        return [cls.from_dict(dict(t)) for t in unique]

    def _bulk_current(self, params):
        return params.values["i1"]

    def _commanded(self, state, params):
        soc = float(state["soc"])
        v = params.values
        if soc < v["soc1"]:
            return v["i1"]
        if soc < v["soc2"]:
            return v["i2"]
        return v["i3"]

    def init_context(self, params):
        ctx = super().init_context(params)
        ctx.i_level = params.values["i1"]
        return ctx

    def target_current(self, state, ctx, params):
        ctx.i_level = self._commanded(state, params)
        return -ctx.i_level

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        if ceiling_hit:
            ctx.i_level = max(MIN_CHARGE_A, ctx.i_level - CV_STEP_A)
        return ctx, None


class PulsedFamily(ProfileFamily):
    family_id = "pulsed"
    label = "Pulsed charge/rest"

    @classmethod
    def param_bounds(cls) -> Dict[str, Tuple[float, float]]:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        return {
            "i_charge": (i_lo, i_hi),
            "pulse_on_min": (b.pulse_on_min_min, b.pulse_on_max_min),
            "rest_fraction": (b.rest_fraction_min, b.rest_fraction_max),
            "i_floor": (i_lo, i_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        i_cc = float(np.clip(values["i_charge"], i_lo, i_hi))
        pulse_on = float(np.clip(values["pulse_on_min"], b.pulse_on_min_min, b.pulse_on_max_min))
        rest_frac = float(np.clip(values["rest_fraction"], b.rest_fraction_min, b.rest_fraction_max))
        i_floor = float(np.clip(values["i_floor"], i_lo, min(i_cc - 0.05, i_hi)))
        pulse_rest = max(0.5, pulse_on * rest_frac)
        return ProfileParams(family_id=cls.family_id, values={
            "i_charge": i_cc,
            "pulse_on_min": pulse_on,
            "pulse_rest_min": pulse_rest,
            "rest_fraction": rest_frac,
            "i_floor": i_floor,
        })

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        b = active_bounds()
        i_lo, i_hi = b.i_bounds()
        i_charge = float(rng.uniform(i_lo, i_hi))
        i_floor = float(rng.uniform(i_lo, max(i_lo, i_charge - 0.05)))
        return cls.from_dict({
            "i_charge": i_charge,
            "pulse_on_min": float(rng.uniform(b.pulse_on_min_min, b.pulse_on_max_min)),
            "rest_fraction": float(rng.uniform(b.rest_fraction_min, b.rest_fraction_max)),
            "i_floor": i_floor,
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        b = active_bounds()
        pul = dict(b.seed_pulsed)
        i_lo, i_hi = b.i_bounds()
        seeds = [pul]
        for i_cc in (i_hi, (i_lo + i_hi) / 2):
            seeds.append({
                "i_charge": i_cc,
                "pulse_on_min": pul["pulse_on_min"],
                "rest_fraction": pul["rest_fraction"],
                "i_floor": pul["i_floor"],
            })
        unique = {tuple(sorted(s.items())) for s in seeds}
        return [cls.from_dict(dict(t)) for t in unique]

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
        pulse_on_s = ctx.extra["pulse_on_s"]
        pulse_rest_s = ctx.extra["pulse_rest_s"]
        if pulse_rest_s > 0.0 and ctx.charge_elapsed >= pulse_on_s:
            ctx.in_rest = True
            ctx.rest_elapsed = 0.0
            return 0.0
        return -ctx.i_level

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        i_floor = ctx.extra["i_floor"]
        pulse_rest_s = ctx.extra["pulse_rest_s"]

        if ctx.in_rest:
            ctx.rest_elapsed += v_traj.size
            if ctx.rest_elapsed >= pulse_rest_s:
                ctx.in_rest = False
                ctx.charge_elapsed = 0.0
        elif ceiling_hit and not ctx.in_rest and ctx.i_level > i_floor + 1e-9:
            ctx.i_level = max(i_floor, ctx.i_level - CV_STEP_A)
            ctx.charge_elapsed = 0.0
        return ctx, None

    def end_check(self, state, ctx, params, *, ceiling_hit, step_samples, target_i):
        i_floor = ctx.extra["i_floor"]
        if (
            ceiling_hit
            and ctx.i_level <= i_floor + 1e-9
            and target_i != 0.0
            and step_samples <= 1
        ):
            return "V ceiling @ min current"
        return None


FAMILY_REGISTRY: Dict[str, Type[ProfileFamily]] = {
    CCCVFamily.family_id: CCCVFamily,
    TwoStepFamily.family_id: TwoStepFamily,
    ThreeStepFamily.family_id: ThreeStepFamily,
    PulsedFamily.family_id: PulsedFamily,
}

DEFAULT_FAMILIES = ["cccv", "two_step", "three_step", "pulsed"]


def get_family(family_id: str) -> ProfileFamily:
    if family_id not in FAMILY_REGISTRY:
        raise KeyError(f"Unknown family {family_id!r}")
    return FAMILY_REGISTRY[family_id]()
