"""Parametric charging profile families for constrained optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np

from Constrained_BO.profile_catalog import ProfileCatalog, snap_to_nearest

CV_STEP_A = 0.25
MIN_CHARGE_A = 0.05

_active_catalog: Optional[ProfileCatalog] = None


def set_profile_catalog(catalog: ProfileCatalog) -> None:
    global _active_catalog
    _active_catalog = catalog


def active_catalog() -> ProfileCatalog:
    if _active_catalog is None:
        return ProfileCatalog.nasa_defaults("RW9")
    return _active_catalog


def _amps(cat: ProfileCatalog) -> List[float]:
    return cat.rw_charge_currents_a


def _i_bounds(cat: ProfileCatalog) -> Tuple[float, float]:
    amps = _amps(cat)
    return amps[0], amps[-1]


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
        cat = active_catalog()
        i_lo, i_hi = _i_bounds(cat)
        v_lo, v_hi = min(cat.cv_levels_v), max(cat.cv_levels_v)
        return {
            "i_cc": (i_lo, i_hi),
            "v_cv": (v_lo, v_hi),
            "i_cutoff": (0.01, min(0.50, i_hi - CV_STEP_A)),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        cat = active_catalog()
        i_lo, i_hi = _i_bounds(cat)
        i_cc = snap_to_nearest(float(values["i_cc"]), _amps(cat))
        i_cc = float(np.clip(i_cc, i_lo, i_hi))
        v_cv = snap_to_nearest(float(values["v_cv"]), cat.cv_levels_v)
        v_cv = float(np.clip(v_cv, min(cat.cv_levels_v), max(cat.cv_levels_v)))
        i_cutoff = float(np.clip(
            values["i_cutoff"],
            0.01,
            min(0.50, i_cc - CV_STEP_A),
        ))
        return ProfileParams(family_id=cls.family_id, values={
            "i_cc": i_cc, "v_cv": v_cv, "i_cutoff": i_cutoff,
        })

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        cat = active_catalog()
        amps = _amps(cat)
        return cls.from_dict({
            "i_cc": float(rng.choice(amps)),
            "v_cv": float(rng.choice(cat.cv_levels_v)),
            "i_cutoff": float(rng.uniform(0.01, min(0.50, amps[-1] - CV_STEP_A))),
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        cat = active_catalog()
        ref = cat.reference_cccv
        seeds = [dict(ref)]
        for i_cc in _amps(cat)[::2]:
            seeds.append({"i_cc": i_cc, "v_cv": max(cat.cv_levels_v), "i_cutoff": ref["i_cutoff"]})
        for v_cv in cat.cv_levels_v:
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
        cat = active_catalog()
        i_lo, i_hi = _i_bounds(cat)
        soc_lo, soc_hi = min(cat.soc_switch_levels), max(cat.soc_switch_levels)
        return {
            "i1": (i_lo, i_hi),
            "i2": (i_lo, i_hi),
            "soc_switch": (soc_lo, soc_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        cat = active_catalog()
        amps = _amps(cat)
        i_lo, i_hi = _i_bounds(cat)
        i1 = snap_to_nearest(float(values["i1"]), amps)
        i1 = float(np.clip(i1, i_lo, i_hi))
        i2 = snap_to_nearest(float(values["i2"]), amps)
        i2 = float(np.clip(i2, i_lo, min(i1, i_hi)))
        soc_sw = snap_to_nearest(float(values["soc_switch"]), cat.soc_switch_levels)
        soc_sw = float(np.clip(soc_sw, 0.10, 0.90))
        return ProfileParams(family_id=cls.family_id, values={
            "i1": i1, "i2": i2, "soc_switch": soc_sw,
        })

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        cat = active_catalog()
        amps = _amps(cat)
        i1, i2 = sorted(rng.choice(amps, size=2, replace=False), reverse=True)
        return cls.from_dict({
            "i1": float(i1),
            "i2": float(i2),
            "soc_switch": float(rng.choice(cat.soc_switch_levels)),
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        cat = active_catalog()
        amps = _amps(cat)
        ref_i = cat.reference_cccv["i_cc"]
        seeds = []
        for soc in cat.soc_switch_levels[:3]:
            for i1 in (amps[-1], ref_i, amps[len(amps) // 2]):
                low = [a for a in amps if a < i1]
                i2 = low[len(low) // 2] if low else amps[0]
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
        cat = active_catalog()
        i_lo, i_hi = _i_bounds(cat)
        socs = cat.soc_switch_levels
        soc_lo, soc_hi = min(socs), max(socs)
        return {
            "i1": (i_lo, i_hi),
            "i2": (i_lo, i_hi),
            "i3": (i_lo, i_hi),
            "soc1": (soc_lo, min(0.55, soc_hi)),
            "soc2": (max(0.35, soc_lo), soc_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        cat = active_catalog()
        amps = _amps(cat)
        i_lo, i_hi = _i_bounds(cat)
        i1 = snap_to_nearest(float(values["i1"]), amps)
        i1 = float(np.clip(i1, i_lo, i_hi))
        i2 = snap_to_nearest(float(values["i2"]), amps)
        i2 = float(np.clip(i2, i_lo, min(i1, i_hi)))
        i3 = snap_to_nearest(float(values["i3"]), amps)
        i3 = float(np.clip(i3, i_lo, min(i2, i_hi)))
        soc1 = snap_to_nearest(float(values["soc1"]), cat.soc_switch_levels)
        soc2 = snap_to_nearest(float(values["soc2"]), cat.soc_switch_levels)
        soc1 = float(np.clip(soc1, 0.10, 0.55))
        soc2 = float(np.clip(soc2, max(0.35, soc1 + 0.05), 0.90))
        return ProfileParams(family_id=cls.family_id, values={
            "i1": i1, "i2": i2, "i3": i3, "soc1": soc1, "soc2": soc2,
        })

    @classmethod
    def sample_random(cls, rng: np.random.Generator) -> ProfileParams:
        cat = active_catalog()
        amps = _amps(cat)
        i1, i2, i3 = sorted(rng.choice(amps, size=3, replace=False), reverse=True)
        socs = sorted(rng.choice(cat.soc_switch_levels, size=2, replace=False))
        if socs[1] <= socs[0] + 0.05:
            socs[1] = min(0.90, socs[0] + 0.10)
        return cls.from_dict({
            "i1": float(i1), "i2": float(i2), "i3": float(i3),
            "soc1": float(socs[0]), "soc2": float(socs[1]),
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        cat = active_catalog()
        amps = _amps(cat)
        socs = cat.soc_switch_levels
        if len(socs) < 2:
            socs = socs + [0.50, 0.70]
        seeds = []
        for i1 in (amps[-1], amps[len(amps) // 2]):
            mid = amps[len(amps) // 2]
            low = amps[0]
            seeds.append({
                "i1": i1, "i2": mid, "i3": low,
                "soc1": socs[0], "soc2": socs[min(1, len(socs) - 1)],
            })
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
        cat = active_catalog()
        i_lo, i_hi = _i_bounds(cat)
        pul = cat.pulsed_charge
        return {
            "i_charge": (i_lo, i_hi),
            "pulse_on_min": (1.0, max(12.0, pul["pulse_on_min"])),
            "rest_fraction": (0.5, max(2.5, pul["rest_fraction"])),
            "i_floor": (i_lo, i_hi),
        }

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        cat = active_catalog()
        amps = _amps(cat)
        i_lo, i_hi = _i_bounds(cat)
        i_cc = snap_to_nearest(float(values["i_charge"]), amps)
        i_cc = float(np.clip(i_cc, i_lo, i_hi))
        pulse_on = float(np.clip(values["pulse_on_min"], 1.0, 15.0))
        rest_frac = float(np.clip(values["rest_fraction"], 0.5, 2.5))
        i_floor = snap_to_nearest(float(values["i_floor"]), amps)
        i_floor = float(np.clip(i_floor, i_lo, min(i_cc - 0.05, i_hi)))
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
        cat = active_catalog()
        amps = _amps(cat)
        i_charge = float(rng.choice(amps))
        low = [a for a in amps if a < i_charge]
        i_floor = float(rng.choice(low)) if low else amps[0]
        return cls.from_dict({
            "i_charge": i_charge,
            "pulse_on_min": float(rng.uniform(2.0, 12.0)),
            "rest_fraction": float(rng.uniform(0.5, 2.5)),
            "i_floor": i_floor,
        })

    @classmethod
    def seed_params(cls) -> List[ProfileParams]:
        cat = active_catalog()
        pul = cat.pulsed_charge
        amps = _amps(cat)
        seeds = [
            {
                "i_charge": pul["i_charge"],
                "pulse_on_min": pul["pulse_on_min"],
                "rest_fraction": pul["rest_fraction"],
                "i_floor": amps[0],
            },
        ]
        for i_cc in (amps[-1], amps[len(amps) // 2]):
            seeds.append({
                "i_charge": i_cc,
                "pulse_on_min": pul["pulse_on_min"],
                "rest_fraction": pul["rest_fraction"],
                "i_floor": amps[0],
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
