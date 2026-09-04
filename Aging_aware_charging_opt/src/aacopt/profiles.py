"""Parametric charging-profile families (CCCV, 2-step, 3-step, pulsed)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np

from aacopt.config import OptimizationSpec, SearchSpace, SessionSpec

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
