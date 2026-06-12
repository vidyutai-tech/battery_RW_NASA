"""
Pluggable charging profile families for multi-strategy optimization.

Each family defines its own parameter vector, BO search space, and current
schedule logic consumed by :class:`ProfileSimulator.simulate_params`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type

import numpy as np
from skopt.space import Real

CV_STEP_A = 0.25
MIN_REST_MIN = 5.0
MIN_CHARGE_A = 0.05
REDUCED_CV_LEVELS = (4.05, 4.10, 4.15, 4.20)


@dataclass
class ProfileParams:
    """Family-tagged parameter bundle."""

    family_id: str
    values: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {"family_id": self.family_id, **self.values}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ProfileParams:
        family_id = str(d["family_id"])
        values = {k: float(v) for k, v in d.items() if k != "family_id"}
        return cls(family_id=family_id, values=values)


@dataclass
class SimulationContext:
    """Mutable rollout state shared across decision steps."""

    phase: str = "cc"
    i_level: float = 0.0
    charge_elapsed: float = 0.0
    rest_elapsed: float = 0.0
    in_rest: bool = False
    extra: Dict[str, float] = field(default_factory=dict)


class ChargingProfileFamily(ABC):
    """Common interface for parametric charging templates."""

    family_id: ClassVar[str]
    label: ClassVar[str]

    @classmethod
    @abstractmethod
    def search_space(cls) -> List[Real]:
        """scikit-optimize dimensions for this family."""

    @classmethod
    @abstractmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        """Decode and clip a BO candidate vector."""

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        """Deterministic initial BO evaluations."""
        return []

    @classmethod
    def params_from_dict(cls, values: Dict[str, float]) -> ProfileParams:
        return ProfileParams(family_id=cls.family_id, values=dict(values))

    def init_context(self, params: ProfileParams) -> SimulationContext:
        return SimulationContext(phase="cc", i_level=self._bulk_current(params))

    @abstractmethod
    def _bulk_current(self, params: ProfileParams) -> float:
        """Nominal bulk charge magnitude (A, positive)."""

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
        """Voltage ceiling passed to the BDT for this step."""
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
        """Update context; optional early termination reason."""
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


def _snap_reduced_cv(v: float) -> float:
    arr = np.asarray(REDUCED_CV_LEVELS, dtype=np.float64)
    return float(arr[int(np.argmin(np.abs(arr - v)))])


class CCCVFamily(ChargingProfileFamily):
    family_id = "cccv"
    label = "CCCV"

    @classmethod
    def search_space(cls) -> List[Real]:
        return [
            Real(0.75, 3.0, name="i_cc"),
            Real(4.05, 4.20, name="v_cv"),
            Real(0.05, 0.50, name="i_cutoff"),
        ]

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        i_cc, v_cv, i_cutoff = [float(v) for v in x]
        i_cc = float(np.clip(i_cc, 0.75, 3.0))
        v_cv = float(np.clip(v_cv, 4.05, 4.20))
        i_cutoff = float(np.clip(i_cutoff, 0.05, min(0.50, i_cc - CV_STEP_A)))
        return cls.params_from_dict({"i_cc": i_cc, "v_cv": v_cv, "i_cutoff": i_cutoff})

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [
            [1.0, 4.20, 0.25],
            [1.25, 4.15, 0.25],
            [1.5, 4.10, 0.20],
            [2.0, 4.20, 0.30],
            [2.5, 4.15, 0.25],
        ]

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
        if ctx.phase != "cv" or target_i == 0.0:
            return None
        if ctx.i_level <= params.values["i_cutoff"] + 1e-6 and ceiling_hit and step_samples <= 1:
            return "CV cutoff current"
        return None


class ReducedCVCCCVFamily(CCCVFamily):
    """CCCV with CV setpoint snapped to {4.05, 4.10, 4.15, 4.20} V."""

    family_id = "reduced_cv_cccv"
    label = "Reduced-CV CCCV"

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        p = super().from_vector(x)
        p.values["v_cv"] = _snap_reduced_cv(p.values["v_cv"])
        return ProfileParams(family_id=cls.family_id, values=p.values)

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [
            [1.25, 4.20, 0.25],
            [1.25, 4.15, 0.25],
            [1.25, 4.10, 0.25],
            [1.25, 4.05, 0.25],
            [1.5, 4.10, 0.20],
            [2.0, 4.15, 0.30],
        ]


class AdaptiveTwoStepFamily(ChargingProfileFamily):
    family_id = "adaptive_two_step"
    label = "Adaptive 2-step (SoC)"

    @classmethod
    def search_space(cls) -> List[Real]:
        return [
            Real(0.75, 3.0, name="i1"),
            Real(0.75, 2.0, name="i2"),
            Real(0.20, 0.80, name="soc_switch"),
        ]

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        i1, i2, soc_sw = [float(v) for v in x]
        i1 = float(np.clip(i1, 0.75, 3.0))
        i2 = float(np.clip(i2, 0.75, min(2.0, i1)))
        soc_sw = float(np.clip(soc_sw, 0.20, 0.80))
        return cls.params_from_dict({"i1": i1, "i2": i2, "soc_switch": soc_sw})

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [
            [2.0, 0.8, 0.60],
            [1.5, 0.75, 0.50],
            [2.5, 1.0, 0.55],
            [1.25, 0.75, 0.65],
        ]

    def _bulk_current(self, params):
        return params.values["i1"]

    def _commanded(self, state, params):
        soc = float(state["soc"])
        if soc < params.values["soc_switch"]:
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


class AdaptiveThreeStepFamily(ChargingProfileFamily):
    family_id = "adaptive_three_step"
    label = "Adaptive 3-step (SoC)"

    @classmethod
    def search_space(cls) -> List[Real]:
        return [
            Real(0.75, 3.0, name="i1"),
            Real(0.75, 2.5, name="i2"),
            Real(0.75, 1.5, name="i3"),
            Real(0.10, 0.50, name="soc1"),
            Real(0.40, 0.85, name="soc2"),
        ]

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        i1, i2, i3, soc1, soc2 = [float(v) for v in x]
        i1 = float(np.clip(i1, 0.75, 3.0))
        i2 = float(np.clip(i2, 0.75, min(2.5, i1)))
        i3 = float(np.clip(i3, 0.75, min(1.5, i2)))
        soc1 = float(np.clip(soc1, 0.10, 0.50))
        soc2 = float(np.clip(soc2, 0.40, 0.85))
        if soc2 <= soc1 + 0.05:
            soc2 = min(0.85, soc1 + 0.10)
        return cls.params_from_dict(
            {"i1": i1, "i2": i2, "i3": i3, "soc1": soc1, "soc2": soc2},
        )

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [
            [2.0, 1.25, 0.75, 0.30, 0.70],
            [2.5, 1.0, 0.75, 0.25, 0.65],
            [1.5, 1.0, 0.75, 0.35, 0.75],
        ]

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


class ExponentialTaperFamily(ChargingProfileFamily):
    family_id = "exponential_taper"
    label = "Exponential taper I(SoC)"

    @classmethod
    def search_space(cls) -> List[Real]:
        return [
            Real(1.0, 1.25, name="i0"),
            Real(0.15, 0.45, name="k"),
        ]

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        i0, k = [float(v) for v in x]
        i0 = float(np.clip(i0, 1.0, 1.25))
        k = float(np.clip(k, 0.15, 0.45))
        return cls.params_from_dict({"i0": i0, "k": k})

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [
            [1.05, 0.28],
            [1.08, 0.25],
            [1.1, 0.22],
            [1.15, 0.2],
            [1.2, 0.18],
        ]

    def _bulk_current(self, params):
        return params.values["i0"]

    def init_context(self, params):
        ctx = super().init_context(params)
        ctx.extra["soc_start"] = None
        return ctx

    def _progress(self, state: Dict[str, float], ctx: SimulationContext, soc_target: float = 0.95) -> float:
        if ctx.extra.get("soc_start") is None:
            ctx.extra["soc_start"] = float(state["soc"])
        soc0 = float(ctx.extra["soc_start"])
        span = max(0.05, float(soc_target) - soc0)
        return float(np.clip((float(state["soc"]) - soc0) / span, 0.0, 1.0))

    def _current(self, state, ctx, params, soc_target: float = 0.95) -> float:
        progress = self._progress(state, ctx, soc_target=soc_target)
        i = params.values["i0"] * np.exp(-params.values["k"] * progress)
        return float(np.clip(i, MIN_CHARGE_A, params.values["i0"]))

    def target_current(self, state, ctx, params):
        ctx.i_level = self._current(state, ctx, params)
        return -ctx.i_level

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        return ctx, None


class CcTaperLegacyFamily(ChargingProfileFamily):
    """Original voltage-triggered CC-taper (+ optional pulse rest)."""

    family_id = "cc_taper_legacy"
    label = "CC-taper (legacy)"

    @classmethod
    def search_space(cls, *, allow_pulsed: bool = False) -> List[Real]:
        space = [
            Real(0.75, 4.0, name="i_charge"),
            Real(5.0, 30.0, name="pulse_on_min"),
        ]
        if allow_pulsed:
            space.append(Real(0.0, 15.0, name="pulse_rest_min"))
        space.append(Real(0.75, 2.25, name="i_floor"))
        return space

    @classmethod
    def from_vector(cls, x: List[float], *, allow_pulsed: bool = False) -> ProfileParams:
        from charging_opt.profile_simulator import ProfileSpec

        if allow_pulsed:
            spec = ProfileSpec.from_vector(x)
        elif len(x) == 3:
            spec = ProfileSpec.from_vector([x[0], x[1], 0.0, x[2]])
        else:
            spec = ProfileSpec.from_vector([x[0], x[1], 0.0, x[3]])
        return ProfileParams(
            family_id=cls.family_id,
            values={
                "i_charge": spec.i_charge,
                "pulse_on_min": spec.pulse_on_min,
                "pulse_rest_min": spec.pulse_rest_min,
                "i_floor": spec.i_floor,
            },
        )

    @classmethod
    def seed_points(cls, *, allow_pulsed: bool = False) -> List[List[float]]:
        from charging_opt.profile_simulator import TAPER_STEP_A

        seeds = []
        for i in [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
            floor = 0.75 if i <= 0.75 + TAPER_STEP_A else 0.75
            seeds.append([float(i), 30.0, 0.0, float(floor)])
        if not allow_pulsed:
            return [[s[0], s[1], s[3]] for s in seeds]
        return seeds

    def _bulk_current(self, params):
        return params.values["i_charge"]

    def init_context(self, params):
        from charging_opt.profile_simulator import MIN_REST_MIN

        ctx = SimulationContext(
            phase="legacy",
            i_level=params.values["i_charge"],
        )
        ctx.extra["pulse_on_s"] = params.values["pulse_on_min"] * 60.0
        rest = float(params.values.get("pulse_rest_min", 0.0))
        min_rest = 0.5 if params.family_id == "pulsed" else MIN_REST_MIN
        ctx.extra["pulse_rest_s"] = rest * 60.0 if rest >= min_rest else 0.0
        ctx.extra["i_floor"] = params.values["i_floor"]
        return ctx

    def target_current(self, state, ctx, params):
        from charging_opt.profile_simulator import TAPER_STEP_A

        if ctx.in_rest:
            return 0.0
        target = -ctx.i_level
        pulse_on_s = ctx.extra["pulse_on_s"]
        pulse_rest_s = ctx.extra["pulse_rest_s"]
        if pulse_rest_s > 0.0 and ctx.charge_elapsed >= pulse_on_s:
            ctx.in_rest = True
            ctx.rest_elapsed = 0.0
            return 0.0
        return target

    def after_step(self, state, ctx, params, *, ceiling_hit, v_traj, global_ceiling):
        from charging_opt.profile_simulator import TAPER_STEP_A

        i_floor = ctx.extra["i_floor"]
        pulse_rest_s = ctx.extra["pulse_rest_s"]

        if ctx.in_rest:
            ctx.rest_elapsed += v_traj.size
            if ctx.rest_elapsed >= pulse_rest_s:
                ctx.in_rest = False
                ctx.charge_elapsed = 0.0
        elif ceiling_hit and not ctx.in_rest and ctx.i_level > i_floor + 1e-9:
            ctx.i_level = max(i_floor, ctx.i_level - TAPER_STEP_A)
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


class CcTaperFamily(CcTaperLegacyFamily):
    """Voltage-triggered CC-taper (typically 2 charge levels)."""

    family_id = "cc_taper"
    label = "CC-taper (2-level)"

    @classmethod
    def search_space(cls) -> List[Real]:
        return super().search_space(allow_pulsed=False)

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        p = super().from_vector(x, allow_pulsed=False)
        return ProfileParams(family_id=cls.family_id, values=p.values)

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [
            [0.75, 30.0, 0.75],
            [1.0, 30.0, 0.75],
            [1.25, 30.0, 0.75],
            [1.5, 30.0, 0.75],
            [2.0, 30.0, 0.75],
        ]


class MultiStepTaperFamily(CcTaperLegacyFamily):
    """High bulk current + voltage ceiling → 3+ taper steps (0.75 A steps)."""

    family_id = "multi_step_taper"
    label = "Multi-step taper (voltage)"

    @classmethod
    def search_space(cls) -> List[Real]:
        return [
            Real(2.0, 4.0, name="i_charge"),
            Real(5.0, 30.0, name="pulse_on_min"),
            Real(0.75, 1.25, name="i_floor"),
        ]

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        vals = [float(v) for v in x]
        if len(vals) == 3:
            p = super().from_vector([vals[0], vals[1], vals[2]], allow_pulsed=False)
        else:
            p = super().from_vector(vals, allow_pulsed=False)
        return ProfileParams(family_id=cls.family_id, values=p.values)

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [
            [2.5, 10.0, 0.75],
            [3.0, 15.0, 0.75],
            [3.5, 20.0, 0.75],
            [4.0, 25.0, 0.75],
        ]


class PulsedFamily(CcTaperLegacyFamily):
    """Charge/rest cycles: rest duration = fraction × pulse-on (short pulses)."""

    family_id = "pulsed"
    label = "Pulsed charge/rest"

    PULSE_ON_MIN = (1.0, 6.0)
    REST_FRAC_RANGE = (0.08, 0.35)
    MIN_REST_MIN = 0.5  # 30 s at 1 Hz decision steps

    @classmethod
    def search_space(cls) -> List[Real]:
        return [
            Real(1.0, 3.0, name="i_charge"),
            Real(1.0, 6.0, name="pulse_on_min"),
            Real(0.08, 0.35, name="rest_fraction"),
            Real(0.75, 2.0, name="i_floor"),
        ]

    @classmethod
    def from_vector(cls, x: List[float]) -> ProfileParams:
        i_cc, pulse_on, rest_frac, i_floor = [float(v) for v in x]
        i_cc = float(np.clip(i_cc, 1.0, 3.0))
        pulse_on = float(np.clip(pulse_on, cls.PULSE_ON_MIN[0], cls.PULSE_ON_MIN[1]))
        rest_frac = float(np.clip(rest_frac, cls.REST_FRAC_RANGE[0], cls.REST_FRAC_RANGE[1]))
        i_floor_max = i_cc - 0.05
        i_floor = float(np.clip(i_floor, 0.75, min(2.0, i_floor_max)))
        pulse_rest = max(cls.MIN_REST_MIN, pulse_on * rest_frac)
        return ProfileParams(
            family_id=cls.family_id,
            values={
                "i_charge": i_cc,
                "pulse_on_min": pulse_on,
                "pulse_rest_min": pulse_rest,
                "rest_fraction": rest_frac,
                "i_floor": i_floor,
            },
        )

    @classmethod
    def seed_points(cls) -> List[List[float]]:
        return [
            [2.0, 3.0, 0.15, 0.75],
            [1.5, 4.0, 0.20, 0.75],
            [2.5, 2.0, 0.12, 0.75],
            [2.0, 5.0, 0.25, 0.80],
        ]


FAMILY_REGISTRY: Dict[str, Type[ChargingProfileFamily]] = {
    CCCVFamily.family_id: CCCVFamily,
    ReducedCVCCCVFamily.family_id: ReducedCVCCCVFamily,
    AdaptiveTwoStepFamily.family_id: AdaptiveTwoStepFamily,
    AdaptiveThreeStepFamily.family_id: AdaptiveThreeStepFamily,
    ExponentialTaperFamily.family_id: ExponentialTaperFamily,
    CcTaperLegacyFamily.family_id: CcTaperLegacyFamily,
    CcTaperFamily.family_id: CcTaperFamily,
    MultiStepTaperFamily.family_id: MultiStepTaperFamily,
    PulsedFamily.family_id: PulsedFamily,
}

DEFAULT_FAMILY_IDS = [
    "cccv",
    "reduced_cv_cccv",
    "adaptive_two_step",
    "adaptive_three_step",
    "exponential_taper",
    "cc_taper",
    "multi_step_taper",
    "pulsed",
]

FAMILY_LABELS = {fid: FAMILY_REGISTRY[fid]().label for fid in FAMILY_REGISTRY}


def get_family(family_id: str) -> ChargingProfileFamily:
    if family_id not in FAMILY_REGISTRY:
        raise KeyError(f"Unknown family {family_id!r}; choose from {list(FAMILY_REGISTRY)}")
    return FAMILY_REGISTRY[family_id]()
