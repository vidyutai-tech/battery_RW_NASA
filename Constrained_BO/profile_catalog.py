"""Continuous practical bounds for charging-profile search (not discrete NASA grids)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Tuple


@dataclass
class ProfileBounds:
    """Inclusive search ranges for profile parameters."""

    cell_id: str
    i_min_a: float = 0.75
    # i_max_a: float = 4.5
    i_max_a: float = 6.0
    v_cv_min_v: float = 4.05
    v_cv_max_v: float = 4.20
    soc_switch_min: float = 0.10
    soc_switch_max: float = 0.90
    pulse_on_min_min: float = 1.0
    pulse_on_max_min: float = 15.0
    rest_fraction_min: float = 0.5
    rest_fraction_max: float = 2.5
    seed_cccv: Dict[str, float] = field(default_factory=lambda: {
        "i_cc": 2.0, "v_cv": 4.2, "i_cutoff": 0.01,
    })
    seed_pulsed: Dict[str, float] = field(default_factory=lambda: {
        "i_charge": 1.0,
        "pulse_on_min": 10.0,
        "rest_fraction": 2.0,
        "i_floor": 0.75,
    })

    @classmethod
    def defaults(cls, cell_id: str) -> ProfileBounds:
        return cls(cell_id=cell_id.upper())

    def i_bounds(self) -> Tuple[float, float]:
        return self.i_min_a, self.i_max_a

    def v_cv_bounds(self) -> Tuple[float, float]:
        return self.v_cv_min_v, self.v_cv_max_v

    def soc_bounds(self) -> Tuple[float, float]:
        return self.soc_switch_min, self.soc_switch_max

    def to_dict(self) -> Dict:
        return asdict(self)
