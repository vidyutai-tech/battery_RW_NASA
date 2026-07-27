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
    soc_switch_min: float = 0.32  # ≥ SOC_START(0.20)+0.12 for visible 2-step stage-1
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
        return cls.lfp_defaults() if cell_id.upper() == "LFP" else cls(cell_id=cell_id.upper())

    @classmethod
    def lfp_defaults(cls) -> ProfileBounds:
        """Search bounds for LFP (~3 V plateau, ~3.65 V ceiling)."""
        return cls(
            cell_id="LFP",
            i_min_a=0.5,
            i_max_a=3.0,
            v_cv_min_v=3.40,
            v_cv_max_v=3.65,
            soc_switch_min=0.10,
            soc_switch_max=0.90,
            seed_cccv={"i_cc": 1.5, "v_cv": 3.60, "i_cutoff": 0.05},
            seed_pulsed={
                "i_charge": 1.0,
                "pulse_on_min": 10.0,
                "rest_fraction": 2.0,
                "i_floor": 0.5,
            },
        )

    def i_bounds(self) -> Tuple[float, float]:
        return self.i_min_a, self.i_max_a

    def v_cv_bounds(self) -> Tuple[float, float]:
        return self.v_cv_min_v, self.v_cv_max_v

    def soc_bounds(self) -> Tuple[float, float]:
        return self.soc_switch_min, self.soc_switch_max

    def to_dict(self) -> Dict:
        return asdict(self)
