"""Current and voltage samples from NASA RW Matlab steps for profile parameters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MATLAB_DIR = (
    REPO_ROOT
    / "NASA_RW/dataset/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post"
    / "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab"
)

NASA_RW_CHARGE_A = [0.75, 1.5, 2.25, 3.0, 3.75, 4.5]
NASA_V_MIN = 3.2
NASA_V_MAX = 4.2
NASA_CV_LEVELS_V = [4.05, 4.10, 4.15, 4.20]
NASA_REFERENCE_CCCV = {"i_cc": 2.0, "v_cv": 4.2, "i_cutoff": 0.01}
NASA_PULSED_CHARGE = {
    "i_charge": 1.0,
    "pulse_on_min": 10.0,
    "pulse_rest_min": 20.0,
    "rest_fraction": 2.0,
}
DEFAULT_SOC_SWITCHES = [0.30, 0.50, 0.65, 0.75]


@dataclass
class ProfileCatalog:
    cell_id: str
    rw_charge_currents_a: List[float]
    v_min: float
    v_max: float
    cv_levels_v: List[float]
    reference_cccv: Dict[str, float]
    pulsed_charge: Dict[str, float]
    soc_switch_levels: List[float] = field(default_factory=lambda: list(DEFAULT_SOC_SWITCHES))

    @classmethod
    def nasa_defaults(cls, cell_id: str) -> ProfileCatalog:
        return cls(
            cell_id=cell_id.upper(),
            rw_charge_currents_a=list(NASA_RW_CHARGE_A),
            v_min=NASA_V_MIN,
            v_max=NASA_V_MAX,
            cv_levels_v=list(NASA_CV_LEVELS_V),
            reference_cccv=dict(NASA_REFERENCE_CCCV),
            pulsed_charge=dict(NASA_PULSED_CHARGE),
            soc_switch_levels=list(DEFAULT_SOC_SWITCHES),
        )

    def to_dict(self) -> Dict:
        return asdict(self)


def catalog_path(cell_id: str) -> Path:
    return DATA_DIR / cell_id.upper() / "profile_catalog.json"


def save_catalog(catalog: ProfileCatalog, path: Optional[Path] = None) -> Path:
    path = path or catalog_path(catalog.cell_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(catalog.to_dict(), f, indent=2)
    return path


def load_catalog(cell_id: str, path: Optional[Path] = None) -> ProfileCatalog:
    path = path or catalog_path(cell_id)
    with open(path) as f:
        data = json.load(f)
    return ProfileCatalog(**data)


def snap_to_nearest(value: float, grid: List[float]) -> float:
    arr = np.asarray(grid, dtype=np.float64)
    return float(arr[int(np.argmin(np.abs(arr - float(value))))])


def _extract_reference_cccv(steps) -> Optional[Dict[str, float]]:
    for step in steps:
        if step.comment.strip().lower() != "reference charge":
            continue
        i = np.asarray(step.current_a, dtype=np.float64)
        v = np.asarray(step.voltage_v, dtype=np.float64)
        if i.size < 20:
            continue
        i_mag = np.maximum(-i, 0.0)
        cc_mask = i_mag > 0.05
        if not np.any(cc_mask):
            continue
        i_cc = float(np.median(i_mag[cc_mask][: max(60, int(0.1 * i_mag.size))]))
        v_cv = float(np.clip(np.max(v), NASA_V_MIN, NASA_V_MAX))
        tail = i_mag[i_mag > 0]
        i_cutoff = float(np.min(tail[-max(10, tail.size // 20):])) if tail.size else 0.01
        i_cutoff = max(0.01, min(i_cutoff, 0.5))
        return {"i_cc": round(i_cc, 3), "v_cv": round(v_cv, 3), "i_cutoff": round(i_cutoff, 3)}
    return None


def _extract_pulsed_charge(steps) -> Optional[Dict[str, float]]:
    for step in steps:
        if step.comment.strip().lower() != "pulsed charge (charge)":
            continue
        i = np.asarray(step.current_a, dtype=np.float64)
        t = np.asarray(step.time_s, dtype=np.float64)
        if i.size < 2:
            continue
        i_charge = float(np.median(np.maximum(-i, 0.0)))
        pulse_on_min = float(max((t[-1] - t[0]) / 60.0, 1.0))
        return {
            "i_charge": round(i_charge, 3),
            "pulse_on_min": round(pulse_on_min, 2),
            "pulse_rest_min": round(pulse_on_min * 2.0, 2),
            "rest_fraction": 2.0,
        }
    return None


def _extract_soc_switches(steps, q_rated_as: float) -> List[float]:
    for step in steps:
        if step.comment.strip().lower() != "reference charge":
            continue
        i = np.asarray(step.current_a, dtype=np.float64)
        t = np.asarray(step.time_s, dtype=np.float64)
        v = np.asarray(step.voltage_v, dtype=np.float64)
        if i.size < 30:
            continue
        dt = np.diff(t, prepend=t[0])
        dt[0] = 0.0
        i_charge = np.maximum(-i, 0.0)
        delivered = np.cumsum(i_charge * dt)
        soc = np.clip(delivered / q_rated_as, 0.0, 1.0)
        cv_start = int(np.argmax(v >= 4.15 - 1e-3))
        if cv_start <= 0:
            cv_start = int(0.7 * soc.size)
        switches = [
            float(np.clip(soc[int(0.35 * soc.size)], 0.1, 0.9)),
            float(np.clip(soc[cv_start], 0.1, 0.9)),
        ]
        extra = [0.30, 0.50, 0.65]
        merged = sorted(set(round(s, 2) for s in switches + extra))
        return merged
    return list(DEFAULT_SOC_SWITCHES)


def extract_profile_catalog(
    cell_id: str,
    *,
    matlab_dir: Optional[Path] = None,
    q_rated_as: float = 7200.0,
) -> ProfileCatalog:
    """Sample I/V levels from NASA Matlab steps; fall back to README defaults."""
    defaults = ProfileCatalog.nasa_defaults(cell_id)
    mat_dir = Path(matlab_dir or DEFAULT_MATLAB_DIR)
    try:
        from charging_opt.soc_utils import load_steps_with_age

        steps, _ = load_steps_with_age(mat_dir, cell_id)
    except Exception:
        return defaults

    ref = _extract_reference_cccv(steps)
    pulsed = _extract_pulsed_charge(steps)
    soc_sw = _extract_soc_switches(steps, q_rated_as)

    currents = list(NASA_RW_CHARGE_A)

    cv_levels = sorted(set(defaults.cv_levels_v))
    if ref:
        cv_levels.append(min(ref["v_cv"], NASA_V_MAX))
        cv_levels = sorted(set(round(v, 2) for v in cv_levels))

    return ProfileCatalog(
        cell_id=cell_id.upper(),
        rw_charge_currents_a=currents,
        v_min=NASA_V_MIN,
        v_max=NASA_V_MAX,
        cv_levels_v=cv_levels,
        reference_cccv=ref or defaults.reference_cccv,
        pulsed_charge=pulsed or defaults.pulsed_charge,
        soc_switch_levels=soc_sw,
    )


def load_or_extract_catalog(
    cell_id: str,
    *,
    matlab_dir: Optional[Path] = None,
    refit: bool = False,
    q_rated_as: float = 7200.0,
) -> ProfileCatalog:
    path = catalog_path(cell_id)
    if path.exists() and not refit:
        try:
            return load_catalog(cell_id, path)
        except Exception:
            pass
    catalog = extract_profile_catalog(cell_id, matlab_dir=matlab_dir, q_rated_as=q_rated_as)
    save_catalog(catalog, path)
    return catalog
