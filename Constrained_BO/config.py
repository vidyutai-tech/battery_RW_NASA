"""Cell configs, BDT checkpoint paths, and default start states."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from Constrained_BO.objective import V_NOM_FALLBACK
from Constrained_BO.profile_catalog import ProfileCatalog

REPO_ROOT = Path(__file__).resolve().parents[1]

Q_RATED_AS = 7200.0
SOC_TARGET = 0.95
SOC_START = 0.20
MAX_DURATION_MIN = 150.0
V_MAX = 4.2
DECISION_INTERVAL_S = 30

TWIN_SOURCE = REPO_ROOT / "outputs/twin_source/20260610_111409/twin_source_RW9.pt"
FINETUNE_FRAC = "0.40"


@dataclass
class CellConfig:
    cell_id: str
    bdt_ckpt: Path
    start_state: Dict[str, float] = field(default_factory=dict)
    q_rated_as: float = Q_RATED_AS
    soc_target: float = SOC_TARGET
    max_duration_min: float = MAX_DURATION_MIN
    energy_fraction: Optional[float] = None
    v_nom: float = V_NOM_FALLBACK
    constraint_mode: str = "soc"
    profile_catalog: Optional[ProfileCatalog] = None

    def with_run_overrides(
        self,
        *,
        soc_target: Optional[float] = None,
        soc_delta: Optional[float] = None,
        energy_fraction: Optional[float] = None,
        max_duration_min: Optional[float] = None,
        v_nom: Optional[float] = None,
    ) -> "CellConfig":
        """Apply CLI overrides; soc_delta sets energy_fraction when energy mode is used."""
        frac = energy_fraction if energy_fraction is not None else soc_delta
        v = v_nom if v_nom is not None else self.v_nom
        max_d = max_duration_min if max_duration_min is not None else self.max_duration_min

        if frac is not None:
            soc_start = float(self.start_state.get("soc", SOC_START))
            target = soc_target if soc_target is not None else min(soc_start + frac, 1.0)
            return CellConfig(
                cell_id=self.cell_id,
                bdt_ckpt=self.bdt_ckpt,
                start_state=dict(self.start_state),
                q_rated_as=self.q_rated_as,
                soc_target=target,
                max_duration_min=max_d,
                energy_fraction=frac,
                v_nom=v,
                constraint_mode="energy",
                profile_catalog=self.profile_catalog,
            )

        target = soc_target if soc_target is not None else self.soc_target
        return CellConfig(
            cell_id=self.cell_id,
            bdt_ckpt=self.bdt_ckpt,
            start_state=dict(self.start_state),
            q_rated_as=self.q_rated_as,
            soc_target=target,
            max_duration_min=max_d,
            energy_fraction=None,
            v_nom=v,
            constraint_mode="soc",
            profile_catalog=self.profile_catalog,
        )


def _finetune_ckpt(cell: str) -> Path:
    return (
        REPO_ROOT
        / f"outputs/finetune_two_stage_{cell}/registry/finetune_{cell}_frac{FINETUNE_FRAC}.pt"
    )


def default_start_state() -> Dict[str, float]:
    """RW9-aligned start at 20% SoC using OCV curve (fallback if OCV unavailable)."""
    from Constrained_BO.ocv import build_start_state

    try:
        return build_start_state("RW9", soc=SOC_START, age=0.0)
    except Exception:
        return {
            "soc": SOC_START,
            "v0": 3.78,
            "t0": 24.7,
            "age": 0.0,
            "prev_i": 0.0,
        }


def extract_start_state(cell_id: str, matlab_dir: Optional[Path] = None) -> Dict[str, float]:
    """OCV-aligned rest state at SOC_START for the given cell."""
    from Constrained_BO.ocv import build_start_state

    return build_start_state(cell_id, soc=SOC_START, age=0.0, matlab_dir=matlab_dir)


def get_cell_config(
    cell_id: str,
    *,
    matlab_dir: Optional[Path] = None,
    refit_ocv: bool = False,
    refit_catalog: bool = False,
) -> CellConfig:
    cell_id = cell_id.upper()
    if cell_id == "RW9":
        ckpt = TWIN_SOURCE
    elif cell_id in ("RW10", "RW11", "RW12"):
        ckpt = _finetune_ckpt(cell_id)
    else:
        raise ValueError(f"Unknown cell {cell_id!r}; expected RW9–RW12")

    if not ckpt.exists():
        raise FileNotFoundError(f"BDT checkpoint not found: {ckpt}")

    from Constrained_BO.ocv import (
        build_start_state,
        load_or_fit_ocv,
        nominal_voltage_from_ocv,
    )

    ocv_spline = load_or_fit_ocv(
        cell_id, matlab_dir=matlab_dir, refit=refit_ocv,
    )
    state = build_start_state(
        cell_id,
        soc=SOC_START,
        age=0.0,
        matlab_dir=matlab_dir,
        ocv_spline=ocv_spline,
        refit_ocv=False,
    )
    try:
        v_nom = nominal_voltage_from_ocv(cell_id, ocv_spline=ocv_spline)
    except Exception:
        v_nom = V_NOM_FALLBACK

    from Constrained_BO.profile_catalog import load_or_extract_catalog

    catalog = load_or_extract_catalog(
        cell_id,
        matlab_dir=matlab_dir,
        refit=refit_catalog,
        q_rated_as=Q_RATED_AS,
    )

    return CellConfig(
        cell_id=cell_id,
        bdt_ckpt=ckpt,
        start_state=state,
        v_nom=v_nom,
        profile_catalog=catalog,
    )


ALL_CELLS: List[str] = ["RW9", "RW10", "RW11", "RW12"]
