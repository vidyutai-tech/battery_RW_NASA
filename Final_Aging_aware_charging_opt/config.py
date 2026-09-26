"""Settings for the final aging-aware charging optimization.

Updated GP-BO (the run this package reproduces):

- Reward weights put SOC, degradation, and time on the same scale as a 0.5C
  CCCV charge: ``w_soc = 2.5``, ``w_loss = 7500``, ``w_time = 1.1``.
- The finished reward is not multiplied by 1000 (``REWARD_SCALE = 1``).
- Every GP-BO search uses seed 42.
- Charge current is capped at the minimum training-window maximum across
  RW9–RW12 (4.616 A), not the old 6 A bound.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
RESULTS_DIR = PACKAGE_DIR / "results"
OCV_NPZ = PACKAGE_DIR / "ocv_curves.npz"

CELLS = ["RW9", "RW10", "RW11", "RW12"]

# Minimum of the four cells' maximum charging currents in the twin training windows.
I_MAX_A = 4.616000175476074

W_SOC = 2.5
W_LOSS = 7500.0
W_TIME = 1.1
Z_TIME = 0.55
REWARD_SCALE = 1.0
GP_BO_SEED = 42

MATLAB_DIR = (
    REPO_ROOT
    / "NASA_RW/dataset/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post"
    / "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab"
)

CHECKPOINTS = {
    "RW9": REPO_ROOT / "outputs/twin_source/20260610_111409/twin_source_RW9.pt",
    "RW10": REPO_ROOT / "outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.60.pt",
    "RW11": REPO_ROOT / "outputs/finetune_two_stage_RW11/registry/finetune_RW11_frac0.60.pt",
    "RW12": REPO_ROOT / "outputs/finetune_two_stage_RW12/registry/finetune_RW12_frac0.60.pt",
}

# Frozen operating-window fit (calendar literature-anchored, cyclic on NASA RW).
DEGRADATION = {
    "A_cal": 71011.01912693257,
    "B_cal": 1.0986122886681098,
    "C_cal": 0.0,
    "Ea_cal": 50000.0,
    "z_cal": 0.5,
    "c_rate_bins": [0.5, 1.0, 2.0],
    "B_cyc": [4293453.763085374, 18820573.106659856, 82500940.17749056],
    "Ea_cyc": [69312.6, 69312.6, 69312.6],
    "z_cyc": [1.4001, 1.4001, 1.4001],
    "R": 8.314,
    "q_nominal_ah": 2.2,
    "provenance": "frozen_operating_window_4p_pooled",
}

Q_NOMINAL_AH = float(DEGRADATION["q_nominal_ah"])


@dataclass(frozen=True)
class Paths:
    matlab_dir: Path
    cells: List[str]
    bdt_checkpoints: Dict[str, Path]

    @classmethod
    def load(cls) -> "Paths":
        return cls(matlab_dir=MATLAB_DIR, cells=list(CELLS), bdt_checkpoints=dict(CHECKPOINTS))

    def mat_path(self, cell: str) -> Path:
        p = self.matlab_dir / f"{cell.upper()}.mat"
        if not p.is_file():
            raise FileNotFoundError(f"NASA .mat not found for {cell}: {p}")
        return p

    def checkpoint(self, cell: str) -> Path:
        p = self.bdt_checkpoints[cell.upper()]
        if not p.is_file():
            raise FileNotFoundError(f"BDT checkpoint not found for {cell}: {p}")
        return p


@dataclass(frozen=True)
class SessionSpec:
    soc_start: float = 0.20
    ambient_t0_c: float = 24.0
    v_max: float = 4.20
    v_nom_fallback: float = 3.70
    max_duration_min: float = 150.0
    decision_interval_s: int = 30
    bdt_switch_pad: int = 5
    constraint_mode: str = "energy"
    energy_fraction: float = 0.40


@dataclass(frozen=True)
class SearchSpace:
    i_min_a: float = 0.75
    i_max_a: float = I_MAX_A
    v_cv_min_v: float = 4.05
    v_cv_max_v: float = 4.20
    soc_switch_min: float = 0.32
    soc_switch_max: float = 0.90
    pulse_on_min_min: float = 1.0
    pulse_on_max_min: float = 15.0
    rest_fraction_min: float = 0.5
    rest_fraction_max: float = 2.5
    min_step_di_a: float = 0.25
    min_stage_dsoc: float = 0.12
    min_stage1_dsoc: float = 0.10
    min_stage_gap_dsoc: float = 0.12
    cv_step_a: float = 0.25
    min_charge_a: float = 0.05

    def i_bounds(self) -> Tuple[float, float]:
        return self.i_min_a, self.i_max_a

    def v_cv_bounds(self) -> Tuple[float, float]:
        return self.v_cv_min_v, self.v_cv_max_v

    def soc_bounds(self) -> Tuple[float, float]:
        return self.soc_switch_min, self.soc_switch_max


@dataclass(frozen=True)
class RewardSpec:
    w_soc: float = W_SOC
    w_loss: float = W_LOSS
    w_time: float = W_TIME
    z_time: float = Z_TIME
    duration_loss_weight: float = 1.0e-3
    energy_shortfall_penalty_scale: float = 300.0
    voltage_ceiling_penalty_scale: float = 100.0

    @classmethod
    def load(cls) -> "RewardSpec":
        return cls()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_soc": self.w_soc,
            "w_loss": self.w_loss,
            "w_time": self.w_time,
            "z_time": self.z_time,
            "duration_loss_weight": self.duration_loss_weight,
            "energy_shortfall_penalty_scale": self.energy_shortfall_penalty_scale,
            "voltage_ceiling_penalty_scale": self.voltage_ceiling_penalty_scale,
            "reward_scale": REWARD_SCALE,
        }


@dataclass(frozen=True)
class OptimizationSpec:
    session: SessionSpec
    space: SearchSpace
    families: List[str]
    n_evals_per_family: int
    gp_bo: Dict[str, Any]
    random_search: Dict[str, Any]
    master_seed: int
    method_index: Dict[str, int]
    baselines: List[Dict[str, Any]]
    lifetime: Dict[str, Any]
    device: str

    @classmethod
    def load(cls) -> "OptimizationSpec":
        return OPT

    def seed_for(self, cell: str, family: str, method: str) -> int:
        if method in ("gp_bo", "random_search"):
            return GP_BO_SEED
        cells = Paths.load().cells
        ci = cells.index(cell.upper()) if cell.upper() in cells else 0
        fi = self.families.index(family) if family in self.families else 0
        mi = int(self.method_index.get(method, 0))
        return int(self.master_seed + 1000 * ci + 100 * fi + mi)


OPT = OptimizationSpec(
    session=SessionSpec(),
    space=SearchSpace(),
    families=["cccv", "two_step", "three_step", "pulsed"],
    n_evals_per_family=80,
    gp_bo={
        "n_initial_points": 15,
        "acq_func": "EI",
        "acq_xi": 0.01,
        "noise": 1.0e-6,
        "warm_start_from_random_search": False,
    },
    random_search={"sampler": "uniform"},
    master_seed=20260904,
    method_index={"random_search": 0, "gp_bo": 1},
    baselines=[
        {"name": "CCCV 0.5C", "c_rate": 0.5, "v_cv": 4.20, "i_cutoff": 0.05},
        {"name": "CCCV 1C", "c_rate": 1.0, "v_cv": 4.20, "i_cutoff": 0.05},
        {"name": "CCCV 2C", "c_rate": 2.0, "v_cv": 4.20, "i_cutoff": 0.05},
    ],
    lifetime={
        "n_equivalent_cycles": 600,
        "report_at_cycles": [100, 200, 400, 600],
        "engineering_reference_retention_pct": 80.0,
    },
    device="auto",
)


def result_path(name: str) -> Path:
    """Flat file under ``results/``. No nested output folders."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / name


def family_result_name(method: str, cell: str, family: str) -> str:
    return f"{method}_{cell.upper()}_{family}.json"


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path: Path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _settings_blob() -> Dict[str, Any]:
    return {
        "weights": RewardSpec.load().to_dict(),
        "reward_scale": REWARD_SCALE,
        "gp_bo_seed": GP_BO_SEED,
        "i_max_a": I_MAX_A,
        "degradation": DEGRADATION,
        "families": OPT.families,
        "n_evals_per_family": OPT.n_evals_per_family,
        "gp_bo": OPT.gp_bo,
        "session": OPT.session.__dict__,
        "energy_fraction": OPT.session.energy_fraction,
    }


def provenance(stage: str, *, configs: Optional[List[str]] = None, inputs: Optional[List[Path]] = None) -> Dict[str, Any]:
    del configs
    raw = json.dumps(_settings_blob(), sort_keys=True, default=_json_default).encode()
    try:
        git = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip() or None
    except Exception:
        git = None
    packages = {"python": sys.version.split()[0]}
    for m in ("numpy", "scipy", "matplotlib", "torch", "skopt"):
        try:
            packages[m] = __import__(m).__version__
        except Exception:
            packages[m] = "unavailable"
    return {
        "stage": stage,
        "project": "Final_Aging_aware_charging_opt",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git,
        "host": platform.node(),
        "packages": packages,
        "settings_sha256": hashlib.sha256(raw).hexdigest(),
        "inputs": [{"path": str(Path(p)), "sha256": file_hash(Path(p))} for p in (inputs or [])],
    }


def ensure_matplotlib_cache() -> None:
    if not os.environ.get("MPLCONFIGDIR"):
        cache = Path("/tmp/final_aacopt_mpl")
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
