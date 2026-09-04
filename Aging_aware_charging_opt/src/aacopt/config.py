"""Configuration loading and provenance stamping.

Every stage of the pipeline resolves its settings through this module so that
``validate_experiment.py`` can prove that two runs used identical configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parent
CONFIG_DIR = PROJECT_DIR / "configs"
RESULTS_DIR = PROJECT_DIR / "results"

CONFIG_FILES = {
    "paths": "paths.yaml",
    "degradation": "degradation.yaml",
    "degradation_fitted": "degradation_fitted.yaml",
    "reward": "reward.yaml",
    "optimization": "optimization.yaml",
}


def load_yaml(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def config_path(name: str) -> Path:
    if name not in CONFIG_FILES:
        raise KeyError(f"unknown config {name!r}; known: {sorted(CONFIG_FILES)}")
    return CONFIG_DIR / CONFIG_FILES[name]


def load_config(name: str) -> Dict[str, Any]:
    return load_yaml(config_path(name))


def config_hash(name: str) -> str:
    """SHA-256 of the raw config bytes — identifies the exact settings used."""
    return hashlib.sha256(config_path(name).read_bytes()).hexdigest()


def file_hash(path: Path) -> Optional[str]:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def package_versions() -> Dict[str, str]:
    mods = ["numpy", "scipy", "pandas", "matplotlib", "torch", "skopt", "yaml"]
    out: Dict[str, str] = {"python": sys.version.split()[0]}
    for m in mods:
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = "unavailable"
    return out


def provenance(stage: str, *, configs: List[str], inputs: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Provenance block embedded in every stage's output JSON."""
    return {
        "stage": stage,
        "project": "Aging_aware_charging_opt",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "host": platform.node(),
        "packages": package_versions(),
        "config_hashes": {c: config_hash(c) for c in configs if config_path(c).is_file()},
        "inputs": [
            {"path": str(Path(p)), "sha256": file_hash(Path(p))}
            for p in (inputs or [])
        ],
    }


# ── resolved views over the YAML ────────────────────────────────────────────


@dataclass(frozen=True)
class Paths:
    matlab_dir: Path
    cells: List[str]
    bdt_checkpoints: Dict[str, Path]
    crosscheck_capacity_fade_csv: Optional[Path]

    @classmethod
    def load(cls) -> "Paths":
        cfg = load_config("paths")
        cc = cfg.get("crosscheck_capacity_fade_csv")
        return cls(
            matlab_dir=REPO_ROOT / str(cfg["nasa_matlab_dir"]).strip(),
            cells=list(cfg["cells"]),
            bdt_checkpoints={
                k: REPO_ROOT / v for k, v in (cfg.get("bdt_checkpoints") or {}).items()
            },
            crosscheck_capacity_fade_csv=(REPO_ROOT / cc) if cc else None,
        )

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
    soc_start: float
    ambient_t0_c: float
    v_max: float
    v_nom_fallback: float
    max_duration_min: float
    decision_interval_s: int
    bdt_switch_pad: int
    constraint_mode: str
    energy_fraction: float


@dataclass(frozen=True)
class SearchSpace:
    i_min_a: float
    i_max_a: float
    v_cv_min_v: float
    v_cv_max_v: float
    soc_switch_min: float
    soc_switch_max: float
    pulse_on_min_min: float
    pulse_on_max_min: float
    rest_fraction_min: float
    rest_fraction_max: float
    min_step_di_a: float
    min_stage_dsoc: float
    min_stage1_dsoc: float
    min_stage_gap_dsoc: float
    cv_step_a: float
    min_charge_a: float

    def i_bounds(self) -> Tuple[float, float]:
        return self.i_min_a, self.i_max_a

    def v_cv_bounds(self) -> Tuple[float, float]:
        return self.v_cv_min_v, self.v_cv_max_v

    def soc_bounds(self) -> Tuple[float, float]:
        return self.soc_switch_min, self.soc_switch_max


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
        cfg = load_config("optimization")
        return cls(
            session=SessionSpec(**cfg["session"]),
            space=SearchSpace(**cfg["search_space"]),
            families=list(cfg["families"]),
            n_evals_per_family=int(cfg["budget"]["n_evals_per_family"]),
            gp_bo=dict(cfg["budget"]["gp_bo"]),
            random_search=dict(cfg["budget"]["random_search"]),
            master_seed=int(cfg["seed"]["master"]),
            method_index=dict(cfg["seed"]["method_index"]),
            baselines=list(cfg["baselines"]),
            lifetime=dict(cfg["lifetime"]),
            device=str(cfg.get("device", "cpu")),
        )

    def seed_for(self, cell: str, family: str, method: str) -> int:
        cells = Paths.load().cells
        ci = cells.index(cell.upper()) if cell.upper() in cells else 0
        fi = self.families.index(family) if family in self.families else 0
        mi = int(self.method_index.get(method, 0))
        return int(self.master_seed + 1000 * ci + 100 * fi + mi)


@dataclass(frozen=True)
class RewardSpec:
    w_soc: float
    w_loss: float
    w_time: float
    z_time: float
    duration_loss_weight: float
    energy_shortfall_penalty_scale: float
    voltage_ceiling_penalty_scale: float

    @classmethod
    def load(cls) -> "RewardSpec":
        cfg = load_config("reward")
        w = cfg["weights"]
        cons = cfg["constraints"]
        return cls(
            w_soc=float(w["w_soc"]),
            w_loss=float(w["w_loss"]),
            w_time=float(w["w_time"]),
            z_time=float(w["z_time"]),
            duration_loss_weight=float(w["duration_loss_weight"]),
            energy_shortfall_penalty_scale=float(cons["energy_shortfall_penalty_scale"]),
            voltage_ceiling_penalty_scale=float(cons["voltage_ceiling_penalty_scale"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_soc": self.w_soc,
            "w_loss": self.w_loss,
            "w_time": self.w_time,
            "z_time": self.z_time,
            "duration_loss_weight": self.duration_loss_weight,
            "energy_shortfall_penalty_scale": self.energy_shortfall_penalty_scale,
            "voltage_ceiling_penalty_scale": self.voltage_ceiling_penalty_scale,
        }


def stage_dir(name: str, *, create: bool = True) -> Path:
    d = RESULTS_DIR / name
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


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


def ensure_matplotlib_cache() -> None:
    """Point MPLCONFIGDIR somewhere writable (the repo runs as varied users)."""
    if not os.environ.get("MPLCONFIGDIR"):
        cache = RESULTS_DIR / ".mplcache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
