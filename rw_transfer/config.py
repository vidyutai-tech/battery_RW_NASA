"""Load YAML experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from rw_transfer.constants import NASA_NOMINAL_Q_AH, SECONDS_PER_AH


def _resolve_q_rated_as(data: Dict[str, Any]) -> None:
    """Set ``q_rated_as`` (A·s) from ``q_rated_ah`` (Ah) when present."""
    if "q_rated_ah" in data:
        data["q_rated_as"] = float(data["q_rated_ah"]) * SECONDS_PER_AH
    elif "q_rated_as" not in data:
        data["q_rated_as"] = NASA_NOMINAL_Q_AH * SECONDS_PER_AH


def load_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = path.resolve().parents[1]
    data_dir = root / cfg["data"]["matlab_dir"]
    cfg["data"]["matlab_dir"] = str(data_dir)
    cfg["output"]["root"] = str(root / cfg["output"]["root"])
    _resolve_q_rated_as(cfg["data"])
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
