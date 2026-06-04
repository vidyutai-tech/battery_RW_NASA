"""Load YAML experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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
    return cfg


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
