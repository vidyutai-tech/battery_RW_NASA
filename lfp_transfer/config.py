"""Load LFP fine-tuning YAML and resolve paths relative to the project root."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rw_transfer.config import _resolve_q_rated_as, load_config


def load_lfp_config(path: Optional[str | Path] = None) -> Dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs" / "lfp_finetune.yaml"
    path = Path(path)
    cfg = load_config(path)

    root = path.resolve().parents[1]
    data = cfg.setdefault("data", {})

    lfp_mat = data.get("lfp_mat")
    if lfp_mat:
        mat_path = Path(lfp_mat)
        if not mat_path.is_absolute():
            mat_path = root / mat_path
        data["lfp_mat"] = str(mat_path)

    _resolve_q_rated_as(data)
    return cfg
