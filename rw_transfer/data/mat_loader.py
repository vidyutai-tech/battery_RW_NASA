"""Load NASA RW MATLAB ``.mat`` step records."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from scipy.io import loadmat

from rw_transfer.constants import STEP_MODE_COMMENTS


@dataclass(frozen=True)
class BatteryStep:
    comment: str
    step_type: str
    time_s: np.ndarray
    relative_time_s: np.ndarray
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray


def _normalize_comment(raw: object) -> str:
    if isinstance(raw, np.ndarray):
        if raw.size == 0:
            return ""
        raw = raw.flat[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def _as_1d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64).ravel()
    return a


def load_cell_steps(
    mat_path: str | Path,
    step_mode: str = "rw_operational",
    allowed_comments: Optional[Set[str]] = None,
) -> List[BatteryStep]:
    """
    Parse one ``RW*.mat`` file into filtered step objects.

    Parameters
    ----------
    step_mode
        ``rw_operational`` | ``rw_plus_reference`` | ``all``
    allowed_comments
        Override comment whitelist (for custom ablations).
    """
    mat_path = Path(mat_path)
    if allowed_comments is None:
        allowed = STEP_MODE_COMMENTS.get(step_mode)
        if allowed is None and step_mode != "all":
            raise ValueError(f"Unknown step_mode: {step_mode!r}")
    else:
        allowed = allowed_comments

    raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)["data"]
    steps_out: List[BatteryStep] = []

    for step in raw.step:
        comment = _normalize_comment(step.comment)
        if allowed is not None and comment not in allowed:
            continue
        stype = _normalize_comment(step.type)
        steps_out.append(
            BatteryStep(
                comment=comment,
                step_type=stype,
                time_s=_as_1d(step.time),
                relative_time_s=_as_1d(step.relativeTime),
                voltage_v=_as_1d(step.voltage),
                current_a=_as_1d(step.current),
                temperature_c=_as_1d(step.temperature),
            )
        )
    return steps_out


def mat_path_for_cell(matlab_dir: str | Path, cell_id: str) -> Path:
    matlab_dir = Path(matlab_dir)
    cell_id = cell_id.upper()
    if not cell_id.startswith("RW"):
        cell_id = f"RW{cell_id}"
    path = matlab_dir / f"{cell_id}.mat"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path
