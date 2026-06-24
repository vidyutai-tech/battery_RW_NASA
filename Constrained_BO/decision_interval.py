"""Pick closed-loop re-anchor interval (s) by minimum BDT rollout error."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DECISION_INTERVAL_S = 30
DECISION_INTERVAL_CANDIDATES: Tuple[int, ...] = (10, 15, 30, 60)
CALIBRATION_HORIZON_S = 600
CALIBRATION_I_CHARGE_A = 2.0


def _contiguous_1hz_spans(time_s: np.ndarray, tol: float = 0.15) -> List[Tuple[int, int]]:
    dt = np.diff(time_s)
    ok = (dt > 0) & (dt <= 1.0 + tol)
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, good in enumerate(ok):
        if good and start is None:
            start = i
        elif not good and start is not None:
            spans.append((start, i + 1))
            start = None
    if start is not None:
        spans.append((start, len(time_s)))
    return spans


def _chained_rollout(
    bdt,
    *,
    v0: float,
    t0: float,
    age: float,
    i_charge_a: float,
    decision_interval_s: int,
    n_decisions: int,
) -> Tuple[np.ndarray, np.ndarray]:
    state: Dict[str, float] = {
        "v0": v0, "t0": t0, "age": age, "prev_i": 0.0,
    }
    action = -abs(float(i_charge_a))
    v_all: List[float] = []
    t_all: List[float] = []
    for _ in range(n_decisions):
        next_state, v_traj, t_traj, _ = bdt.single_step(
            state, action, n_steps=int(decision_interval_s),
        )
        v_all.extend(v_traj.tolist())
        t_all.extend(t_traj.tolist())
        state = next_state
    return np.asarray(v_all, dtype=np.float64), np.asarray(t_all, dtype=np.float64)


def _segment_score(
    v_pred: np.ndarray,
    t_pred: np.ndarray,
    v_meas: np.ndarray,
    t_meas: np.ndarray,
) -> float:
    n = min(v_pred.size, v_meas.size, t_pred.size, t_meas.size)
    if n < 10:
        return float("inf")
    dv = v_pred[:n] - v_meas[:n]
    dt = t_pred[:n] - t_meas[:n]
    v_rmse = float(np.sqrt(np.mean(dv * dv)))
    t_rmse = float(np.sqrt(np.mean(dt * dt)))
    return v_rmse + 0.01 * t_rmse


def _find_calibration_window(
    series,
    start_state: Dict[str, float],
    *,
    horizon_s: int,
    i_charge_a: float,
) -> Optional[Tuple[int, int]]:
    """Return [start, end) indices for a ~constant-current charge segment."""
    v0 = float(start_state.get("v0", series.voltage_v[0]))
    t0 = float(start_state.get("t0", series.temperature_c[0]))
    age = float(start_state.get("age", series.age[0]))

    target_i = -abs(i_charge_a)
    cur = series.current_a
    v = series.voltage_v
    t = series.temperature_c
    ages = series.age

    best: Optional[Tuple[int, int, float]] = None
    for a, b in _contiguous_1hz_spans(series.time_s):
        if b - a < horizon_s + 5:
            continue
        for s in range(a, b - horizon_s, max(1, horizon_s // 4)):
            e = s + horizon_s
            w = cur[s:e]
            if np.median(w) > -0.5:
                continue
            if np.abs(np.median(w) - target_i) > 0.35 or w.std() > 0.15:
                continue
            dist = (
                abs(v[s] - v0)
                + 0.05 * abs(t[s] - t0)
                + 0.5 * abs(ages[s] - age)
            )
            if best is None or dist < best[2]:
                best = (s, e, dist)
    if best is None:
        return None
    return best[0], best[1]


def _score_from_conformal_margins(
    margins_path: Path,
    candidates: Sequence[int],
) -> Optional[int]:
    if not margins_path.exists():
        return None
    m = np.load(margins_path)
    hmax = int(m["horizon_s"].size)
    best_dt: Optional[int] = None
    best_score = float("inf")
    for dt in candidates:
        h = min(int(dt), hmax) - 1
        if h < 0:
            continue
        score = float(m["v_q95"][h]) + 0.01 * float(m["t_q95"][h])
        if score < best_score:
            best_score = score
            best_dt = int(dt)
    return best_dt


def resolve_margins_path(cell_id: str) -> Optional[Path]:
    cell_id = cell_id.upper()
    canonical = REPO_ROOT / "outputs/charging_opt/models/stage1_drift/conformal_margins.npz"
    if canonical.exists():
        return canonical
    user_root = REPO_ROOT / "outputs/charging_opt_user"
    if user_root.exists():
        matches = sorted(user_root.glob(f"*/cells/{cell_id}/stage1_drift/conformal_margins.npz"))
        if matches:
            return matches[-1]
    return None


def select_decision_interval_s(
    bdt,
    cell_id: str,
    start_state: Dict[str, float],
    *,
    candidates: Sequence[int] = DECISION_INTERVAL_CANDIDATES,
    matlab_dir: Optional[Path] = None,
    margins_path: Optional[Path] = None,
    horizon_s: int = CALIBRATION_HORIZON_S,
    i_charge_a: float = CALIBRATION_I_CHARGE_A,
    verbose: bool = True,
) -> Tuple[int, Dict[str, float]]:
    """
    Pick re-anchor interval with lowest chained-rollout V/T error on NASA RW data.

    Falls back to conformal p95 margins, then ``DEFAULT_DECISION_INTERVAL_S``.
    """
    candidates = tuple(int(c) for c in candidates)
    scores: Dict[int, float] = {}
    method = "default"

    try:
        from rw_transfer.config import load_config
        from rw_transfer.data.series import load_battery_series

        cfg = load_config()
        mat_dir = matlab_dir or Path(cfg["data"]["matlab_dir"])
        series = load_battery_series(
            mat_dir, cell_id.upper(), step_mode="rw_operational", decimation=1,
        )
        window = _find_calibration_window(
            series, start_state, horizon_s=horizon_s, i_charge_a=i_charge_a,
        )
        if window is not None:
            s, e = window
            n_dec = max(1, (e - s) // min(candidates))
            v_meas = series.voltage_v[s:e]
            t_meas = series.temperature_c[s:e]
            for dt in candidates:
                v_pred, t_pred = _chained_rollout(
                    bdt,
                    v0=float(series.voltage_v[s]),
                    t0=float(series.temperature_c[s]),
                    age=float(series.age[s]),
                    i_charge_a=i_charge_a,
                    decision_interval_s=dt,
                    n_decisions=n_dec,
                )
                scores[dt] = _segment_score(v_pred, t_pred, v_meas, t_meas)
            if scores and min(scores.values()) < float("inf"):
                best = min(scores, key=scores.get)
                method = "chained_rollout"
                if verbose:
                    ranked = ", ".join(f"{k}s={scores[k]:.4f}" for k in sorted(scores))
                    print(f"Decision interval calibration ({cell_id}): {ranked} → {best}s")
                return best, {
                    "method": method,
                    "source": method,
                    "selected_s": int(best),
                    "scores": scores,
                    "horizon_s": float(e - s),
                }
    except Exception as exc:
        if verbose:
            print(f"Decision interval: RW calibration skipped ({exc})")

    margins = margins_path or resolve_margins_path(cell_id)
    if margins is not None:
        picked = _score_from_conformal_margins(margins, candidates)
        if picked is not None:
            if verbose:
                print(
                    f"Decision interval from conformal margins ({margins.name}): {picked}s"
                )
            return picked, {
                "method": "conformal_margins",
                "source": "conformal_margins",
                "selected_s": int(picked),
                "margins_path": str(margins),
            }

    if verbose:
        print(f"Decision interval fallback: {DEFAULT_DECISION_INTERVAL_S}s")
    return DEFAULT_DECISION_INTERVAL_S, {
        "method": method,
        "source": "default",
        "selected_s": DEFAULT_DECISION_INTERVAL_S,
    }
