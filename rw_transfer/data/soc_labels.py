"""Coulomb-counting SOC labels for NASA RW (per-step, author-aligned)."""

from __future__ import annotations

import numpy as np

from rw_transfer.constants import NASA_NOMINAL_Q_AS
from rw_transfer.data.series import BatteryTimeSeries


def voltage_to_soc_anchor(v: float, v_min: float = 3.0, v_max: float = 4.2) -> float:
    return float(np.clip((v - v_min) / (v_max - v_min), 0.0, 1.0))


def _dt_seconds(time_s: np.ndarray) -> np.ndarray:
    tt = np.asarray(time_s, dtype=np.float64)
    dt = np.diff(tt, prepend=tt[0] if tt.size else 0.0)
    if dt.size:
        dt[0] = 0.0
    return dt


def coulomb_soc_discharge_segment(
    time_s: np.ndarray,
    current_a: np.ndarray,
) -> np.ndarray:
    """
    Author discharge label within one step:

        SOC(t) = 1 - (∫₀ᵗ |I| dt) / Q_seg
    """
    tt = np.asarray(time_s, dtype=np.float64)
    cur = np.abs(np.asarray(current_a, dtype=np.float64))
    if tt.size == 0:
        return np.array([], dtype=np.float32)
    dt = _dt_seconds(tt)
    delivered = np.cumsum(cur * dt)
    q_seg = float(delivered[-1])
    if q_seg < 1e-9:
        return np.ones(len(delivered), dtype=np.float32)
    soc = 1.0 - delivered / q_seg
    return np.clip(soc, 0.0, 1.0).astype(np.float32)


def coulomb_soc_charge_segment(
    time_s: np.ndarray,
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    *,
    q_rated_as: float = NASA_NOMINAL_Q_AS,
    q_norm: str = "per_file",
    min_segment_charge_as: float = 80.0,
) -> np.ndarray:
    """
    Charge-side Coulomb counting **within one step** (NASA RW: I < 0 is charge).

    ``per_file``: SOC(t) = soc₀ + (∫ I₊ dt / Q_seg) · (1 − soc₀) when Q_seg is large enough.
    ``global``: SOC(t) = soc₀ + ∫ I₊ dt / Q_rated.
    """
    if q_norm not in ("global", "per_file"):
        raise ValueError("q_norm must be 'global' or 'per_file'")

    tt = np.asarray(time_s, dtype=np.float64)
    i = np.asarray(current_a, dtype=np.float64)
    v = np.asarray(voltage_v, dtype=np.float64)
    if tt.size == 0:
        return np.array([], dtype=np.float32)

    # NASA RW: negative current = charge
    i_charge = np.maximum(-i, 0.0)
    dt = _dt_seconds(tt)
    delivered = np.cumsum(i_charge * dt)
    soc0 = voltage_to_soc_anchor(float(v[0]))
    q_seg = float(delivered[-1])

    if q_norm == "per_file" and q_seg >= min_segment_charge_as:
        soc = soc0 + delivered * ((1.0 - soc0) / q_seg)
    else:
        qr = float(q_rated_as) if q_rated_as > 1.0 else NASA_NOMINAL_Q_AS
        soc = soc0 + delivered / qr
    return np.clip(soc, 0.0, 1.0).astype(np.float32)


def _step_comment(comment_arr: np.ndarray) -> str:
    c = comment_arr.flat[0]
    return str(c).strip().lower()


def coulomb_soc_stitched_operational(
    series: BatteryTimeSeries,
    *,
    q_rated_as: float = NASA_NOMINAL_Q_AS,
    q_norm: str = "per_file",
    min_segment_charge_as: float = 80.0,
) -> np.ndarray:
    """
    Build SOC labels on a stitched RW operational timeline.

    Each ``step_index`` block is labeled independently (charge / discharge / rest),
    matching the author notebook — **not** one global integral over the full file.
    """
    n = series.time_s.size
    soc = np.full(n, np.nan, dtype=np.float32)
    last_soc = 0.5

    for sid in sorted(np.unique(series.step_index)):
        mask = series.step_index == sid
        if not np.any(mask):
            continue
        cmt = _step_comment(series.comment[mask])
        t = series.time_s[mask]
        i = series.current_a[mask]
        v = series.voltage_v[mask]

        if "discharge" in cmt and "random walk" in cmt:
            seg = coulomb_soc_discharge_segment(t, i)
        elif "charge" in cmt and "random walk" in cmt:
            seg = coulomb_soc_charge_segment(
                t, i, v,
                q_rated_as=q_rated_as,
                q_norm=q_norm,
                min_segment_charge_as=min_segment_charge_as,
            )
        elif cmt == "reference discharge":
            seg = coulomb_soc_discharge_segment(t, i)
        elif cmt == "reference charge":
            seg = coulomb_soc_charge_segment(
                t, i, v,
                q_rated_as=q_rated_as,
                q_norm="per_file",
                min_segment_charge_as=min_segment_charge_as,
            )
        elif "rest" in cmt:
            seg = np.full(int(mask.sum()), last_soc, dtype=np.float32)
        else:
            continue

        soc[mask] = seg
        last_soc = float(seg[-1])

    valid = np.isfinite(soc)
    if not np.all(valid):
        idx = np.where(valid)[0]
        if idx.size == 0:
            soc[:] = 0.5
        else:
            first = idx[0]
            soc[:first] = soc[first]
            for j in range(first + 1, n):
                if not np.isfinite(soc[j]):
                    soc[j] = soc[j - 1]
    return np.clip(soc, 0.0, 1.0).astype(np.float32)


def coulomb_soc_from_voltage_anchor(
    time_s: np.ndarray,
    current_a: np.ndarray,
    voltage_v: np.ndarray,
    q_rated_as: float = NASA_NOMINAL_Q_AS,
    q_norm: str = "global",
    min_segment_charge_as: float = 80.0,
) -> np.ndarray:
    """
    Legacy single-segment charge-only integral (avoid on long stitched RW series).

    Prefer :func:`coulomb_soc_stitched_operational` for RW operational data.
    """
    return coulomb_soc_charge_segment(
        time_s,
        current_a,
        voltage_v,
        q_rated_as=q_rated_as,
        q_norm=q_norm,
        min_segment_charge_as=min_segment_charge_as,
    )


def coulomb_soc_rw_series(
    time_s: np.ndarray,
    current_a: np.ndarray,
    q_rated_as: float = NASA_NOMINAL_Q_AS,
    q_norm: str = "global",
    min_segment_charge_as: float = 80.0,
) -> np.ndarray:
    """Deprecated mixed global integral — kept for compatibility."""
    tt = np.asarray(time_s, dtype=np.float64)
    i = np.asarray(current_a, dtype=np.float64)
    if tt.size == 0:
        return np.array([], dtype=np.float32)
    dt = _dt_seconds(tt)
    discharged = np.cumsum(np.maximum(i, 0.0) * dt)
    charged = np.cumsum(np.maximum(-i, 0.0) * dt)
    net = charged - discharged
    if q_norm == "per_file":
        q_seg = float(np.abs(net[-1])) if net.size else 0.0
        if q_seg >= min_segment_charge_as:
            soc = 0.5 + net / (2.0 * q_seg)
            return np.clip(soc.astype(np.float32), 0.0, 1.0)
    qr = float(q_rated_as) if q_rated_as > 1.0 else NASA_NOMINAL_Q_AS
    soc = 0.5 + net / qr
    return np.clip(soc.astype(np.float32), 0.0, 1.0)


def discharge_soc_series(
    time_s: np.ndarray,
    current_a: np.ndarray,
    q_rated_as: float = NASA_NOMINAL_Q_AS,
) -> np.ndarray:
    """Global-rated discharge ramp (prefer :func:`coulomb_soc_discharge_segment`)."""
    tt = np.asarray(time_s, dtype=np.float64)
    cu = np.abs(np.asarray(current_a, dtype=np.float64))
    dt = _dt_seconds(tt)
    discharged = np.cumsum(cu * dt)
    soc = 1.0 - discharged / float(q_rated_as)
    return np.clip(soc.astype(np.float32), 0.0, 1.0)
