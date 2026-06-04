"""Signal preprocessing before windowing."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from rw_transfer.data.series import BatteryTimeSeries


def smooth_temperature_series(
    series: BatteryTimeSeries,
    window: int = 31,
    polyorder: int = 2,
) -> BatteryTimeSeries:
    """
    Light Savitzky–Golay smoothing on the full timeline (once, not per window).

    Reduces sensor noise while preserving thermal dynamics better than window=149.
    """
    n = series.temperature_c.size
    if window <= 0 or n < 5:
        return series
    wl = int(window)
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, n if n % 2 == 1 else n - 1)
    wl = max(wl, polyorder + 2)
    if wl % 2 == 0:
        wl -= 1
    if wl < 5:
        return series
    temp = savgol_filter(series.temperature_c.astype(np.float64), wl, polyorder).astype(np.float32)
    return BatteryTimeSeries(
        cell_id=series.cell_id,
        time_s=series.time_s,
        relative_time_s=series.relative_time_s,
        voltage_v=series.voltage_v,
        current_a=series.current_a,
        temperature_c=temp,
        age=series.age,
        comment=series.comment,
        step_index=series.step_index,
    )
