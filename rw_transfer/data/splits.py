"""Chronological adaptation / evaluation splits for transfer experiments."""

from __future__ import annotations

from typing import Tuple

from rw_transfer.data.series import BatteryTimeSeries


def adaptation_and_eval_split(
    series: BatteryTimeSeries,
    eval_tail_frac: float = 0.20,
) -> Tuple[BatteryTimeSeries, BatteryTimeSeries]:
    """
    Fixed held-out **eval tail** (last ``eval_tail_frac`` of timeline).

    Adaptation budgets are taken from the **prefix** before that tail.
    """
    frac = float(eval_tail_frac)
    if series.time_s.size < 2:
        return series, series
    t0, t1 = float(series.time_s[0]), float(series.time_s[-1])
    t_eval = t0 + (1.0 - frac) * (t1 - t0)
    adapt_pool = series.slice_by_time(t_eval)
    eval_series = series.remainder_after(adapt_pool)
    return adapt_pool, eval_series


def prefix_by_fraction(pool: BatteryTimeSeries, frac: float) -> BatteryTimeSeries:
    return pool.slice_by_fraction(frac)


def prefix_by_hours(pool: BatteryTimeSeries, hours: float) -> BatteryTimeSeries:
    if pool.time_s.size == 0:
        return pool
    t_end = float(pool.time_s[0]) + hours * 3600.0
    return pool.slice_by_time(t_end)
