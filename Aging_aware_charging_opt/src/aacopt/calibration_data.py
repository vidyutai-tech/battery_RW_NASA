"""Build the degradation-calibration dataset from the raw NASA RW record.

One row per interval between consecutive accepted reference discharges, holding
the stress the cell experienced during that interval and the capacity loss
measured at its end. This is a pure measurement product: no degradation model
appears here.

Modelling choice recorded explicitly (see docs/methodology.md and RULE 8):
cyclic throughput is accumulated from **charge** samples only, binned by the
C-rate of the charge current. The optimization intervenes on the charge
protocol, so attributing cyclic fade to charge throughput keeps calibration and
later profile scoring on the same footing. The (unchanged) random-walk
discharge duty is therefore a fixed co-factor absorbed into the fitted
coefficients; discharge throughput is recorded in its own columns so an
alternative attribution can be fitted from the same file without re-parsing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from aacopt.capacity import (
    OcvCurve,
    ReferenceMeasurement,
    fit_ocv_curve,
    reference_capacity_table,
    validate_ocv_curve,
)
from aacopt.degradation import tent_weights
from aacopt.nasa_data import REF_CHARGE, REF_DISCHARGE, Step, load_steps, summarize_cell


@dataclass
class IntervalRow:
    """Stress accumulated between reference discharge m-1 and m."""

    cell: str
    interval_index: int          # m, so the row's target is y at reference m
    ref_number: int
    t_start_h: float
    t_end_h: float
    duration_h: float
    rest_hours: float
    mean_soc: float
    mean_temperature_c: float          # duration-weighted (calendar term)
    cyclic_mean_temperature_c: float   # charge-throughput-weighted (cyclic term)
    max_temperature_c: float
    dah_charge: List[float] = field(default_factory=list)     # per C-rate bin
    dah_discharge: List[float] = field(default_factory=list)  # per C-rate bin
    dah_charge_total: float = 0.0
    dah_discharge_total: float = 0.0
    q_full_ah: float = 0.0
    y_measured: float = 0.0      # fractional capacity loss at reference m


def _soc_trace(
    steps: Sequence[Step],
    refs: Sequence[ReferenceMeasurement],
    ocv: OcvCurve,
) -> List[np.ndarray]:
    """Coulomb-counted SOC per step, re-anchored at reference events.

    Anchoring bounds integration drift over the ~3500 h record: SOC is reset to
    1.0 after every ``reference charge`` and to the OCV-implied SOC at the end
    of every reference discharge. The counting denominator is the cell's
    *present* capacity, linearly interpolated in absolute time between the
    accepted reference measurements.
    """
    t_ref = np.asarray([r.t_end_h for r in refs], dtype=np.float64)
    q_ref = np.asarray([r.q_full_ah for r in refs], dtype=np.float64)

    def q_now(t_h: np.ndarray) -> np.ndarray:
        return np.interp(t_h, t_ref, q_ref, left=q_ref[0], right=q_ref[-1])

    traces: List[np.ndarray] = []
    soc = 0.5
    for s in steps:
        dt = s.dt_s()
        t_h = s.time_s / 3600.0 if s.time_s.size == s.n else np.full(s.n, t_ref[0])
        cap_as = np.maximum(q_now(t_h), 1e-6) * 3600.0
        # negative current = charge => SOC increases
        d_soc = (-s.current_a * dt) / cap_as
        tr = np.clip(soc + np.cumsum(d_soc), 0.0, 1.0)
        traces.append(tr)

        if s.comment == REF_CHARGE:
            soc = 1.0
        elif s.comment == REF_DISCHARGE:
            soc = float(ocv.soc_from_voltage(float(s.voltage_v[-1])))
        else:
            soc = float(tr[-1])
    return traces


def build_intervals(
    cell: str,
    steps: List[Step],
    *,
    c_rate_bins: Sequence[float],
    q_nominal_ah: float,
    rest_current_threshold_a: float,
) -> Tuple[List[IntervalRow], Dict[str, object]]:
    """Interval rows plus a diagnostics dict for the stage-1 report."""
    ocv = fit_ocv_curve(steps)
    refs, ref_info = reference_capacity_table(steps, ocv)
    if len(refs) < 3:
        raise ValueError(f"{cell}: only {len(refs)} accepted reference discharges")

    q0 = refs[0].q_full_ah
    soc_traces = _soc_trace(steps, refs, ocv)
    bins = np.asarray(c_rate_bins, dtype=np.float64)
    n_bins = bins.size

    rows: List[IntervalRow] = []
    bin_population = np.zeros(n_bins)

    for m in range(1, len(refs)):
        lo = refs[m - 1].step_index + 1   # first step after the previous reference
        hi = refs[m].step_index           # inclusive: the reference itself
        if hi < lo:
            continue

        dah_c = np.zeros(n_bins)
        dah_d = np.zeros(n_bins)
        soc_num = 0.0
        t_num = 0.0
        t_den = 0.0
        rest_s = 0.0
        t_max = -np.inf
        cyc_t_num = 0.0
        cyc_t_den = 0.0

        for k in range(lo, hi + 1):
            s = steps[k]
            dt = s.dt_s()
            i_abs = np.abs(s.current_a)
            active = i_abs >= rest_current_threshold_a
            rest_s += float(np.sum(dt[~active]))
            if s.temperature_c.size:
                t_max = max(t_max, float(s.temperature_c.max()))

            # duration-weighted means so long rests are not under-counted
            w = float(np.sum(dt))
            t_den += w
            if s.temperature_c.size:
                t_num += float(np.sum(s.temperature_c * dt))
            soc_num += float(np.sum(soc_traces[k] * dt))

            if active.any():
                c_rate = i_abs[active] / q_nominal_ah
                ah = i_abs[active] * dt[active] / 3600.0
                # Tent (linear) weights between neighbouring grid nodes: the
                # same continuous assignment the model uses when scoring a
                # charging profile, so calibration and scoring agree exactly.
                w_bins = tent_weights(c_rate, bins)
                charging = s.current_a[active] < 0.0
                dah_c += (w_bins[charging] * ah[charging, None]).sum(axis=0)
                dah_d += (w_bins[~charging] * ah[~charging, None]).sum(axis=0)
                if charging.any() and s.temperature_c.size:
                    t_act = s.temperature_c[active][charging]
                    a_chg = ah[charging]
                    cyc_t_num += float(np.sum(t_act * a_chg))
                    cyc_t_den += float(np.sum(a_chg))

        t_start = float(steps[lo].time_s[0] / 3600.0)
        t_end = float(steps[hi].time_s[-1] / 3600.0)
        duration_h = max(t_end - t_start, t_den / 3600.0)
        bin_population += dah_c

        rows.append(IntervalRow(
            cell=cell,
            interval_index=m,
            ref_number=refs[m].ref_number,
            t_start_h=t_start,
            t_end_h=t_end,
            duration_h=duration_h,
            rest_hours=rest_s / 3600.0,
            mean_soc=(soc_num / t_den) if t_den > 0 else 0.5,
            mean_temperature_c=(t_num / t_den) if t_den > 0 else 25.0,
            cyclic_mean_temperature_c=(
                cyc_t_num / cyc_t_den if cyc_t_den > 0
                else ((t_num / t_den) if t_den > 0 else 25.0)
            ),
            max_temperature_c=float(t_max) if np.isfinite(t_max) else 25.0,
            dah_charge=[float(x) for x in dah_c],
            dah_discharge=[float(x) for x in dah_d],
            dah_charge_total=float(dah_c.sum()),
            dah_discharge_total=float(dah_d.sum()),
            q_full_ah=refs[m].q_full_ah,
            y_measured=float(1.0 - refs[m].q_full_ah / q0),
        ))

    info = {
        "cell": cell,
        "ocv": ocv.to_dict(),
        "ocv_validation": validate_ocv_curve(steps, ocv),
        "reference_capacity": ref_info,
        "q0_full_ah": q0,
        "q_end_full_ah": refs[-1].q_full_ah,
        "y_end_measured": float(1.0 - refs[-1].q_full_ah / q0),
        "n_intervals": len(rows),
        "c_rate_bins": [float(b) for b in bins],
        "charge_ah_per_bin": [float(x) for x in bin_population],
        "charge_ah_total": float(bin_population.sum()),
        "cumulative_charge_ah": float(sum(r.dah_charge_total for r in rows)),
        "cumulative_discharge_ah": float(sum(r.dah_discharge_total for r in rows)),
        "calendar_span_h": float(rows[-1].t_end_h - rows[0].t_start_h) if rows else 0.0,
    }
    return rows, info


def rows_to_dataframe(rows: Sequence[IntervalRow], c_rate_bins: Sequence[float]):
    import pandas as pd

    recs = []
    for r in rows:
        d = asdict(r)
        for b, c in enumerate(c_rate_bins):
            d[f"dah_chg_{_bin_tag(c)}"] = r.dah_charge[b]
            d[f"dah_dis_{_bin_tag(c)}"] = r.dah_discharge[b]
        d.pop("dah_charge")
        d.pop("dah_discharge")
        recs.append(d)
    return pd.DataFrame(recs)


def _bin_tag(c_rate: float) -> str:
    return f"{c_rate:g}C".replace(".", "p")


def load_calibration_dataframe(path: Path, c_rate_bins: Sequence[float]):
    """Read the stage-1 CSV back into rows usable by the fitter."""
    import pandas as pd

    df = pd.read_csv(path)
    tags = [_bin_tag(c) for c in c_rate_bins]
    missing = [f"dah_chg_{t}" for t in tags if f"dah_chg_{t}" not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing throughput columns {missing}")
    return df


def cell_arrays(df, cell: str, c_rate_bins: Sequence[float]) -> Dict[str, np.ndarray]:
    """Per-cell arrays in interval order, as consumed by the degradation model."""
    sub = df[df["cell"] == cell].sort_values("interval_index")
    tags = [_bin_tag(c) for c in c_rate_bins]
    dah = np.stack(
        [sub[f"dah_chg_{t}"].to_numpy(dtype=np.float64) for t in tags], axis=1,
    )
    return {
        "duration_h": sub["duration_h"].to_numpy(dtype=np.float64),
        "mean_soc": sub["mean_soc"].to_numpy(dtype=np.float64),
        "mean_temperature_c": sub["mean_temperature_c"].to_numpy(dtype=np.float64),
        "cyclic_mean_temperature_c": sub["cyclic_mean_temperature_c"].to_numpy(dtype=np.float64),
        "dah": dah,
        "y": sub["y_measured"].to_numpy(dtype=np.float64),
        "cum_ah": np.cumsum(dah.sum(axis=1)),
        "cum_duration_h": np.cumsum(sub["duration_h"].to_numpy(dtype=np.float64)),
        "t_end_h": sub["t_end_h"].to_numpy(dtype=np.float64),
        "ref_number": sub["ref_number"].to_numpy(dtype=np.int64),
        "q_full_ah": sub["q_full_ah"].to_numpy(dtype=np.float64),
    }
