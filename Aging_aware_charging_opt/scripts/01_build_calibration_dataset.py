#!/usr/bin/env python
"""STAGE 1 — build the degradation-calibration dataset from raw NASA RW data.

Reads only ``NASA_RW/.../RW*.mat``. Writes
``results/01_calibration_dataset/``:

    calibration_intervals.csv   one row per inter-reference interval (all cells)
    capacity_fade_measured.csv  measured reference-discharge capacity per cell
    dataset_report.json         diagnostics, provenance, cross-check
    ocv_curves.npz              fitted OCV-SOC curves per cell

Gate: at least ``--min-refs`` accepted reference discharges per cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.calibration_data import build_intervals, rows_to_dataframe
from aacopt.capacity import fit_ocv_curve, reference_capacity_table
from aacopt.config import Paths, load_config, provenance, stage_dir, write_json
from aacopt.nasa_data import load_steps, summarize_cell

STAGE = "01_calibration_dataset"


def crosscheck(df_fade: pd.DataFrame, path: Path | None) -> dict:
    """Compare against the previous project's measured-fade CSV.

    Reported only. An independent reimplementation agreeing with the earlier
    extraction is evidence the parsing is right; it is not a pipeline input and
    a disagreement does not change anything computed here.
    """
    if path is None or not Path(path).is_file():
        return {"available": False}
    old = pd.read_csv(path)
    out = {"available": True, "source": str(path), "per_cell": {}}
    for cell, sub in df_fade.groupby("cell"):
        o = old[old["cell"] == cell]
        if o.empty:
            out["per_cell"][cell] = {"matched": 0}
            continue
        n = min(len(sub), len(o))
        new_q = sub.sort_values("ref_number")["q_full_ah"].to_numpy()[:n]
        old_q = o.sort_values("ref_number")["capacity_Ah"].to_numpy()[:n]
        d = new_q - old_q
        out["per_cell"][cell] = {
            "n_compared": int(n),
            "n_new": int(len(sub)),
            "n_old": int(len(o)),
            "capacity_ah_mean_abs_diff": float(np.mean(np.abs(d))),
            "capacity_ah_max_abs_diff": float(np.max(np.abs(d))),
            "correlation": float(np.corrcoef(new_q, old_q)[0, 1]) if n > 2 else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--min-refs", type=int, default=70)
    args = ap.parse_args()

    paths = Paths.load()
    deg_cfg = load_config("degradation")
    bins = list(deg_cfg["c_rate_bins"])
    q_nom = float(deg_cfg["physical_constants"]["q_nominal_ah"])
    rest_thr = float(deg_cfg["rest_current_threshold_a"])
    cells = [c.upper() for c in (args.cells or paths.cells)]

    out_dir = stage_dir(STAGE)
    all_rows = []
    fade_rows = []
    per_cell = {}
    ocv_arrays = {}
    inputs = []

    for cell in cells:
        mat = paths.mat_path(cell)
        inputs.append(mat)
        print(f"[{cell}] parsing {mat.name} …", flush=True)
        steps = load_steps(mat)

        rows, info = build_intervals(
            cell, steps,
            c_rate_bins=bins,
            q_nominal_ah=q_nom,
            rest_current_threshold_a=rest_thr,
        )
        info["dataset_summary"] = summarize_cell(steps, q_nominal_ah=q_nom)
        per_cell[cell] = info
        all_rows.extend(rows)

        ocv = fit_ocv_curve(steps)
        ocv_arrays[f"{cell}_soc"] = ocv.soc_grid
        ocv_arrays[f"{cell}_ocv"] = ocv.ocv_grid

        refs, _ = reference_capacity_table(steps, ocv)
        q0 = refs[0].q_full_ah
        for r in refs:
            fade_rows.append({
                "cell": cell,
                "ref_number": r.ref_number,
                "t_end_h": r.t_end_h,
                "q_measured_ah": r.q_measured_ah,
                "q_full_ah": r.q_full_ah,
                "remaining_pct": 100.0 * r.q_full_ah / q0,
                "y_measured": 1.0 - r.q_full_ah / q0,
                "v_start": r.v_start,
                "v_end": r.v_end,
                "soc_window": r.soc_window,
                "mean_temperature_c": r.mean_temperature_c,
            })

        n_acc = info["reference_capacity"]["n_accepted"]
        print(
            f"[{cell}] refs accepted {n_acc}/{info['reference_capacity']['n_reference_steps_seen']}"
            f"  intervals {info['n_intervals']}"
            f"  Q0 {info['q0_full_ah']:.3f} Ah -> {info['q_end_full_ah']:.3f} Ah"
            f"  (y_end {info['y_end_measured']:.3f})"
        )
        print(
            f"[{cell}] charge Ah per C-rate bin "
            + "  ".join(
                f"{b:g}C={a:.1f}" for b, a in zip(bins, info["charge_ah_per_bin"])
            )
            + f"   calendar span {info['calendar_span_h']:.0f} h"
        )

    df = rows_to_dataframe(all_rows, bins)
    df_fade = pd.DataFrame(fade_rows)
    df.to_csv(out_dir / "calibration_intervals.csv", index=False)
    df_fade.to_csv(out_dir / "capacity_fade_measured.csv", index=False)
    np.savez(out_dir / "ocv_curves.npz", **ocv_arrays)

    report = {
        "provenance": provenance(STAGE, configs=["paths", "degradation"], inputs=inputs),
        "cells": cells,
        "c_rate_bins": bins,
        "q_nominal_ah": q_nom,
        "rest_current_threshold_a": rest_thr,
        "n_intervals_total": int(len(df)),
        "throughput_attribution": "charge_only",
        "throughput_attribution_note": (
            "Cyclic throughput is charge-only, binned by charge C-rate. The "
            "unchanged random-walk discharge duty is a fixed co-factor absorbed "
            "into the fitted coefficients; discharge Ah is recorded separately."
        ),
        "per_cell": per_cell,
        "crosscheck_vs_previous_project": crosscheck(
            df_fade, paths.crosscheck_capacity_fade_csv,
        ),
    }
    write_json(out_dir / "dataset_report.json", report)

    print(f"\nWrote → {out_dir}")
    cc = report["crosscheck_vs_previous_project"]
    if cc.get("available"):
        print("Cross-check vs previous extraction (reported only, not an input):")
        for cell, v in cc["per_cell"].items():
            if v.get("n_compared"):
                print(
                    f"  {cell}: n={v['n_compared']}  mean|ΔQ|={v['capacity_ah_mean_abs_diff']:.4f} Ah"
                    f"  max|ΔQ|={v['capacity_ah_max_abs_diff']:.4f} Ah"
                    f"  r={v['correlation']:.5f}"
                )

    failures = []
    for cell, info in per_cell.items():
        n = info["reference_capacity"]["n_accepted"]
        if n < args.min_refs:
            failures.append(f"{cell}: only {n} accepted references (< {args.min_refs})")
        pop = np.asarray(info["charge_ah_per_bin"])
        if (pop <= 0).any():
            empty = [f"{b:g}C" for b, a in zip(bins, pop) if a <= 0]
            failures.append(f"{cell}: empty C-rate bin(s) {empty}")

    if failures:
        print("\nGATE FAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nGATE PASS: all cells have enough references and populated C-rate bins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
