#!/usr/bin/env python
"""STAGE 3 — calibrate the hybrid degradation coefficients on measured RW fade.

Reads ``results/01_calibration_dataset/calibration_intervals.csv`` and
``configs/degradation.yaml``. Writes ``results/03_degradation_fit/`` and the
single downstream source of truth ``configs/degradation_fitted.yaml``.

Structure of this stage, and why (see docs/methodology.md for the full
argument):

1. The CALENDAR branch is literature-anchored, not fitted. NASA RW has no
   storage arm — rest is ~9 % of each record and sits in a narrow low-SOC
   band, and cumulative time is collinear with cumulative throughput at
   r ~ 0.99 — so its calendar coefficients are not estimable. ``A_cal`` is
   solved from interpretable anchors declared in ``configs/degradation.yaml``.
2. The CYCLIC branch IS fitted, reduced to the parameters the data supports.
3. Ablations that instead fit the calendar branch are run and reported, as the
   standing evidence for step 1 rather than an assertion (RULE 8).
4. The fit is repeated restricted to the operating window (retention >= 80 %),
   because a single power law cannot span the degradation knee and every
   downstream claim lives inside that window. The deployed model is chosen by
   the rule pre-declared in the config.
5. The anchored calendar level is swept over its declared multipliers to show
   what the calibration can and cannot distinguish.

Nothing from any optimization stage is read here (RULE 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.calibration import (
    CellParameters,
    calendar_scale_sweep,
    calendar_share,
    datasets_from_dataframe,
    fit,
    goodness_of_fit,
    layouts_from_config,
    params_for,
    predict_cell_split,
    restrict_to_window,
)
from aacopt.config import (
    Paths,
    config_path,
    load_config,
    provenance,
    stage_dir,
    write_json,
)
from aacopt.degradation import calendar_from_anchors

STAGE = "03_degradation_fit"
DEPLOYED_VARIANT = "primary"


def _report(res, gof, layout, bins, cells, indent="  ") -> None:
    p = params_for(res.params, cells[0])
    se = res.standard_errors
    names = layout.names

    def fmt(name: str) -> str:
        if se is None or name not in names:
            return ""
        i = names.index(name)
        return f" ± {se[i]:.4g}" if np.isfinite(se[i]) else " ± n/a"

    print(f"{indent}converged={res.success} ({res.message.strip()})")
    print(
        f"{indent}multi-start: {res.n_starts} starts, best cost={min(res.start_costs):.6e}, "
        f"{res.to_dict()['n_starts_within_1pct_of_best']} within 1 % of best"
    )
    print(
        f"{indent}pooled     R²={gof['pooled']['r2']:.4f}  "
        f"RMSE={gof['pooled']['rmse_pct_capacity']:.3f} %-cap  "
        f"MAE={gof['pooled']['mae_pct_capacity']:.3f} %-cap  "
        f"bias={gof['pooled']['bias_pct_capacity']:+.3f} %-cap"
    )
    for cell in cells:
        m = gof["per_cell"][cell]
        print(
            f"{indent}  {cell:5s} n={m['n']:3d}  R²={m['r2']:.4f}  "
            f"RMSE={m['rmse_pct_capacity']:5.3f}  bias={m['bias_pct_capacity']:+6.3f} %-cap"
        )

    R = p.R
    print(f"\n{indent}CALENDAR branch — literature-anchored, NOT fitted here:")
    print(f"{indent}  A_cal  = {p.A_cal:.6e}  (solved from the declared anchors)")
    print(f"{indent}  B_cal  = {p.B_cal:+.4f}   (= ln of the SOC acceleration ratio)")
    print(f"{indent}  C_cal  = {p.C_cal:+.1f} J/mol per unit SOC")
    print(f"{indent}  Ea_cal = {p.Ea_cal:.1f} J/mol")
    print(f"{indent}  z_cal  = {p.z_cal:.4f}")

    print(
        f"\n{indent}CYCLIC branch — FITTED (± Jacobian standard error), "
        f"reference T = {layout.t_ref_k:.2f} K:"
    )
    if layout.c_rate_power_law:
        print(
            f"{indent}  p_c_rate = {res.x[layout.names.index('p_c_rate')]:.4f}"
            f"{fmt('p_c_rate')}   (k_cyc_ref ∝ C^p)"
        )
    if layout.scale_cells:
        print(f"{indent}  cell-specific scale k_cyc_ref at {layout.c_rate_ref:g}C:")
        for c in layout.scale_cells:
            pc = res.params[c]
            b_ref = int(np.argmin(np.abs(np.asarray(bins) - layout.c_rate_ref)))
            k = pc.B_cyc[b_ref] * np.exp(-pc.Ea_cyc[b_ref] / (R * layout.t_ref_k))
            print(
                f"{indent}    {c:5s} = {k:.6e} /Ah^z  (log10 = {np.log10(k):+.4f})"
                f"{fmt(f'log10_k_cyc_ref[{layout.c_rate_ref:g}C|{c}]')}"
            )
        ks = [
            res.params[c].B_cyc[
                int(np.argmin(np.abs(np.asarray(bins) - layout.c_rate_ref)))
            ] for c in layout.scale_cells
        ]
        print(
            f"{indent}    spread max/min = {max(ks) / min(ks):.2f}x "
            f"— measured cell-to-cell damage-rate variability"
        )
    else:
        for b, c in enumerate(bins):
            tag = (
                f"log10_k_cyc_ref[{layout.c_rate_ref:g}C]" if layout.c_rate_power_law
                else (f"log10_k_cyc_ref_base[{c:g}C]" if (layout.monotone_B_cyc and b == 0)
                      else (f"dlog10_k_cyc_ref[{c:g}C]" if layout.monotone_B_cyc
                            else f"log10_k_cyc_ref[{c:g}C]"))
            )
            k_ref = p.B_cyc[b] * np.exp(-p.Ea_cyc[b] / (R * layout.t_ref_k))
            anchored = layout.c_rate_power_law and abs(c - layout.c_rate_ref) < 1e-9
            print(
                f"{indent}  {c:g}C: k_cyc_ref={k_ref:.6e} /Ah^z  "
                f"(log10 = {np.log10(k_ref):+.4f})"
                + (fmt(tag) if (anchored or not layout.c_rate_power_law) else "  [from C^p]")
            )
    if layout.shared_ea_cyc:
        print(f"{indent}  Ea_cyc = {p.Ea_cyc[0]:.1f} J/mol (shared){fmt('Ea_cyc[shared]')}")
    else:
        for b, c in enumerate(bins):
            print(f"{indent}  Ea_cyc[{c:g}C] = {p.Ea_cyc[b]:.1f} J/mol{fmt(f'Ea_cyc[{c:g}C]')}")
    if layout.shared_z_cyc:
        print(f"{indent}  z_cyc  = {p.z_cyc[0]:.4f} (shared){fmt('z_cyc[shared]')}")
    else:
        for b, c in enumerate(bins):
            print(f"{indent}  z_cyc[{c:g}C]  = {p.z_cyc[b]:.4f}{fmt(f'z_cyc[{c:g}C]')}")
    if not layout.scale_cells:
        print(f"{indent}  recovered B_cyc = "
              + ", ".join(f"{v:.4e}" for v in p.B_cyc))

    if res.flags:
        print(f"\n{indent}Identifiability flags (reported, not suppressed):")
        for f in res.flags:
            print(f"{indent}  ! {f}")
    else:
        print(f"\n{indent}Identifiability: no flags raised.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-starts", type=int, default=None)
    ap.add_argument("--skip-sweep", action="store_true")
    args = ap.parse_args()

    paths = Paths.load()
    cfg = load_config("degradation")
    cal_cfg = cfg["calibration"]
    bins = list(cfg["c_rate_bins"])
    q_nom = float(cfg["physical_constants"]["q_nominal_ah"])
    R = float(cfg["physical_constants"]["R_universal"])
    window = float(cal_cfg["operating_window_max_y"])
    cells = [c.upper() for c in paths.cells]

    src = stage_dir("01_calibration_dataset", create=False) / "calibration_intervals.csv"
    if not src.is_file():
        print(f"missing {src} — run scripts/01_build_calibration_dataset.py first")
        return 1
    df = pd.read_csv(src)
    datasets = datasets_from_dataframe(df, cells, bins)
    windowed = restrict_to_window(datasets, window)

    print(f"Calibration data: {sum(d.n for d in datasets)} intervals across {len(cells)} cells")
    for d, w in zip(datasets, windowed):
        print(
            f"  {d.cell}: n={d.n:3d}  y_end={d.y[-1]:.4f}  Ah={d.cum_ah[-1]:.0f}  "
            f"t={d.duration_h.sum():.0f} h  T_cal={d.mean_temperature_c.mean():.1f} °C  "
            f"T_cyc={d.cyclic_mean_temperature_c.mean():.1f} °C   "
            f"| operating window (y<={window:g}): n={w.n}"
        )

    cal_branch = cfg["calendar_branch"]
    anchors = cal_branch["anchors"]
    cal_coef = calendar_from_anchors(anchors, R=R)
    print("\n" + "=" * 88)
    print("A. Calendar branch — LITERATURE-ANCHORED (not calibrated on NASA RW)")
    print("=" * 88)
    print(f"  status: {cal_branch['status']}")
    print(
        f"  anchor: {100 * float(anchors['fade_fraction_at_anchor']):.2f} % capacity loss "
        f"after {float(anchors['anchor_time_h']) / 24:.0f} d at "
        f"{float(anchors['anchor_temperature_k']) - 273.15:.0f} °C, "
        f"SOC {float(anchors['anchor_soc']):.2f}"
    )
    print(
        f"  shape:  z_cal={cal_coef['z_cal']:.2f}, Ea_cal={cal_coef['Ea_cal']:.0f} J/mol, "
        f"SOC acceleration ratio={float(anchors['soc_acceleration_ratio']):.1f}x "
        f"(=> B_cal={cal_coef['B_cal']:.4f}), C_cal={cal_coef['C_cal']:.0f}"
    )
    print(f"  solved: A_cal={cal_coef['A_cal']:.6e}")

    variants = layouts_from_config(cfg, R=R, cells=cells)
    print("\n" + "=" * 88)
    print("B. Parameterization comparison — fitted on the FULL record")
    print("=" * 88)
    full_fits = {}
    for name, layout in variants.items():
        res = fit(
            datasets, layout=layout, cfg=cfg, q_nominal_ah=q_nom, R=R,
            provenance=f"full_record:{name}", n_starts=args.n_starts, verbose=False,
        )
        gof = goodness_of_fit(res.params, datasets, window_max_y=window)
        full_fits[name] = (res, gof)
        n_bound = sum(1 for f in res.flags if "bound" in f)
        w = gof["pooled_window"]
        print(
            f"  {name:18s} {layout.label}\n"
            f"      full: R²={gof['pooled']['r2']:.4f} RMSE={gof['pooled']['rmse_pct_capacity']:5.3f}"
            f" | window: R²={w['r2']:+.4f} RMSE={w['rmse_pct_capacity']:5.3f}"
            f" | at-bound={n_bound}/{layout.size} | cal.share={calendar_share(res.params, datasets):.3f}"
        )
    print(
        "\n  Note: the two 'fit_calendar*' rows are the documented evidence for\n"
        "  anchoring the calendar branch — compare their at-bound counts and\n"
        "  flag lists in degradation_fit.json against the deployed model's."
    )

    print("\n" + "=" * 88)
    print(f"C. Deployed parameterization '{DEPLOYED_VARIANT}' — fitted on the OPERATING WINDOW")
    print("=" * 88)
    layout = variants[DEPLOYED_VARIANT]
    win_res = fit(
        windowed, layout=layout, cfg=cfg, q_nominal_ah=q_nom, R=R,
        provenance=f"operating_window:{DEPLOYED_VARIANT}",
        n_starts=args.n_starts, verbose=False,
    )
    win_gof_in = goodness_of_fit(win_res.params, windowed)
    win_gof_full = goodness_of_fit(win_res.params, datasets, window_max_y=window)
    print(
        f"  window-fitted model: in-window R²={win_gof_in['pooled']['r2']:.4f}  "
        f"RMSE={win_gof_in['pooled']['rmse_pct_capacity']:.3f} %-cap  "
        f"(extrapolated to full record: R²={win_gof_full['pooled']['r2']:.4f})"
    )

    full_res, full_gof = full_fits[DEPLOYED_VARIANT]
    rmse_full_in_window = full_gof["pooled_window"]["rmse_pct_capacity"]
    rmse_win_in_window = win_gof_in["pooled"]["rmse_pct_capacity"]
    print("\n" + "=" * 88)
    print("D. Deployment decision — rule pre-declared in configs/degradation.yaml:")
    print(f"   '{cal_cfg['deploy_rule']}'")
    print("=" * 88)
    print(f"  full-record fit,       RMSE inside window = {rmse_full_in_window:.4f} %-cap")
    print(f"  window-restricted fit, RMSE inside window = {rmse_win_in_window:.4f} %-cap")
    if rmse_win_in_window <= rmse_full_in_window:
        deployed_key = "operating_window"
        res, fit_datasets = win_res, windowed
    else:
        deployed_key = "full_record"
        res, fit_datasets = full_res, datasets
    gof = goodness_of_fit(res.params, fit_datasets, window_max_y=window)
    print(f"  → DEPLOYED: {deployed_key} fit of '{DEPLOYED_VARIANT}'")

    print("\n" + "=" * 88)
    print(f"E. Deployed model: {layout.label}  [{deployed_key}]")
    print("=" * 88)
    _report(res, gof, layout, bins, cells)

    sweep = None
    if not args.skip_sweep:
        print("\n" + "=" * 88)
        print("F. Calendar-anchor sensitivity — cyclic branch refitted per multiplier")
        print("=" * 88)
        sweep = calendar_scale_sweep(
            fit_datasets, cfg=cfg, q_nominal_ah=q_nom, R=R,
            variant=DEPLOYED_VARIANT, n_starts=args.n_starts,
        )
        print(f"  {'scale':>6s}  {'cal.share':>9s}  {'R²':>8s}  {'RMSE':>7s}  {'bias':>7s}")
        for r in sweep["rows"]:
            print(
                f"  {r['calendar_scale']:6.2f}  {r['calendar_share']:9.4f}  "
                f"{r['r2']:8.4f}  {r['rmse_pct_capacity']:7.3f}  {r['bias_pct_capacity']:+7.3f}"
            )
        print(f"  R² spread across the sweep = {sweep['r2_spread']:.4f}")
        if sweep["r2_spread"] < 0.01:
            print(
                "  → calibration quality is flat: the measured fade does NOT\n"
                "    distinguish these calendar levels, the cyclic branch absorbs\n"
                "    the difference. Carried forward as a declared assumption."
            )

    # per-interval predictions for the fit figures and validate_experiment
    pred_rows = []
    for d in datasets:
        split = predict_cell_split(params_for(res.params, d.cell), d)
        for k in range(d.n):
            pred_rows.append({
                "cell": d.cell,
                "interval_index": k + 1,
                "t_end_h": d.t_end_h[k],
                "cum_duration_h": d.cum_duration_h[k],
                "cum_ah": d.cum_ah[k],
                "mean_temperature_c": d.mean_temperature_c[k],
                "mean_soc": d.mean_soc[k],
                "y_measured": d.y[k],
                "y_predicted": split["q_total"][k],
                "y_pred_calendar": split["q_calendar"][k],
                "y_pred_cyclic": split["q_cyclic"][k],
                "residual": split["q_total"][k] - d.y[k],
                "in_operating_window": bool(d.y[k] <= window),
            })
    out_dir = stage_dir(STAGE)
    pd.DataFrame(pred_rows).to_csv(out_dir / "fit_predictions.csv", index=False)

    payload = {
        "provenance": provenance(STAGE, configs=["paths", "degradation"], inputs=[src]),
        "cells": cells,
        "c_rate_bins": bins,
        "n_intervals_full": int(sum(d.n for d in datasets)),
        "n_intervals_window": int(sum(d.n for d in windowed)),
        "operating_window_max_y": window,
        "deploy_rule": cal_cfg["deploy_rule"],
        "deployed_variant": DEPLOYED_VARIANT,
        "deployed_fit_scope": deployed_key,
        "calendar_branch": {
            "mode": cal_branch["mode"],
            "status": cal_branch["status"],
            "anchors": dict(anchors),
            "derived_coefficients": cal_coef,
            "references": list(cal_branch["references"]),
            "why_not_fitted": (
                "NASA RW has no calendar-storage arm: rest is 8.8-9.1 % of each "
                "record, the longest rest is ~12 h, mean SOC during the longest "
                "rests is 0.12-0.21, and cumulative elapsed time is collinear "
                "with cumulative throughput at r = 0.988-0.991 per cell. The "
                "calendar coefficients and the calendar/cyclic apportionment are "
                "therefore not estimable from this dataset. See the "
                "'fit_calendar' and 'fit_calendar_full' entries under "
                "full_record_variants for what happens when they are fitted "
                "anyway."
            ),
        },
        "full_record_variants": {
            name: {**r.to_dict(datasets), "variant": name, "scope": "full_record"}
            for name, (r, _) in full_fits.items()
        },
        "operating_window_fit": {
            **win_res.to_dict(windowed), "variant": DEPLOYED_VARIANT,
            "scope": "operating_window",
            "extrapolated_to_full_record": win_gof_full,
        },
        "deployment_decision": {
            "rule": cal_cfg["deploy_rule"],
            "rmse_in_window_full_record_fit": rmse_full_in_window,
            "rmse_in_window_window_fit": rmse_win_in_window,
            "selected": deployed_key,
        },
        "calendar_scale_sweep": sweep,
        "measured_fade_shape": {
            "note": (
                "Measured fade is ACCELERATING in throughput: pooled log-log "
                "slope 1.11-1.21, rising to 1.7-2.3 over the final third of "
                "each record. These cells are aged to 46-62 % capacity loss, "
                "i.e. past the degradation knee. A sub-linear cyclic power law "
                "(z < 1) is therefore refuted by this dataset and the z bounds "
                "were widened to [0.2, 3.0] to avoid imposing a prior the data "
                "contradicts. The deployed model is nonetheless restricted to "
                "the pre-knee operating window."
            ),
        },
        "notes": [
            "Target is measured fractional capacity loss y = 1 - Q_k/Q_0 from "
            "OCV-corrected reference discharges.",
            "Cyclic throughput is charge-only, split between neighbouring "
            "C-rate grid nodes with linear (tent) weights — identical to how a "
            "charging profile is later scored.",
            "Accumulation uses the exact equivalent-exposure form "
            "Q = (sum_m k_m**(1/z) * delta_m) ** z, so the power laws are "
            "advanced correctly under time-varying SOC/T/C-rate and the result "
            "is independent of the order stresses are applied in.",
            "Cyclic prefactors are fitted as rate coefficients at T_ref and "
            "B_cyc recovered analytically. Fitting raw Arrhenius prefactors "
            "puts the estimated parameter at the unreachable T -> infinity "
            "intercept and produced |corr(prefactor, Ea)| = 0.9923 with the "
            "prefactor pinned at its bound. Change of variables only.",
            "Ea_cyc and z_cyc are shared across C-rate bins and k_cyc_ref(C) is "
            "constrained non-decreasing in C: the bins' cumulative throughputs "
            "are collinear (r = 0.87-0.98), which without these restrictions "
            "admits degenerate solutions that switch a middle bin off.",
            "Acceptance of the model rests on the out-of-sample metrics in "
            "stage 4, not on individual coefficient values.",
        ],
    }
    write_json(out_dir / "degradation_fit.json", payload)

    fitted_yaml = {
        "_comment": (
            "GENERATED by scripts/03_fit_degradation.py — do not edit by hand. "
            "Single source of truth for the degradation coefficients used by "
            "every downstream stage. The calendar coefficients are "
            "LITERATURE-ANCHORED (see calendar_branch); only the cyclic "
            "coefficients are calibrated on NASA RW."
        ),
        "generated_utc": payload["provenance"]["created_utc"],
        "source_dataset": str(src),
        "deployed_fit": f"{deployed_key}:{DEPLOYED_VARIANT}",
        "deployed": res.params.to_dict(),
        "layout": layout.to_dict(),
        "calendar_branch": payload["calendar_branch"],
        "operating_window_max_y": window,
        "fitted_cyclic_at_reference_temperature": {
            "T_ref_K": layout.t_ref_k,
            "k_cyc_ref_per_cell": {
                cell: {
                    f"{c:g}C": float(
                        params_for(res.params, cell).B_cyc[b]
                        * np.exp(
                            -params_for(res.params, cell).Ea_cyc[b]
                            / (params_for(res.params, cell).R * layout.t_ref_k)
                        )
                    )
                    for b, c in enumerate(bins)
                }
                for cell in (layout.scale_cells or (cells[0],))
            },
            "names": layout.names,
            "values": [float(v) for v in res.x],
            "standard_errors": (
                None if res.standard_errors is None
                else [float(v) for v in res.standard_errors]
            ),
        },
        "fit_summary": {
            "cost": res.cost,
            "n_obs": res.n_obs,
            "n_params": res.n_params,
            "converged": res.success,
            "pooled_r2": gof["pooled"]["r2"],
            "pooled_rmse_pct_capacity": gof["pooled"]["rmse_pct_capacity"],
            "pooled_mae_pct_capacity": gof["pooled"]["mae_pct_capacity"],
            "pooled_bias_pct_capacity": gof["pooled"]["bias_pct_capacity"],
            "calendar_share_end_of_fit_range": calendar_share(res.params, fit_datasets),
        },
        "identifiability_flags": res.flags,
        "calendar_scale_sensitivity_multipliers": list(
            cal_branch["sensitivity_multipliers"]
        ),
        "ablation_parameters": {
            f"full_record:{name}": r.params.to_dict()
            for name, (r, _) in full_fits.items()
        },
    }
    dst = config_path("degradation_fitted")
    dst.write_text(
        yaml.safe_dump(fitted_yaml, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )

    print(f"\nWrote → {out_dir}")
    print(f"Wrote → {dst}")
    if not res.success and res.status <= 0:
        print("GATE FAIL: optimizer did not converge")
        return 1
    print("GATE PASS: deployed fit converged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
