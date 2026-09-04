#!/usr/bin/env python
"""STAGE 4 — INDEPENDENT validation of the degradation model (hard gate).

Leave-one-cell-out: for each cell, the coefficients are re-fitted on the OTHER
three cells only and then used to predict the held-out cell. The held-out cell
is never seen by the optimizer that produced the coefficients scoring it, so
these are genuinely out-of-sample metrics (RULE 5 — the validation data is
never tuned on).

Reads ``results/01_calibration_dataset/calibration_intervals.csv`` and
``configs/degradation.yaml``. Reads NOTHING from stage 3 — the folds are fitted
from scratch here, so a leaked pooled fit cannot flatter the result. Writes
``results/04_degradation_validation/``.

This stage is a GATE. If the model does not generalize, the exit code is
non-zero and the pipeline stops for diagnosis rather than proceeding to
optimization.

Nothing from any optimization stage is read here (RULE 6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.calibration import (
    datasets_from_dataframe,
    fit,
    goodness_of_fit,
    layouts_from_config,
    predict_cell_split,
    restrict_to_window,
)
from aacopt.config import (
    Paths,
    load_config,
    provenance,
    stage_dir,
    write_json,
)

STAGE = "04_degradation_validation"
DEPLOYED_VARIANT = "primary"


def _residual_diagnostics(y: np.ndarray, yhat: np.ndarray, x: np.ndarray) -> dict:
    """Systematic-bias tests on the residuals (predicted minus measured)."""
    r = yhat - y
    out = {
        "n": int(r.size),
        "mean_residual_pct_capacity": float(100.0 * r.mean()),
        "std_residual_pct_capacity": float(100.0 * r.std(ddof=1)) if r.size > 1 else None,
        "max_abs_residual_pct_capacity": float(100.0 * np.abs(r).max()),
    }
    # is the mean residual distinguishable from zero?
    if r.size > 2:
        t, p = stats.ttest_1samp(r, 0.0)
        out["bias_t_statistic"] = float(t)
        out["bias_p_value"] = float(p)
        out["bias_significant_at_5pct"] = bool(p < 0.05)
    # does the residual trend with the magnitude of the loss? (a trend means
    # the functional form is wrong, not just noisy)
    if r.size > 2 and np.ptp(x) > 0:
        sl, ic, rr, pp, se = stats.linregress(x, r)
        out["residual_vs_measured_slope"] = float(sl)
        out["residual_vs_measured_p_value"] = float(pp)
        out["residual_trend_significant_at_5pct"] = bool(pp < 0.05)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-starts", type=int, default=None)
    args = ap.parse_args()

    paths = Paths.load()
    cfg = load_config("degradation")
    cal_cfg = cfg["calibration"]
    val_cfg = cfg["validation"]
    bins = list(cfg["c_rate_bins"])
    q_nom = float(cfg["physical_constants"]["q_nominal_ah"])
    R = float(cfg["physical_constants"]["R_universal"])
    window = float(cal_cfg["operating_window_max_y"])
    cells = [c.upper() for c in paths.cells]
    folds = [c.upper() for c in val_cfg["folds"]]

    src = stage_dir("01_calibration_dataset", create=False) / "calibration_intervals.csv"
    if not src.is_file():
        print(f"missing {src} — run scripts/01_build_calibration_dataset.py first")
        return 1
    df = pd.read_csv(src)

    # Validation is performed in the same regime the model is deployed in.
    all_data = restrict_to_window(datasets_from_dataframe(df, cells, bins), window)
    by_cell = {d.cell: d for d in all_data}
    

    print("=" * 88)
    print("Leave-one-cell-out validation")
    print("=" * 88)
    print(f"  parameterization: {layout.label}")
    print(f"  regime:           operating window, measured loss <= {window:g}")
    print(f"  gate:             R² >= {val_cfg['gate_min_r2']:.2f} and "
          f"|bias| <= {100 * float(val_cfg['gate_max_abs_bias']):.1f} %-capacity "
          f"on every held-out cell")
    print(
        "  NOTE: coefficients are re-fitted per fold from scratch; stage 3 "
        "output is not read.\n"
    )

    fold_rows = []
    pred_rows = []
    per_fold = {}
    for held in folds:
        train = [by_cell[c] for c in cells if c != held]
        test = by_cell[held]
        res = fit(
            train, layout=layout, cfg=cfg, q_nominal_ah=q_nom, R=R,
            provenance=f"loco:holdout={held}", n_starts=args.n_starts, verbose=False,
        )
        in_sample = goodness_of_fit(res.params, train)["pooled"]
        out_sample = goodness_of_fit(res.params, [test])["per_cell"][held]
        split = predict_cell_split(res.params, test)
        diag = _residual_diagnostics(test.y, split["q_total"], test.y)

        print(
            f"  holdout {held:5s} | train n={sum(d.n for d in train):3d} "
            f"R²={in_sample['r2']:.4f} RMSE={in_sample['rmse_pct_capacity']:.3f}"
            f"  ||  TEST n={out_sample['n']:3d} "
            f"R²={out_sample['r2']:+.4f} RMSE={out_sample['rmse_pct_capacity']:.3f} "
            f"MAE={out_sample['mae_pct_capacity']:.3f} "
            f"bias={out_sample['bias_pct_capacity']:+.3f} %-cap"
        )

        fold_rows.append({
            "holdout_cell": held,
            "n_train": int(sum(d.n for d in train)),
            "n_test": out_sample["n"],
            "train_r2": in_sample["r2"],
            "train_rmse_pct_capacity": in_sample["rmse_pct_capacity"],
            "test_r2": out_sample["r2"],
            "test_rmse_pct_capacity": out_sample["rmse_pct_capacity"],
            "test_mae_pct_capacity": out_sample["mae_pct_capacity"],
            "test_bias_pct_capacity": out_sample["bias_pct_capacity"],
            "p_c_rate": float(
                res.x[layout.names.index("p_c_rate")]
                if "p_c_rate" in layout.names else np.nan
            ),
            "z_cyc": float(res.params.z_cyc[0]),
            "Ea_cyc": float(res.params.Ea_cyc[0]),
            "converged": bool(res.success),
            "n_flags": len(res.flags),
        })
        per_fold[held] = {
            "fit": res.to_dict(train),
            "in_sample": in_sample,
            "out_of_sample": out_sample,
            "residual_diagnostics": diag,
        }
        for k in range(test.n):
            pred_rows.append({
                "cell": held,
                "fold": f"holdout_{held}",
                "interval_index": k + 1,
                "cum_duration_h": test.cum_duration_h[k],
                "cum_ah": test.cum_ah[k],
                "mean_soc": test.mean_soc[k],
                "mean_temperature_c": test.mean_temperature_c[k],
                "y_measured": test.y[k],
                "y_predicted": split["q_total"][k],
                "y_pred_calendar": split["q_calendar"][k],
                "y_pred_cyclic": split["q_cyclic"][k],
                "residual": split["q_total"][k] - test.y[k],
            })

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)

    # pooled out-of-sample performance: every point predicted by a model that
    # never saw its cell
    y = pred_df["y_measured"].to_numpy()
    yhat = pred_df["y_predicted"].to_numpy()
    ss_res = float(np.sum((yhat - y) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    pooled = {
        "n": int(y.size),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "rmse_pct_capacity": float(100.0 * np.sqrt(ss_res / y.size)),
        "mae_pct_capacity": float(100.0 * np.mean(np.abs(yhat - y))),
        "bias_pct_capacity": float(100.0 * np.mean(yhat - y)),
    }
    print(
        f"\n  POOLED out-of-sample (n={pooled['n']}): R²={pooled['r2']:.4f}  "
        f"RMSE={pooled['rmse_pct_capacity']:.3f} %-cap  "
        f"MAE={pooled['mae_pct_capacity']:.3f} %-cap  "
        f"bias={pooled['bias_pct_capacity']:+.3f} %-cap"
    )

    print("\n  Coefficient stability across folds (a stable coefficient means the")
    print("  functional form transfers, not just the numbers):")
    for col, unit in (("p_c_rate", ""), ("z_cyc", ""), ("Ea_cyc", " J/mol")):
        v = fold_df[col].to_numpy(dtype=float)
        if np.all(np.isfinite(v)):
            print(
                f"    {col:9s} = {v.mean():.4g} ± {v.std(ddof=1):.3g}{unit}  "
                f"(range {v.min():.4g} … {v.max():.4g}, CV={100 * v.std(ddof=1) / abs(v.mean()):.1f} %)"
            )

    print("\n  Systematic-bias tests on the held-out residuals:")
    for held in folds:
        d = per_fold[held]["residual_diagnostics"]
        bias_note = (
            "bias significant" if d.get("bias_significant_at_5pct")
            else "bias not significant"
        )
        trend_note = (
            "trend with loss magnitude SIGNIFICANT"
            if d.get("residual_trend_significant_at_5pct")
            else "no significant trend"
        )
        print(
            f"    {held:5s} mean={d['mean_residual_pct_capacity']:+6.3f} %-cap  "
            f"max|r|={d['max_abs_residual_pct_capacity']:5.3f} %-cap  "
            f"{bias_note} (p={d.get('bias_p_value', float('nan')):.3f}), {trend_note}"
        )

    # ---- GATE -------------------------------------------------------------
    min_r2 = float(val_cfg["gate_min_r2"])
    max_bias = float(val_cfg["gate_max_abs_bias"]) * 100.0
    failures = []
    for r in fold_rows:
        if not (r["test_r2"] >= min_r2):
            failures.append(
                f"{r['holdout_cell']}: out-of-sample R² = {r['test_r2']:.4f} < {min_r2:.2f}"
            )
        if abs(r["test_bias_pct_capacity"]) > max_bias:
            failures.append(
                f"{r['holdout_cell']}: |bias| = {abs(r['test_bias_pct_capacity']):.3f} "
                f"> {max_bias:.1f} %-capacity"
            )
        if not r["converged"]:
            failures.append(f"{r['holdout_cell']}: fold fit did not converge")

    out_dir = stage_dir(STAGE)
    fold_df.to_csv(out_dir / "loco_folds.csv", index=False)
    pred_df.to_csv(out_dir / "loco_predictions.csv", index=False)
    write_json(out_dir / "validation.json", {
        "provenance": provenance(STAGE, configs=["paths", "degradation"], inputs=[src]),
        "scheme": val_cfg["scheme"],
        "folds": folds,
        "parameterization": layout.to_dict(),
        "regime": {
            "operating_window_max_y": window,
            "note": (
                "Validation is run in the regime the model is deployed in "
                "(measured loss <= the operating-window limit). The model is "
                "NOT claimed valid past the degradation knee, and no "
                "downstream conclusion is read from outside this window."
            ),
        },
        "independence": (
            "Coefficients are re-fitted from scratch within each fold using "
            "only the three training cells; the held-out cell contributes "
            "nothing to the fit that scores it. Stage 3 output is not read by "
            "this script."
        ),
        "per_fold": per_fold,
        "pooled_out_of_sample": pooled,
        "coefficient_stability": {
            col: {
                "mean": float(fold_df[col].mean()),
                "std": float(fold_df[col].std(ddof=1)),
                "min": float(fold_df[col].min()),
                "max": float(fold_df[col].max()),
            }
            for col in ("p_c_rate", "z_cyc", "Ea_cyc")
        },
        "gate": {
            "min_r2": min_r2,
            "max_abs_bias_pct_capacity": max_bias,
            "failures": failures,
            "passed": not failures,
        },
    })

    print(f"\nWrote → {out_dir}")
    print("=" * 88)
    if failures:
        print("GATE FAIL — the model does not generalize. Diagnose before optimizing.")
        for f in failures:
            print(f"  ✗ {f}")
        print(
            "\nPer the protocol, the pipeline STOPS here rather than carrying a "
            "non-generalizing degradation model into the reward and optimization."
        )
        return 1
    print(
        f"GATE PASS — every held-out cell meets R² >= {min_r2:.2f} and "
        f"|bias| <= {max_bias:.1f} %-capacity out of sample."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
