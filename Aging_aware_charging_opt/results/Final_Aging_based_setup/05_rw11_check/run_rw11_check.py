#!/usr/bin/env python
"""RW11 data vs twin check. Writes only into this folder. Does not run GP-BO."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
REPO = PROJECT.parent
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(REPO))

from aacopt.bdt import FrozenBDT  # noqa: E402
from aacopt.calibration_data import _soc_trace  # noqa: E402
from aacopt.capacity import OcvCurve, reference_capacity_table  # noqa: E402
from aacopt.config import Paths, RESULTS_DIR, ensure_matplotlib_cache, read_json  # noqa: E402
from aacopt.nasa_data import load_steps  # noqa: E402
from aacopt.viz.style import apply_paper_style, savefig  # noqa: E402

CELLS = ("RW9", "RW10", "RW11", "RW12")
CHUNK = 150
I_BINS = (
    ("1.5-2.5 A", 1.5, 2.5, False),
    ("2.5-3.5 A", 2.5, 3.5, False),
    ("3.5-4.62 A", 3.5, 4.62, True),
)
SOC_BINS = (
    ("50-60%", 0.50, 0.60, False),
    ("60-70%", 0.60, 0.70, False),
    ("70-80%", 0.70, 0.80, False),
    ("80-90%", 0.80, 0.90, True),
)
CELL_COLORS = {"RW9": "#2563eb", "RW10": "#16a34a", "RW11": "#ea580c", "RW12": "#7c3aed"}
V_MAX = 4.20
V_TOL = 1e-3
MIN_N = 30


def _in_bin(x: np.ndarray, lo: float, hi: float, right_closed: bool) -> np.ndarray:
    if right_closed:
        return (x >= lo) & (x <= hi)
    return (x >= lo) & (x < hi)


def _median_iqr(a: np.ndarray) -> Tuple[float, float, float, float]:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    q25, med, q75 = np.percentile(a, [25.0, 50.0, 75.0])
    return float(med), float(q75 - q25), float(q25), float(q75)


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def load_raw_with_soc(cell: str, paths: Paths) -> Dict[str, np.ndarray]:
    steps = load_steps(paths.mat_path(cell))
    ocv = OcvCurve.from_npz(
        RESULTS_DIR / "01_calibration_dataset" / "ocv_curves.npz", cell,
    )
    refs, info = reference_capacity_table(steps, ocv)
    traces = _soc_trace(steps, refs, ocv)
    voltage = np.concatenate([s.voltage_v for s in steps]).astype(np.float64)
    current = np.concatenate([s.current_a for s in steps]).astype(np.float64)
    temp = np.concatenate([s.temperature_c for s in steps]).astype(np.float64)
    soc = np.concatenate(traces).astype(np.float64)
    age = np.concatenate([
        np.full(s.n, i / max(len(steps), 1), dtype=np.float64) for i, s in enumerate(steps)
    ])
    if not (voltage.size == current.size == temp.size == soc.size == age.size):
        raise RuntimeError(f"{cell}: concatenated array length mismatch")
    i_chg = np.maximum(-current, 0.0)
    return {
        "voltage": voltage,
        "current_nasa": current,
        "temperature": temp,
        "soc": soc,
        "age": age,
        "i_chg": i_chg,
        "n_steps": np.array([len(steps)]),
        "n_refs_accepted": np.array([info["n_accepted"]]),
    }


def window_xy(series: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Author-chunk layout: actions/next-states are samples [start+1, start+CHUNK]."""
    n = int(series["voltage"].size)
    n_chunks = n // CHUNK
    n_win = max(0, n_chunks - 1)
    X, yv, soc, ichg, temp = [], [], [], [], []
    for i in range(n_win):
        start = CHUNK * i
        end = start + CHUNK + 1
        X.append(np.concatenate([
            [series["age"][start], series["voltage"][start], series["temperature"][start]],
            series["current_nasa"][start + 1:end],
        ]))
        yv.append(series["voltage"][start + 1:end])
        soc.append(series["soc"][start + 1:end])
        ichg.append(series["i_chg"][start + 1:end])
        temp.append(series["temperature"][start + 1:end])
    return (
        np.stack(X).astype(np.float32),
        np.stack(yv).astype(np.float64),
        np.stack(soc).astype(np.float64),
        np.stack(ichg).astype(np.float64),
        np.stack(temp).astype(np.float64),
    )


def bin_stats(i_chg, soc, v, t) -> List[Dict[str, Any]]:
    rows = []
    for i_name, i_lo, i_hi, i_rc in I_BINS:
        for s_name, s_lo, s_hi, s_rc in SOC_BINS:
            m = _in_bin(i_chg, i_lo, i_hi, i_rc) & _in_bin(soc, s_lo, s_hi, s_rc)
            n = int(np.sum(m))
            v_med, v_iqr, v_q25, v_q75 = _median_iqr(v[m]) if n else (float("nan"),) * 4
            t_med, t_iqr, t_q25, t_q75 = _median_iqr(t[m]) if n else (float("nan"),) * 4
            rows.append({
                "i_bin": i_name, "soc_bin": s_name, "n": n,
                "insufficient": n < MIN_N,
                "v_median": v_med, "v_iqr": v_iqr, "v_q25": v_q25, "v_q75": v_q75,
                "t_median": t_med, "t_iqr": t_iqr, "t_q25": t_q25, "t_q75": t_q75,
            })
    return rows


def plot_v_vs_soc(per_cell: Dict[str, Dict[str, np.ndarray]], out: Path) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.8), sharey=True)
    soc_edges = np.arange(0.50, 0.90 + 1e-9, 0.02)
    soc_cent = 0.5 * (soc_edges[:-1] + soc_edges[1:])
    for ax, (i_name, i_lo, i_hi, i_rc) in zip(axes, I_BINS):
        for cell in CELLS:
            s = per_cell[cell]
            m_i = _in_bin(s["i_chg"], i_lo, i_hi, i_rc)
            meds, xs = [], []
            for lo, hi in zip(soc_edges[:-1], soc_edges[1:]):
                m = m_i & (s["soc"] >= lo) & (s["soc"] < hi if hi < 0.90 - 1e-12 else s["soc"] <= hi)
                if int(np.sum(m)) < MIN_N:
                    continue
                meds.append(float(np.median(s["voltage"][m])))
                xs.append(0.5 * (lo + hi))
            ax.plot(xs, meds, color=CELL_COLORS[cell], lw=2.2, marker="o", ms=4, label=cell)
        ax.set_title(i_name)
        ax.set_xlabel("SOC")
        ax.set_xlim(0.48, 0.92)
    axes[0].set_ylabel("Measured voltage (V)")
    axes[2].legend(frameon=False, loc="best")
    fig.suptitle("Measured V vs SOC on charging samples (median in 2% SOC bins, n≥30)")
    fig.tight_layout()
    savefig(fig, out)


def main() -> int:
    ensure_matplotlib_cache()
    out = HERE
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = Paths.load()

    per_cell_series = {}
    data_rows = []
    high_rows = []
    print("A) loading raw NASA + aacopt._soc_trace …", flush=True)
    for cell in CELLS:
        s = load_raw_with_soc(cell, paths)
        per_cell_series[cell] = s
        charge = s["i_chg"] > 0.0
        print(
            f"  {cell}: n={s['voltage'].size}  n_charge={int(np.sum(charge))}  "
            f"I_chg max={float(s['i_chg'].max())}  refs={int(s['n_refs_accepted'][0])}",
            flush=True,
        )
        for rec in bin_stats(s["i_chg"][charge], s["soc"][charge],
                             s["voltage"][charge], s["temperature"][charge]):
            rec["cell"] = cell
            rec["source"] = "raw_nasa_charging_I_chg>0"
            data_rows.append(rec)
        hi = (s["i_chg"] > 4.0) & (s["voltage"] >= 4.1)
        hi_c = hi & charge
        high_rows.append({
            "cell": cell,
            "n_I_chg_gt_4_and_V_ge_4_1": int(np.sum(hi_c)),
            "n_charging": int(np.sum(charge)),
            "n_I_chg_gt_4": int(np.sum((s["i_chg"] > 4.0) & charge)),
            "n_V_ge_4_1_charging": int(np.sum((s["voltage"] >= 4.1) & charge)),
            "max_V_when_I_chg_gt_4": float(s["voltage"][(s["i_chg"] > 4.0) & charge].max())
            if np.any((s["i_chg"] > 4.0) & charge) else float("nan"),
        })

    write_csv(out / "data_bins.csv", data_rows, [
        "cell", "i_bin", "soc_bin", "n", "insufficient",
        "v_median", "v_iqr", "v_q25", "v_q75",
        "t_median", "t_iqr", "t_q25", "t_q75", "source",
    ])
    write_csv(out / "high_I_high_V_counts.csv", high_rows, [
        "cell", "n_I_chg_gt_4_and_V_ge_4_1", "n_charging", "n_I_chg_gt_4",
        "n_V_ge_4_1_charging", "max_V_when_I_chg_gt_4",
    ])
    plot_v_vs_soc(per_cell_series, fig_dir / "measured_V_vs_SOC_by_current_bin.png")

    # B) twin vs measured voltage in the same bins
    print("B) twin predictions on author-style windows …", flush=True)
    src_ckpt = paths.checkpoint("RW9")
    twin_rows = []
    high_soc_high_i = []
    for cell in CELLS:
        s = per_cell_series[cell]
        X, yv, soc_w, ichg_w, _t = window_xy(s)
        print(f"  {cell}: n_windows={X.shape[0]}", flush=True)
        models = [
            ("cell_twin", paths.checkpoint(cell)),
            ("twin_source_RW9.pt", src_ckpt),
        ]
        for mid, ckpt in models:
            twin = FrozenBDT(ckpt, device="cuda:0")
            v_hat, _ = twin.predict_windows(X, batch_size=64)
            v_hat = np.asarray(v_hat, dtype=np.float64)
            pred = v_hat.ravel()
            meas = yv.ravel()
            i_flat = ichg_w.ravel()
            soc_flat = soc_w.ravel()
            charge = i_flat > 0.0
            for rec in bin_stats(i_flat[charge], soc_flat[charge], meas[charge],
                                 np.zeros_like(meas[charge])):
                # replace T stats with voltage error stats in this loop via extra fields
                m = (
                    _in_bin(i_flat, *[b[1:] for b in I_BINS if b[0] == rec["i_bin"]][0])
                    & _in_bin(soc_flat, *[b[1:] for b in SOC_BINS if b[0] == rec["soc_bin"]][0])
                    & charge
                )
                n = int(np.sum(m))
                if n == 0:
                    bias = rmse = float("nan")
                else:
                    err = pred[m] - meas[m]
                    bias = float(np.mean(err))
                    rmse = float(np.sqrt(np.mean(err ** 2)))
                twin_rows.append({
                    "cell": cell,
                    "model_id": mid,
                    "checkpoint": str(ckpt),
                    "i_bin": rec["i_bin"],
                    "soc_bin": rec["soc_bin"],
                    "n": n,
                    "insufficient": n < MIN_N,
                    "v_meas_median": rec["v_median"],
                    "v_meas_iqr": rec["v_iqr"],
                    "bias_pred_minus_meas_v": bias,
                    "rmse_v": rmse,
                    "underpredicts_voltage": bool(n >= MIN_N and bias < 0.0),
                })
            m_hi = charge & _in_bin(i_flat, 3.5, 4.62, True) & _in_bin(soc_flat, 0.80, 0.90, True)
            n_hi = int(np.sum(m_hi))
            if n_hi:
                err = pred[m_hi] - meas[m_hi]
                high_soc_high_i.append({
                    "cell": cell, "model_id": mid, "n": n_hi,
                    "insufficient": n_hi < MIN_N,
                    "bias_v": float(np.mean(err)),
                    "rmse_v": float(np.sqrt(np.mean(err ** 2))),
                    "meas_median_v": float(np.median(meas[m_hi])),
                    "pred_median_v": float(np.median(pred[m_hi])),
                })
            else:
                high_soc_high_i.append({
                    "cell": cell, "model_id": mid, "n": 0, "insufficient": True,
                    "bias_v": float("nan"), "rmse_v": float("nan"),
                    "meas_median_v": float("nan"), "pred_median_v": float("nan"),
                })
            del twin

    write_csv(out / "twin_vs_data_bins.csv", twin_rows, [
        "cell", "model_id", "i_bin", "soc_bin", "n", "insufficient",
        "v_meas_median", "v_meas_iqr", "bias_pred_minus_meas_v", "rmse_v",
        "underpredicts_voltage", "checkpoint",
    ])
    write_csv(out / "twin_highI_highSOC.csv", high_soc_high_i, [
        "cell", "model_id", "n", "insufficient", "bias_v", "rmse_v",
        "meas_median_v", "pred_median_v",
    ])

    # C) post-hoc C3 margin (verdict unchanged)
    print("C) post-hoc C3 margin from Arm A cross_cell.csv …", flush=True)
    cross = list(csv.DictReader(
        (RESULTS_DIR / "Final_Aging_based_setup" / "02_armA" / "litmus" / "cross_cell.csv").open()
    ))
    rmse_blob = read_json(
        RESULTS_DIR / "Final_Aging_based_setup" / "00_diagnostics" / "identical_windows_rmse.json"
    )
    twin_rmse_mv = {
        c: float(rmse_blob["per_cell"][c]["finetuned_voltage_rmse_mv"]) for c in CELLS
    }
    by_pid: Dict[str, List[Dict[str, str]]] = {}
    for r in cross:
        by_pid.setdefault(r["profile_id"], []).append(r)
    c3_fail_profiles = []
    for pid, hits in by_pid.items():
        if not all(h["feasible"] == "True" for h in hits):
            c3_fail_profiles.append(pid)

    fail_rows = []
    for pid in c3_fail_profiles:
        for h in by_pid[pid]:
            if h["feasible"] == "True":
                continue
            vpeak = float(h["peak_v"])
            over_mv = 1000.0 * (vpeak - V_MAX)
            eval_cell = h["eval_cell"]
            rmse_mv = twin_rmse_mv[eval_cell]
            voltage_reject = vpeak > (V_MAX + V_TOL)
            fail_rows.append({
                "profile_id": pid,
                "source_cell": h["source_cell"],
                "family_id": h["family_id"],
                "rejecting_twin": eval_cell,
                "native": h["native"],
                "v_peak": vpeak,
                "overshoot_mV_above_4_20": over_mv,
                "voltage_reject": voltage_reject,
                "end_reason": h["end_reason"],
                "twin_voltage_rmse_mv": rmse_mv,
                "overshoot_lt_twin_rmse": over_mv < rmse_mv,
                "voltage_overshoot_positive_and_lt_rmse": bool(
                    voltage_reject and (over_mv < rmse_mv)
                ),
            })

    write_csv(out / "c3_posthoc_failures.csv", fail_rows, [
        "profile_id", "source_cell", "family_id", "rejecting_twin", "native",
        "v_peak", "overshoot_mV_above_4_20", "voltage_reject", "end_reason",
        "twin_voltage_rmse_mv", "overshoot_lt_twin_rmse",
        "voltage_overshoot_positive_and_lt_rmse",
    ])

    n_reject = len(fail_rows)
    n_volt = sum(1 for r in fail_rows if r["voltage_reject"])
    n_over_lt = sum(1 for r in fail_rows if r["overshoot_lt_twin_rmse"])
    n_volt_lt = sum(1 for r in fail_rows if r["voltage_overshoot_positive_and_lt_rmse"])

    # Answer paragraph ingredients
    def _bin_n(cell, i_name, s_name):
        hits = [r for r in data_rows if r["cell"] == cell and r["i_bin"] == i_name and r["soc_bin"] == s_name]
        return hits[0] if hits else None

    def _twin_hi(cell, mid):
        hits = [r for r in high_soc_high_i if r["cell"] == cell and r["model_id"] == mid]
        return hits[0] if hits else None

    rw11_hi_data = _bin_n("RW11", "3.5-4.62 A", "80-90%")
    others_hi = {c: _bin_n(c, "3.5-4.62 A", "80-90%") for c in CELLS}
    rw11_twin_hi = _twin_hi("RW11", "cell_twin")
    rw11_src_hi = _twin_hi("RW11", "twin_source_RW9.pt")

    # median V comparison in high-I bins at 80-90% if sufficient
    vmed_lines = []
    for i_name, *_ in I_BINS:
        parts = []
        for c in CELLS:
            rec = _bin_n(c, i_name, "80-90%")
            if rec is None or rec["insufficient"]:
                parts.append(f"{c}=insufficient(n={rec['n'] if rec else 0})")
            else:
                parts.append(f"{c} Vmed={rec['v_median']!r} (n={rec['n']})")
        vmed_lines.append(f"{i_name}, SOC 80–90%: " + "; ".join(parts))

    high_iv = {r["cell"]: r["n_I_chg_gt_4_and_V_ge_4_1"] for r in high_rows}

    data_insufficient_high = any(
        (others_hi[c] is None) or others_hi[c]["insufficient"] for c in CELLS
    )

    if data_insufficient_high:
        answer = (
            "The high-current / high-SOC bin (3.5–4.62 A × 80–90% SOC) is too sparsely "
            "populated on at least one cell (n<30) to decide from measurements whether RW11 "
            "physically tolerates higher charge current than RW9/RW10/RW12; the RW11 twin's "
            "lower predicted V_peak at high current is therefore not established as a "
            "property of the measured data."
        )
    else:
        v11 = others_hi["RW11"]["v_median"]
        v_others = [others_hi[c]["v_median"] for c in CELLS if c != "RW11"]
        # If RW11 measured median V is lower than others at same I and SOC, data support
        # higher current tolerance (less overpotential). If similar or higher, twin-only.
        if v11 < min(v_others):
            data_says = (
                f"In that bin RW11's measured median voltage ({v11!r} V) is below the other "
                f"cells ({v_others!r} V), so a milder high-SOC voltage rise is present in the "
                "measurements, not only in the twin."
            )
        else:
            data_says = (
                f"In that bin RW11's measured median voltage ({v11!r} V) is not below the other "
                f"cells ({v_others!r} V), so the RW11 twin's extra current headroom is not "
                "supported by the measured voltages."
            )
        twin_note = ""
        if rw11_twin_hi and not rw11_twin_hi["insufficient"]:
            twin_note = (
                f" The RW11 cell twin bias (pred−meas) in this bin is {rw11_twin_hi['bias_v']!r} V"
                f" (n={rw11_twin_hi['n']})"
                + (" and therefore under-predicts voltage." if rw11_twin_hi["bias_v"] < 0 else
                   " and therefore does not under-predict voltage.")
            )
        answer = data_says + twin_note

    # Build REPORT
    def fmt_row(r):
        flag = "insufficient" if r["insufficient"] else "ok"
        return (
            f"| {r['cell']} | {r['i_bin']} | {r['soc_bin']} | {r['n']} | {flag} | "
            f"{r['v_median']!r} | {r['v_iqr']!r} | {r['t_median']!r} | {r['t_iqr']!r} |"
        )

    data_md = "\n".join(fmt_row(r) for r in data_rows)
    high_md = "\n".join(
        f"| {r['cell']} | {r['n_I_chg_gt_4_and_V_ge_4_1']} | {r['n_I_chg_gt_4']} | "
        f"{r['n_V_ge_4_1_charging']} | {r['max_V_when_I_chg_gt_4']!r} |"
        for r in high_rows
    )
    twin_hi_md = "\n".join(
        f"| {r['cell']} | {r['model_id']} | {r['n']} | "
        f"{'insufficient' if r['insufficient'] else 'ok'} | {r['bias_v']!r} | {r['rmse_v']!r} | "
        f"{r['meas_median_v']!r} | {r['pred_median_v']!r} |"
        for r in high_soc_high_i
    )
    # compact twin table: only sufficient bins, or all with flag
    twin_md_lines = []
    for r in twin_rows:
        twin_md_lines.append(
            f"| {r['cell']} | {r['model_id']} | {r['i_bin']} | {r['soc_bin']} | {r['n']} | "
            f"{'insufficient' if r['insufficient'] else 'ok'} | {r['bias_pred_minus_meas_v']!r} | "
            f"{r['rmse_v']!r} | {r['underpredicts_voltage']} |"
        )
    fail_md = "\n".join(
        f"| {r['profile_id']} | {r['rejecting_twin']} | {r['v_peak']!r} | "
        f"{r['overshoot_mV_above_4_20']!r} | {r['voltage_reject']} | {r['end_reason']} | "
        f"{r['twin_voltage_rmse_mv']!r} | {r['overshoot_lt_twin_rmse']} |"
        for r in fail_rows
    )

    rw11_under = None
    if rw11_twin_hi and not rw11_twin_hi["insufficient"]:
        rw11_under = rw11_twin_hi["bias_v"] < 0.0

    report = f"""# 05 — RW11 measured-data vs twin check

This folder is diagnostic only. Arm A / Phase 3 / Arm B GP-BO was **not** re-run.
The pre-registered C1–C3 verdict in `../02_armA/armA_verdict.json` is **unchanged** (FAIL).

NASA sign: recorded current I < 0 is charge. Charging samples are those with **I_chg = max(−I, 0) > 0** (the user request “I > 0” after converting to charging current). SOC is `aacopt.calibration_data._soc_trace` on `aacopt.nasa_data.load_steps` (Coulomb counting re-anchored at reference charge/discharge, present capacity interpolated between accepted reference measurements, OCV from `results/01_calibration_dataset/ocv_curves.npz`).

Bins: current [1.5, 2.5), [2.5, 3.5), [3.5, 4.62] A; SOC [0.50, 0.60), [0.60, 0.70), [0.70, 0.80), [0.80, 0.90]. A bin with n < 30 is **insufficient**.

## A) Model-free data

| Cell | I bin | SOC bin | n | status | V median | V IQR | T median | T IQR |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
{data_md}

### I_chg > 4 A and V ≥ 4.1 V (charging samples)

| Cell | n (I>4 A and V≥4.1 V) | n (I>4 A) | n (V≥4.1 V, charging) | max V at I>4 A |
| --- | ---: | ---: | ---: | ---: |
{high_md}

Figure: `figures/measured_V_vs_SOC_by_current_bin.png`.

SOC 80–90% median voltages by current bin:
{chr(10).join('- ' + x for x in vmed_lines)}

## B) Twin vs measured voltage (same bins)

Predictions: each cell twin and `twin_source_RW9.pt`, on author-style non-overlapping 150-sample windows built from the same concatenated raw record. Errors are pred − meas on charging samples in the bin.

| Cell | model | I bin | SOC bin | n | status | bias (pred−meas) V | RMSE V | under-predicts V? |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- |
{chr(10).join(twin_md_lines)}

### Focused bin: 3.5–4.62 A × 80–90% SOC

| Cell | model | n | status | bias V | RMSE V | meas median V | pred median V |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
{twin_hi_md}

RW11 cell twin under-predicts voltage at high current / high SOC? **{rw11_under}**
(bias {None if not rw11_twin_hi else rw11_twin_hi['bias_v']!r} V; insufficient bin? {None if not rw11_twin_hi else rw11_twin_hi['insufficient']}).

## C) POST-HOC C3 margin (verdict unchanged)

Arm A C3 remains **FAIL**. This section only describes the failed (profile, twin) pairs.

Profiles that are not feasible on all four twins: {len(c3_fail_profiles)} / 24.
Rejecting (profile, twin) pairs: {n_reject}.
Voltage rejections (V_peak > 4.20 + 1e-3): {n_volt}.
Rejecting pairs with overshoot_mV < that twin's held-out voltage RMSE: {n_over_lt} / {n_reject}.
Voltage rejections with 0 < overshoot < twin RMSE: {n_volt_lt} / {n_volt}.

Twin RMSE (Phase 0 identical test windows, cell-specific checkpoint):
{chr(10).join(f'- {c}: {twin_rmse_mv[c]!r} mV' for c in CELLS)}

| profile | rejecting twin | V_peak | overshoot mV | voltage reject? | end_reason | twin RMSE mV | overshoot < RMSE |
| --- | --- | ---: | ---: | --- | --- | ---: | --- |
{fail_md}

## Is RW11's higher current tolerance in the data or only in the twin?

{answer}
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")
    meta = {
        "nasa_charge_sign": "I<0 is charge; I_chg=max(-I,0)",
        "soc_function": "aacopt.calibration_data._soc_trace",
        "i_bins": I_BINS,
        "soc_bins": SOC_BINS,
        "min_n": MIN_N,
        "c3_verdict_unchanged": "FAIL",
        "n_c3_fail_profiles": len(c3_fail_profiles),
        "n_rejecting_pairs": n_reject,
        "n_voltage_reject": n_volt,
        "n_overshoot_lt_rmse": n_over_lt,
        "n_voltage_overshoot_lt_rmse": n_volt_lt,
        "rw11_twin_underpredicts_highI_highSOC": rw11_under,
        "high_I_and_V_counts": high_iv,
        "answer": answer,
    }
    (out / "summary.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out / 'REPORT.md'}", flush=True)
    print("RW11 under-predict highI/highSOC:", rw11_under, flush=True)
    print(answer, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
