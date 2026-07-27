"""Grounded figures from NASA RW9–RW12 measured data (not model index).

Sources
-------
* Capacity: 1 A ``reference discharge`` steps → Ah throughput, optionally
  OCV-corrected to a full SoC window (same method as ``charging_opt.soc_utils``).
* Temperature: measured ``temperature_c`` on every sample between consecutive
  reference discharges (life-mean T and per-interval mean T).
* OCV–SoC: ``Constrained_BO/data/RW*/ocv_soc_curve.npz`` when present.

NASA RW does **not** record humidity — no RH figures are invented here.

Usage
-----
    python -m Constrained_BO.grounded_dataset_figures \\
        --out-dir Constrained_BO/results/grounded_figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

from charging_opt.soc_utils import (
    REF_DISCHARGE_COMMENT,
    _dt_seconds,
    capacity_fade_table,
    load_steps_with_age,
)
from rw_transfer.data.mat_loader import BatteryStep

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATLAB = ROOT / (
    "NASA_RW/dataset/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/"
    "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab"
)
CELLS = ("RW9", "RW10", "RW11", "RW12")
COLORS = {
    "RW9": "#1f77b4",
    "RW10": "#ff7f0e",
    "RW11": "#2ca02c",
    "RW12": "#d62728",
}


def _load_ocv(cell_id: str) -> Optional[PchipInterpolator]:
    candidates = [
        ROOT / "Constrained_BO" / "data" / cell_id / "ocv_soc_curve.npz",
        ROOT / "outputs" / "charging_opt" / "models" / "stage1_state_estimation" / "ocv_soc_curve.npz",
    ]
    for p in candidates:
        if not p.is_file():
            continue
        d = np.load(p)
        keys = set(d.files)
        if "voltage_v" in keys:
            v = np.asarray(d["voltage_v"], dtype=float)
        elif "ocv" in keys:
            v = np.asarray(d["ocv"], dtype=float)
        else:
            v = np.asarray(d["v"], dtype=float)
        if "soc" in keys:
            s = np.asarray(d["soc"], dtype=float)
        else:
            s = np.asarray(d["soc_ocv"], dtype=float)
        order = np.argsort(v)
        return PchipInterpolator(v[order], s[order], extrapolate=True)
    return None


def _ref_indices(steps: List[BatteryStep]) -> np.ndarray:
    return np.asarray(
        [
            i
            for i, s in enumerate(steps)
            if s.comment.strip().lower() == REF_DISCHARGE_COMMENT
        ],
        dtype=int,
    )


def _interval_mean_temp(
    steps: List[BatteryStep],
    ref_idx: np.ndarray,
) -> np.ndarray:
    """Mean measured T (°C) from previous ref (or start) up to this ref inclusive."""
    means = np.full(len(ref_idx), np.nan)
    prev = 0
    for k, i in enumerate(ref_idx):
        chunks = []
        for j in range(prev, int(i) + 1):
            t = np.asarray(steps[j].temperature_c, dtype=float)
            if t.size:
                chunks.append(t)
        if chunks:
            means[k] = float(np.nanmean(np.concatenate(chunks)))
        prev = int(i) + 1
    return means


def _cumulative_ah_before_refs(
    steps: List[BatteryStep],
    ref_idx: np.ndarray,
) -> np.ndarray:
    """Cumulative |I|·dt (Ah) from experiment start up to each reference discharge."""
    out = np.zeros(len(ref_idx), dtype=float)
    cum = 0.0
    next_k = 0
    for j, s in enumerate(steps):
        dt = _dt_seconds(s.relative_time_s)
        i = np.asarray(s.current_a, dtype=float)
        if i.size and dt.size:
            cum += float(np.sum(np.abs(i) * dt)) / 3600.0
        if next_k < len(ref_idx) and j == int(ref_idx[next_k]):
            out[next_k] = cum
            next_k += 1
    return out


def extract_cell(
    cell_id: str,
    matlab_dir: Path,
) -> Dict[str, np.ndarray]:
    steps, step_age = load_steps_with_age(matlab_dir, cell_id)
    ocv = _load_ocv(cell_id)
    table = capacity_fade_table(steps, step_age, ocv_spline=ocv)
    ref_idx = _ref_indices(steps)
    # Align temps to the same filtered reference set as capacity_fade_table
    # (aborted discharges dropped). Rebuild by matching ages.
    all_ages = step_age[ref_idx]
    all_T = _interval_mean_temp(steps, ref_idx)
    all_ah = _cumulative_ah_before_refs(steps, ref_idx)

    age = np.asarray(table["age"], dtype=float)
    q_as = np.asarray(table["q_full_as"], dtype=float)
    m = np.isfinite(q_as) & (q_as > 0)
    age, q_as = age[m], q_as[m]

    # nearest-neighbor match of filtered ages onto full ref list
    T = np.full(len(age), np.nan)
    ah = np.full(len(age), np.nan)
    for k, a in enumerate(age):
        j = int(np.argmin(np.abs(all_ages - a)))
        T[k] = all_T[j]
        ah[k] = all_ah[j]

    q_ah = q_as / 3600.0
    q0 = float(q_ah[0])
    return {
        "cell": cell_id,
        "age": age,
        "ref_number": np.arange(1, len(age) + 1, dtype=float),
        "q_ah": q_ah,
        "remaining_pct": 100.0 * q_ah / q0,
        "mean_T_c": T,
        "cum_ah": ah,
        "life_mean_T_c": float(np.nanmean(T)),
        "q0_ah": q0,
        "q_end_ah": float(q_ah[-1]),
    }


def _save_csv(rows: List[Dict[str, np.ndarray]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "cell",
                "ref_number",
                "age_norm",
                "capacity_Ah",
                "remaining_pct",
                "interval_mean_T_C",
                "cum_throughput_Ah",
                "life_mean_T_C",
            ]
        )
        for d in rows:
            for i in range(len(d["age"])):
                w.writerow(
                    [
                        d["cell"],
                        int(d["ref_number"][i]),
                        f"{d['age'][i]:.8f}",
                        f"{d['q_ah'][i]:.6f}",
                        f"{d['remaining_pct'][i]:.4f}",
                        f"{d['mean_T_c'][i]:.4f}",
                        f"{d['cum_ah'][i]:.4f}",
                        f"{d['life_mean_T_c']:.4f}",
                    ]
                )


def fig1_capacity_vs_ref(rows: List[Dict], out: Path) -> None:
    """Analog of literature 'capacity vs cycle' — measured Ah at each ref discharge."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    for d in rows:
        ax.plot(
            d["ref_number"],
            d["q_ah"],
            "-o",
            ms=3,
            lw=1.6,
            color=COLORS[d["cell"]],
            label=f"{d['cell']}  (Q₀={d['q0_ah']:.2f} Ah → {d['q_end_ah']:.2f} Ah)",
        )
    ax.set_xlabel("Reference discharge number")
    ax.set_ylabel("Capacity [Ah]")
    ax.set_title(
        "NASA RW: measured capacity from 1 A reference discharges\n"
        "(OCV-corrected full-window Ah; not a model)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 2.4)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig2_remaining_vs_age_by_temp(rows: List[Dict], out: Path) -> None:
    """Analog of literature 'remaining capacity vs time at storage T' — cycling age + life-mean T."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    # sort by life-mean T so legend reads cold→hot
    for d in sorted(rows, key=lambda x: x["life_mean_T_c"]):
        ax.plot(
            d["age"],
            d["remaining_pct"],
            "-",
            lw=2.0,
            color=COLORS[d["cell"]],
            label=(
                f"{d['cell']}: life-mean T = {d['life_mean_T_c']:.1f} °C "
                f"(end {d['remaining_pct'][-1]:.0f}%)"
            ),
        )
    ax.set_xlabel("Normalized age (step index / (N−1))")
    ax.set_ylabel("Remaining capacity [%]")
    ax.set_title(
        "NASA RW: remaining capacity vs age\n"
        "(legend: measured life-mean cell temperature during cycling)"
    )
    ax.set_ylim(20, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig3_capacity_vs_throughput(rows: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    for d in rows:
        ax.plot(
            d["cum_ah"],
            d["remaining_pct"],
            "-",
            lw=1.8,
            color=COLORS[d["cell"]],
            label=d["cell"],
        )
    ax.set_xlabel("Cumulative |I|·dt throughput [Ah] (from experiment start)")
    ax.set_ylabel("Remaining capacity [%]")
    ax.set_title("NASA RW: capacity retention vs measured ampere-hour throughput")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(20, 105)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig4_temp_history(rows: List[Dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    for d in rows:
        ax.plot(
            d["ref_number"],
            d["mean_T_c"],
            "-",
            lw=1.5,
            color=COLORS[d["cell"]],
            label=f"{d['cell']} (mean {d['life_mean_T_c']:.1f} °C)",
        )
    ax.set_xlabel("Reference discharge number")
    ax.set_ylabel("Interval-mean cell temperature [°C]")
    ax.set_title(
        "NASA RW: measured temperature between consecutive reference discharges\n"
        "(same intervals used for capacity points)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig5_fade_vs_local_temp(rows: List[Dict], out: Path) -> None:
    """Per-interval fade rate vs local mean T — grounded thermal sensitivity check."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    for d in rows:
        q = d["q_ah"]
        T = d["mean_T_c"]
        if len(q) < 5:
            continue
        # fade per reference interval (Ah lost); positive = loss
        dq = -np.diff(q)
        Tmid = 0.5 * (T[:-1] + T[1:])
        m = np.isfinite(dq) & np.isfinite(Tmid) & (dq > -0.05)
        ax.scatter(
            Tmid[m],
            dq[m],
            s=18,
            alpha=0.55,
            color=COLORS[d["cell"]],
            label=d["cell"],
        )
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("Interval-mean cell temperature [°C]")
    ax.set_ylabel("Capacity drop per reference interval [Ah]")
    ax.set_title(
        "NASA RW: measured capacity drop vs local temperature\n"
        "(each point = one interval between reference discharges)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig6_ocv_soc(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    any_plot = False
    for cell in CELLS:
        p = ROOT / "Constrained_BO" / "data" / cell / "ocv_soc_curve.npz"
        if not p.is_file():
            continue
        d = np.load(p)
        keys = set(d.files)
        if "voltage_v" in keys:
            v = np.asarray(d["voltage_v"], dtype=float)
        elif "ocv" in keys:
            v = np.asarray(d["ocv"], dtype=float)
        else:
            v = np.asarray(d["v"], dtype=float)
        s = np.asarray(d["soc"] if "soc" in keys else d["soc_ocv"], dtype=float)
        order = np.argsort(s)
        ax.plot(s[order] * 100.0, v[order], lw=2.0, color=COLORS[cell], label=cell)
        any_plot = True
    if not any_plot:
        plt.close(fig)
        return
    ax.set_xlabel("SoC [%]")
    ax.set_ylabel("OCV [V]")
    ax.set_title("NASA RW: fitted OCV–SoC curves (from low-current discharges)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig7_summary_bar(rows: List[Dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), dpi=140)
    cells = [d["cell"] for d in rows]
    colors = [COLORS[c] for c in cells]
    end_pct = [float(d["remaining_pct"][-1]) for d in rows]
    mean_T = [d["life_mean_T_c"] for d in rows]
    axes[0].bar(cells, end_pct, color=colors, edgecolor="k", lw=0.4)
    axes[0].set_ylabel("End-of-test remaining capacity [%]")
    axes[0].set_title("Measured end capacity")
    axes[0].set_ylim(0, 100)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[1].bar(cells, mean_T, color=colors, edgecolor="k", lw=0.4)
    axes[1].set_ylabel("Life-mean cell temperature [°C]")
    axes[1].set_title("Measured thermal exposure")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "NASA RW grounded summary (reference-discharge capacity + onboard T)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--matlab-dir",
        type=Path,
        default=DEFAULT_MATLAB,
        help="Directory containing RW9.mat … RW12.mat",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "Constrained_BO" / "results" / "grounded_figures",
    )
    ap.add_argument(
        "--cells",
        nargs="+",
        default=list(CELLS),
    )
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for cell in args.cells:
        print(f"Extracting {cell} …")
        d = extract_cell(cell, args.matlab_dir)
        rows.append(d)
        print(
            f"  n_ref={len(d['age'])}  Q0={d['q0_ah']:.3f} Ah  "
            f"Qend={d['q_end_ah']:.3f} Ah ({d['remaining_pct'][-1]:.1f}%)  "
            f"life-mean T={d['life_mean_T_c']:.2f} °C"
        )

    _save_csv(rows, out / "capacity_fade_measured.csv")
    fig1_capacity_vs_ref(rows, out / "fig1_capacity_vs_ref_discharge.png")
    fig2_remaining_vs_age_by_temp(rows, out / "fig2_remaining_vs_age_by_mean_T.png")
    fig3_capacity_vs_throughput(rows, out / "fig3_remaining_vs_throughput_Ah.png")
    fig4_temp_history(rows, out / "fig4_temperature_history.png")
    fig5_fade_vs_local_temp(rows, out / "fig5_fade_vs_local_temperature.png")
    fig6_ocv_soc(out / "fig6_ocv_soc_curves.png")
    fig7_summary_bar(rows, out / "fig7_summary_capacity_and_temperature.png")

    # short provenance note
    note = out / "README.txt"
    note.write_text(
        "Grounded NASA RW figures\n"
        "========================\n"
        "All capacity points come from measured 1 A 'reference discharge' steps\n"
        "in RW9–RW12 .mat files (OCV-corrected full-window Ah when an OCV curve\n"
        "is available). Temperatures are onboard cell thermocouple readings.\n"
        "\n"
        "Humidity is not recorded in NASA RW — no RH plots are included.\n"
        "\n"
        "CSV: capacity_fade_measured.csv (one row per reference discharge).\n",
        encoding="utf-8",
    )
    print(f"Wrote figures + CSV → {out}")


if __name__ == "__main__":
    main()
