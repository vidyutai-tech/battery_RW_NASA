"""Pulsed-charge sets with the same initial state and the same charging current.

A pulsed charge in these files is a rest at about 0 A followed by a hold at
about -1 A. Negative current is charging. A set is kept when RW9–RW12 agree on
the voltage, temperature, and age at the start of that hold, and the four hold
currents span at most 0.15 A.

Random-walk step sequences are counted with the same rule. A step sequence is
two charge holds separated only by a short rest. None of those sequences repeat
across all four cells inside 0.15 A, so they are reported and not plotted.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial import cKDTree

MATLAB_DIR = Path(
    "/data/battery_RW_NASA/NASA_RW/dataset/"
    "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/"
    "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab"
)
FIGURE_DIR = Path("/data/battery_RW_NASA/eda/figures/rw11_pulsed_current_gallery")
OUTPUT_DIR = Path("/data/battery_RW_NASA/eda/outputs")

CELLS = ["RW9", "RW10", "RW11", "RW12"]
OTHERS = ["RW10", "RW11", "RW12"]
CELL_COLOR = {"RW9": "#4C78A8", "RW10": "#F58518", "RW11": "#E45756", "RW12": "#54A24B"}
TOL_V, TOL_T, TOL_AGE = 0.02, 1.0, 0.02
I_SPAN_A = 0.15
PRE_S, POST_S = 30.0, 120.0

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def _comment(step) -> str:
    raw = step.comment
    if isinstance(raw, np.ndarray):
        raw = "" if raw.size == 0 else raw.flat[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip().lower()


def _median_current(step) -> float:
    cur = np.asarray(step.current, dtype=np.float64).ravel()
    cur = cur[np.isfinite(cur)]
    return float(np.median(cur)) if cur.size else np.nan


def _pulsed_holds(cell: str) -> list[dict]:
    raw = loadmat(str(MATLAB_DIR / f"{cell}.mat"), squeeze_me=True, struct_as_record=False)["data"]
    steps = list(raw.step)
    n_steps = len(steps)
    holds = []
    for index, step in enumerate(steps):
        if _comment(step) != "pulsed charge (charge)":
            continue
        voltage = np.asarray(step.voltage, dtype=np.float64).ravel()
        temperature = np.asarray(step.temperature, dtype=np.float64).ravel()
        if voltage.size == 0 or temperature.size == 0:
            continue
        prev = index - 1 if index > 0 and _comment(steps[index - 1]) == "pulsed charge (rest)" else None
        holds.append({
            "cell": cell,
            "step_index": index,
            "rest_index": prev,
            "V": float(voltage[0]),
            "T": float(temperature[0]),
            "age": index / max(n_steps - 1, 1),
            "I": _median_current(step),
        })
    print(f"  {cell}: {len(holds)} pulsed charge holds")
    return holds


def _match(tables: dict[str, list[dict]]) -> list[dict]:
    scales = np.array([TOL_V, TOL_T, TOL_AGE])
    trees = {
        cell: cKDTree(np.array([[b["V"], b["T"], b["age"]] for b in tables[cell]], dtype=float) / scales)
        for cell in OTHERS
    }
    candidates = []
    for i, ref in enumerate(tables["RW9"]):
        if not np.isfinite(ref["I"]) or ref["I"] >= 0:
            continue
        query = np.array([ref["V"], ref["T"], ref["age"]]) / scales
        partners = {}
        seq_gap = 0.0
        ok = True
        for cell in OTHERS:
            hits = trees[cell].query_ball_point(query, r=1.0, p=np.inf)
            close = []
            for hit in hits:
                other = tables[cell][hit]
                if not np.isfinite(other["I"]) or other["I"] >= 0:
                    continue
                gap = abs(ref["I"] - other["I"])
                state = abs(np.array([other["V"], other["T"], other["age"]]) / scales - query).max()
                close.append((gap, state, hit))
            if not close:
                ok = False
                break
            gap, _, hit = min(close)
            partners[cell] = hit
            seq_gap = max(seq_gap, gap)
        if not ok or seq_gap > I_SPAN_A:
            continue
        dist = 0.0
        choice = {"RW9": ref}
        for cell in OTHERS:
            other = tables[cell][partners[cell]]
            choice[cell] = other
            dist += abs(ref["V"] - other["V"]) / TOL_V
            dist += abs(ref["T"] - other["T"]) / TOL_T
            dist += abs(ref["age"] - other["age"]) / TOL_AGE
        candidates.append((dist, i, partners, choice, seq_gap))

    candidates.sort()
    used = {cell: set() for cell in CELLS}
    rows = []
    for dist, i, partners, choice, seq_gap in candidates:
        if i in used["RW9"] or any(partners[cell] in used[cell] for cell in OTHERS):
            continue
        used["RW9"].add(i)
        for cell in OTHERS:
            used[cell].add(partners[cell])
        rows.append({"distance": dist, "I_span": seq_gap, "choice": choice})
    return rows


def _step_sequence_count() -> int:
    """How many stepped random-walk bursts share the first two current levels."""
    from plot_step_current_gallery import _charge_bursts, _match as _step_match

    tables = {cell: _charge_bursts(cell) for cell in CELLS}
    matched = _step_match(tables)
    if matched.empty:
        return 0
    return int((matched["I_span"] <= I_SPAN_A).sum())


def _stitch(steps, rest_index: int | None, charge_index: int) -> dict[str, np.ndarray]:
    chunks = []
    if rest_index is not None:
        chunks.append(steps[rest_index])
    chunks.append(steps[charge_index])
    charge_t0 = float(np.asarray(steps[charge_index].time, dtype=np.float64).ravel()[0])
    times, current, voltage, temperature = [], [], [], []
    for step in chunks:
        tim = np.asarray(step.time, dtype=np.float64).ravel() - charge_t0
        cur = np.asarray(step.current, dtype=np.float64).ravel()
        vol = np.asarray(step.voltage, dtype=np.float64).ravel()
        tmp = np.asarray(step.temperature, dtype=np.float64).ravel()
        n = min(tim.size, cur.size, vol.size, tmp.size)
        if n == 0:
            continue
        keep = (tim >= -PRE_S - 1e-6) & (tim <= POST_S + 1e-6)
        times.append(tim[:n][keep[:n]])
        current.append(cur[:n][keep[:n]])
        voltage.append(vol[:n][keep[:n]])
        temperature.append(tmp[:n][keep[:n]])
    if not times:
        empty = np.array([])
        return {"time": empty, "current": empty, "voltage": empty, "temperature": empty}
    return {key: np.concatenate(value) for key, value in {
        "time": times, "current": current, "voltage": voltage, "temperature": temperature
    }.items()}


def _load_traces(rows: list[dict]) -> dict[int, dict]:
    needed = {cell: set() for cell in CELLS}
    for row in rows:
        for cell in CELLS:
            choice = row["choice"][cell]
            needed[cell].add(choice["step_index"])
            if choice["rest_index"] is not None:
                needed[cell].add(choice["rest_index"])
    raw_steps = {}
    for cell in CELLS:
        raw = loadmat(str(MATLAB_DIR / f"{cell}.mat"), squeeze_me=True, struct_as_record=False)["data"]
        raw_steps[cell] = list(raw.step)
        print(f"  {cell}: reading {len(needed[cell])} pulse steps")
    traces = {}
    for row in rows:
        traces[row["set_id"]] = {
            cell: _stitch(raw_steps[cell], row["choice"][cell]["rest_index"], row["choice"][cell]["step_index"])
            for cell in CELLS
        }
    return traces


def _plot_page(rows: list[dict], traces: dict, path: Path, page: int, n_pages: int) -> None:
    n = len(rows)
    fig, axes = plt.subplots(3, n, figsize=(3.15 * n, 8.6), sharex=True)
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    keys = [("current", "Current (A)"), ("voltage", "Voltage (V)"), ("temperature", "Temperature (°C)")]
    for col, row in enumerate(rows):
        ref = row["choice"]["RW9"]
        for r, (key, ylabel) in enumerate(keys):
            ax = axes[r, col]
            for cell in CELLS:
                tr = traces[row["set_id"]][cell]
                ax.plot(
                    tr["time"], tr[key],
                    color=CELL_COLOR[cell],
                    lw=1.6 if cell == "RW11" else 1.05,
                    label=cell if col == 0 and r == 0 else None,
                )
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(-PRE_S, POST_S)
            ax.axvline(0.0, color="0.5", lw=0.6, ls=":")
            if r == 0:
                ax.set_ylim(-1.6, 0.25)
                ax.set_title(
                    f"set {row['set_id']}  I {ref['I']:.2f} A\n"
                    f"V {ref['V']:.3f} V  T {ref['T']:.1f} °C\n"
                    f"age {ref['age']:.3f}  I span {row['I_span']:.3f} A",
                    fontsize=8,
                )
            if r == 2:
                ax.set_xlabel("Time from pulse start (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        f"Pulsed charge, same initial V, T, age, and the same charging current — page {page} of {n_pages}\n"
        "Rest before the hold is shown. The dotted line is the start of the −1 A charge.",
        y=1.10,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Extracting pulsed charge holds...")
    tables = {cell: _pulsed_holds(cell) for cell in CELLS}
    matched = _match(tables)
    print(f"Pulsed sets with the same initial state and the same charging current: {len(matched)}")
    matched.sort(key=lambda row: (row["choice"]["RW9"]["V"], row["choice"]["RW9"]["age"]))
    for i, row in enumerate(matched, start=1):
        row["set_id"] = i
        row["gallery_page"] = (i - 1) // 6 + 1
        row["gallery_column"] = (i - 1) % 6 + 1

    records = []
    for row in matched:
        rec = {
            "set_id": row["set_id"],
            "gallery_page": row["gallery_page"],
            "gallery_column": row["gallery_column"],
            "I_span_A": row["I_span"],
            "profile": "pulsed charge",
        }
        for cell in CELLS:
            hold = row["choice"][cell]
            rec[f"V_{cell}"] = hold["V"]
            rec[f"T_{cell}"] = hold["T"]
            rec[f"age_{cell}"] = hold["age"]
            rec[f"I_{cell}"] = hold["I"]
            rec[f"step_{cell}"] = hold["step_index"]
        records.append(rec)
    table = pd.DataFrame(records)
    if len(table):
        table["V_range"] = table[[f"V_{c}" for c in CELLS]].max(axis=1) - table[[f"V_{c}" for c in CELLS]].min(axis=1)
        table["T_range"] = table[[f"T_{c}" for c in CELLS]].max(axis=1) - table[[f"T_{c}" for c in CELLS]].min(axis=1)
        table["age_range"] = table[[f"age_{c}" for c in CELLS]].max(axis=1) - table[[f"age_{c}" for c in CELLS]].min(axis=1)
        table["figure"] = table["gallery_page"].map(
            lambda page: f"eda/figures/rw11_pulsed_current_gallery/page_{int(page):02d}.png"
        )

    print("Counting random-walk step sequences with the same current rule...")
    n_step = _step_sequence_count()
    print(f"Stepped random-walk sequences within {I_SPAN_A:.2f} A: {n_step}")

    notes = pd.DataFrame({
        "rule": [
            "Initial voltage within 0.02 V, temperature within 1.0 °C, age within 0.02.",
            "Charging current is negative. Pulsed charge holds are about -1 A.",
            "The four hold currents must span at most 0.15 A.",
            "The figure shows 30 s of the preceding rest and 120 s of the charge hold.",
            "A random-walk step is two charge holds separated only by rest.",
            f"Stepped random-walk sequences meeting the same 0.15 A rule: {n_step}.",
            "Rows are sorted by initial voltage, then age. Six sets per figure page.",
        ]
    })
    xlsx = OUTPUT_DIR / "rw11_pulsed_current_gallery.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="pulsed_charge", index=False)
        notes.to_excel(writer, sheet_name="rules", index=False)
    print("Wrote", xlsx, "rows", len(table))
    if table.empty:
        return

    traces = _load_traces(matched)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    n_pages = int(table["gallery_page"].max())
    by_page: dict[int, list[dict]] = {}
    for row in matched:
        by_page.setdefault(row["gallery_page"], []).append(row)
    for page, rows in by_page.items():
        path = FIGURE_DIR / f"page_{page:02d}.png"
        _plot_page(rows, traces, path, page, n_pages)
        print("Wrote", path)


if __name__ == "__main__":
    main()
