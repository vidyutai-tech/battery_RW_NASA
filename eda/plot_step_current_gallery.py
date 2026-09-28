"""Gallery of multi-step charge bursts with a similar current sequence.

One Matlab "charge (random walk)" step is a flat current. A step profile here
is a burst: two or more of those holds separated only by "rest (random walk)",
stopping at the next discharge. Bursts are matched on the initial voltage,
temperature, and age of the first hold, then kept only when the current level
changes by at least 0.4 A and the first levels agree across RW9–RW12.
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
FIGURE_DIR = Path("/data/battery_RW_NASA/eda/figures")
OUTPUT_DIR = Path("/data/battery_RW_NASA/eda/outputs")

CELLS = ["RW9", "RW10", "RW11", "RW12"]
OTHERS = ["RW10", "RW11", "RW12"]
CELL_COLOR = {"RW9": "#4C78A8", "RW10": "#F58518", "RW11": "#E45756", "RW12": "#54A24B"}

TOL_V = 0.02
TOL_T = 1.0
TOL_AGE = 0.02
LEVEL_TOL_A = 0.30          # each corresponding hold must agree within this
MIN_STEP_A = 0.40           # |I| must change by at least this inside the burst
MIN_HOLD_S = 10.0
N_LEVELS = 3                # compare this many holds
PLOT_S = 620.0              # long enough for two 300 s holds

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def _comment(step) -> str:
    raw = step.comment
    if isinstance(raw, np.ndarray):
        raw = "" if raw.size == 0 else raw.flat[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip().lower()


def _charge_bursts(cell: str) -> list[dict]:
    raw = loadmat(str(MATLAB_DIR / f"{cell}.mat"), squeeze_me=True, struct_as_record=False)["data"]
    steps = list(raw.step)
    n_steps = len(steps)
    bursts: list[dict] = []
    pieces: list[tuple[str, int]] = []

    def flush() -> None:
        charges = [(k, i) for k, i in pieces if k == "charge"]
        if len(charges) < 2:
            pieces.clear()
            return
        holds = []
        for _, index in charges:
            step = steps[index]
            cur = np.asarray(step.current, dtype=np.float64).ravel()
            tim = np.asarray(step.time, dtype=np.float64).ravel()
            if cur.size < 2 or tim.size < 2:
                continue
            duration = float(tim[-1] - tim[0])
            if duration < MIN_HOLD_S:
                continue
            finite = cur[np.isfinite(cur)]
            if finite.size == 0:
                continue
            holds.append((index, float(np.median(finite)), duration))
        if len(holds) >= 2 and abs(holds[0][1] - holds[1][1]) >= MIN_STEP_A:
            first = steps[holds[0][0]]
            voltage = np.asarray(first.voltage, dtype=np.float64).ravel()
            temperature = np.asarray(first.temperature, dtype=np.float64).ravel()
            bursts.append({
                "cell": cell,
                "local": len(bursts),
                "V": float(voltage[0]),
                "T": float(temperature[0]),
                "age": holds[0][0] / max(n_steps - 1, 1),
                "levels": [h[1] for h in holds[:N_LEVELS]],
                "durations": [h[2] for h in holds[:N_LEVELS]],
                "pieces": list(pieces),
            })
        pieces.clear()

    for index, step in enumerate(steps):
        kind = _comment(step)
        if kind == "charge (random walk)":
            pieces.append(("charge", index))
        elif kind == "rest (random walk)" and pieces:
            pieces.append(("rest", index))
        else:
            if pieces:
                flush()
    if pieces:
        flush()
    print(f"  {cell}: {len(bursts)} stepped bursts")
    return bursts


def _match(tables: dict[str, list[dict]]) -> pd.DataFrame:
    scales = np.array([TOL_V, TOL_T, TOL_AGE])
    trees = {}
    for cell in OTHERS:
        feat = np.array([[b["V"], b["T"], b["age"]] for b in tables[cell]], dtype=float) / scales
        trees[cell] = cKDTree(feat)

    candidates = []
    for i, ref in enumerate(tables["RW9"]):
        query = np.array([ref["V"], ref["T"], ref["age"]]) / scales
        partners = {}
        ok = True
        seq_gap = 0.0
        for cell in OTHERS:
            hits = trees[cell].query_ball_point(query, r=1.0, p=np.inf)
            close = []
            for hit in hits:
                other = tables[cell][hit]
                n = min(len(ref["levels"]), len(other["levels"]), 2)
                if n < 2:
                    continue
                gap = max(abs(ref["levels"][k] - other["levels"][k]) for k in range(n))
                state = abs(np.array([other["V"], other["T"], other["age"]]) / scales - query).max()
                close.append((gap, state, hit))
            if not close:
                ok = False
                break
            gap, _, hit = min(close)
            partners[cell] = hit
            seq_gap = max(seq_gap, gap)
        if ok:
            dist = 0.0
            for cell in OTHERS:
                other = tables[cell][partners[cell]]
                dist += abs(ref["V"] - other["V"]) / TOL_V
                dist += abs(ref["T"] - other["T"]) / TOL_T
                dist += abs(ref["age"] - other["age"]) / TOL_AGE
            candidates.append((seq_gap, dist, i, partners))

    candidates.sort()
    used = {cell: set() for cell in CELLS}
    rows = []
    for seq_gap, dist, i, partners in candidates:
        if i in used["RW9"] or any(partners[c] in used[c] for c in OTHERS):
            continue
        used["RW9"].add(i)
        choice = {"RW9": tables["RW9"][i]}
        for cell in OTHERS:
            used[cell].add(partners[cell])
            choice[cell] = tables[cell][partners[cell]]
        rows.append({
            "match_id": len(rows) + 1,
            "distance": dist,
            "I_span": seq_gap,
            "choice": choice,
        })
    return pd.DataFrame(rows)


def _trace(steps, pieces: list[tuple[str, int]]) -> dict[str, np.ndarray]:
    times, current, voltage, temperature = [], [], [], []
    t0 = None
    for _, index in pieces:
        step = steps[index]
        tim = np.asarray(step.time, dtype=np.float64).ravel()
        cur = np.asarray(step.current, dtype=np.float64).ravel()
        vol = np.asarray(step.voltage, dtype=np.float64).ravel()
        tmp = np.asarray(step.temperature, dtype=np.float64).ravel()
        n = min(tim.size, cur.size, vol.size, tmp.size)
        if n == 0:
            continue
        if t0 is None:
            t0 = float(tim[0])
        times.append(tim[:n] - t0)
        current.append(cur[:n])
        voltage.append(vol[:n])
        temperature.append(tmp[:n])
    if not times:
        empty = np.array([])
        return {"time": empty, "current": empty, "voltage": empty, "temperature": empty}
    return {
        "time": np.concatenate(times),
        "current": np.concatenate(current),
        "voltage": np.concatenate(voltage),
        "temperature": np.concatenate(temperature),
    }


def _load_traces(selected: list[dict]) -> dict[int, dict]:
    needed = {cell: set() for cell in CELLS}
    for row in selected:
        for cell in CELLS:
            for _, index in row["choice"][cell]["pieces"]:
                needed[cell].add(index)
    loaded = {cell: {} for cell in CELLS}
    raw_steps = {}
    for cell in CELLS:
        raw = loadmat(str(MATLAB_DIR / f"{cell}.mat"), squeeze_me=True, struct_as_record=False)["data"]
        raw_steps[cell] = list(raw.step)
        print(f"  {cell}: reading {len(needed[cell])} burst steps")
    traces = {}
    for row in selected:
        traces[row["match_id"]] = {
            cell: _trace(raw_steps[cell], row["choice"][cell]["pieces"]) for cell in CELLS
        }
    return traces


def _fmt_levels(levels: list[float]) -> str:
    return " → ".join(f"{abs(v):.2f}" for v in levels[:3])


def _pick_diverse(matched: pd.DataFrame, n: int = 6) -> list[dict]:
    rows = matched.to_dict("records")
    rows.sort(key=lambda r: (r["I_span"], r["distance"]))
    chosen = []
    seen = set()
    for row in rows:
        levels = row["choice"]["RW9"]["levels"][:2]
        key = tuple(np.round(np.abs(levels) * 2) / 2)
        if key in seen:
            continue
        seen.add(key)
        chosen.append(row)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        used = {r["match_id"] for r in chosen}
        for row in rows:
            if row["match_id"] in used:
                continue
            chosen.append(row)
            if len(chosen) >= n:
                break
    return chosen


def _plot(chosen: list[dict], traces: dict) -> None:
    n = len(chosen)
    fig, axes = plt.subplots(3, n, figsize=(3.3 * n, 9.2), sharex=True)
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    for col, row in enumerate(chosen):
        match_id = row["match_id"]
        ref = row["choice"]["RW9"]
        for r, (key, ylabel) in enumerate([
            ("current", "Current (A)"),
            ("voltage", "Voltage (V)"),
            ("temperature", "Temperature (°C)"),
        ]):
            ax = axes[r, col]
            for cell in CELLS:
                tr = traces[match_id][cell]
                ax.plot(
                    tr["time"], tr[key],
                    color=CELL_COLOR[cell],
                    lw=1.5 if cell == "RW11" else 1.0,
                    label=cell if col == 0 and r == 0 else None,
                )
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(0, PLOT_S)
            if r == 0:
                ax.set_ylim(-4.6, 0.3)
                level_lines = "\n".join(
                    f"{cell} { _fmt_levels(row['choice'][cell]['levels'])}"
                    for cell in CELLS
                )
                ax.set_title(
                    f"closest level gap {row['I_span']:.2f} A\n"
                    f"V {ref['V']:.3f} V  T {ref['T']:.1f} °C  age {ref['age']:.3f}\n"
                    f"{level_lines}",
                    fontsize=7,
                )
            if r == 2:
                ax.set_xlabel("Time from burst start (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "Closest stepped charge bursts at a matched initial V, T, and age\n"
        "A single charge step is flat. These are the next holds in the same burst. Current axis is shared.",
        y=1.12,
        fontsize=12,
    )
    fig.tight_layout()
    path = FIGURE_DIR / "rw11_step_current_gallery.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", path)


def main() -> None:
    print("Extracting stepped charge bursts...")
    tables = {cell: _charge_bursts(cell) for cell in CELLS}
    matched = _match(tables)
    print(f"Stepped bursts with a matched initial V, T, and age: {len(matched)}")
    if len(matched):
        for limit in (0.30, 0.50, 0.75, 1.50):
            print(f"  first two levels within {limit:.2f} A on every cell: {int((matched['I_span'] <= limit).sum())}")
    if matched.empty:
        print("No stepped burst had the same initial state and the same current sequence.")
        return
    chosen = _pick_diverse(matched)
    print(f"Plotting {len(chosen)} different step patterns")
    traces = _load_traces(chosen)
    _plot(chosen, traces)

    records = []
    for row in chosen:
        rec = {
            "match_id": row["match_id"],
            "I_span_A": row["I_span"],
            "RW9_levels_A": _fmt_levels(row["choice"]["RW9"]["levels"]),
        }
        for cell in CELLS:
            burst = row["choice"][cell]
            rec[f"V_{cell}"] = burst["V"]
            rec[f"T_{cell}"] = burst["T"]
            rec[f"age_{cell}"] = burst["age"]
            rec[f"levels_{cell}"] = _fmt_levels(burst["levels"])
        records.append(rec)
    table = pd.DataFrame(records)
    dest = OUTPUT_DIR / "rw11_step_current_cases.csv"
    table.to_csv(dest, index=False)
    print("Wrote", dest)
    print(table.round(3).to_string(index=False))
    print(f"Total exclusive stepped matches available: {len(matched)}")


if __name__ == "__main__":
    main()
