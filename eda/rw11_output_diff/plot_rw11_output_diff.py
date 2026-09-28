"""Short-segment check of RW11 voltage and temperature.

Initial voltage, temperature, and age still have to match, and the charging
current over the plotted piece has to match. The piece may be shorter than the
120 s window used earlier: a hold is kept from 15 s up to 60 s, whatever is
actually available. Pulsed-charge holds are sampled along their length, because
the temperature overlap with RW9 is not always at the first sample.

Negative current is charging. Samples that are not charging are not scored.
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
OUT_DIR = Path("/data/battery_RW_NASA/eda/rw11_output_diff")
FIG_DIR = OUT_DIR / "figures"

CELLS = ["RW9", "RW10", "RW11", "RW12"]
OTHERS = ["RW10", "RW11", "RW12"]
COLOR = {"RW9": "#4C78A8", "RW10": "#F58518", "RW11": "#E45756", "RW12": "#54A24B"}
TOL_V, TOL_T, TOL_AGE = 0.02, 1.0, 0.02
I_SPAN_A = 0.15
MIN_S, MAX_S = 15.0, 60.0
V_RMSE_FLAG = 0.03
T_RMSE_FLAG = 0.50

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def _comment(step) -> str:
    raw = step.comment
    if isinstance(raw, np.ndarray):
        raw = "" if raw.size == 0 else raw.flat[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip().lower()


def _on_grid(time_s, values, t0: float, length: float) -> np.ndarray:
    grid = np.arange(0.0, length + 1e-9, 1.0)
    t = np.asarray(time_s, dtype=np.float64) - t0
    y = np.asarray(values, dtype=np.float64)
    out = np.full(grid.shape, np.nan)
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if t.size < 2:
        return out
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, y = t[keep], y[keep]
    if t.size < 2:
        return out
    out = np.interp(grid, t, y)
    out[grid > t[-1] + 1e-6] = np.nan
    return out


def _window(step, t0: float, length: float, age: float, profile: str, step_index: int) -> dict | None:
    time_s = np.asarray(step.time, dtype=np.float64).ravel()
    current = _on_grid(time_s, step.current, t0, length)
    if np.nanmedian(current) >= 0:
        return None
    return {
        "V": float(np.interp(t0, time_s, np.asarray(step.voltage, dtype=np.float64).ravel())),
        "T": float(np.interp(t0, time_s, np.asarray(step.temperature, dtype=np.float64).ravel())),
        "age": age,
        "I": float(np.nanmedian(current)),
        "profile": profile,
        "segment_s": float(length),
        "step_index": step_index,
        "current": current,
        "voltage": _on_grid(time_s, step.voltage, t0, length),
        "temperature": _on_grid(time_s, step.temperature, t0, length),
    }


def _collect(cell: str) -> list[dict]:
    raw = loadmat(str(MATLAB_DIR / f"{cell}.mat"), squeeze_me=True, struct_as_record=False)["data"]
    steps = list(raw.step)
    n_steps = len(steps)
    windows: list[dict] = []
    seen_steps: set[int] = set()

    def age_of(index: int) -> float:
        return index / max(n_steps - 1, 1)

    for index, step in enumerate(steps):
        kind = _comment(step)
        if kind not in {"pulsed charge (charge)", "charge (random walk)"}:
            continue
        time_s = np.asarray(step.time, dtype=np.float64).ravel()
        if time_s.size < 2:
            continue
        duration = float(time_s[-1] - time_s[0])
        if duration < MIN_S:
            continue
        profile = "pulsed" if kind.startswith("pulsed") else "charge"
        # Pulsed holds are long and change temperature, so sample along the hold.
        # A random-walk hold is flat, so the start of the hold is the segment.
        marks = np.arange(0.0, max(duration - MIN_S, 0.0) + 1e-9, 30.0) if profile == "pulsed" else np.array([0.0])
        for mark in marks:
            length = min(MAX_S, duration - float(mark))
            if length < MIN_S:
                continue
            item = _window(step, float(time_s[0] + mark), length, age_of(index), profile, index)
            if item is not None:
                windows.append(item)
                seen_steps.add(index)
    print(f"  {cell}: {len(windows)} short charging segments ({len(seen_steps)} steps)")
    return windows


def _match(tables: dict[str, list[dict]]) -> list[dict]:
    scales = np.array([TOL_V, TOL_T, TOL_AGE])
    trees = {
        cell: cKDTree(np.array([[w["V"], w["T"], w["age"]] for w in tables[cell]], dtype=float) / scales)
        for cell in OTHERS
    }
    candidates = []
    for i, ref in enumerate(tables["RW9"]):
        query = np.array([ref["V"], ref["T"], ref["age"]]) / scales
        partners: dict[str, int] = {}
        gap = 0.0
        ok = True
        for cell in OTHERS:
            hits = trees[cell].query_ball_point(query, r=1.0, p=np.inf)
            close = []
            for hit in hits:
                other = tables[cell][hit]
                if other["profile"] != ref["profile"]:
                    continue
                this_gap = abs(ref["I"] - other["I"])
                state = abs(np.array([other["V"], other["T"], other["age"]]) / scales - query).max()
                close.append((this_gap, state, hit))
            if not close:
                ok = False
                break
            this_gap, _, hit = min(close)
            partners[cell] = hit
            gap = max(gap, this_gap)
        if not ok or gap > I_SPAN_A:
            continue
        dist = sum(
            abs(ref["V"] - tables[cell][partners[cell]]["V"]) / TOL_V
            + abs(ref["T"] - tables[cell][partners[cell]]["T"]) / TOL_T
            + abs(ref["age"] - tables[cell][partners[cell]]["age"]) / TOL_AGE
            for cell in OTHERS
        )
        candidates.append((dist, i, partners))
    candidates.sort()
    used = {cell: set() for cell in CELLS}
    rows = []
    for dist, i, partners in candidates:
        if i in used["RW9"] or any(partners[cell] in used[cell] for cell in OTHERS):
            continue
        used["RW9"].add(i)
        choice = {"RW9": tables["RW9"][i]}
        for cell in OTHERS:
            used[cell].add(partners[cell])
            choice[cell] = tables[cell][partners[cell]]
        rows.append({"distance": dist, "choice": choice})
    return rows


def _score(row: dict) -> dict:
    length = int(min(w["current"].size for w in row["choice"].values()))
    current = np.vstack([row["choice"][c]["current"][:length] for c in CELLS])
    voltage = np.vstack([row["choice"][c]["voltage"][:length] for c in CELLS])
    temperature = np.vstack([row["choice"][c]["temperature"][:length] for c in CELLS])
    agree = np.isfinite(current).all(axis=0) & (current < 0).all(axis=0)
    agree &= (current.max(axis=0) - current.min(axis=0)) <= I_SPAN_A
    others = [CELLS.index(c) for c in OTHERS]

    def _rmse(stack: np.ndarray) -> float:
        if agree.sum() < 5:
            return np.nan
        ref = stack[others][:, agree].mean(axis=0)
        err = stack[CELLS.index("RW11"), agree] - ref
        return float(np.sqrt(np.mean(err ** 2)))

    ref = row["choice"]["RW9"]
    v_rmse = _rmse(voltage)
    t_rmse = _rmse(temperature)
    flag = "RW11_OUTPUT_DIFFERENT" if (
        (np.isfinite(v_rmse) and v_rmse > V_RMSE_FLAG) or (np.isfinite(t_rmse) and t_rmse > T_RMSE_FLAG)
    ) else "RW11_OUTPUT_SIMILAR"
    if agree.sum() < 5:
        flag = "TOO_SHORT"
    return {
        "segment_s": int(agree.sum()),
        "profile": ref["profile"],
        "V_rmse": v_rmse,
        "T_rmse": t_rmse,
        "flag": flag,
        "current": current,
        "voltage": voltage,
        "temperature": temperature,
        "agree": agree,
    }


def _plot_pages(rows: list[dict]) -> None:
    # Show the largest RW11 voltage differences and a few similar cases.
    usable = [r for r in rows if r["score"]["flag"] != "TOO_SHORT"]
    usable.sort(key=lambda r: (-(r["score"]["V_rmse"] if np.isfinite(r["score"]["V_rmse"]) else -1), r["score"]["profile"]))
    different = [r for r in usable if r["score"]["flag"] == "RW11_OUTPUT_DIFFERENT"]
    similar = [r for r in usable if r["score"]["flag"] == "RW11_OUTPUT_SIMILAR"]
    show = different[:36] + similar[:12]
    if not show:
        show = usable[:12]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pages = [show[i:i + 6] for i in range(0, len(show), 6)]
    for page, chunk in enumerate(pages, start=1):
        n = len(chunk)
        fig, axes = plt.subplots(3, n, figsize=(3.2 * n, 8.4), sharex=False)
        if n == 1:
            axes = np.array(axes).reshape(3, 1)
        for col, row in enumerate(chunk):
            score = row["score"]
            grid = np.arange(score["current"].shape[1])
            ref = row["choice"]["RW9"]
            for r, (key, ylabel, ylim) in enumerate([
                ("current", "Current (A)", (-4.6, 0.3)),
                ("voltage", "Voltage (V)", None),
                ("temperature", "Temperature (°C)", None),
            ]):
                ax = axes[r, col]
                for i, cell in enumerate(CELLS):
                    ax.plot(grid, score[key][i], color=COLOR[cell], lw=1.7 if cell == "RW11" else 1.05,
                            label=cell if col == 0 and r == 0 else None)
                ax.set_xlim(0, max(grid[-1], 1) if grid.size else 1)
                if ylim:
                    ax.set_ylim(*ylim)
                if col == 0:
                    ax.set_ylabel(ylabel)
                if r == 2:
                    ax.set_xlabel("Time (s)")
                if r == 0:
                    ax.set_title(
                        f"{score['profile']}  set {row['set_id']}\n"
                        f"V {ref['V']:.3f}  T {ref['T']:.1f}  age {ref['age']:.3f}\n"
                        f"scored {score['segment_s']} s\n"
                        f"RW11 V RMSE {score['V_rmse']:.3f} V\n"
                        f"RW11 T RMSE {score['T_rmse']:.2f} °C",
                        fontsize=7.5,
                    )
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.suptitle(
            f"RW11 output on short matched charging segments — page {page} of {len(pages)}\n"
            "Same initial V, T, age, and charging current. Segment length is whatever the four cells share.",
            y=1.12, fontsize=12,
        )
        fig.tight_layout()
        path = FIG_DIR / f"page_{page:02d}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print("Wrote", path)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    v = np.array([r["score"]["V_rmse"] for r in usable], dtype=float)
    t = np.array([r["score"]["T_rmse"] for r in usable], dtype=float)
    axes[0].hist(v[np.isfinite(v)], bins=30, color="#E45756")
    axes[1].hist(t[np.isfinite(t)], bins=30, color="#F58518")
    axes[0].axvline(V_RMSE_FLAG, color="k", ls="--", lw=1)
    axes[1].axvline(T_RMSE_FLAG, color="k", ls="--", lw=1)
    axes[0].set_xlabel("RW11 voltage RMSE (V)")
    axes[1].set_xlabel("RW11 temperature RMSE (°C)")
    for ax in axes:
        ax.set_ylabel("Segments")
    fig.suptitle("RW11 output difference where initial state and charging current match")
    fig.tight_layout()
    hist = FIG_DIR / "rw11_output_rmse_hist.png"
    fig.savefig(hist, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", hist)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Collecting short charging segments...")
    tables = {cell: _collect(cell) for cell in CELLS}
    print("Matching...")
    matched = _match(tables)
    print(f"Matched short segments: {len(matched)}")
    records = []
    for i, row in enumerate(matched, start=1):
        row["set_id"] = i
        row["score"] = _score(row)
        score = row["score"]
        rec = {
            "set_id": i,
            "profile": score["profile"],
            "segment_s": score["segment_s"],
            "RW11_voltage_rmse_V": score["V_rmse"],
            "RW11_temperature_rmse_C": score["T_rmse"],
            "RW11_output": score["flag"],
        }
        for cell in CELLS:
            w = row["choice"][cell]
            rec[f"V_{cell}"] = w["V"]
            rec[f"T_{cell}"] = w["T"]
            rec[f"age_{cell}"] = w["age"]
            rec[f"I_{cell}"] = w["I"]
            rec[f"step_{cell}"] = w["step_index"]
        records.append(rec)
    table = pd.DataFrame(records)
    xlsx = OUT_DIR / "rw11_output_diff.xlsx"
    notes = pd.DataFrame({
        "rule": [
            "Initial voltage within 0.02 V, temperature within 1.0 °C, age within 0.02.",
            "Charging current is negative. The four currents span at most 0.15 A on the scored samples.",
            "Segment length is 15 to 60 s, limited by the shortest of the four cells.",
            "Pulsed-charge holds are sampled every 30 s, because temperature changes along the hold.",
            "RW11_OUTPUT_DIFFERENT means voltage RMSE above 0.03 V or temperature RMSE above 0.50 °C versus the mean of RW9, RW10, and RW12.",
            "Figures show the largest voltage differences first, then similar cases. The workbook lists every match.",
        ]
    })
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="short_segments", index=False)
        notes.to_excel(writer, sheet_name="rules", index=False)
    print("Wrote", xlsx, "rows", len(table))
    if table.empty:
        return
    counts = table.groupby(["profile", "RW11_output"]).size()
    print(counts.to_string())
    summary = OUT_DIR / "summary.txt"
    n = len(table)
    n_diff = int((table["RW11_output"] == "RW11_OUTPUT_DIFFERENT").sum())
    n_sim = int((table["RW11_output"] == "RW11_OUTPUT_SIMILAR").sum())
    scored = table[table["RW11_output"] != "TOO_SHORT"]
    text = (
        f"Short charging segments with the same initial voltage, temperature, age, and charging current: {n}.\n"
        f"RW11 voltage or temperature differs on {n_diff} of them. It stays within the thresholds on {n_sim}.\n"
        f"Median RW11 voltage RMSE {scored['RW11_voltage_rmse_V'].median():.4f} V. "
        f"Median RW11 temperature RMSE {scored['RW11_temperature_rmse_C'].median():.3f} °C.\n"
        "This compares the cell response on the matched piece. It does not identify a cause.\n"
    )
    summary.write_text(text, encoding="utf-8")
    print(text)
    _plot_pages(matched)


if __name__ == "__main__":
    main()
