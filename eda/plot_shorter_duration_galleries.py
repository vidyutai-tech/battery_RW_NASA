"""Short-window galleries for every matched initial condition.

Each page has six sets. The window is the first 30 s after the matched
charge sample, resampled at 1 s. Current, voltage, and temperature are
drawn for RW9, RW10, RW11, and RW12. RW11 is the thicker red line.

Groups:
  different              RW11_DIFFERENT
  similar                RW11_SIMILAR
  same_current           four charging currents already on one level
  insufficient_overlap   matched, but the 120 s score did not have 30 samples
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat

MATLAB_DIR = Path(
    "/data/battery_RW_NASA/NASA_RW/dataset/"
    "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/"
    "Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/data/Matlab"
)
MATCHES = Path("/data/battery_RW_NASA/eda/outputs/rw11_matched_instances.csv")
OUT_DIR = Path("/data/battery_RW_NASA/eda/figures/shorter_duration")
INDEX_XLSX = Path("/data/battery_RW_NASA/eda/outputs/shorter_duration_index.xlsx")

CELLS = ["RW9", "RW10", "RW11", "RW12"]
CELL_COLOR = {"RW9": "#4C78A8", "RW10": "#F58518", "RW11": "#E45756", "RW12": "#54A24B"}
WINDOW_S = 30.0
GRID = np.arange(0.0, WINDOW_S + 1e-9, 1.0)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 8})


def _resample(time_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    t = np.asarray(time_s, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    out = np.full(GRID.shape, np.nan, dtype=np.float64)
    if t.size < 2 or y.size < 2:
        return out
    t = t - t[0]
    mask = np.isfinite(t) & np.isfinite(y)
    t, y = t[mask], y[mask]
    if t.size < 2:
        return out
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    t, y = t[keep], y[keep]
    if t.size < 2:
        return out
    out = np.interp(GRID, t, y)
    out[GRID > t[-1] + 1e-6] = np.nan
    return out


def _load(needed: dict[str, set[int]]) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    found = {cell: {} for cell in CELLS}
    for cell in CELLS:
        raw = loadmat(str(MATLAB_DIR / f"{cell}.mat"), squeeze_me=True, struct_as_record=False)["data"]
        steps = list(raw.step)
        for index in needed[cell]:
            step = steps[index]
            time_s = np.asarray(step.time, dtype=np.float64).ravel()
            found[cell][index] = {
                "current": _resample(time_s, np.asarray(step.current, dtype=np.float64).ravel()),
                "voltage": _resample(time_s, np.asarray(step.voltage, dtype=np.float64).ravel()),
                "temperature": _resample(time_s, np.asarray(step.temperature, dtype=np.float64).ravel()),
            }
        print(f"  {cell}: {len(found[cell])} steps")
    return found


def _traces_for(row: pd.Series, found: dict) -> dict[str, dict[str, np.ndarray]]:
    return {cell: found[cell][int(row[f"{cell}_index"])] for cell in CELLS}


def _plot_page(rows: pd.DataFrame, found: dict, path: Path, title: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(3, n, figsize=(3.05 * n, 7.6), sharex=True)
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    keys = [("current", "Current (A)"), ("voltage", "Voltage (V)"), ("temperature", "Temperature (°C)")]
    for col, (_, row) in enumerate(rows.iterrows()):
        series = _traces_for(row, found)
        for r, (key, ylabel) in enumerate(keys):
            ax = axes[r, col]
            for cell in CELLS:
                ax.plot(
                    GRID,
                    series[cell][key],
                    color=CELL_COLOR[cell],
                    lw=1.8 if cell == "RW11" else 1.05,
                    label=cell if col == 0 and r == 0 else None,
                )
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(0, WINDOW_S)
            if r == 0:
                ax.set_ylim(-4.7, 0.3)
                ax.set_title(
                    f"match {int(row['match_id'])}\n"
                    f"V {row['V_initial_RW11']:.3f}  T {row['T_initial_RW11']:.1f} °C\n"
                    f"age {row['Age_initial_RW11']:.3f}\n"
                    f"I rmse {row['RW11_current_rmse_A']:.2f} A\n"
                    f"V rmse {row['RW11_voltage_rmse_V']:.3f} V",
                    fontsize=7.5,
                )
            if r == 2:
                ax.set_xlabel("Time from match (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _pages(frame: pd.DataFrame, found: dict, folder: Path, heading: str) -> pd.DataFrame:
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    n_pages = int(np.ceil(len(frame) / 6))
    for page, start in enumerate(range(0, len(frame), 6), start=1):
        chunk = frame.iloc[start:start + 6]
        path = folder / f"page_{page:03d}.png"
        _plot_page(
            chunk,
            found,
            path,
            f"{heading} — first {WINDOW_S:.0f} s — page {page} of {n_pages}",
        )
        for column, (_, row) in enumerate(chunk.iterrows(), start=1):
            records.append({
                "group": folder.name,
                "page": page,
                "column": column,
                "match_id": int(row["match_id"]),
                "RW11_flag": row["RW11_flag"],
                "V_initial_RW11": row["V_initial_RW11"],
                "T_initial_RW11": row["T_initial_RW11"],
                "Age_initial_RW11": row["Age_initial_RW11"],
                "V_range": row["V_range"],
                "T_range": row["T_range"],
                "age_range": row["age_range"],
                "RW11_current_rmse_A": row["RW11_current_rmse_A"],
                "RW11_voltage_rmse_V": row["RW11_voltage_rmse_V"],
                "RW11_temperature_rmse_C": row["RW11_temperature_rmse_C"],
                "figure": str(path.relative_to(path.parents[3])),
            })
        if page % 20 == 0 or page == n_pages:
            print(f"  {folder.name}: page {page}/{n_pages}")
    return pd.DataFrame(records)


def main() -> None:
    matches = pd.read_csv(MATCHES)
    window_cols = [f"I_mean_window_{c}" for c in CELLS]
    charging = (matches[window_cols] < 0).all(axis=1)
    span = (matches[window_cols].max(axis=1) - matches[window_cols].min(axis=1)).abs()
    same_current = matches[charging & (span <= 0.15) & (matches["n_overlap_samples"] >= 30)].copy()
    same_current["level"] = matches.loc[same_current.index, window_cols].abs().mean(axis=1)
    same_current = same_current.sort_values(["level", "V_initial_RW11", "match_id"])

    groups = {
        "different": matches[matches["RW11_flag"] == "RW11_DIFFERENT"].sort_values(
            "RW11_current_rmse_A", ascending=False
        ),
        "similar": matches[matches["RW11_flag"] == "RW11_SIMILAR"].sort_values(
            "RW11_current_rmse_A", ascending=True
        ),
        "same_current": same_current,
        "insufficient_overlap": matches[matches["RW11_flag"] == "INSUFFICIENT_OVERLAP"].sort_values(
            "n_overlap_samples", ascending=False
        ),
    }

    needed = {cell: set() for cell in CELLS}
    for frame in groups.values():
        for cell in CELLS:
            needed[cell].update(frame[f"{cell}_index"].astype(int))
    print("Loading matched charge steps...")
    found = _load(needed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    headings = {
        "different": "RW11 current differs from RW9, RW10, and RW12",
        "similar": "RW11 current stays near RW9, RW10, and RW12",
        "same_current": "Same charging current on all four cells",
        "insufficient_overlap": "Matched, short overlap, not scored on the 120 s window",
    }
    tables = []
    for name, frame in groups.items():
        print(f"{name}: {len(frame)} sets")
        tables.append(_pages(frame, found, OUT_DIR / name, headings[name]))
    index = pd.concat(tables, ignore_index=True)
    rules = pd.DataFrame({
        "rule": [
            "Initial voltage within 0.02 V, temperature within 1 °C, age within 0.02.",
            "Each panel is the first 30 s after that matched sample, at 1 s spacing.",
            "Current keeps the Matlab sign: negative is charging. The current axis is shared.",
            "Voltage and temperature use the scale of that panel.",
            "RW11 is the red line.",
            "different and similar use the 120 s current flag. insufficient_overlap was not scored.",
            "same_current keeps sets whose four window-mean currents are negative and span at most 0.15 A.",
            "Pages hold six sets. The index gives the page and column for each match.",
        ]
    })
    with pd.ExcelWriter(INDEX_XLSX, engine="openpyxl") as writer:
        index.to_excel(writer, sheet_name="index", index=False)
        rules.to_excel(writer, sheet_name="rules", index=False)
    print("Wrote", INDEX_XLSX, "rows", len(index))


if __name__ == "__main__":
    main()
