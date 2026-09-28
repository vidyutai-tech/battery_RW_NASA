"""Extra trajectory figures for matched RW9–RW12 charge steps.

Reads the matched-instance table and redraws a spread of RW11_DIFFERENT and
RW11_SIMILAR cases. Alignment matches the notebook: time is relative to the
first sample of the charge step, linearly resampled at 1 s, and plotted for
the first 120 s. Samples past the end of a step stay blank.
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
OUTPUT_DIR = Path("/data/battery_RW_NASA/eda/outputs")
FIGURE_DIR = Path("/data/battery_RW_NASA/eda/figures")
MATCHES_CSV = OUTPUT_DIR / "rw11_matched_instances.csv"

CELLS = ["RW9", "RW10", "RW11", "RW12"]
CELL_COLOR = {"RW9": "#4C78A8", "RW10": "#F58518", "RW11": "#E45756", "RW12": "#54A24B"}
FUTURE_WINDOW_S = 120.0
RESAMPLE_DT_S = 1.0
GRID = np.arange(0.0, FUTURE_WINDOW_S + 1e-9, RESAMPLE_DT_S)

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


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


def _pick(frame: pd.DataFrame, flag: str, per_band: int = 2) -> pd.DataFrame:
    """Two cases in each voltage third: a mild one and an extreme one."""
    part = frame[(frame["RW11_flag"] == flag) & (frame["n_overlap_samples"] >= 90)].copy()
    window_cols = [f"I_mean_window_{cell}" for cell in CELLS]
    part["I_span"] = part[window_cols].max(axis=1) - part[window_cols].min(axis=1)
    # SIMILAR must mean all four currents agree, not only that RW11 is near their average.
    if flag == "RW11_SIMILAR":
        part = part[part["I_span"].abs() <= 0.35]
    edges = part["V_initial_RW11"].quantile([0.0, 1 / 3, 2 / 3, 1.0]).to_numpy()
    chosen = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = part[(part["V_initial_RW11"] >= lo) & (part["V_initial_RW11"] <= hi)].copy()
        if band.empty:
            continue
        band["tight"] = band["V_range"] / 0.02 + band["T_range"] / 1.0 + band["age_range"] / 0.02
        # Keep the tighter half of the band so the initial conditions really are close.
        band = band.nsmallest(max(8, len(band) // 2), "tight")
        if flag == "RW11_DIFFERENT":
            order = [
                band.iloc[(band["RW11_current_rmse_A"] - 0.75).abs().argmin()],
                band.loc[band["RW11_current_rmse_A"].idxmax()],
            ]
        else:
            order = [
                band.loc[band["RW11_current_rmse_A"].idxmin()],
                band.iloc[(band["RW11_current_rmse_A"] - 0.15).abs().argmin()],
            ]
        for row in order[:per_band]:
            if int(row["match_id"]) not in {int(r["match_id"]) for r in chosen}:
                chosen.append(row)
    out = pd.DataFrame(chosen)
    return out.sort_values("V_initial_RW11").reset_index(drop=True)


def _load_selected(selected: pd.DataFrame) -> dict[int, dict[str, dict[str, np.ndarray]]]:
    needed = {cell: set(selected[f"{cell}_index"].astype(int)) for cell in CELLS}
    found: dict[str, dict[int, dict[str, np.ndarray]]] = {cell: {} for cell in CELLS}
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
        print(f"  {cell}: loaded {len(found[cell])} selected steps")
    traces = {}
    for _, row in selected.iterrows():
        match_id = int(row["match_id"])
        traces[match_id] = {
            cell: found[cell][int(row[f"{cell}_index"])] for cell in CELLS
        }
    return traces


def _plot_case(row: pd.Series, traces: dict, path: Path) -> None:
    match_id = int(row["match_id"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = [
        (axes[0, 0], "current", "Current (A)"),
        (axes[0, 1], "voltage", "Voltage (V)"),
        (axes[1, 0], "temperature", "Temperature (°C)"),
    ]
    for ax, key, ylabel in panels:
        for cell in CELLS:
            ax.plot(
                GRID,
                traces[match_id][cell][key],
                color=CELL_COLOR[cell],
                lw=1.8 if cell == "RW11" else 1.15,
                label=cell,
            )
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, FUTURE_WINDOW_S)
        ax.set_xlabel("Time from charge start (s)")
        ax.legend(frameon=False, fontsize=8)
    voltages = [row[f"V_initial_{cell}"] for cell in CELLS]
    temps = [row[f"T_initial_{cell}"] for cell in CELLS]
    x = np.arange(len(CELLS))
    axes[1, 1].bar(x, voltages, color=[CELL_COLOR[c] for c in CELLS])
    ax_t = axes[1, 1].twinx()
    ax_t.plot(x, temps, color="k", ls="--", marker="o", label="T initial")
    axes[1, 1].set_xticks(x, CELLS)
    axes[1, 1].set_ylabel("Initial voltage (V)")
    ax_t.set_ylabel("Initial temperature (°C)")
    axes[1, 1].set_title(
        f"Initial V and T\nage {row['Age_initial_RW9']:.3f} / {row['Age_initial_RW10']:.3f} / "
        f"{row['Age_initial_RW11']:.3f} / {row['Age_initial_RW12']:.3f}"
    )
    axes[1, 1].grid(False)
    fig.suptitle(
        f"match {match_id} | {row['RW11_flag']} | "
        f"V {row['V_initial_RW11']:.3f} V | T {row['T_initial_RW11']:.2f} °C | "
        f"age {row['Age_initial_RW11']:.3f}\n"
        f"current RMSE {row['RW11_current_rmse_A']:.3f} A | "
        f"mean-current gap {row['RW11_mean_current_gap_A']:+.2f} A | "
        f"V range {row['V_range']*1000:.1f} mV | T range {row['T_range']:.2f} °C"
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_gallery(rows: pd.DataFrame, traces: dict, path: Path, title: str) -> None:
    n = len(rows)
    fig, axes = plt.subplots(3, n, figsize=(3.1 * n, 8.2), sharex=True)
    keys = [("current", "Current (A)"), ("voltage", "Voltage (V)"), ("temperature", "Temperature (°C)")]
    for col, (_, row) in enumerate(rows.iterrows()):
        match_id = int(row["match_id"])
        for r, (key, ylabel) in enumerate(keys):
            ax = axes[r, col]
            for cell in CELLS:
                ax.plot(
                    GRID,
                    traces[match_id][cell][key],
                    color=CELL_COLOR[cell],
                    lw=1.6 if cell == "RW11" else 1.0,
                    label=cell if col == 0 and r == 0 else None,
                )
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(0, FUTURE_WINDOW_S)
            if r == 2:
                ax.set_xlabel("Time (s)")
            if r == 0:
                ax.set_title(
                    f"match {match_id}\n"
                    f"V {row['V_initial_RW11']:.3f} V\n"
                    f"T {row['T_initial_RW11']:.1f} °C  age {row['Age_initial_RW11']:.3f}\n"
                    f"I RMSE {row['RW11_current_rmse_A']:.2f} A",
                    fontsize=8,
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.08, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    matches = pd.read_csv(MATCHES_CSV)
    different = _pick(matches, "RW11_DIFFERENT")
    similar = _pick(matches, "RW11_SIMILAR")
    selected = pd.concat([different, similar], ignore_index=True)
    print(f"Selected {len(different)} different and {len(similar)} similar matches")
    traces = _load_selected(selected)

    diff_dir = FIGURE_DIR / "rw11_different"
    sim_dir = FIGURE_DIR / "rw11_similar"
    diff_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)

    for _, row in different.iterrows():
        _plot_case(row, traces, diff_dir / f"match_{int(row['match_id'])}.png")
    for _, row in similar.iterrows():
        _plot_case(row, traces, sim_dir / f"match_{int(row['match_id'])}.png")

    _plot_gallery(
        different,
        traces,
        FIGURE_DIR / "rw11_different_gallery.png",
        "RW11_DIFFERENT — comparable initial V, T, and age, then the next 120 s",
    )
    _plot_gallery(
        similar,
        traces,
        FIGURE_DIR / "rw11_similar_gallery.png",
        "RW11_SIMILAR — comparable initial V, T, and age, then the next 120 s",
    )

    keep = [
        "match_id", "RW11_flag", "V_initial_RW11", "T_initial_RW11", "Age_initial_RW11",
        "V_range", "T_range", "age_range", "RW11_current_rmse_A", "RW11_mean_current_gap_A",
        "RW11_voltage_rmse_V", "RW11_temperature_rmse_C",
    ]
    index = selected[keep].copy()
    index["figure"] = index.apply(
        lambda r: f"eda/figures/rw11_{'different' if r['RW11_flag']=='RW11_DIFFERENT' else 'similar'}/match_{int(r['match_id'])}.png",
        axis=1,
    )
    index_path = OUTPUT_DIR / "rw11_example_cases.csv"
    index.to_csv(index_path, index=False)
    print("Wrote", index_path)
    print(index.round(3).to_string(index=False))


def _pick_same_current(matches: pd.DataFrame) -> pd.DataFrame:
    """Matches whose four future currents already lie on one level.

    Window-mean currents must span at most 0.15 A. One case is kept at each
    dataset level, plus a second 0.75 A case at a different voltage.
    """
    part = matches[matches["n_overlap_samples"] >= 90].copy()
    window_cols = [f"I_mean_window_{cell}" for cell in CELLS]
    # Charging current is stored negative in the Matlab files. A positive
    # sample is not a charge and is excluded from this gallery.
    part = part[(part[window_cols] < 0).all(axis=1)]
    part["I_span"] = (part[window_cols].max(axis=1) - part[window_cols].min(axis=1)).abs()
    part = part[part["I_span"] <= 0.15]
    part["I_abs"] = part[window_cols].abs().mean(axis=1)
    part["level"] = (part["I_abs"] / 0.25).round() * 0.25
    part["tight"] = part["V_range"] / 0.02 + part["T_range"] / 1.0 + part["age_range"] / 0.02
    chosen = []
    seen = set()
    plan = [(0.75, 2), (1.50, 1), (2.25, 1), (3.00, 1), (3.75, 1)]
    for level, n_keep in plan:
        band = part[(part["level"] - level).abs() < 0.05].sort_values(
            ["RW11_voltage_rmse_V", "tight"]
        )
        if band.empty:
            continue
        if n_keep == 1:
            picks = [band.iloc[0]]
        else:
            half = band.head(max(n_keep, len(band) // 2))
            low = half.loc[half["V_initial_RW11"].idxmin()]
            high = half.loc[half["V_initial_RW11"].idxmax()]
            picks = [low] if int(low["match_id"]) == int(high["match_id"]) else [low, high]
        for row in picks:
            match_id = int(row["match_id"])
            if match_id not in seen:
                seen.add(match_id)
                chosen.append(row)
    out = pd.DataFrame(chosen)
    return out.sort_values(["level", "V_initial_RW11"]).reset_index(drop=True)


def _plot_same_current_gallery(rows: pd.DataFrame, traces: dict, path: Path, title: str | None = None) -> None:
    n = len(rows)
    fig, axes = plt.subplots(3, n, figsize=(3.15 * n, 8.4), sharex=True)
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    keys = [("current", "Current (A)"), ("voltage", "Voltage (V)"), ("temperature", "Temperature (°C)")]
    # One current scale for every column, so a shared level looks shared
    # and a different level is obvious from its height.
    i_lo, i_hi = -4.3, -0.4
    for col, (_, row) in enumerate(rows.iterrows()):
        match_id = int(row["match_id"])
        level = float(row["level"])
        for r, (key, ylabel) in enumerate(keys):
            ax = axes[r, col]
            for cell in CELLS:
                ax.plot(
                    GRID,
                    traces[match_id][cell][key],
                    color=CELL_COLOR[cell],
                    lw=1.7 if cell == "RW11" else 1.05,
                    label=cell if col == 0 and r == 0 else None,
                )
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_xlim(0, FUTURE_WINDOW_S)
            if r == 0:
                ax.set_ylim(i_lo, i_hi)
                ax.set_title(
                    f"|I| ≈ {level:.2f} A\n"
                    f"match {match_id}\n"
                    f"V {row['V_initial_RW11']:.3f} V   T {row['T_initial_RW11']:.1f} °C\n"
                    f"age {row['Age_initial_RW11']:.3f}   I span {row['I_span']:.3f} A",
                    fontsize=8,
                )
            if r == 2:
                ax.set_xlabel("Time (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        title
        or (
            "Same initial V, T, and age, and the same charging current (I < 0 only)\n"
            "Positive current is omitted. Current axis is shared. Columns are the closest voltage agreement at each level."
        ),
        y=1.10,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def same_current() -> None:
    matches = pd.read_csv(MATCHES_CSV)
    selected = _pick_same_current(matches)
    print(f"Selected {len(selected)} same-current matches")
    traces = _load_selected(selected)
    # Drop any sample that is not charging. Voltage and temperature at that
    # instant are dropped with it so a discharge blip cannot pull the curves apart.
    for match_traces in traces.values():
        for cell in CELLS:
            series = match_traces[cell]
            charging = series["current"] < 0
            for key in ("current", "voltage", "temperature"):
                series[key] = np.where(charging, series[key], np.nan)
    out_dir = FIGURE_DIR / "rw11_same_current"
    out_dir.mkdir(parents=True, exist_ok=True)
    for _, row in selected.iterrows():
        _plot_case(row, traces, out_dir / f"match_{int(row['match_id'])}.png")
    gallery = FIGURE_DIR / "rw11_same_current_gallery.png"
    _plot_same_current_gallery(selected, traces, gallery)
    keep = [
        "match_id", "level", "I_span", "V_initial_RW11", "T_initial_RW11", "Age_initial_RW11",
        "V_range", "T_range", "age_range", "RW11_current_rmse_A",
        "RW11_voltage_rmse_V", "RW11_temperature_rmse_C",
    ]
    table = selected[keep].copy()
    table["figure"] = table["match_id"].map(lambda i: f"eda/figures/rw11_same_current/match_{int(i)}.png")
    table_path = OUTPUT_DIR / "rw11_same_current_cases.csv"
    table.to_csv(table_path, index=False)
    print("Wrote", gallery)
    print("Wrote", table_path)
    print(table.round(3).to_string(index=False))


def _same_current_table(matches: pd.DataFrame) -> pd.DataFrame:
    """Every matched quadruple whose charging current is the same on all four cells."""
    window_cols = [f"I_mean_window_{cell}" for cell in CELLS]
    part = matches[matches["n_overlap_samples"] >= 90].copy()
    part = part[(part[window_cols] < 0).all(axis=1)]
    part["I_span"] = (part[window_cols].max(axis=1) - part[window_cols].min(axis=1)).abs()
    part = part[part["I_span"] <= 0.15].copy()
    part["level"] = (part[window_cols].abs().mean(axis=1) / 0.25).round() * 0.25
    part = part.sort_values(["level", "V_initial_RW11", "match_id"]).reset_index(drop=True)
    part.insert(0, "set_id", np.arange(1, len(part) + 1))
    part.insert(1, "gallery_page", (part["set_id"] - 1) // 6 + 1)
    part.insert(2, "gallery_column", (part["set_id"] - 1) % 6 + 1)
    return part


def same_current_r1() -> None:
    matches = pd.read_csv(MATCHES_CSV)
    selected = _same_current_table(matches)
    print(f"Same initial state and same charging current: {len(selected)}")

    export_cols = [
        "set_id", "gallery_page", "gallery_column", "match_id", "level", "I_span",
        "V_initial_RW9", "V_initial_RW10", "V_initial_RW11", "V_initial_RW12", "V_range",
        "T_initial_RW9", "T_initial_RW10", "T_initial_RW11", "T_initial_RW12", "T_range",
        "Age_initial_RW9", "Age_initial_RW10", "Age_initial_RW11", "Age_initial_RW12", "age_range",
        "I_mean_window_RW9", "I_mean_window_RW10", "I_mean_window_RW11", "I_mean_window_RW12",
        "RW11_current_rmse_A", "RW11_voltage_rmse_V", "RW11_temperature_rmse_C",
        "RW9_index", "RW10_index", "RW11_index", "RW12_index",
        "duration_s_RW9", "duration_s_RW10", "duration_s_RW11", "duration_s_RW12",
    ]
    table = selected[export_cols].copy()
    table["figure"] = table["gallery_page"].map(
        lambda page: f"eda/figures/rw11_same_current_gallery_r1/page_{int(page):02d}.png"
    )
    xlsx = OUTPUT_DIR / "rw11_same_current_gallery_r1.xlsx"
    notes = pd.DataFrame({
        "rule": [
            "Initial voltage, temperature, and age already match (0.02 V, 1.0 °C, 0.02 age).",
            "Charging current is negative in the Matlab files. All four window means are < 0.",
            "The four window-mean currents span at most 0.15 A.",
            "Each case has at least 90 s of overlap. Plots show the first 120 s.",
            "Samples with current >= 0 are omitted from the figures.",
            "Rows are sorted by charging-current level, then initial voltage.",
            "Each figure page holds six sets. gallery_page and gallery_column point at the panel.",
        ]
    })
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="same_current", index=False)
        notes.to_excel(writer, sheet_name="rules", index=False)
    print("Wrote", xlsx, "rows", len(table))

    traces = _load_selected(selected)
    for match_traces in traces.values():
        for cell in CELLS:
            series = match_traces[cell]
            charging = series["current"] < 0
            for key in ("current", "voltage", "temperature"):
                series[key] = np.where(charging, series[key], np.nan)

    out_dir = FIGURE_DIR / "rw11_same_current_gallery_r1"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_pages = int(selected["gallery_page"].max())
    for page, rows in selected.groupby("gallery_page"):
        path = out_dir / f"page_{int(page):02d}.png"
        _plot_same_current_gallery(
            rows,
            traces,
            path,
            title=(
                f"Same initial V, T, age, and the same charging current — page {int(page)} of {n_pages}\n"
                "I < 0 only. Current axis is shared. Six sets per page."
            ),
        )
        print("Wrote", path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "same-current":
        same_current()
    elif len(sys.argv) > 1 and sys.argv[1] == "same-current-r1":
        same_current_r1()
    else:
        main()
