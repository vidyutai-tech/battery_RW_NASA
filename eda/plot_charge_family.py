"""Same initial state: is the charging family the same even when the amperes differ?

Family is the shape of the charge burst that contains the matched step.
A burst is one or more random-walk charge holds separated only by rest.

    constant     one current level
    two_step     two levels, at least 0.75 A apart
    three_step   three levels
    multi_step   four or more levels

The level is the hold current rounded to the dataset grid of 0.75 A.
The pattern is that level. The family ignores the ampere values.
Pulsed charge is a separate protocol and is not inside these matched steps.
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
OUT_DIR = Path("/data/battery_RW_NASA/eda/outputs")
FIG_DIR = Path("/data/battery_RW_NASA/eda/figures")
CELLS = ["RW9", "RW10", "RW11", "RW12"]
GRID_A = 0.75


def _comment(step) -> str:
    raw = step.comment
    if isinstance(raw, np.ndarray):
        raw = "" if raw.size == 0 else raw.flat[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip().lower()


def _level(step) -> float | None:
    cur = np.asarray(step.current, dtype=np.float64).ravel()
    tim = np.asarray(step.time, dtype=np.float64).ravel()
    cur = cur[np.isfinite(cur)]
    if cur.size == 0 or tim.size < 2:
        return None
    if float(tim[-1] - tim[0]) < 10.0:
        return None
    return float(round((abs(float(np.median(cur))) / GRID_A)) * GRID_A)


def _family(n_levels: int) -> str:
    if n_levels <= 1:
        return "constant"
    if n_levels == 2:
        return "two_step"
    if n_levels == 3:
        return "three_step"
    return "multi_step"


def _label_cell(cell: str) -> dict[int, dict]:
    raw = loadmat(str(MATLAB_DIR / f"{cell}.mat"), squeeze_me=True, struct_as_record=False)["data"]
    steps = list(raw.step)
    labels: dict[int, dict] = {}
    burst: list[tuple[int, float | None]] = []

    def flush() -> None:
        if not burst:
            return
        levels = [level for _, level in burst if level is not None]
        family = _family(len(set(levels)) if levels else 1)
        for index, level in burst:
            labels[index] = {"family": family, "level_A": level, "n_levels": len(set(levels)) if levels else 1}
        burst.clear()

    for index, step in enumerate(steps):
        kind = _comment(step)
        if kind == "charge (random walk)":
            burst.append((index, _level(step)))
        elif kind == "rest (random walk)" and burst:
            continue
        elif burst:
            flush()
    flush()
    print(f"  {cell}: labeled {len(labels)} charge steps")
    return labels


def main() -> None:
    print("Labeling charge bursts...")
    labels = {cell: _label_cell(cell) for cell in CELLS}
    matches = pd.read_csv(MATCHES)
    for cell in CELLS:
        families, levels = [], []
        for index in matches[f"{cell}_index"].astype(int):
            info = labels[cell].get(index, {"family": "unlabeled", "level_A": np.nan})
            families.append(info["family"])
            levels.append(info["level_A"])
        matches[f"family_{cell}"] = families
        matches[f"level_{cell}"] = levels

    fam_cols = [f"family_{c}" for c in CELLS]
    lvl_cols = [f"level_{c}" for c in CELLS]
    matches["same_family"] = matches[fam_cols].nunique(axis=1) == 1
    matches["same_pattern"] = matches[lvl_cols].nunique(axis=1) == 1
    matches["family_same_pattern_different"] = matches["same_family"] & ~matches["same_pattern"]

    n = len(matches)
    n_family = int(matches["same_family"].sum())
    n_pattern = int(matches["same_pattern"].sum())
    n_family_only = int(matches["family_same_pattern_different"].sum())
    n_both = int((matches["same_family"] & matches["same_pattern"]).sum())
    n_neither = int((~matches["same_family"]).sum())

    summary = pd.DataFrame([
        ["Matched initial conditions", n, 100.0],
        ["Same family and same current level", n_both, 100 * n_both / n],
        ["Same family, different current level", n_family_only, 100 * n_family_only / n],
        ["Same family, either pattern", n_family, 100 * n_family / n],
        ["Different family", n_neither, 100 * n_neither / n],
        ["Same current level, either family", n_pattern, 100 * n_pattern / n],
    ], columns=["result", "matches", "percent"])

    counts = (
        matches.melt(value_vars=fam_cols, var_name="cell", value_name="family")
        .assign(cell=lambda d: d["cell"].str.replace("family_", "", regex=False))
        .groupby(["cell", "family"]).size().unstack(fill_value=0)
    )
    counts = counts.reindex(CELLS)
    for col in ["constant", "two_step", "three_step", "multi_step"]:
        if col not in counts.columns:
            counts[col] = 0
    counts = counts[["constant", "two_step", "three_step", "multi_step"]]

    # What family the other cells have when RW11 is each family.
    rw11_vs = (
        matches.groupby("family_RW11")[fam_cols]
        .apply(lambda g: pd.Series({
            "n": len(g),
            "all_four_same": int(g.nunique(axis=1).eq(1).sum()) if False else int((g.nunique(axis=1) == 1).sum()),
            "RW9_same": int((g["family_RW9"] == g["family_RW11"]).sum()),
            "RW10_same": int((g["family_RW10"] == g["family_RW11"]).sum()),
            "RW12_same": int((g["family_RW12"] == g["family_RW11"]).sum()),
        }))
        .reset_index()
    )

    detail_cols = [
        "match_id", "same_family", "same_pattern", "family_same_pattern_different",
        *fam_cols, *lvl_cols,
        "V_initial_RW11", "T_initial_RW11", "Age_initial_RW11",
        "RW11_flag",
    ]
    detail = matches[detail_cols].copy()

    xlsx = OUT_DIR / "rw11_charge_family.xlsx"
    rules = pd.DataFrame({
        "rule": [
            "A family is the shape of the charge burst, not the ampere value.",
            "A burst is random-walk charge holds separated only by rest, ending at the next discharge.",
            "Levels are rounded to the 0.75 A dataset grid. Holds shorter than 10 s are ignored.",
            "constant = one level, two_step = two, three_step = three, multi_step = four or more.",
            "Same pattern means the matched hold itself is on the same 0.75 A level for all four cells.",
            "These matches already share initial voltage within 0.02 V, temperature within 1 °C, and age within 0.02.",
            "Pulsed charge is a different protocol and is not in this matched set.",
        ]
    })
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        counts.reset_index().to_excel(writer, sheet_name="family_counts", index=False)
        rw11_vs.to_excel(writer, sheet_name="rw11_agreement", index=False)
        detail.to_excel(writer, sheet_name="matches", index=False)
        rules.to_excel(writer, sheet_name="rules", index=False)
    print("Wrote", xlsx)
    print(summary.to_string(index=False))
    print(counts)
    print(rw11_vs.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels_x = ["Same family\nand same level", "Same family,\ndifferent level", "Different family"]
    values = [n_both, n_family_only, n_neither]
    colors = ["#54A24B", "#4C78A8", "#E45756"]
    bars = ax.bar(labels_x, [100 * v / n for v in values], color=colors)
    for bar, count in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2, f"{count}", ha="center", va="bottom")
    ax.set_ylabel("Percent of matched initial conditions")
    ax.set_ylim(0, 100)
    ax.set_title("Same initial V, T, and age: family versus current level")
    fig.tight_layout()
    fig_path = FIG_DIR / "rw11_charge_family.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", fig_path)


if __name__ == "__main__":
    main()
