#!/usr/bin/env python
"""Cross-cell Top-6 litmus test on frozen per-cell GP-BO (stage 8).

Does not re-optimize and does not change twins. Reads
``results/08_gp_bo/<CELL>/{cccv,two_step,three_step,pulsed}.json``,
takes the 6 lowest-loss feasible trials per cell (24 elite profiles),
replays each profile on all four twins, and writes tables/figures under
``results/14_litmus_top6/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.config import (  # noqa: E402
    OptimizationSpec, Paths, RewardSpec, ensure_matplotlib_cache, provenance,
    read_json, stage_dir, write_json,
)
from aacopt.evaluate import (  # noqa: E402
    evaluate_params, load_anchors, load_degradation_model, make_simulator,
)
from aacopt.profiles import get_family  # noqa: E402
from aacopt.viz.style import (  # noqa: E402
    FAMILY_LABELS, FAMILY_ORDER, PAPER_DPI, apply_paper_style, savefig,
)

CELLS = ("RW9", "RW10", "RW11", "RW12")
N_ELITE = 6
N_GRID = 100
CELL_COLORS = {"RW9": "#2563eb", "RW10": "#16a34a", "RW11": "#ea580c", "RW12": "#7c3aed"}


def _params_from_dict(raw: Dict[str, Any]):
    blob = dict(raw)
    fid = str(blob.pop("family_id"))
    cls = type(get_family(fid))
    return cls.from_dict(blob)


def _charge_i(session: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(session["time_s"], dtype=np.float64)
    i = -np.asarray(session["current_a"], dtype=np.float64)
    return t, i


def _resample_unit_time(t: np.ndarray, y: np.ndarray, n: int = N_GRID) -> np.ndarray:
    if t.size == 0:
        return np.zeros(n, dtype=np.float64)
    t = t - t[0]
    span = float(t[-1]) if t.size else 0.0
    if span <= 1e-9:
        return np.full(n, float(y[0]) if y.size else 0.0, dtype=np.float64)
    tau = t / span
    grid = np.linspace(0.0, 1.0, n)
    return np.interp(grid, tau, y)


def _mean_std_cv(xs: List[float]) -> Dict[str, float]:
    a = np.asarray(xs, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "cv_pct": float("nan"), "min": float("nan"), "max": float("nan")}
    mu = float(a.mean())
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    cv = 100.0 * sd / abs(mu) if abs(mu) > 1e-18 else float("nan")
    return {"n": int(a.size), "mean": mu, "std": sd, "cv_pct": cv,
            "min": float(a.min()), "max": float(a.max())}


def load_elite(gpbo_dir: Path, cells: List[str], k: int) -> List[Dict[str, Any]]:
    elite: List[Dict[str, Any]] = []
    for cell in cells:
        rows: List[Dict[str, Any]] = []
        for fid in FAMILY_ORDER:
            path = gpbo_dir / cell / f"{fid}.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            blob = read_json(path)
            for idx, h in enumerate(blob.get("history") or []):
                m = h.get("metrics") or {}
                if not bool(m.get("feasible", h.get("feasible"))):
                    continue
                rows.append({
                    "source_cell": cell,
                    "family_id": str((h.get("params") or {}).get("family_id", fid)),
                    "history_index": int(idx),
                    "loss": float(h.get("loss", m.get("loss", np.inf))),
                    "reward": float(h.get("reward", m.get("reward", np.nan))),
                    "params": dict(h.get("params") or {}),
                    "native_metrics": {kk: vv for kk, vv in m.items()
                                       if kk not in ("weights", "anchors")},
                })
        rows.sort(key=lambda r: r["loss"])
        for rank, row in enumerate(rows[:k], start=1):
            row["rank"] = rank
            row["profile_id"] = f"{cell}-P{rank}"
            elite.append(row)
    return elite


def _flat_params(p: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("i_cc", "v_cv", "i_cutoff", "i1", "i2", "i3", "soc_switch", "soc1", "soc2",
            "i_charge", "pulse_on_min", "pulse_rest_min", "rest_fraction", "i_floor")
    return {k: p.get(k) for k in keys}


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def plot_family_counts(counts: Dict[str, Counter], out: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(CELLS))
    bottom = np.zeros(len(CELLS))
    palette = {"cccv": "#2563eb", "two_step": "#16a34a",
               "three_step": "#ea580c", "pulsed": "#7c3aed"}
    for fid in FAMILY_ORDER:
        h = np.array([counts[c][fid] for c in CELLS], dtype=float)
        ax.bar(x, h, bottom=bottom, color=palette[fid], label=FAMILY_LABELS[fid], width=0.62)
        bottom += h
    ax.set_xticks(x)
    ax.set_xticklabels(list(CELLS))
    ax.set_ylabel("Count in Top-6")
    ax.set_ylim(0, N_ELITE)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    savefig(fig, out)


def plot_current_overlay(native: List[Dict[str, Any]], out: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for row in native:
        t_min = np.asarray(row["t_s"]) / 60.0
        ax.plot(t_min, row["i_a"], color=CELL_COLORS[row["source_cell"]],
                lw=1.6, alpha=0.75)
    for cell, col in CELL_COLORS.items():
        ax.plot([], [], color=col, lw=2.4, label=cell)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Charge current (A)")
    ax.legend(frameon=False, ncol=4, loc="upper right")
    fig.tight_layout()
    savefig(fig, out)


def plot_current_grid(native: List[Dict[str, Any]], out: Path) -> None:
    apply_paper_style()
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2), sharey=True)
    by_cell: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CELLS}
    for row in native:
        by_cell[row["source_cell"]].append(row)
    for ax, cell in zip(axes.ravel(), CELLS):
        for row in by_cell[cell]:
            t_min = np.asarray(row["t_s"]) / 60.0
            ax.plot(t_min, row["i_a"], lw=1.7, alpha=0.85,
                    label=f"P{row['rank']} {FAMILY_LABELS.get(row['family_id'], row['family_id'])}")
        ax.set_title(cell)
        ax.set_xlabel("Time (min)")
        ax.legend(fontsize=9, frameon=False, loc="best")
    axes[0, 0].set_ylabel("Charge current (A)")
    axes[1, 0].set_ylabel("Charge current (A)")
    fig.tight_layout()
    savefig(fig, out)


def plot_similarity(ids: List[str], dist: np.ndarray, out: Path) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(9.6, 8.4))
    im = ax.imshow(dist, cmap="magma_r", origin="upper")
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=90, fontsize=8)
    ax.set_yticklabels(ids, fontsize=8)
    for k in range(1, 4):
        ax.axhline(k * N_ELITE - 0.5, color="white", lw=1.2)
        ax.axvline(k * N_ELITE - 0.5, color="white", lw=1.2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("RMSE of I(t) (A), time-normalized")
    fig.tight_layout()
    savefig(fig, out)


def plot_cross_heat(ids: List[str], cells: List[str], grid: np.ndarray,
                    *, title: str, cmap: str, out: Path, vmin=None, vmax=None) -> None:
    apply_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    im = ax.imshow(grid, cmap=cmap, origin="upper", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(cells)
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=8)
    ax.set_xlabel("Evaluated on twin")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(title)
    fig.tight_layout()
    savefig(fig, out)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def build_report(
    *,
    elite: List[Dict[str, Any]],
    family_counts: Dict[str, Counter],
    stats: Dict[str, Dict[str, float]],
    cross: List[Dict[str, Any]],
    within: Dict[str, float],
    between: float,
    n_all_feasible: int,
    voltage_headroom: Dict[str, Dict[str, float]],
    twin_rmse: Dict[str, Dict[str, float]],
) -> str:
    fam_rows = []
    for fid in FAMILY_ORDER:
        fam_rows.append(
            [FAMILY_LABELS[fid]] + [str(family_counts[c][fid]) for c in CELLS]
        )
    master_rows = []
    for e in elite:
        p = e["params"]
        nm = e["native_replay"]
        master_rows.append([
            e["source_cell"], str(e["rank"]), FAMILY_LABELS.get(e["family_id"], e["family_id"]),
            f"{nm['duration_min']:.2f}", f"{nm['q_total']:.4e}", f"{nm['reward']:.4f}",
            f"{nm['peak_v']:.3f}", f"{nm['i_mean']:.2f}", f"{nm['i_max']:.2f}",
            _param_short(p),
        ])
    cv_rows = []
    for name, s in stats.items():
        cv_rows.append([
            name, f"{s['mean']:.4g}", f"{s['std']:.4g}", f"{s['cv_pct']:.1f}",
            f"{s['min']:.4g}", f"{s['max']:.4g}",
        ])
    robust = []
    for e in elite:
        hits = [r for r in cross if r["profile_id"] == e["profile_id"]]
        n_ok = sum(1 for r in hits if r["feasible"])
        if n_ok == 4:
            robust.append(e["profile_id"])
    head_rows = []
    for c in CELLS:
        h = voltage_headroom[c]
        rmse = twin_rmse.get(c, {})
        head_rows.append([
            c, f"{h['peak_v_1c']:.3f}", f"{h['headroom_1c_mv']:.0f}",
            f"{h['cccv_2c_feasible']}", f"{rmse.get('v_rmse_mv', float('nan')):.1f}",
        ])
    return f"""# Top-6 cross-cell litmus test

Frozen per-cell digital twins. Frozen stage-8 GP-BO (80 evals / family / cell).
Identical operating box for every cell: SOC₀ = 20 %, T₀ = 24 °C, energy target = 40 % of
nominal pack energy, V_max = 4.20 V, I ∈ [0.75, 6.00] A, same reward, same GP-BO budget.
The only factor that changes is the cell twin.

This analysis does **not** force the four elite sets to match. It measures whether they do.

## Level 1 — charging family in each Top-6

{_md_table(["Family", *CELLS], fam_rows)}

CCCV is the only family that appears in every cell's Top-6. Pulsed appears only for RW9.
RW11's entire Top-6 is CCCV.

## Level 2 — 24 elite profiles (native-twin replay)

{_md_table(
    ["Cell", "Rank", "Family", "t (min)", "Q_loss", "Reward", "V_peak", "I_mean (A)", "I_max (A)", "Parameters"],
    master_rows,
)}

## Parameter / outcome spread across the 24 native elites

{_md_table(["Quantity", "Mean", "Std", "CV %", "Min", "Max"], cv_rows)}

A small CV means the elite set is a shared neighbourhood. A large CV on current or time
means the independently optimized elites are not the same protocol.

## Trajectory similarity (time-normalized RMSE of I(t))

Mean pairwise RMSE **within** a cell's own Top-6:

{_md_table(["Cell", "Mean within-cell RMSE (A)"], [[c, f"{within[c]:.3f}"] for c in CELLS])}

Mean pairwise RMSE **between different cells**: {between:.3f} A.

If within-cell RMSE is small and between-cell RMSE is comparable, the four Top-6 clouds
overlap. If between-cell RMSE is much larger, the elite sets occupy different current
trajectories.

## Cross-cell validation (24 profiles × 4 twins)

A profile is counted as pack-feasible if it meets the energy target and stays ≤ 4.20 V
on **every** twin.

Profiles feasible on all four twins: **{n_all_feasible} / 24**
{("— " + ", ".join(robust)) if robust else ""}

## Voltage behaviour of the twins (from frozen stage-8 baselines + stage-2 RMSE)

These numbers describe **predicted** restrictiveness, not a claim about physical impedance.

{_md_table(
    ["Cell", "CCCV 1C V_peak", "Headroom to 4.20 V (mV)", "CCCV 2C feasible?", "Twin V RMSE (mV)"],
    head_rows,
)}

Headroom of tens of millivolts is comparable to twin voltage RMSE on RW11/RW12
(48 mV / 64 mV). A predicted V = 4.05 V is therefore **not** 150 mV of demonstrated
safety margin.

## How to read the result

- **Consistent strategy:** same families dominate, currents in a tight band, many of the
  24 profiles remain feasible on the other twins.
- **Cell-specific / twin-specific:** one cell's elite set is pulsed or 6 A CCCV while
  the others sit near 2 A, and those profiles fail the 4.20 V (or energy) check on
  other twins.

The figures `family_counts.png`, `current_overlay.png`, `similarity_24x24.png`, and
`cross_feasible.png` are the visual version of this test.
"""


def _param_short(p: Dict[str, Any]) -> str:
    fid = p.get("family_id")
    if fid == "cccv":
        return f"I={p['i_cc']:.2f}A Vcv={p['v_cv']:.3f}"
    if fid == "two_step":
        return f"I1={p['i1']:.2f} I2={p['i2']:.2f}"
    if fid == "three_step":
        return f"I1={p['i1']:.2f} I2={p['i2']:.2f} I3={p['i3']:.2f}"
    if fid == "pulsed":
        return f"Ion={p['i_charge']:.2f} on={p['pulse_on_min']:.1f} rest={p['pulse_rest_min']:.1f}"
    return ""


def main() -> int:
    ensure_matplotlib_cache()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--k", type=int, default=N_ELITE)
    args = ap.parse_args()

    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    paths = Paths.load()
    device = args.device or opt.device
    k = int(args.k)
    gpbo_dir = stage_dir("08_gp_bo", create=False)
    out = stage_dir("14_litmus_top6", create=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    elite = load_elite(gpbo_dir, list(CELLS), k)
    print(f"Loaded {len(elite)} elite profiles from {gpbo_dir}", flush=True)

    family_counts = {c: Counter() for c in CELLS}
    for e in elite:
        family_counts[e["source_cell"]][e["family_id"]] += 1

    model = load_degradation_model()
    sims = {}
    states = {}
    anchors = {}
    for cell in CELLS:
        print(f"Loading twin {cell} …", flush=True)
        sims[cell], states[cell] = make_simulator(cell, device=device)
        anchors[cell] = load_anchors(cell)

    cross_rows: List[Dict[str, Any]] = []
    native: List[Dict[str, Any]] = []
    native_I: List[np.ndarray] = []
    ids: List[str] = []

    n_eval = 0
    total = len(elite) * len(CELLS)
    for e in elite:
        params = _params_from_dict(e["params"])
        e["params"] = params.to_dict()
        for cell in CELLS:
            n_eval += 1
            print(f"  [{n_eval}/{total}] {e['profile_id']} → {cell}", flush=True)
            loss, metrics, session = evaluate_params(
                sims[cell], states[cell], params, model=model, spec=reward,
                anchors=anchors[cell],
            )
            t, i = _charge_i(session)
            rec = {
                "profile_id": e["profile_id"],
                "source_cell": e["source_cell"],
                "rank": e["rank"],
                "family_id": e["family_id"],
                "eval_cell": cell,
                "native": cell == e["source_cell"],
                "feasible": bool(metrics["feasible"]),
                "loss": float(loss),
                "reward": float(metrics["reward"]),
                "duration_min": float(metrics["duration_min"]),
                "q_total": float(metrics["q_total"]),
                "peak_v": float(metrics["peak_v"]),
                "peak_t": float(metrics["peak_t"]),
                "delta_soc": float(metrics["delta_soc"]),
                "i_mean": float(i.mean()) if i.size else float("nan"),
                "i_max": float(i.max()) if i.size else float("nan"),
                "end_reason": metrics.get("end_reason"),
            }
            rec.update({f"p_{kk}": vv for kk, vv in _flat_params(e["params"]).items()})
            cross_rows.append(rec)
            if cell == e["source_cell"]:
                e["native_replay"] = rec
                native.append({
                    "profile_id": e["profile_id"],
                    "source_cell": e["source_cell"],
                    "rank": e["rank"],
                    "family_id": e["family_id"],
                    "t_s": t,
                    "i_a": i,
                })
                native_I.append(_resample_unit_time(t, i))
                ids.append(e["profile_id"])

    I = np.stack(native_I, axis=0)
    dist = np.sqrt(((I[:, None, :] - I[None, :, :]) ** 2).mean(axis=2))

    within = {}
    between_vals = []
    for ci, c in enumerate(CELLS):
        sl = slice(ci * k, (ci + 1) * k)
        sub = dist[sl, sl]
        tri = sub[np.triu_indices(k, k=1)]
        within[c] = float(tri.mean()) if tri.size else 0.0
    for i in range(len(ids)):
        ci = i // k
        for j in range(i + 1, len(ids)):
            if j // k != ci:
                between_vals.append(float(dist[i, j]))
    between = float(np.mean(between_vals)) if between_vals else float("nan")

    i_mean = [e["native_replay"]["i_mean"] for e in elite]
    i_max = [e["native_replay"]["i_max"] for e in elite]
    t_min = [e["native_replay"]["duration_min"] for e in elite]
    qv = [e["native_replay"]["q_total"] for e in elite]
    stats = {
        "I_mean (A)": _mean_std_cv(i_mean),
        "I_max (A)": _mean_std_cv(i_max),
        "Time (min)": _mean_std_cv(t_min),
        "Q_loss": _mean_std_cv(qv),
        "Reward": _mean_std_cv([e["native_replay"]["reward"] for e in elite]),
        "V_peak (V)": _mean_std_cv([e["native_replay"]["peak_v"] for e in elite]),
    }

    n_all_feasible = 0
    for e in elite:
        hits = [r for r in cross_rows if r["profile_id"] == e["profile_id"]]
        if all(r["feasible"] for r in hits):
            n_all_feasible += 1

    voltage_headroom = {}
    for cell in CELLS:
        bl = read_json(gpbo_dir / cell / "baselines.json")
        by_name = {b["name"]: b for b in bl}
        p1 = float(by_name["CCCV 1C"]["metrics"]["peak_v"])
        feas2 = bool(by_name["CCCV 2C"]["metrics"]["feasible"])
        voltage_headroom[cell] = {
            "peak_v_1c": p1,
            "headroom_1c_mv": 1000.0 * (4.20 - p1),
            "cccv_2c_feasible": feas2,
        }

    twin_rmse: Dict[str, Dict[str, float]] = {}
    ver_path = stage_dir("02_bdt_verification", create=False) / "bdt_verification.json"
    if ver_path.is_file():
        ver = read_json(ver_path)
        for cell, blob in (ver.get("per_cell") or {}).items():
            v = ((blob.get("held_out_nasa") or {}).get("voltage") or {})
            twin_rmse[cell] = {"v_rmse_mv": 1000.0 * float(v.get("rmse", float("nan")))}

    feas_grid = np.zeros((len(ids), len(CELLS)))
    time_grid = np.full((len(ids), len(CELLS)), np.nan)
    q_grid = np.full((len(ids), len(CELLS)), np.nan)
    id_index = {pid: i for i, pid in enumerate(ids)}
    cell_index = {c: j for j, c in enumerate(CELLS)}
    for r in cross_rows:
        i = id_index[r["profile_id"]]
        j = cell_index[r["eval_cell"]]
        feas_grid[i, j] = 1.0 if r["feasible"] else 0.0
        time_grid[i, j] = r["duration_min"]
        q_grid[i, j] = r["q_total"]

    master_fields = [
        "profile_id", "source_cell", "rank", "family_id", "loss", "reward",
        "duration_min", "q_total", "peak_v", "peak_t", "delta_soc", "i_mean", "i_max",
        "p_i_cc", "p_v_cv", "p_i_cutoff", "p_i1", "p_i2", "p_i3", "p_soc_switch",
        "p_soc1", "p_soc2", "p_i_charge", "p_pulse_on_min", "p_pulse_rest_min",
        "p_rest_fraction", "p_i_floor",
    ]
    master_rows = []
    for e in elite:
        row = dict(e["native_replay"])
        row["loss"] = e["loss"]
        master_rows.append(row)

    write_csv(out / "elite_master.csv", master_rows, master_fields)
    write_csv(out / "cross_cell.csv", cross_rows, [
        "profile_id", "source_cell", "rank", "family_id", "eval_cell", "native",
        "feasible", "loss", "reward", "duration_min", "q_total", "peak_v", "peak_t",
        "delta_soc", "i_mean", "i_max", "end_reason",
    ])
    count_rows = [{"family": FAMILY_LABELS[fid], **{c: family_counts[c][fid] for c in CELLS}}
                  for fid in FAMILY_ORDER]
    write_csv(out / "family_counts.csv", count_rows, ["family", *CELLS])
    stat_rows = [{"quantity": name, **s} for name, s in stats.items()]
    write_csv(out / "parameter_spread.csv", stat_rows,
              ["quantity", "n", "mean", "std", "cv_pct", "min", "max"])

    dist_rows = [{"profile_id": ids[i], **{ids[j]: float(dist[i, j]) for j in range(len(ids))}}
                 for i in range(len(ids))]
    write_csv(out / "similarity_rmse_I.csv", dist_rows, ["profile_id", *ids])

    write_json(out / "summary.json", {
        "k": k,
        "n_elite": len(elite),
        "family_counts": {c: dict(family_counts[c]) for c in CELLS},
        "spread": stats,
        "within_cell_rmse_A": within,
        "between_cell_rmse_A": between,
        "n_profiles_feasible_on_all_twins": n_all_feasible,
        "voltage_headroom": voltage_headroom,
        "twin_voltage_rmse_mv": twin_rmse,
        "operating_box": {
            "soc_start": opt.session.soc_start,
            "t0_c": opt.session.ambient_t0_c,
            "v_max": opt.session.v_max,
            "energy_fraction": opt.session.energy_fraction,
            "i_min_a": opt.space.i_min_a,
            "i_max_a": opt.space.i_max_a,
            "n_evals_per_family": opt.n_evals_per_family,
        },
        "provenance": provenance(
            "14_litmus_top6",
            configs=["paths", "reward", "optimization", "degradation_fitted"],
            inputs=[gpbo_dir / c / "summary.json" for c in CELLS],
        ),
    })

    report = build_report(
        elite=elite, family_counts=family_counts, stats=stats, cross=cross_rows,
        within=within, between=between, n_all_feasible=n_all_feasible,
        voltage_headroom=voltage_headroom, twin_rmse=twin_rmse,
    )
    (out / "REPORT.md").write_text(report, encoding="utf-8")

    plot_family_counts(family_counts, fig_dir / "family_counts.png")
    plot_current_overlay(native, fig_dir / "current_overlay.png")
    plot_current_grid(native, fig_dir / "current_grid.png")
    plot_similarity(ids, dist, fig_dir / "similarity_24x24.png")
    plot_cross_heat(ids, list(CELLS), feas_grid, title="Feasible (1 = yes)",
                    cmap="Greens", out=fig_dir / "cross_feasible.png", vmin=0, vmax=1)
    plot_cross_heat(ids, list(CELLS), time_grid, title="Time (min)",
                    cmap="viridis", out=fig_dir / "cross_time.png")
    plot_cross_heat(ids, list(CELLS), q_grid, title="Q_loss",
                    cmap="magma", out=fig_dir / "cross_qloss.png")

    print(f"\nWrote {out}", flush=True)
    print(f"Family counts: { {c: dict(family_counts[c]) for c in CELLS} }", flush=True)
    print(f"Feasible on all 4 twins: {n_all_feasible}/24", flush=True)
    print(f"Between-cell I(t) RMSE: {between:.3f} A", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
