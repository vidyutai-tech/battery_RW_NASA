#!/usr/bin/env python
"""Reward-landscape diagnostic for saved Final_Aging_based_setup GP-BO runs.

Does not rerun or modify GP-BO. Reads history already stored in each
family JSON, writes reward statistics and plots, then checks whether the
digital twin is in eval mode and whether one fixed profile is deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELLS, FAMILY_ORDER, OUTPUT_ROOT, make_simulator_ckpt, output_root,
    params_from_dict, read_json, write_csv, write_json,
)

from aacopt.config import ensure_matplotlib_cache  # noqa: E402
from aacopt.evaluate import evaluate_params, load_anchors, load_degradation_model  # noqa: E402
from aacopt.viz.style import FAMILY_LABELS  # noqa: E402

NEAR_FRAC = 0.01
N_REPEAT = 10

ARMS = {
    "02_armA": ("original reward", OUTPUT_ROOT / "02_armA" / "gp_bo"),
    "03_armA_rescaled": (
        "rescaled reward (normalize_three_terms)",
        OUTPUT_ROOT / "03_armA_rescaled" / "normalize_three_terms" / "gp_bo",
    ),
}


def _finite(values: Sequence[float]) -> np.ndarray:
    a = np.asarray(list(values), dtype=np.float64)
    return a[np.isfinite(a)]


def reward_stats(rewards: Sequence[float]) -> Dict[str, Any]:
    a = _finite(rewards)
    if a.size == 0:
        return {
            "n": 0, "min": None, "p25": None, "median": None, "mean": None,
            "p75": None, "max": None, "std": None, "range": None,
        }
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.percentile(a, 50)),
        "mean": float(a.mean()),
        "p75": float(np.percentile(a, 75)),
        "max": float(a.max()),
        "std": float(a.std(ddof=0)),
        "range": float(a.max() - a.min()),
    }


def _rows_from_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for i, h in enumerate(history):
        metrics = h.get("metrics") or {}
        reward = h.get("reward", metrics.get("reward"))
        loss = h.get("loss", metrics.get("loss"))
        feasible = bool(h.get("feasible", metrics.get("feasible")))
        if reward is None or loss is None:
            continue
        rows.append({
            "eval_index": int(i) + 1,
            "reward": float(reward),
            "loss": float(loss),
            "feasible": feasible,
            "duration_min": metrics.get("duration_min"),
            "penalty": metrics.get("penalty"),
        })
    return rows


def near_best(rewards: np.ndarray, *, frac: float = NEAR_FRAC) -> Dict[str, Any]:
    if rewards.size == 0:
        return {"n": 0, "r_max": None, "n_within_1pct": 0, "threshold": None}
    r_max = float(rewards.max())
    # Gap relative to |R_max|, so the band stays just below the best reward
    # when rewards are negative (rescaled arm).
    threshold = r_max - frac * abs(r_max)
    n_near = int(np.sum(rewards >= threshold - 1e-15))
    return {
        "n": int(rewards.size),
        "r_max": r_max,
        "threshold": float(threshold),
        "rule": "R >= R_max - 0.01*|R_max|",
        "n_within_1pct": n_near,
        "fraction_within_1pct": float(n_near / rewards.size),
    }


def loss_vs_reward(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best-by-loss and best-by-reward can differ: loss adds duration and penalty."""
    if not rows:
        return {}
    feasible = [r for r in rows if r["feasible"]]
    pool = feasible if feasible else rows
    by_loss = min(pool, key=lambda r: r["loss"])
    by_reward = max(pool, key=lambda r: r["reward"])
    return {
        "pool": "feasible" if feasible else "all",
        "n_pool": len(pool),
        "best_by_loss_eval": by_loss["eval_index"],
        "best_by_loss_loss": by_loss["loss"],
        "best_by_loss_reward": by_loss["reward"],
        "best_by_reward_eval": by_reward["eval_index"],
        "best_by_reward_loss": by_reward["loss"],
        "best_by_reward_reward": by_reward["reward"],
        "same_candidate": by_loss["eval_index"] == by_reward["eval_index"],
    }


def analyze_family(blob: Dict[str, Any]) -> Dict[str, Any]:
    rows = _rows_from_history(blob.get("history") or [])
    all_r = np.array([r["reward"] for r in rows], dtype=np.float64)
    feas = np.array([r["reward"] for r in rows if r["feasible"]], dtype=np.float64)
    infeas = np.array([r["reward"] for r in rows if not r["feasible"]], dtype=np.float64)
    return {
        "n_history": len(rows),
        "n_feasible": int(sum(r["feasible"] for r in rows)),
        "n_infeasible": int(sum(not r["feasible"] for r in rows)),
        "all": reward_stats(all_r),
        "feasible": reward_stats(feas),
        "infeasible": reward_stats(infeas),
        "near_best_feasible": near_best(feas),
        "loss_vs_reward": loss_vs_reward(rows),
        "stored_best_reward": blob.get("best_reward"),
        "stored_best_loss": blob.get("best_loss"),
        "rows": rows,
    }


def collect_arm(arm_id: str, gp_dir: Path) -> Dict[str, Any]:
    cells: Dict[str, Any] = {}
    flat_rows: List[Dict[str, Any]] = []
    for cell in CELLS:
        families: Dict[str, Any] = {}
        pooled_feas: List[Dict[str, Any]] = []
        for fid in FAMILY_ORDER:
            path = gp_dir / cell / f"{fid}.json"
            if not path.is_file():
                continue
            blob = read_json(path)
            fam = analyze_family(blob)
            families[fid] = {k: v for k, v in fam.items() if k != "rows"}
            for row in fam["rows"]:
                flat_rows.append({
                    "arm": arm_id, "cell": cell, "family_id": fid, **row,
                })
                if row["feasible"]:
                    pooled_feas.append({"family_id": fid, **row})
        if not families:
            continue
        feas_r = np.array([r["reward"] for r in pooled_feas], dtype=np.float64)
        top = sorted(pooled_feas, key=lambda r: r["loss"])[:6]
        top_rewards = [t["reward"] for t in top]
        r_max = float(feas_r.max()) if feas_r.size else None
        cells[cell] = {
            "families": families,
            "pooled_feasible": reward_stats(feas_r),
            "near_best_feasible_pooled": near_best(feas_r),
            "top6_by_loss": [
                {
                    "rank": i + 1,
                    "family_id": t["family_id"],
                    "eval_index": t["eval_index"],
                    "loss": t["loss"],
                    "reward": t["reward"],
                    "delta_reward_from_best": (
                        None if r_max is None else float(r_max - t["reward"])
                    ),
                    "within_1pct_of_best_reward": (
                        None if r_max is None
                        else bool(t["reward"] >= r_max - NEAR_FRAC * abs(r_max) - 1e-15)
                    ),
                }
                for i, t in enumerate(top)
            ],
        }
    return {"arm": arm_id, "gp_dir": str(gp_dir), "cells": cells, "flat_rows": flat_rows}


def _fmt(v: Optional[float], digits: int = 6) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def write_summary(path: Path, payload: Dict[str, Any], mode: Dict[str, Any], repeat: Dict[str, Any]) -> None:
    lines = [
        "# GP-BO reward landscape",
        "",
        "GP-BO was not rerun and was not modified. Statistics come from `history` already saved in each family JSON.",
        "",
        "## Model mode",
        "",
        f"- Digital twin `training` flag at load: `{mode['twin_training_at_load']}`.",
        f"- `load_twin` calls `model.eval()` and freezes weights. `BatteryDigitalTwin.predict` calls `eval()` again, and chunk inference is under `torch.no_grad()`.",
        f"- Dropout modules: {mode['n_dropout']}. All in eval after load: `{mode['all_dropout_eval']}`.",
        "- The object passed into `optimize_family_gp_bo` as `model` is `HybridDegradationModel`, a closed-form NumPy scorer. It is not an `nn.Module` and has no `eval()` / `train()` switch.",
        f"- Repeated-evaluation device: `{repeat['device']}`.",
        "",
        "## Repeated evaluation of one fixed profile",
        "",
        f"- Cell `{repeat['cell']}`, family `{repeat['family_id']}`, {repeat['n_repeats']} identical calls.",
        f"- Reward std: `{repeat['reward_std']:.3e}`. Reward max−min: `{repeat['reward_range']:.3e}`.",
        f"- Loss std: `{repeat['loss_std']:.3e}`.",
        f"- Deterministic at 1e-12 absolute tolerance: `{repeat['deterministic_atol_1e-12']}`.",
        "",
        "## Loss is not exactly minus reward",
        "",
        "`score_session` sets `loss = -reward + 0.001 * duration_min + penalty`.",
        "A lower loss usually tracks a higher reward, but the duration term and the infeasibility penalty can pick a different winner.",
        "",
    ]
    for arm_id, arm in payload["arms"].items():
        label = ARMS[arm_id][0]
        lines.append(f"## {arm_id} — {label}")
        lines.append("")
        lines.append("| Cell | Family | N | Feasible | Min | P25 | Median | Mean | P75 | Max | Std | Within 1% of best | Loss winner = reward winner |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for cell, cell_blob in arm["cells"].items():
            for fid in FAMILY_ORDER:
                fam = cell_blob["families"].get(fid)
                if fam is None:
                    continue
                st = fam["feasible"] if fam["feasible"]["n"] else fam["all"]
                scope = "feasible" if fam["feasible"]["n"] else "all"
                near = fam["near_best_feasible"]
                same = fam["loss_vs_reward"].get("same_candidate")
                lines.append(
                    f"| {cell} | {FAMILY_LABELS.get(fid, fid)} ({scope}) | {st['n']} | {fam['n_feasible']} | "
                    f"{_fmt(st['min'])} | {_fmt(st['p25'])} | {_fmt(st['median'])} | {_fmt(st['mean'])} | "
                    f"{_fmt(st['p75'])} | {_fmt(st['max'])} | {_fmt(st['std'])} | "
                    f"{near['n_within_1pct']}/{near['n']} | {same} |"
                )
        lines.append("")
        lines.append("Top 6 feasible candidates per cell, ranked by loss (the rule GP-BO uses to pick a winner):")
        lines.append("")
        for cell, cell_blob in arm["cells"].items():
            lines.append(f"### {cell}")
            lines.append("")
            lines.append("| Rank | Family | Eval | Loss | Reward | Δreward from best feasible | Within 1% |")
            lines.append("| ---: | --- | ---: | ---: | ---: | ---: | --- |")
            for row in cell_blob["top6_by_loss"]:
                lines.append(
                    f"| {row['rank']} | {row['family_id']} | {row['eval_index']} | "
                    f"{_fmt(row['loss'])} | {_fmt(row['reward'])} | "
                    f"{_fmt(row['delta_reward_from_best'])} | {row['within_1pct_of_best_reward']} |"
                )
            pooled = cell_blob["near_best_feasible_pooled"]
            lines.append("")
            lines.append(
                f"Feasible candidates within 1% of this cell's best feasible reward: "
                f"**{pooled['n_within_1pct']} / {pooled['n']}** "
                f"(best reward {_fmt(pooled['r_max'])})."
            )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_arm(arm_id: str, arm: Dict[str, Any], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    for cell, cell_blob in arm["cells"].items():
        present = [fid for fid in FAMILY_ORDER if fid in cell_blob["families"]]
        if not present:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=False)
        axes = axes.ravel()
        for ax, fid in zip(axes, FAMILY_ORDER):
            fam_rows = [
                r for r in arm["flat_rows"]
                if r["cell"] == cell and r["family_id"] == fid
            ]
            if not fam_rows:
                ax.set_visible(False)
                continue
            idx = np.array([r["eval_index"] for r in fam_rows])
            rew = np.array([r["reward"] for r in fam_rows])
            feas = np.array([r["feasible"] for r in fam_rows])
            ax.scatter(idx[~feas], rew[~feas], s=18, c="#9ca3af", label="infeasible", zorder=2)
            ax.scatter(idx[feas], rew[feas], s=18, c="#2563eb", label="feasible", zorder=3)
            best_so_far = np.maximum.accumulate(np.where(feas, rew, -np.inf))
            finite = np.isfinite(best_so_far)
            if np.any(finite):
                ax.plot(idx[finite], best_so_far[finite], color="#dc2626", lw=1.4, label="best feasible so far")
            ax.set_title(FAMILY_LABELS.get(fid, fid))
            ax.set_xlabel("Evaluation")
            ax.set_ylabel("Reward")
            ax.grid(True, alpha=0.3)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.suptitle(f"{arm_id}  {cell}  reward vs evaluation")
        fig.tight_layout()
        fig.savefig(out_dir / f"{cell}_reward_vs_eval.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        series = []
        labels = []
        for fid in present:
            vals = [
                r["reward"] for r in arm["flat_rows"]
                if r["cell"] == cell and r["family_id"] == fid and r["feasible"]
            ]
            if not vals:
                vals = [
                    r["reward"] for r in arm["flat_rows"]
                    if r["cell"] == cell and r["family_id"] == fid
                ]
            series.append(vals)
            labels.append(FAMILY_LABELS.get(fid, fid))
        ax.boxplot(series, tick_labels=labels, showfliers=True)
        ax.set_ylabel("Reward (feasible, or all if none feasible)")
        ax.set_title(f"{arm_id}  {cell}  reward distribution")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{cell}_reward_box.png", dpi=160)
        plt.close(fig)


def check_model_mode(sim) -> Dict[str, Any]:
    import torch.nn as nn

    twin = sim.bdt.model
    dropouts = [m for m in twin.modules() if isinstance(m, nn.Dropout)]
    deg = load_degradation_model()
    return {
        "device": str(sim.bdt.device),
        "twin_training_at_load": bool(twin.training),
        "n_dropout": len(dropouts),
        "dropout_p": sorted({float(m.p) for m in dropouts}),
        "all_dropout_eval": all(not m.training for m in dropouts),
        "degradation_model_type": type(deg).__name__,
        "degradation_model_is_nn_module": isinstance(deg, nn.Module),
        "checkpoint": str(sim.bdt.checkpoint),
    }


def repeat_profile(
    sim,
    cell: str,
    *,
    family_id: str,
    params_dict: Dict[str, Any],
    n_repeats: int,
) -> Dict[str, Any]:
    from aacopt.config import RewardSpec

    spec = RewardSpec.load()
    model = load_degradation_model()
    anchors = load_anchors(cell)
    _sim, state = sim if isinstance(sim, tuple) else (sim, None)
    if state is None:
        raise RuntimeError("simulator state missing")
    params = params_from_dict(params_dict)
    rewards = []
    losses = []
    peaks = []
    for _ in range(n_repeats):
        loss, metrics, _session = evaluate_params(
            _sim, state, params, model=model, spec=spec, anchors=anchors,
        )
        rewards.append(float(metrics["reward"]))
        losses.append(float(loss))
        peaks.append(float(metrics["peak_v"]))
    r = np.asarray(rewards, dtype=np.float64)
    ell = np.asarray(losses, dtype=np.float64)
    twin_after = bool(_sim.bdt.model.training)
    return {
        "cell": cell,
        "family_id": family_id,
        "n_repeats": n_repeats,
        "device": str(_sim.bdt.device),
        "params": params_dict,
        "rewards": rewards,
        "losses": losses,
        "peak_v": peaks,
        "reward_std": float(r.std(ddof=0)),
        "reward_range": float(r.max() - r.min()),
        "loss_std": float(ell.std(ddof=0)),
        "loss_range": float(ell.max() - ell.min()),
        "deterministic_atol_1e-12": bool(np.allclose(r, r[0], atol=1e-12, rtol=0.0)),
        "twin_training_after_eval": twin_after,
    }


def main() -> int:
    ensure_matplotlib_cache()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--repeats", type=int, default=N_REPEAT)
    ap.add_argument("--skip-repeat", action="store_true")
    args = ap.parse_args()

    root = output_root() / "05_gpbo_reward_diagnostic"
    root.mkdir(parents=True, exist_ok=True)
    cfg = read_json(output_root(create=False) / "config_used.json")
    i_max = float(cfg["i_max_a_phase2"])

    arms_out: Dict[str, Any] = {}
    stat_rows: List[Dict[str, Any]] = []
    for arm_id, (_label, gp_dir) in ARMS.items():
        if not gp_dir.is_dir():
            print(f"skip missing {gp_dir}", flush=True)
            continue
        print(f"reading {arm_id}", flush=True)
        arm = collect_arm(arm_id, gp_dir)
        plot_arm(arm_id, arm, root / arm_id)
        arms_out[arm_id] = {
            "label": _label,
            "cells": arm["cells"],
        }
        for row in arm["flat_rows"]:
            stat_rows.append(row)
        del arm["flat_rows"]

    sim, state = make_simulator_ckpt("RW9", device=args.device, i_max_a=i_max)
    mode = check_model_mode(sim)
    mode["cell"] = "RW9"
    print(
        f"twin.training={mode['twin_training_at_load']}  "
        f"dropout_modules={mode['n_dropout']}  device={sim.bdt.device}",
        flush=True,
    )

    repeat: Dict[str, Any]
    if args.skip_repeat:
        repeat = {"skipped": True}
    else:
        best = read_json(ARMS["02_armA"][1] / "RW9" / "cccv.json")["best_params"]
        print(f"repeating RW9 CCCV best profile x{args.repeats}", flush=True)
        repeat = repeat_profile(
            (sim, state), "RW9", family_id="cccv", params_dict=best, n_repeats=args.repeats,
        )
        print(
            f"reward std={repeat['reward_std']:.3e}  range={repeat['reward_range']:.3e}  "
            f"deterministic={repeat['deterministic_atol_1e-12']}",
            flush=True,
        )

    slim_repeat = {k: v for k, v in repeat.items() if k != "params"}
    payload = {"arms": arms_out, "model_mode": mode, "repeat_eval": slim_repeat, "i_max_a": i_max}
    write_json(root / "reward_landscape.json", payload)
    write_csv(
        root / "candidate_rewards.csv",
        stat_rows,
        ["arm", "cell", "family_id", "eval_index", "reward", "loss", "feasible", "duration_min", "penalty"],
    )
    write_summary(root / "SUMMARY.md", payload, mode, repeat if not args.skip_repeat else {
        "cell": "—", "family_id": "—", "n_repeats": 0, "device": mode["device"],
        "reward_std": float("nan"), "reward_range": float("nan"), "loss_std": float("nan"),
        "deterministic_atol_1e-12": None,
    })
    print(f"wrote {root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
