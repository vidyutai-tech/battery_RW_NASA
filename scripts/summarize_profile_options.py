#!/usr/bin/env python3
"""
Summarize feasible charging profiles across families (constant CC, CC-taper,
multi-step taper, pulsed).

Reads BO history from ``optimization_result.json``, re-simulates canonical
candidates for under-represented families, picks the best feasible option per
family, and writes a comparison table + plots.

Usage
-----
    venv/bin/python scripts/summarize_profile_options.py
    venv/bin/python scripts/summarize_profile_options.py \\
        --result outputs/charging_opt/models/stage3_optimization/optimization_result.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rw_transfer.config import load_config
from charging_opt import paths as P
from charging_opt.artifacts import CANONICAL, OPTIONAL, resolve_bdt_ckpt
from charging_opt.lifetime_reward import aggregate_lifetime_reward
from charging_opt.profile_families import (
    FAMILY_LABELS,
    charge_levels,
    default_candidate_specs,
    family_from_spec_only,
    profile_family,
    simulate_from_spec_dict,
)
from charging_opt.profile_simulator import ProfileSimulator, ProfileSpec
from charging_opt.soc_utils import load_ocv_curve


def _entry_from_sim(
    option_id: str,
    session: dict,
    metrics: dict,
    *,
    source: str,
) -> dict:
    fam = profile_family(session)
    levels = charge_levels(session)
    return {
        "option_id": option_id,
        "source": source,
        "family": fam,
        "family_label": FAMILY_LABELS.get(fam, fam),
        "spec": session["profile_spec"],
        "charge_levels_a": levels,
        "n_charge_steps": len(levels),
        "merged_segments": ProfileSimulator.merged_segments(session),
        "metrics": metrics,
        "feasible": bool(metrics.get("feasible", False)),
        "loss": float(metrics.get("loss", 1e6)),
    }


def _pick_best_per_family(entries: list[dict]) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for e in sorted(entries, key=lambda x: x["loss"]):
        if not e.get("feasible"):
            continue
        fam = e["family"]
        if fam not in best:
            best[fam] = e
    return best


def _plot_comparison(options: list[dict], out_path: Path) -> None:
    n = len(options)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), squeeze=False)
    for ax_row, opt in zip(axes, options):
        ax = ax_row[0]
        sess_path = opt.get("_session")
        if sess_path is None:
            ax.text(0.5, 0.5, "No session data", ha="center", va="center")
            continue
        session = opt["_session"]
        m = opt["metrics"]
        t_min = session["time_s"] / 60.0
        ax2 = ax.twinx()
        ax.plot(t_min, -session["current_a"], color="tab:blue", lw=1.8, label="I charge")
        ax.plot(t_min, session["voltage_v"], color="tab:red", ls="--", lw=1.2, label="V")
        ax2.plot(t_min, session["soc"] * 100, color="black", ls="-.", lw=1.5, label="SoC")
        ax.set_ylabel("I (A) / V")
        ax2.set_ylabel("SoC (%)")
        ax2.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        levels = opt.get("charge_levels_a", [])
        level_str = " → ".join(f"{x:.2f}" for x in levels) if levels else "—"
        title = (
            f"{opt['family_label']}  |  {opt['option_id']}  |  "
            f"loss={opt['loss']:.1f}  |  {m['duration_min']:.1f} min  |  "
            f"SEI/%SoC={m.get('sei_per_pct_soc', float('nan')):.1f}\n"
            f"Steps: {level_str} A"
        )
        ax.set_title(title, fontsize=9)
        if ax is axes[0, 0]:
            lines1, lab1 = ax.get_legend_handles_labels()
            lines2, lab2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=7)
    fig.suptitle("Charging profile options by family (best feasible each)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_tradeoff_table(options: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, max(3, 0.45 * len(options) + 1)))
    ax.axis("off")
    headers = ["Family", "Option", "I_cc", "Floor", "Steps", "Duration", "SEI/%SoC", "Loss"]
    rows = []
    for o in options:
        s = o["spec"]
        rows.append([
            o["family_label"],
            o["option_id"],
            f"{s['i_charge']:.2f}",
            f"{s['i_floor']:.2f}",
            str(o.get("n_charge_steps", "—")),
            f"{o['metrics']['duration_min']:.1f} min",
            f"{o['metrics'].get('sei_per_pct_soc', float('nan')):.1f}",
            f"{o['loss']:.1f}",
        ])
    table = ax.table(
        cellText=rows, colLabels=headers, loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    ax.set_title("Profile option comparison", fontsize=11, pad=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize multi-family charging profile options.")
    p.add_argument("--result", default=OPTIONAL["lifetime_bo_result"])
    p.add_argument("--config", default=None)
    p.add_argument("--soc_target", type=float, default=0.95)
    p.add_argument("--max_duration_min", type=float, default=105.0)
    p.add_argument("--max_minutes", type=int, default=150)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    result_path = ROOT / args.result
    if not result_path.is_file():
        raise SystemExit(f"Missing {result_path} — run scripts/02_optimize_charging_profile.py first.")

    data = json.loads(result_path.read_text())
    start = data["initial_state"]
    constraints = data.get("constraints", {})
    soc_target = float(constraints.get("soc_target", args.soc_target))
    max_dur = constraints.get("max_duration_min", args.max_duration_min)
    if max_dur is not None:
        max_dur = float(max_dur)

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / P.STAGE3_PLOTS
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = ProfileSimulator(
        bdt_path=resolve_bdt_ckpt(data.get("bdt_checkpoint"), root=ROOT),
        capacity_path=ROOT / CANONICAL["capacity_fade"],
        margins_path=ROOT / CANONICAL["conformal_margins"],
        max_minutes=args.max_minutes,
        soc_target=soc_target,
    )

    entries: list[dict] = []

    # From BO history
    for i, h in enumerate(data.get("history", [])):
        fam = family_from_spec_only(h["spec"])
        entries.append({
            "option_id": f"bo_iter_{i+1}",
            "source": "bo_history",
            "family": fam,
            "family_label": FAMILY_LABELS.get(fam, fam),
            "spec": h["spec"],
            "metrics": h["metrics"],
            "feasible": h.get("feasible", False),
            "loss": float(h["loss"]),
        })

    best_per_family = _pick_best_per_family(entries)
    families_wanted = list(FAMILY_LABELS.keys())[:-1]  # exclude 'other'

    # Re-simulate canonical candidates for missing / weak families
    sessions_cache: dict[str, dict] = {}
    for option_id, fam_hint, spec in default_candidate_specs():
        session = sim.simulate(start, spec)
        _, metrics = aggregate_lifetime_reward(
            session, soc_target=soc_target, max_duration_min=max_dur,
        )
        entry = _entry_from_sim(option_id, session, metrics, source="candidate")
        sessions_cache[option_id] = session
        entries.append(entry)
        if entry["feasible"]:
            prev = best_per_family.get(entry["family"])
            if prev is None or entry["loss"] < prev["loss"]:
                best_per_family[entry["family"]] = entry

    # Build final option list (one per family, stable order)
    order = ["constant_cc", "cc_taper", "multi_step_taper", "pulsed"]
    final_options: list[dict] = []
    for fam in order:
        if fam not in best_per_family:
            continue
        opt = dict(best_per_family[fam])
        # Re-simulate for plot traces
        spec = opt["spec"]
        session = simulate_from_spec_dict(sim, start, spec)
        _, metrics = aggregate_lifetime_reward(
            session, soc_target=soc_target, max_duration_min=max_dur,
        )
        opt = _entry_from_sim(
            opt["option_id"], session, metrics, source=opt.get("source", "selected"),
        )
        opt["_session"] = session
        final_options.append(opt)

    payload = {
        "initial_state": start,
        "constraints": {"soc_target": soc_target, "max_duration_min": max_dur},
        "families_requested": order,
        "options": [
            {k: v for k, v in o.items() if k != "_session"} for o in final_options
        ],
        "all_feasible_count": sum(1 for e in entries if e.get("feasible")),
    }
    json_path = ROOT / P.STAGE3_MODELS / "profile_alternatives.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)

    _plot_comparison(final_options, out_dir / "profile_options_by_family.png")
    _plot_tradeoff_table(final_options, out_dir / "profile_options_table.png")

    print(f"\n{'='*60}")
    print("  Profile options by family")
    print(f"{'='*60}")
    for o in final_options:
        levels = " → ".join(f"{x:.2f}" for x in o.get("charge_levels_a", []))
        print(
            f"  {o['family_label']:22s}  {o['option_id']:22s}  "
            f"loss={o['loss']:.1f}  dur={o['metrics']['duration_min']:.1f} min  "
            f"steps=[{levels}]"
        )
    missing = [f for f in order if f not in best_per_family]
    if missing:
        print(f"\n  No feasible option for: {', '.join(missing)}")
    print(f"\n  Saved {json_path}")
    print(f"  Saved {out_dir / 'profile_options_by_family.png'}")
    print(f"  Saved {out_dir / 'profile_options_table.png'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
