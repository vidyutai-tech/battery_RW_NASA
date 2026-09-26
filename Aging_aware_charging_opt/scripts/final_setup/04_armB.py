#!/usr/bin/env python
"""Phase 4 Arm B: shared-surrogate transfer test on twin_source_RW9.pt.

This is NOT a litmus test. One GP-BO per family on the RW9 source twin,
then one Top-6, replayed on the four cell-specific twins and on the source twin.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CELLS, N_ELITE, REPO_ROOT, SOURCE_PT_REL, output_root, read_json, write_csv,
    write_json, run_gp_bo_cell, load_elite, params_from_dict, make_simulator_ckpt,
    apply_i_max, _md_table, FAMILY_LABELS, make_rescale_transform,
)
from aacopt.config import OptimizationSpec, Paths, RewardSpec, provenance  # noqa: E402
from aacopt.evaluate import (  # noqa: E402
    cccv_params, evaluate_params, load_anchors, load_degradation_model,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    opt = OptimizationSpec.load()
    paths = Paths.load()
    reward = RewardSpec.load()
    device = args.device or opt.device
    root = output_root()
    cfg = read_json(root / "config_used.json")
    i_max = float(cfg["i_max_a_phase2"])
    src_ckpt = REPO_ROOT / SOURCE_PT_REL

    out = root / "04_armB_shared_RW9"
    gp_dir = out / "gp_bo"
    gp_dir.mkdir(parents=True, exist_ok=True)

    armA = root / "02_armA" / "armA_verdict.json"
    use_phase3 = armA.is_file() and read_json(armA).get("verdict") != "PASS"
    loss_transform = None
    halfc_metrics = None
    if use_phase3:
        apply_i_max(i_max)
        model0 = load_degradation_model()
        sim0, state0 = make_simulator_ckpt(
            "RW9", device=device, i_max_a=i_max, checkpoint=src_ckpt,
        )
        _, halfc_metrics, _ = evaluate_params(
            sim0, state0, cccv_params(0.5, v_cv=4.20, i_cutoff=0.05),
            model=model0, spec=reward, anchors=load_anchors("RW9"),
        )
        halfc_metrics = {k: v for k, v in halfc_metrics.items() if k not in ("weights", "anchors")}
        # Primary Phase 3 variant: three-term normalization. Drop-ΔSOC did not pass either.
        loss_transform = make_rescale_transform(halfc_metrics, drop_soc=False)
        print("Arm B uses Phase 3 normalize_three_terms (Arm A FAIL; neither Phase 3 variant passed C1–C3).", flush=True)

    print(f"\n======== Arm B shared surrogate  twin_source_RW9.pt  i_max={i_max} ========", flush=True)
    run_gp_bo_cell(
        "RW9", gp_dir / "RW9", device=device, i_max_a=i_max,
        checkpoint=src_ckpt, seed_cell="RW9", resume=not args.no_resume,
        loss_transform=loss_transform,
    )

    elite = load_elite(gp_dir, ["RW9"], N_ELITE)
    for e in elite:
        e["profile_id"] = f"SRC-P{e['rank']}"
        e["params"] = params_from_dict(e["params"]).to_dict()

    eval_names = list(CELLS) + ["twin_source_RW9"]
    ckpts = {c: paths.checkpoint(c) for c in CELLS}
    ckpts["twin_source_RW9"] = src_ckpt
    ocv_cells = {c: c for c in CELLS}
    ocv_cells["twin_source_RW9"] = "RW9"

    model = load_degradation_model()
    sims = {}
    states = {}
    anchors = {}
    for name in eval_names:
        print(f"Loading {name} …", flush=True)
        apply_i_max(i_max)
        sims[name], states[name] = make_simulator_ckpt(
            ocv_cells[name], device=device, i_max_a=i_max, checkpoint=ckpts[name],
        )
        anchors[name] = load_anchors(ocv_cells[name])

    rows = []
    n = 0
    total = len(elite) * len(eval_names)
    for e in elite:
        params = params_from_dict(e["params"])
        e["params"] = params.to_dict()
        for name in eval_names:
            n += 1
            print(f"  [{n}/{total}] {e['profile_id']} → {name}", flush=True)
            apply_i_max(i_max)
            loss, metrics, session = evaluate_params(
                sims[name], states[name], params, model=model, spec=reward,
                anchors=anchors[name],
            )
            rows.append({
                "profile_id": e["profile_id"],
                "rank": e["rank"],
                "family_id": e["family_id"],
                "eval_twin": name,
                "checkpoint": str(ckpts[name]),
                "feasible": bool(metrics["feasible"]),
                "peak_v": float(metrics["peak_v"]),
                "duration_min": float(metrics["duration_min"]),
                "q_total": float(metrics["q_total"]),
                "reward": float(metrics["reward"]),
                "loss": float(loss),
                "delta_soc": float(metrics["delta_soc"]),
                "end_reason": metrics.get("end_reason"),
                "soc_reward": float(metrics["soc_reward"]),
                "qloss_penalty": float(metrics["qloss_penalty"]),
                "time_penalty": float(metrics["time_penalty"]),
            })

    write_csv(out / "transfer_table.csv", rows, [
        "profile_id", "rank", "family_id", "eval_twin", "feasible", "peak_v",
        "duration_min", "q_total", "reward", "loss", "delta_soc", "end_reason",
        "soc_reward", "qloss_penalty", "time_penalty", "checkpoint",
    ])
    write_json(out / "elites.json", [
        {"profile_id": e["profile_id"], "rank": e["rank"], "family_id": e["family_id"],
         "loss": e["loss"], "reward": e["reward"], "params": e["params"]}
        for e in elite
    ])

    headers = ["profile", "family"] + eval_names
    feas_rows, vp_rows, t_rows, q_rows, r_rows = [], [], [], [], []
    for e in elite:
        hits = {r["eval_twin"]: r for r in rows if r["profile_id"] == e["profile_id"]}
        label = e["profile_id"]
        fam = FAMILY_LABELS.get(e["family_id"], e["family_id"])
        feas_rows.append([label, fam] + [("yes" if hits[n]["feasible"] else "no") for n in eval_names])
        vp_rows.append([label, fam] + [repr(hits[n]["peak_v"]) for n in eval_names])
        t_rows.append([label, fam] + [repr(hits[n]["duration_min"]) for n in eval_names])
        q_rows.append([label, fam] + [repr(hits[n]["q_total"]) for n in eval_names])
        r_rows.append([label, fam] + [repr(hits[n]["reward"]) for n in eval_names])

    same_ckpt = str(paths.checkpoint("RW9")) == str(src_ckpt)
    reward_note = (
        "Phase 3 normalize_three_terms (Arm A failed, so Phase 3 settings are used). "
        "Neither Phase 3 variant passed C1–C3; this arm is still a transfer test, not a litmus test."
        if use_phase3 else
        "Phase 2 published reward (Arm A passed, Phase 3 skipped)."
    )
    md = f"""# Arm B — shared-surrogate transfer test (not a litmus test)

Optimizer twin: `twin_source_RW9.pt`
Settings: current cap, ICs, families, 80 evals, EI, energy f=0.40, V_max=4.20 V as Phase 2.
Reward: {reward_note}
RW9 cell-specific checkpoint is the same file as twin_source_RW9.pt: {same_ckpt}.

## Top-6 (from the shared search)

{_md_table(
    ['id', 'family', 'params'],
    [[e['profile_id'], FAMILY_LABELS.get(e['family_id'], e['family_id']), str(e['params'])] for e in elite],
)}

## Feasibility

{_md_table(headers, feas_rows)}

## V_peak (V)

{_md_table(headers, vp_rows)}

## Time (min)

{_md_table(headers, t_rows)}

## Q_loss

{_md_table(headers, q_rows)}

## Reward

{_md_table(headers, r_rows)}
"""
    (out / "TRANSFER.md").write_text(md, encoding="utf-8")
    write_json(out / "summary.json", {
        "label": "shared-surrogate transfer test",
        "not_a_litmus_test": True,
        "i_max_a": i_max,
        "used_phase3_normalize_three_terms": bool(use_phase3),
        "halfc_baseline_rw9": halfc_metrics,
        "source_checkpoint": str(src_ckpt),
        "rw9_checkpoint_is_source": same_ckpt,
        "n_elite": len(elite),
        "n_replay": len(rows),
        "rows": rows,
        "provenance": provenance(
            "Final_Aging_based_setup_armB",
            configs=["paths", "reward", "optimization", "degradation_fitted"],
            inputs=[src_ckpt],
        ),
    })
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
