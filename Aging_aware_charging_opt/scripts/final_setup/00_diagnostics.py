#!/usr/bin/env python
"""Phase 0 diagnostics for Final_Aging_based_setup. No optimization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AUTHOR_SPLIT_SEED, AUTHOR_TRAIN_FRAC, AUTHOR_VAL_FRAC, CELLS, CHUNK_SIZE,
    FINE_TUNE_FRACTION, OUTPUT_ROOT, REPO_ROOT, SOURCE_PT_REL, author_splits,
    mean_std_cv, output_root, training_subset, write_csv,
)

from aacopt.bdt import FrozenBDT  # noqa: E402
from aacopt.config import (  # noqa: E402
    OptimizationSpec, Paths, RewardSpec, ensure_matplotlib_cache, file_hash,
    provenance, read_json, write_json,
)
from aacopt.evaluate import start_state_and_vnom  # noqa: E402


def _metrics(pred, ref) -> dict:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    err = pred - ref
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(ref), 1e-8)) * 100.0)
    return {
        "n": int(pred.size),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": mape,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "bias": float(np.mean(err)),
    }


def current_stats_from_subset(subset) -> dict:
    """Charging current is −I on NASA samples with I < 0."""
    from torch.utils.data import Subset

    assert isinstance(subset, Subset)
    currents = []
    voltages = []
    voltages_when_charging = []
    for idx in subset.indices:
        state, actions, next_states = subset.dataset[int(idx)]
        i = actions.squeeze(-1).detach().cpu().numpy().astype(np.float64)
        v_next = next_states[:, 0].detach().cpu().numpy().astype(np.float64)
        v0 = float(state[1].item())
        currents.append(i)
        voltages.append(np.concatenate([[v0], v_next]))
        charge = i < 0.0
        if np.any(charge):
            voltages_when_charging.append(v_next[charge])
    i_all = np.concatenate(currents) if currents else np.array([], dtype=np.float64)
    v_all = np.concatenate(voltages) if voltages else np.array([], dtype=np.float64)
    v_chg = (np.concatenate(voltages_when_charging)
             if voltages_when_charging else np.array([], dtype=np.float64))
    charge_mask = i_all < 0.0
    i_chg = -i_all[charge_mask]
    n_chg = int(i_chg.size)

    def _count_above(thr: float) -> int:
        return int(np.sum(i_chg > thr)) if n_chg else 0

    return {
        "n_windows": int(len(subset.indices)),
        "n_current_samples": int(i_all.size),
        "n_charging_samples": n_chg,
        "max_charging_current_a": float(i_chg.max()) if n_chg else float("nan"),
        "p99_charging_current_a": float(np.percentile(i_chg, 99)) if n_chg else float("nan"),
        "p50_charging_current_a": float(np.percentile(i_chg, 50)) if n_chg else float("nan"),
        "mean_charging_current_a": float(i_chg.mean()) if n_chg else float("nan"),
        "n_charging_above_3A": _count_above(3.0),
        "n_charging_above_4A": _count_above(4.0),
        "n_charging_above_4_5A": _count_above(4.5),
        "n_voltage_samples": int(v_all.size),
        "n_voltage_ge_4_1V": int(np.sum(v_all >= 4.1)) if v_all.size else 0,
        "n_charging_voltage_samples": int(v_chg.size),
        "n_charging_voltage_ge_4_1V": int(np.sum(v_chg >= 4.1)) if v_chg.size else 0,
        "max_voltage_v": float(v_all.max()) if v_all.size else float("nan"),
        "min_nasa_current_a": float(i_all.min()) if i_all.size else float("nan"),
        "max_nasa_current_a": float(i_all.max()) if i_all.size else float("nan"),
        "window_indices_n": int(len(subset.indices)),
        "window_indices_min": int(min(subset.indices)) if subset.indices else None,
        "window_indices_max": int(max(subset.indices)) if subset.indices else None,
    }


def load_test_indices(cell: str) -> list:
    _, _, _, _, test = author_splits(cell)
    return sorted(int(i) for i in test.indices)


def eval_twin_on_test(checkpoint: Path, cell: str, device: str) -> dict:
    from rw_transfer.data.author_dataset import author_subset_to_window_batch

    _, _, _, _, test = author_splits(cell)
    batch = author_subset_to_window_batch(test, max_windows=None)
    twin = FrozenBDT(checkpoint, device=device)
    v_hat, t_hat = twin.predict_windows(batch.X, batch_size=64)
    return {
        "cell": cell,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_hash(checkpoint),
        "split": "author_random_60_20_20_seed42_test",
        "n_windows": int(batch.X.shape[0]),
        "test_indices_sorted_sha256": __import__("hashlib").sha256(
            ",".join(str(i) for i in sorted(int(x) for x in test.indices)).encode()
        ).hexdigest(),
        "voltage": _metrics(v_hat, batch.Y_voltage),
        "temperature": _metrics(t_hat, batch.Y_temperature),
    }


def decompose_reward_terms(row: dict, spec: RewardSpec) -> dict:
    delta_soc = float(row["delta_soc"])
    q_total = float(row["q_total"])
    duration_min = float(row["duration_min"])
    t_h = duration_min / 60.0
    time_raw = (t_h ** spec.z_time) if t_h > 0.0 else 0.0
    term_soc = spec.w_soc * delta_soc
    term_qloss = spec.w_loss * q_total
    term_time = spec.w_time * time_raw
    reconstructed = term_soc - term_qloss - term_time
    recorded = float(row["reward"])
    return {
        "profile_id": row.get("profile_id"),
        "source_cell": row.get("source_cell"),
        "rank": int(row["rank"]) if str(row.get("rank", "")).strip() else None,
        "family_id": row.get("family_id"),
        "delta_soc": delta_soc,
        "q_total": q_total,
        "duration_min": duration_min,
        "duration_h": t_h,
        "term_w_soc_dSOC": term_soc,
        "term_w_loss_Q": term_qloss,
        "term_w_time_t_z": term_time,
        "reward_reconstructed": reconstructed,
        "reward_recorded": recorded,
        "reward_reconstruction_abs_err": abs(reconstructed - recorded),
        "abs_term_soc": abs(term_soc),
        "abs_term_qloss": abs(term_qloss),
        "abs_term_time": abs(term_time),
    }


def main() -> int:
    ensure_matplotlib_cache()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    opt = OptimizationSpec.load()
    reward = RewardSpec.load()
    paths = Paths.load()
    device = args.device or opt.device

    out = output_root() / "00_diagnostics"
    out.mkdir(parents=True, exist_ok=True)

    # ── 0.1 training-window currents ─────────────────────────────────────
    current_rows = []
    per_cell_current = {}
    for cell in CELLS:
        subset, split_name = training_subset(cell)
        stats = current_stats_from_subset(subset)
        stats.update({
            "cell": cell,
            "twin_kind": "source" if cell == "RW9" else "finetune_frac0.60",
            "checkpoint": str(paths.checkpoint(cell)),
            "checkpoint_sha256": file_hash(paths.checkpoint(cell)),
            "split": split_name,
        })
        per_cell_current[cell] = stats
        current_rows.append(stats)
        print(f"[0.1] {cell} max_chg={stats['max_charging_current_a']} A  "
              f"p99={stats['p99_charging_current_a']} A  "
              f"n>3/4/4.5={stats['n_charging_above_3A']}/"
              f"{stats['n_charging_above_4A']}/{stats['n_charging_above_4_5A']}",
              flush=True)

    src_subset, src_split = training_subset("RW9", source_twin=True)
    src_stats = current_stats_from_subset(src_subset)
    src_ckpt = REPO_ROOT / SOURCE_PT_REL
    src_stats.update({
        "cell": "twin_source_RW9.pt",
        "twin_kind": "source_RW9",
        "checkpoint": str(src_ckpt),
        "checkpoint_sha256": file_hash(src_ckpt),
        "split": src_split,
    })
    current_rows.append(src_stats)
    per_cell_current["twin_source_RW9.pt"] = src_stats
    print(f"[0.1] twin_source_RW9.pt max_chg={src_stats['max_charging_current_a']} A", flush=True)

    cell_maxima = {c: per_cell_current[c]["max_charging_current_a"] for c in CELLS}
    i_max_cap = float(min(cell_maxima.values()))
    i_max_cap_cell = min(cell_maxima, key=cell_maxima.get)

    write_json(out / "training_currents.json", {
        "per_cell": per_cell_current,
        "max_charging_current_per_cell_a": cell_maxima,
        "phase2_i_max_a": i_max_cap,
        "phase2_i_max_from_cell": i_max_cap_cell,
        "rule": "smallest of the four cells' training-set max charging currents",
    })
    write_csv(out / "training_currents.csv", current_rows, [
        "cell", "twin_kind", "split", "n_windows", "n_current_samples", "n_charging_samples",
        "max_charging_current_a", "p99_charging_current_a", "p50_charging_current_a",
        "mean_charging_current_a", "n_charging_above_3A", "n_charging_above_4A",
        "n_charging_above_4_5A", "n_voltage_samples", "n_voltage_ge_4_1V",
        "n_charging_voltage_samples", "n_charging_voltage_ge_4_1V", "max_voltage_v",
        "checkpoint", "checkpoint_sha256",
    ])

    # ── 0.2 identical held-out windows ───────────────────────────────────
    test_index_sets = {c: load_test_indices(c) for c in CELLS}
    identical_windows_note = (
        "Fine-tuned twins (frac 0.60) and twin_source_RW9.pt are both scored on "
        "the author 60/20/20 seed-42 TEST split of that cell. Training subsets "
        "differ (full train vs 60% prefix), but the held-out test windows are "
        "the same function of (cell, seed=42). Re-evaluation below uses one "
        "materialised test batch per cell for both models."
    )
    identical_by_construction = True
    eval_rows = []
    eval_payload = {"note": identical_windows_note, "per_cell": {}}
    for cell in CELLS:
        test_idx = test_index_sets[cell]
        print(f"[0.2] {cell} n_test_windows={len(test_idx)} — scoring both twins", flush=True)
        ft = eval_twin_on_test(paths.checkpoint(cell), cell, device)
        zs = eval_twin_on_test(src_ckpt, cell, device)
        same_hash = ft["test_indices_sorted_sha256"] == zs["test_indices_sorted_sha256"]
        same_n = ft["n_windows"] == zs["n_windows"]
        if not (same_hash and same_n):
            identical_by_construction = False
        v_ft, v_zs = ft["voltage"]["rmse"], zs["voltage"]["rmse"]
        t_ft, t_zs = ft["temperature"]["rmse"], zs["temperature"]["rmse"]
        rec = {
            "cell": cell,
            "n_test_windows": ft["n_windows"],
            "test_indices_sha256": ft["test_indices_sorted_sha256"],
            "identical_test_batch_both_models": bool(same_hash and same_n),
            "finetuned_or_source_cell_ckpt": str(paths.checkpoint(cell)),
            "zero_shot_ckpt": str(src_ckpt),
            "finetuned_voltage_rmse_v": v_ft,
            "zero_shot_voltage_rmse_v": v_zs,
            "finetuned_voltage_rmse_mv": 1000.0 * v_ft,
            "zero_shot_voltage_rmse_mv": 1000.0 * v_zs,
            "finetuned_temp_rmse_c": t_ft,
            "zero_shot_temp_rmse_c": t_zs,
            "finetuned_beats_zero_shot_voltage": bool(v_ft < v_zs),
            "finetuned_beats_zero_shot_temperature": bool(t_ft < t_zs),
            "voltage_rmse_delta_ft_minus_zs_v": v_ft - v_zs,
            "temp_rmse_delta_ft_minus_zs_c": t_ft - t_zs,
            "finetuned_voltage": ft["voltage"],
            "zero_shot_voltage": zs["voltage"],
            "finetuned_temperature": ft["temperature"],
            "zero_shot_temperature": zs["temperature"],
        }
        eval_payload["per_cell"][cell] = rec
        eval_rows.append(rec)
        print(
            f"      V RMSE ft={1000*v_ft} mV  zs={1000*v_zs} mV  "
            f"T RMSE ft={t_ft} C  zs={t_zs} C",
            flush=True,
        )
    eval_payload["identical_held_out_windows"] = identical_by_construction
    eval_payload["n_cells_finetune_better_voltage"] = int(sum(
        1 for c in CELLS if eval_payload["per_cell"][c]["finetuned_beats_zero_shot_voltage"]
    ))
    write_json(out / "identical_windows_rmse.json", eval_payload)
    write_csv(out / "identical_windows_rmse.csv", eval_rows, [
        "cell", "n_test_windows", "identical_test_batch_both_models",
        "finetuned_voltage_rmse_v", "zero_shot_voltage_rmse_v",
        "finetuned_voltage_rmse_mv", "zero_shot_voltage_rmse_mv",
        "finetuned_temp_rmse_c", "zero_shot_temp_rmse_c",
        "finetuned_beats_zero_shot_voltage", "finetuned_beats_zero_shot_temperature",
        "voltage_rmse_delta_ft_minus_zs_v", "temp_rmse_delta_ft_minus_zs_c",
    ])

    # ── 0.3 original GP-BO initial conditions ────────────────────────────
    orig_ics = {}
    for cell in CELLS:
        summary_path = Path(paths.matlab_dir)  # dummy to satisfy type checkers
        del summary_path
        gp_summary = read_json(REPO_ROOT / "Aging_aware_charging_opt" / "results" /
                               "08_gp_bo" / cell / "summary.json")
        recomputed, v_nom = start_state_and_vnom(cell, opt.session)
        orig_ics[cell] = {
            "from_08_gp_bo_summary": gp_summary["start_state"],
            "v_nom": gp_summary.get("v_nom"),
            "checkpoint": gp_summary.get("checkpoint"),
            "recomputed_from_ocv_soc": recomputed,
            "recomputed_v_nom": v_nom,
            "matches_start_state_and_vnom": (
                abs(float(gp_summary["start_state"]["soc"]) - float(recomputed["soc"])) < 1e-12
                and abs(float(gp_summary["start_state"]["t0"]) - float(recomputed["t0"])) < 1e-12
                and abs(float(gp_summary["start_state"]["age"]) - float(recomputed["age"])) < 1e-12
                and abs(float(gp_summary["start_state"]["v0"]) - float(recomputed["v0"])) < 1e-9
            ),
        }
    write_json(out / "original_initial_conditions.json", orig_ics)

    shared_soc0 = float(opt.session.soc_start)
    shared_t0 = float(opt.session.ambient_t0_c)
    shared_age = 0.0
    v0_per_cell = {c: orig_ics[c]["from_08_gp_bo_summary"]["v0"] for c in CELLS}

    # ── 0.4 elite reward-term decomposition ──────────────────────────────
    elite_path = (REPO_ROOT / "Aging_aware_charging_opt" / "results" /
                  "14_litmus_top6" / "elite_master.csv")
    import csv
    elite_rows = []
    with elite_path.open("r", encoding="utf-8", newline="") as fh:
        elite_rows = list(csv.DictReader(fh))
    term_rows = [decompose_reward_terms(r, reward) for r in elite_rows]
    write_csv(out / "elite_reward_terms.csv", term_rows, [
        "profile_id", "source_cell", "rank", "family_id", "delta_soc", "q_total",
        "duration_min", "duration_h", "term_w_soc_dSOC", "term_w_loss_Q",
        "term_w_time_t_z", "reward_reconstructed", "reward_recorded",
        "reward_reconstruction_abs_err", "abs_term_soc", "abs_term_qloss", "abs_term_time",
    ])
    term_spread = {
        "term_w_soc_dSOC": mean_std_cv([r["term_w_soc_dSOC"] for r in term_rows]),
        "term_w_loss_Q": mean_std_cv([r["term_w_loss_Q"] for r in term_rows]),
        "term_w_time_t_z": mean_std_cv([r["term_w_time_t_z"] for r in term_rows]),
        "reward_recorded": mean_std_cv([r["reward_recorded"] for r in term_rows]),
    }
    mean_abs = {
        "mean_abs_term_soc": float(np.mean([r["abs_term_soc"] for r in term_rows])),
        "mean_abs_term_qloss": float(np.mean([r["abs_term_qloss"] for r in term_rows])),
        "mean_abs_term_time": float(np.mean([r["abs_term_time"] for r in term_rows])),
    }
    write_json(out / "elite_reward_terms.json", {
        "n": len(term_rows),
        "weights": reward.to_dict(),
        "spread": term_spread,
        "mean_abs_magnitude": mean_abs,
        "ratio_mean_abs_soc_over_qloss": mean_abs["mean_abs_term_soc"] / mean_abs["mean_abs_term_qloss"],
        "ratio_mean_abs_soc_over_time": mean_abs["mean_abs_term_soc"] / mean_abs["mean_abs_term_time"],
        "ratio_mean_abs_time_over_qloss": mean_abs["mean_abs_term_time"] / mean_abs["mean_abs_term_qloss"],
        "max_reconstruction_abs_err": max(r["reward_reconstruction_abs_err"] for r in term_rows),
        "rows": term_rows,
    })

    payload = {
        "phase": 0,
        "training_currents": per_cell_current,
        "phase2_i_max_a": i_max_cap,
        "phase2_i_max_from_cell": i_max_cap_cell,
        "identical_held_out_windows": identical_by_construction,
        "window_eval": eval_payload,
        "original_initial_conditions": orig_ics,
        "shared_init_for_phase2": {
            "soc0": shared_soc0,
            "t0_c": shared_t0,
            "age": shared_age,
            "v0_policy": "cell-specific OCV(SOC0) via start_state_and_vnom",
            "v0_per_cell": v0_per_cell,
            "note": (
                "Original 08_gp_bo runs already used this same (SOC0, T0, age) "
                "with cell OCV V0. Phase 2 change (b) therefore matches the "
                "original IC policy; only the current cap is a new search-space change."
            ),
        },
        "elite_reward_term_spread": term_spread,
        "elite_reward_term_mean_abs": mean_abs,
        "provenance": provenance(
            "Final_Aging_based_setup_00",
            configs=["paths", "reward", "optimization"],
            inputs=[paths.checkpoint(c) for c in CELLS] + [src_ckpt],
        ),
    }
    write_json(out / "diagnostics.json", payload)

    # config_used.json (filled with everything known before optimization)
    opt_seeds = {
        c: {fid: opt.seed_for(c, fid, "gp_bo") for fid in opt.families} for c in CELLS
    }
    config_used = {
        "output_root": str(OUTPUT_ROOT),
        "cells": list(CELLS),
        "do_not_tune_to_pass": True,
        "phase2_changes_only": [
            "current upper bound = min over RW9–RW12 of max charging current in that twin's training windows",
            "identical SOC0/T0/age; cell V0 from OCV(SOC0)",
        ],
        "i_min_a": opt.space.i_min_a,
        "i_max_a_original": opt.space.i_max_a,
        "i_max_a_phase2": i_max_cap,
        "i_max_a_phase2_from_cell": i_max_cap_cell,
        "i_max_a_phase2_per_cell_training_max": cell_maxima,
        "soc0": shared_soc0,
        "t0_c": shared_t0,
        "age": shared_age,
        "v0_policy": "cell-specific OCV(SOC0) via aacopt.evaluate.start_state_and_vnom",
        "v0_per_cell": v0_per_cell,
        "energy_fraction": opt.session.energy_fraction,
        "v_max": opt.session.v_max,
        "n_evals_per_family": opt.n_evals_per_family,
        "families": list(opt.families),
        "acq_func": opt.gp_bo.get("acq_func"),
        "acq_xi": opt.gp_bo.get("acq_xi"),
        "n_initial_points": opt.gp_bo.get("n_initial_points"),
        "noise": opt.gp_bo.get("noise"),
        "reward": reward.to_dict(),
        "seeds_gp_bo": opt_seeds,
        "master_seed": opt.master_seed,
        "checkpoints": {c: str(paths.checkpoint(c)) for c in CELLS},
        "checkpoint_sha256": {c: file_hash(paths.checkpoint(c)) for c in CELLS},
        "twin_source_RW9_pt": str(src_ckpt),
        "twin_source_RW9_pt_sha256": file_hash(src_ckpt),
        "author_split": {
            "seed": AUTHOR_SPLIT_SEED,
            "train_frac": AUTHOR_TRAIN_FRAC,
            "val_frac": AUTHOR_VAL_FRAC,
            "chunk_size": CHUNK_SIZE,
            "fine_tune_fraction": FINE_TUNE_FRACTION,
        },
        "device": device,
    }
    write_json(OUTPUT_ROOT / "config_used.json", config_used)

    def fmt(x, nd=None):
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            return repr(float(x))
        return str(x)

    cur_lines = []
    for r in current_rows:
        cur_lines.append(
            f"- **{r['cell']}** ({r['twin_kind']}, {r['split']}): "
            f"n_windows={r['n_windows']}, n_charging={r['n_charging_samples']}, "
            f"max_charging={fmt(r['max_charging_current_a'])} A, "
            f"p99={fmt(r['p99_charging_current_a'])} A, "
            f"n>3A={r['n_charging_above_3A']}, n>4A={r['n_charging_above_4A']}, "
            f"n>4.5A={r['n_charging_above_4_5A']}, "
            f"n(V≥4.1 V)={r['n_voltage_ge_4_1V']} of {r['n_voltage_samples']} voltage samples "
            f"(charging-only V≥4.1: {r['n_charging_voltage_ge_4_1V']} of {r['n_charging_voltage_samples']})."
        )
    win_lines = []
    for c in CELLS:
        r = eval_payload["per_cell"][c]
        win_lines.append(
            f"- **{c}**: n_test={r['n_test_windows']}, identical_batch={r['identical_test_batch_both_models']}. "
            f"Voltage RMSE finetuned/source-cell twin = {fmt(r['finetuned_voltage_rmse_v'])} V "
            f"({fmt(r['finetuned_voltage_rmse_mv'])} mV); "
            f"zero-shot twin_source_RW9.pt = {fmt(r['zero_shot_voltage_rmse_v'])} V "
            f"({fmt(r['zero_shot_voltage_rmse_mv'])} mV); "
            f"finetuned better on V? {r['finetuned_beats_zero_shot_voltage']}. "
            f"Temp RMSE finetuned = {fmt(r['finetuned_temp_rmse_c'])} °C; "
            f"zero-shot = {fmt(r['zero_shot_temp_rmse_c'])} °C; "
            f"finetuned better on T? {r['finetuned_beats_zero_shot_temperature']}."
        )
    ic_lines = []
    for c in CELLS:
        s = orig_ics[c]["from_08_gp_bo_summary"]
        ic_lines.append(
            f"- **{c}**: SOC0={fmt(s['soc'])}, V0={fmt(s['v0'])} V, "
            f"T0={fmt(s['t0'])} °C, age={fmt(s['age'])}, prev_i={fmt(s['prev_i'])}. "
            f"Matches start_state_and_vnom: {orig_ics[c]['matches_start_state_and_vnom']}."
        )
    ts = term_spread
    md = f"""# Phase 0 diagnostics

No optimization was run. Numbers are as computed (Python `repr` / full JSON).

## 0.1 Training-window charging currents

NASA sign: negative current is charge. Charging current reported here is `−I` on samples with `I < 0`.

Windows used:
- RW9 / `twin_source_RW9.pt`: author random 60/20/20 seed-42 **train** split.
- RW10–RW12 frac0.60 twins: deterministic **prefix** of that train split at fraction {FINE_TUNE_FRACTION}.

{chr(10).join(cur_lines)}

**Phase 2 current upper bound** = min of the four cell training maxima = **{fmt(i_max_cap)} A** (from {i_max_cap_cell}).

JSON: `training_currents.json`.

## 0.2 Identical held-out windows

{identical_windows_note}

Identical test batches for both models on every cell: **{identical_by_construction}**.

{chr(10).join(win_lines)}

Cells where the cell-specific checkpoint has lower voltage RMSE than zero-shot `twin_source_RW9.pt`: {eval_payload['n_cells_finetune_better_voltage']} / 4.

JSON: `identical_windows_rmse.json`.

## 0.3 Original GP-BO initial conditions (`results/08_gp_bo/<CELL>/summary.json`)

{chr(10).join(ic_lines)}

Shared (SOC0, T0, age) already used: SOC0={fmt(shared_soc0)}, T0={fmt(shared_t0)} °C, age={fmt(shared_age)}.
V0 differs by cell because it is taken from that cell's OCV–SOC curve at SOC0 (`start_state_and_vnom`). Phase 2 will keep this policy.

JSON: `original_initial_conditions.json`.

## 0.4 Reward-term decomposition of existing 14_litmus_top6 elites

R = w_soc·ΔSOC − w_loss·Q_loss − w_time·t_h^z_time
with w_soc={fmt(reward.w_soc)}, w_loss={fmt(reward.w_loss)}, w_time={fmt(reward.w_time)}, z_time={fmt(reward.z_time)}.

Mean |term| across {len(term_rows)} elites:
- |w_soc·ΔSOC| = {fmt(mean_abs['mean_abs_term_soc'])}
- |w_loss·Q_loss| = {fmt(mean_abs['mean_abs_term_qloss'])}
- |w_time·t^z| = {fmt(mean_abs['mean_abs_term_time'])}

Ratios of those means: SOC/Q = {fmt(mean_abs['mean_abs_term_soc'] / mean_abs['mean_abs_term_qloss'])}; SOC/time = {fmt(mean_abs['mean_abs_term_soc'] / mean_abs['mean_abs_term_time'])}; time/Q = {fmt(mean_abs['mean_abs_term_time'] / mean_abs['mean_abs_term_qloss'])}.

Spread (mean, std, CV%, min, max):
- w_soc·ΔSOC: {fmt(ts['term_w_soc_dSOC']['mean'])}, {fmt(ts['term_w_soc_dSOC']['std'])}, {fmt(ts['term_w_soc_dSOC']['cv_pct'])} %, {fmt(ts['term_w_soc_dSOC']['min'])}, {fmt(ts['term_w_soc_dSOC']['max'])}
- w_loss·Q: {fmt(ts['term_w_loss_Q']['mean'])}, {fmt(ts['term_w_loss_Q']['std'])}, {fmt(ts['term_w_loss_Q']['cv_pct'])} %, {fmt(ts['term_w_loss_Q']['min'])}, {fmt(ts['term_w_loss_Q']['max'])}
- w_time·t^z: {fmt(ts['term_w_time_t_z']['mean'])}, {fmt(ts['term_w_time_t_z']['std'])}, {fmt(ts['term_w_time_t_z']['cv_pct'])} %, {fmt(ts['term_w_time_t_z']['min'])}, {fmt(ts['term_w_time_t_z']['max'])}

Max |reconstructed R − recorded R| = {fmt(max(r['reward_reconstruction_abs_err'] for r in term_rows))}.

CSV: `elite_reward_terms.csv`.
"""
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    print(f"\nWrote {out / 'SUMMARY.md'}", flush=True)
    print(f"Phase 2 i_max_a = {i_max_cap} A (from {i_max_cap_cell})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
