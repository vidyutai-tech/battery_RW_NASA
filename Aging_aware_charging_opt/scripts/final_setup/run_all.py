#!/usr/bin/env python
"""Orchestrate Final_Aging_based_setup phases 0–4 and write REPORT.md."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CELLS, OUTPUT_ROOT, output_root, read_json  # noqa: E402


HERE = Path(__file__).resolve().parent


def run_script(name: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(HERE / name), *extra]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def fmt(x):
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        return repr(x)
    return str(x)


def write_report() -> None:
    root = output_root()
    d0 = read_json(root / "00_diagnostics" / "diagnostics.json")
    cfg = read_json(root / "config_used.json")
    armA_path = root / "02_armA" / "armA_verdict.json"
    armA = read_json(armA_path) if armA_path.is_file() else {}
    lit = read_json(root / "02_armA" / "litmus" / "summary.json") if (
        root / "02_armA" / "litmus" / "summary.json"
    ).is_file() else {}

    p3_skip = (root / "03_armA_rescaled" / "SKIPPED.md").is_file()
    p3 = None
    if (root / "03_armA_rescaled" / "summary.json").is_file():
        p3 = read_json(root / "03_armA_rescaled" / "summary.json")

    armB_md = ""
    if (root / "04_armB_shared_RW9" / "TRANSFER.md").is_file():
        armB_md = (root / "04_armB_shared_RW9" / "TRANSFER.md").read_text(encoding="utf-8")

    i_max = cfg["i_max_a_phase2"]
    maxima = cfg["i_max_a_phase2_per_cell_training_max"]
    win = d0["window_eval"]["per_cell"]
    mean_abs = d0["elite_reward_term_mean_abs"]

    c1 = armA.get("C1", {})
    c2 = armA.get("C2", {})
    c3 = armA.get("C3", {})

    def c1_block():
        lines = []
        for cell in CELLS:
            d = (c1.get("per_cell") or {}).get(cell, {})
            lines.append(
                f"- {cell}: dominant={d.get('dominant_family')} count={d.get('count')} "
                f"→ {'PASS' if d.get('pass') else 'FAIL'} (counts={d.get('counts')})"
            )
        return "\n".join(lines)

    def c3_block():
        lines = []
        for cell in CELLS:
            d = (c3.get("per_cell") or {}).get(cell, {})
            lines.append(
                f"- {cell}: {d.get('n_feasible_on_all_twins')} / {d.get('of')} "
                f"feasible on all four twins → {'PASS' if d.get('pass') else 'FAIL'}"
            )
        return "\n".join(lines)

    win_lines = []
    for cell in CELLS:
        r = win[cell]
        win_lines.append(
            f"- {cell}: V RMSE cell-twin={fmt(r['finetuned_voltage_rmse_v'])} V vs "
            f"zero-shot={fmt(r['zero_shot_voltage_rmse_v'])} V "
            f"(cell-twin better? {r['finetuned_beats_zero_shot_voltage']}); "
            f"T RMSE cell-twin={fmt(r['finetuned_temp_rmse_c'])} °C vs "
            f"zero-shot={fmt(r['zero_shot_temp_rmse_c'])} °C "
            f"(cell-twin better? {r['finetuned_beats_zero_shot_temperature']})."
        )

    if p3_skip:
        p3_text = "Phase 3 was not run because Arm A passed C1–C3."
    elif p3:
        p3_text = (
            f"normalize_three_terms: {p3['normalize_three_terms']['verdict']}\n\n"
            f"normalize_drop_dsoc: {p3['normalize_drop_dsoc']['verdict']}\n\n"
            f"Any variant pass: {p3['any_pass']}"
        )
    else:
        p3_text = "Phase 3 outputs not found."

    # Which change produced consistency
    orig_cv = None
    orig_spread = REPO_ROOT_SPREAD()
    consistency_line = (
        "Arm A changed only the current upper bound relative to the original GP-BO "
        "(SOC0/T0/age were already shared, with cell OCV V0). "
        f"Original 14_litmus_top6 I_max CV was {orig_spread}. "
        f"Arm A I_max CV is {c2.get('cv_pct_peak_current')}. "
        f"Arm A verdict is {armA.get('verdict')}. "
        + (
            "Reward rescaling was not applied."
            if p3_skip else
            "Reward rescaling was applied because Arm A failed; see Phase 3 verdicts. "
            "No criterion was changed after seeing results."
        )
    )

    report = f"""# Final Aging-based setup — pre-registered Top-6 re-test

This folder is a **new** run. Existing `results/08_*`, `results/14_litmus_top6/`, and `results/15_zero_shot_rw9_twins/` were not modified. Criteria C1–C3 were written before Arm A and were not edited afterwards.

Logged settings: `config_used.json`.

## Phase 0 findings

### Max supported charging current (training windows)

NASA sign: negative current is charge. Values are max(−I) on I<0 samples in the windows actually used to train each twin.

- RW9: {fmt(maxima['RW9'])} A
- RW10: {fmt(maxima['RW10'])} A
- RW11: {fmt(maxima['RW11'])} A
- RW12: {fmt(maxima['RW12'])} A

Phase 2 / Arm A / Arm B current upper bound = min of those four = **{fmt(i_max)} A** (from {cfg['i_max_a_phase2_from_cell']}). Original search used 6.00 A.

Details: `00_diagnostics/SUMMARY.md`.

### Fine-tuned twins vs zero-shot `twin_source_RW9.pt` on identical held-out windows

Held-out windows are the author 60/20/20 seed-42 test split of each cell. Both models were scored on the same materialised batch per cell. Identical batches: {d0['identical_held_out_windows']}.

{chr(10).join(win_lines)}

### Original GP-BO initial conditions

All four original runs used SOC0={fmt(cfg['soc0'])}, T0={fmt(cfg['t0_c'])} °C, age={fmt(cfg['age'])}, with cell-specific V0 from OCV(SOC0): {cfg['v0_per_cell']}. Phase 2 keeps that policy.

### Original elite reward-term scales (14_litmus_top6)

Mean |w_soc·ΔSOC| = {fmt(mean_abs['mean_abs_term_soc'])}; mean |w_loss·Q| = {fmt(mean_abs['mean_abs_term_qloss'])}; mean |w_time·t^z| = {fmt(mean_abs['mean_abs_term_time'])}.

## Phase 1 — frozen criteria

See `01_criteria.md`. Verdict is PASS only if C1, C2, and C3 all hold.

## Arm A verdict (primary litmus)

**{armA.get('verdict', 'MISSING')}**

C1: **{'PASS' if c1.get('pass') else 'FAIL'}**
{c1_block()}

C2: **{'PASS' if c2.get('pass') else 'FAIL'}**
CV of peak current across {c2.get('n_elites')} elites = {fmt(c2.get('cv_pct_peak_current'))} % (threshold 15).
mean={fmt(c2.get('mean'))} A, std={fmt(c2.get('std'))} A, min={fmt(c2.get('min'))} A, max={fmt(c2.get('max'))} A.

C3: **{'PASS' if c3.get('pass') else 'FAIL'}**
{c3_block()}

Family counts: {lit.get('family_counts')}
Profiles feasible on all four twins: {lit.get('n_profiles_feasible_on_all_eval_twins')} / {lit.get('n_elite')}

Tables and figures: `02_armA/litmus/`.

## Phase 3 (reward rescaling)

{p3_text}

## Arm B — shared-surrogate transfer test

{armB_md}

## What produced any consistency

{consistency_line}

Any failure is reported as a failure. Criteria were not changed post hoc.
"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Wrote {root / 'REPORT.md'}", flush=True)


def REPO_ROOT_SPREAD():
    p = Path("/data/battery_RW_NASA/Aging_aware_charging_opt/results/14_litmus_top6/summary.json")
    if not p.is_file():
        return "unavailable"
    blob = read_json(p)
    return (blob.get("spread") or {}).get("I_max (A)", {}).get("cv_pct", "unavailable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-diagnostics", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    extra = ["--device", args.device] if args.device else []

    if args.report_only:
        write_report()
        return 0

    output_root()
    if not args.skip_diagnostics:
        run_script("00_diagnostics.py", extra)

    criteria = """# Pre-registered pass criteria

Written before any new GP-BO run. These criteria are not changed later.

- **C1:** For every cell, the same dominant family appears in >= 4 of its Top-6.
- **C2:** Coefficient of variation of peak current across all 24 elites < 15%.
- **C3:** For every cell, >= 4 of its 6 elites are feasible (V_peak <= 4.20 V and energy target met) on ALL four twins.

Verdict: PASS only if C1, C2, C3 all hold.
"""
    (OUTPUT_ROOT / "01_criteria.md").write_text(criteria, encoding="utf-8")
    print("Wrote 01_criteria.md", flush=True)

    run_script("02_armA.py", extra)
    run_script("03_armA_rescaled.py", extra)
    run_script("04_armB.py", extra)
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
