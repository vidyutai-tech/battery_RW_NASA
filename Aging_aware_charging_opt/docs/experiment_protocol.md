# Experiment Protocol

The exact, ordered procedure. Each stage is a single script, reads only the
artifacts listed as its inputs, and writes only into its own results
sub-directory. No stage writes outside `Aging_aware_charging_opt/results/`.

Run everything with the repository virtualenv:

```bash
cd /data/battery_RW_NASA
export PYTHONPATH=Aging_aware_charging_opt/src
VENV=venv/bin/python
```

---

## Stage table

| # | Script | Reads | Writes | Gate |
|---|---|---|---|---|
| 0 | `scripts/00_verify_environment.py` | — | `results/00_environment/environment.json` | packages importable, checkpoints and `.mat` files present |
| 1 | `scripts/01_build_calibration_dataset.py` | `NASA_RW/.../RW*.mat` | `results/01_calibration_dataset/` | ≥ 70 accepted references per cell |
| 2 | `scripts/02_verify_bdt.py` | BDT `.pt` checkpoints, `RW*.mat` | `results/02_bdt_verification/` | max-abs equivalence < 1e-5 vs original implementation |
| 3 | `scripts/03_fit_degradation.py` | `results/01_calibration_dataset/calibration_intervals.csv` | `results/03_degradation_fit/`, `configs/degradation_fitted.yaml` | converged, finite covariance |
| 4 | `scripts/04_validate_degradation.py` | stage-1 dataset, `configs/degradation.yaml` | `results/04_degradation_validation/` | **pooled LOCO R² ≥ 0.90 and \|bias\| ≤ 0.02** |
| 5 | `scripts/05_build_reward_anchors.py` | `configs/degradation_fitted.yaml`, BDT ckpts | `results/05_reward_anchors/reward_anchors.json` | CCCV 1C feasible on all cells, anchors finite |
| 6 | `scripts/06_sanity_checks.py` | stages 3 and 5 | `results/06_sanity_checks/` | all assertions pass |
| 7 | `scripts/07_run_random_search.py` | stages 3 and 5 | `results/07_random_search/<CELL>/` | ≥ 1 feasible point per cell |
| 8 | `scripts/08_run_gp_bo.py` | stages 3 and 5 | `results/08_gp_bo/<CELL>/` | ≥ 1 feasible point per cell |
| 9 | `scripts/09_compare_optimizers.py` | stages 5, 7, 8 | `results/09_comparison/` | budgets equal, objectives identical |
| 10 | `scripts/10_cross_cell_analysis.py` | stage 9 | `results/10_cross_cell/` | — |
| 11 | `scripts/11_lifetime_analysis.py` | stages 3, 9 | `results/11_lifetime/` | — |
| 12 | `scripts/12_make_figures.py` | stages 1–11 | `results/figures/` | — |
| 13 | `scripts/13_make_tables.py` | stages 1–11 | `results/tables/` | — |
| 14 | `scripts/validate_experiment.py` | everything | `results/validation_report.txt` | all checks PASS |

Convenience driver: `bash scripts/run_all.sh [--device cuda]`.

---

## Separation of concerns (RULE 6 and RULE 7)

The pipeline is a strict DAG and the arrows only go one way:

```
stage 1 (measurement) ──► stage 3 (fit) ──► stage 4 (validate)
                                   │
                                   └──► stage 5 (anchors) ──► stages 7,8 (optimize)
                                                                    │
                                                                    ▼
                                                    stages 9-11 (analysis) ──► 12,13 (report)
```

Enforced consequences:

* Stages 7–11 **cannot** influence stage 3 or 4: they run later and write to
  different directories. `validate_experiment.py` asserts that
  `configs/degradation_fitted.yaml` has an `mtime` and a content hash that
  predate every optimization result file.
* Stage 4 (validation) writes metrics only. No stage reads stage 4's output as
  an input to any fit or search, so the validation data cannot be tuned on
  (RULE 5).
* Stage 5's anchors are written once. Stages 7 and 8 read the same file and
  `validate_experiment.py` asserts both runs recorded the same anchor values.
* Stage 7 (Random Search) and stage 8 (GP-BO) do not read each other's output.
  There is no `qloss_cap`, no elite warm-start, no shared best-so-far.
  `validate_experiment.py` asserts that neither results file references the
  other's path and that the recorded reward configuration is byte-identical.

---

## Fixed experimental settings

| Setting | Value | Where |
|---|---|---|
| Cells | RW9, RW10, RW11, RW12 | `configs/paths.yaml` |
| Start state | OCV-consistent rest at SOC = 0.20, 24 °C | `configs/optimization.yaml` |
| Session constraint | delivered energy = 40 % of nominal pack energy | `configs/optimization.yaml` |
| Time budget | 150 min | `configs/optimization.yaml` |
| Voltage ceiling | 4.20 V | `configs/optimization.yaml` |
| Current range | 0.75 – 6.00 A | `configs/optimization.yaml` |
| Decision interval | 30 s | `configs/optimization.yaml` |
| Families | cccv, two_step, three_step, pulsed | `configs/optimization.yaml` |
| Evaluations / family / cell | 80 (both methods) | `configs/optimization.yaml` |
| GP-BO initial design | 15 (family seeds + random fill) | `configs/optimization.yaml` |
| GP-BO acquisition | EI, `xi = 0.01` | `configs/optimization.yaml` |
| Master seed | 20260904 | `configs/optimization.yaml` |
| Reward | `R = 1·ΔSOC − 2·Q_loss − 0.1·t_h^{0.55}` (paper Eq. 10, calibrated Q) | `configs/reward.yaml` |
| C-rate bins | 0.5, 1.0, 2.0 | `configs/degradation.yaml` |
| Calibration fit starts | 24 Latin-hypercube + 1 anchored | `configs/degradation.yaml` |
| Validation scheme | leave-one-cell-out, 4 folds | `configs/degradation.yaml` |
| Equivalent-cycle horizon | 600 equal-energy sessions | `configs/optimization.yaml` |

---

## Seeding

A single master seed `20260904` is expanded deterministically:

```
seed(cell, family, method) = master_seed
                           + 1000 * cell_index
                           +  100 * family_index
                           +    1 * method_index      (random=0, gp_bo=1)
```

Latin-hypercube starts for the degradation fit use `master_seed`. Every saved
result records the exact integer seed used.

---

## Change log

Any deviation from `docs/implementation_plan.md` must be appended here with the
date, the change, and the scientific reason.

| Date | Change | Reason |
|---|---|---|
| 2026-09-04 | Project created; plan and methodology written before any pipeline code. | Workflow STEP 1–2. |
