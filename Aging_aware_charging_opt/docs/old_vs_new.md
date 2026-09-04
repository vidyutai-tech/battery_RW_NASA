# Old vs New — component-by-component

Old project: `/data/battery_RW_NASA/Constrained_BO` (+ `rw_transfer`, `charging_opt`).
New project: `/data/battery_RW_NASA/Aging_aware_charging_opt`.

The new project **imports nothing** from the old one at runtime. The single
exception is `tests/test_bdt_equivalence.py`, which imports
`rw_transfer.training.twin_trainer` purely to prove the reimplemented twin is
numerically identical to the original. Nothing under `Constrained_BO/` or
`outputs/` is written to.

---

## A. Reusable infrastructure — inherited artifacts, used read-only

| Artifact | Path (read-only) | Why it is safe to reuse |
|---|---|---|
| NASA RW9–RW12 raw data | `NASA_RW/dataset/Battery_Uniform_Distribution_Charge_Discharge_DataSet_2Post/.../Matlab/RW{9,10,11,12}.mat` | Primary measurement data; independent of any model choice. |
| BDT checkpoint RW9 (source) | `outputs/twin_source/20260610_111409/twin_source_RW9.pt` | Predicts V/T from a current profile. Its training objective never involved the degradation model, so a change of degradation formulation cannot invalidate it. |
| BDT checkpoints RW10–12 (fine-tuned) | `outputs/finetune_two_stage_RW{10,11,12}/registry/finetune_RW{10,11,12}_frac0.60.pt` | Same reasoning. `frac0.60` selected because it had the lower held-out V RMSE for all three cells. |

Reused **conceptually** (idea and interface kept, code rewritten in
`src/aacopt/` so the project is self-contained):

| Concept | Old location | New location | Change |
|---|---|---|---|
| `.mat` step parsing | `rw_transfer/data/mat_loader.py` | `src/aacopt/nasa_data.py` | Rewritten; adds absolute-time interval cutting at reference discharges. |
| OCV curve fit from 0.04 A discharge | `Constrained_BO/ocv.py`, `charging_opt/soc_utils.py` | `src/aacopt/capacity.py` | Rewritten; PCHIP on 5 mV bins, re-fit per cell, own cache. |
| Reference-discharge capacity table | `charging_opt/soc_utils.capacity_fade_table` | `src/aacopt/capacity.py` | Rewritten; same OCV-window correction, now also emits calendar time, mean SOC, mean T and C-rate-binned Ah. |
| Transformer twin architecture | `rw_transfer/models/digital_twin.py` | `src/aacopt/bdt.py` | Inference-only copy (no training code); verified bit-equivalent. |
| Frozen-twin rollout / decision-interval stepping | `Constrained_BO/simulator.py` | `src/aacopt/simulator.py` | Rewritten; same 1 Hz decision stepping and voltage-ceiling truncation. |
| Charging-profile families | `Constrained_BO/profiles.py` | `src/aacopt/profiles.py` | Rewritten; same four families and the same hard ordering constraints. |
| Continuous search bounds | `Constrained_BO/profile_catalog.py` | `src/aacopt/config.py` (`configs/optimization.yaml`) | Moved to configuration so both optimizers provably read one source. |
| GP-BO driver | `Constrained_BO/bayesian_optimizer.py` | `src/aacopt/optimizers/gp_bo.py` | Rewritten; `qloss_cap` and elite warm-start from Random removed. |
| Random-search driver | `Constrained_BO/run.py` | `src/aacopt/optimizers/random_search.py` | Rewritten; identical budget accounting. |
| Figure style | `Constrained_BO/viz.py` | `src/aacopt/viz/style.py` | Same palette / DPI / font conventions so figures match the previous paper's look. |
| Final export layout | `Constrained_BO/export_final_charging_opt_results.py` | `scripts/12_make_figures.py`, `scripts/13_make_tables.py` | Same figure filenames (`fig8*`, `fig9*`, `fig10`) and per-cell folder layout. |

---

## B. Newly implemented components

| Component | New file | Purpose |
|---|---|---|
| Calibration dataset builder | `src/aacopt/calibration_data.py`, `scripts/01_build_calibration_dataset.py` | Per-interval calendar time, mean SOC, mean T, rest hours and C-rate-binned Ah throughput, plus measured fractional capacity loss per reference discharge. Did not exist as a first-class, documented artifact before. |
| Path-dependent degradation integrator | `src/aacopt/degradation.py` | Equivalent-time (state-shift) accumulation of the power-law terms under time-varying SOC/T/C-rate. The old code either evaluated closed forms on session means or summed increments of a power law at a fixed stress. |
| NLS calibration | `src/aacopt/calibration.py`, `scripts/03_fit_degradation.py` | Bounded, multi-start trust-region least squares in log space; Jacobian-based standard errors and parameter correlation matrix. |
| Leave-one-cell-out validation | `scripts/04_validate_degradation.py` | Strictly out-of-sample generalization test with a hard go/no-go gate. |
| Reward normalization anchors | `src/aacopt/reward.py`, `scripts/05_build_reward_anchors.py` | Frozen per-cell `Q_ref`, `t_ref`, `E_ref` from the CCCV 1C reference protocol so the three reward terms are commensurable. |
| Physics/reward sanity suite | `scripts/06_sanity_checks.py`, `tests/` | Pre-optimization assertions on profile shape, energy/coulomb closure and degradation monotonicity. |
| Consistency checker | `scripts/validate_experiment.py` | Re-derives every reported number from saved files and verifies provenance. |

---

## C. Intentionally discarded old assumptions

| Discarded | Where it lived | Why |
|---|---|---|
| Cyclic coefficients on the `{0.5, 2, 6, 10}C` grid: `B = {30330, 19300, 12000, 11500}`, `Ea = {31500, 31000, 29500, 28000}`, `z = {0.552, 0.554, 0.560, 0.560}` | `Constrained_BO/hybrid_degradation.py` `TABLE7_*` | Literature values for a different cell and chemistry, never fit to RW9–RW12. Three of four grid nodes (6C, 10C) lie far outside anything these cells experienced (max ≈ 2.2C), so the interpolation was extrapolation. The published grid is also non-monotonic in C-rate, which propagated a physically wrong ordering (`Q_cyc(C/2) > Q_cyc(2C)`) into the reward. |
| Calendar coefficients `A_cal=1e-3, B_cal=2.5, C_cal=3500, Ea_cal=32000, z_cal=0.55` | `Constrained_BO/hybrid_degradation.py` | The old README states these are "illustrative stubs, not fit to any dataset". |
| `Q_loss` interpreted as a "Relative Capacity-Loss Index" | `Constrained_BO/objective.py` `QLOSS_TERMINOLOGY` | Correct labelling given uncalibrated coefficients, but it means no fade, retention or lifetime number was physically meaningful. Replaced by a calibrated fractional capacity loss. |
| `qloss_cap` soft constraint | `Constrained_BO/objective.py`, `export_final_charging_opt_results.py` (`IMPROVED_QLOSS_CAP_SCALE = 120`) | The cap was set to the Random-search winner's `Q`, so GP-BO minimized a *different* function than Random Search. That invalidates the head-to-head comparison. Removed; both methods now minimize the identical objective. |
| `duration_loss_weight = 1e-3` tie-break | `Constrained_BO/objective.py` | An undeclared second time penalty on top of `w_time`, not present in the stated reward equation. |
| `w_qloss` raised to 2.0 *to compensate for the tiny raw `Q_loss` magnitude* | `export_final_charging_opt_results.py` | The magnitude mismatch is now fixed by explicit normalization, so the weight expresses a genuine priority instead of a scale correction. |
| Lifetime fade scaling `scale = (100 − 80) / Q_total(CCCV ½C, cycle 400)` | `Constrained_BO/lifetime_fade_projection.project_fade` | Forced CCCV ½C to reach exactly 80 % retention at cycle 400 by construction, then read other policies' retention off that scale. The anchor was an input, not a finding. Replaced by direct application of the calibrated model, with the 80 % line drawn only as a labelled engineering reference. |
| Legacy temperature/time shaped reward (`temperature_reward`, `time_reward`, plateau 1.5, zero at 150 s) | `Constrained_BO/objective.py` | Hand-shaped, unit-free reward surfaces with no physical derivation. Not carried over. |
| All previous numerical results | `Constrained_BO/results/**` | The objective function changed, so no previous optimum, reward, profile, table or figure transfers (RULE 3). |
| `LFP` cell path | `Constrained_BO/config.py`, `lfp_processed.mat` | Out of scope: this study calibrates against NASA RW reference-discharge fade, and no comparable aging ground truth was established for the LFP set here. Excluded rather than assumed. |

---

## D. What is deliberately *not* re-derived

| Item | Treatment |
|---|---|
| BDT weights | Inherited. Re-verified for numerical equivalence and re-measured for held-out V/T accuracy, but not retrained. Reported as an inherited, verified artifact. |
| BDT architecture / training procedure | Unchanged by design (the instruction was not to change it for its own sake). |
| Universal gas constant, nominal capacity 2.2 Ah, 4.2 V ceiling | Physical/datasheet constants, carried over. |
| Profile families (CCCV, 2-step, 3-step, pulsed) | Retained as scientifically justified, but each re-verified for correct implementation in `scripts/06_sanity_checks.py`. |
