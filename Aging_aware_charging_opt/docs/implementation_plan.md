# Implementation Plan — Aging-Aware Charging Optimization (fresh study)

Status: **written before any pipeline code was implemented** (STEP 1–2 of the
workflow). This is the contract the rest of the project is held to. Deviations
must be recorded in `docs/experiment_protocol.md` with a reason.

---

## 0. Why this project exists

The previous study (`Constrained_BO/`) is structurally sound but rests on a
degradation model whose coefficients were **never calibrated to the cells being
optimized**. Concretely, `Constrained_BO/hybrid_degradation.py` hard-codes:

* Calendar coefficients `A_cal=1e-3`, `B_cal=2.5`, `C_cal=3500`, `Ea_cal=32000`,
  `z_cal=0.55` — described in the old README itself as *"illustrative stubs, not
  fit to any dataset"*.
* Cyclic coefficients from a literature "Table 7" on a C-rate grid
  `{0.5, 2, 6, 10}C` for a **different cell and chemistry**.

Consequences that invalidate every downstream number in the old study:

1. `Q_loss` is a dimensionless ranking index, not capacity fade, so no
   percentage-fade or lifetime claim is supported.
2. The lifetime figures rescale `Q_loss` by an affine factor chosen so that
   CCCV ½C hits 80 % retention at cycle 400 — the "80 % @ 400 cycles" anchor is
   an assumption, not a result.
3. Because the reward mixes an uncalibrated `Q_loss` (magnitude ~1e-4) with
   `ΔSoC` (~0.5) and `t^0.55` (~0.7), the degradation term had almost no
   influence on the optimizer unless propped up by ad-hoc devices
   (`qloss_cap` soft constraint seeded from the Random-search winner, a
   `duration_loss_weight` tie-break, `w_qloss` bumped to 2.0). These couple the
   two optimizers together and break the fairness of the GP-BO vs Random
   comparison.

This project fixes (1) by **calibrating the same equations to NASA RW9–RW12
reference-discharge capacity data**, which then fixes (2) and (3) because the
degradation term acquires real physical units and a real magnitude.

---

## 1. Pipeline (target)

```
NASA RW9-RW12 .mat
   │
   ├─(A)─ reference-discharge capacity extraction ─────────► calibration dataset
   │        + calendar time + mean SOC + mean T
   │        + Ah throughput binned by C-rate {C/2, 1C, 2C}
   │
   ├─(B)─ BDT checkpoints (.pt, reused verbatim) ─────► inference-only twin
   │
   ▼
(A) ──► NLS calibration of Q_cal / Q_cyc coefficients
   ──► leave-one-cell-out validation  (R², RMSE, MAE, bias, residuals)
   ──► FREEZE  configs/degradation_fitted.yaml
   │
   ▼
(B) + frozen degradation
   ──► baseline protocols (CCCV 0.5C, 1C, 2C)
   ──► normalization anchors (Q_ref, t_ref, E_ref from CCCV 1C)  [FROZEN]
   ──► reward  R = 0.5·e_norm − 2.0·q_norm − 1.0·t_norm
   │
   ▼
charging-profile families (CCCV, 2-step, 3-step, pulsed) × 80 evals each
   ──► Random Search      (seed 20260904)
   ──► GP-BO              (identical space / budget / objective, seed 20260904)
   ──► comparison, cross-cell analysis, equivalent-cycle analysis
   ──► figures 1-10, tables 1-6
   ──► validate_experiment.py  (PASS/FAIL)
```

## 2. Stage-by-stage plan

### STEP 3 — BDT (reuse, verify, do not retrain)

The transformer digital twin and its trained checkpoints are **not** affected by
the degradation formulation, so retraining would add risk without adding
validity. Plan:

* Copy the model *architecture* into `src/aacopt/bdt.py` so the new project is
  self-contained (no import of `rw_transfer`).
* Load the existing checkpoints read-only:
  * RW9  → `outputs/twin_source/20260610_111409/twin_source_RW9.pt`
  * RW10 → `outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.60.pt`
  * RW11 → `outputs/finetune_two_stage_RW11/registry/finetune_RW11_frac0.60.pt`
  * RW12 → `outputs/finetune_two_stage_RW12/registry/finetune_RW12_frac0.60.pt`
* **Equivalence test**: for each cell, assert the new implementation's
  `predict()` matches the original `rw_transfer` implementation to
  `< 1e-5` max-abs on randomized current profiles. This is the only place the
  old code is imported, and only inside a test.
* **Held-out accuracy check**: re-measure V/T RMSE on NASA RW segments that were
  never used to fit the checkpoint, per cell, and save to
  `results/02_bdt_verification/`. This is a *verification* of an inherited
  artifact, reported as such — not a new training claim.

Deliverable: `results/02_bdt_verification/bdt_verification.json`, figure 0.

### STEP 4 — Calibration dataset (built from raw `.mat`, independently)

For each cell, walk all steps in chronological order and cut the timeline at
every `reference discharge` step. For reference discharge *k*:

* `q_ah[k]` = `∫|I| dt / 3600` over the step, corrected to a full 0–100 % window
  using the cell's own low-current OCV curve (re-fit in this project from the
  `low current discharge at 0.04A` step).
* `y[k]` = `1 - q_ah[k] / q_ah[0]` → **measured fractional capacity loss**
  (this is the regression target; it is a real physical fraction, not an index).

For the interval between reference discharge *k* and *k+1*:

* `duration_h` — wall-clock calendar hours from the absolute `time` field.
* `mean_soc` — coulomb-counted SOC averaged over the interval.
* `mean_temperature_c` — sample mean of the measured temperature.
* `dah_b` for `b ∈ {0.5C, 1C, 2C}` — Ah throughput assigned to the nearest
  C-rate bin, sample by sample, using `C = |I| / Q_nominal`.

Rationale for binning: the NASA RW duty cycle draws `|I| ∈ [0.75, 4.8] A`,
i.e. `0.34C – 2.18C` on a 2.2 Ah cell, so the three bins `{C/2, 1C, 2C}` span
the observed data. The old model's `{0.5, 2, 6, 10}C` grid extrapolated 3 of 4
nodes far outside anything these cells ever experienced.

Deliverable: `results/01_calibration_dataset/calibration_intervals.csv`
(+ per-cell `capacity_fade_measured.csv`), figure 0b.

Cross-check only: agreement with `Constrained_BO/results/grounded_figures/capacity_fade_measured.csv`
is *reported* as an independent-reimplementation check; it is never read as an input.

### STEP 5 — Degradation model + calibration

Equations are **unchanged from the paper methodology**; only coefficients change.

```
Q_cal(SOC, T, t) = A_cal · exp(B_cal·SOC) · exp( -(Ea_cal + C_cal·SOC) / (R·T_K) ) · t_h^z_cal
Q_cyc(Ah, T, I)  = B_cyc(C) · exp( -Ea_cyc(C) / (R·T_K) ) · Ah^z_cyc(C)
Q_total          = Q_cal + Q_cyc
```

Time-varying stress is handled with the standard **equivalent-time (state-shift)
integration** for power laws, so the cumulative prediction is path-dependent and
never sums increments of a nonlinear power law evaluated at different stresses:

```
given accumulated Q and new stress k:   t_eq = (Q/k)^(1/z);  Q ← k·(t_eq + Δ)^z
```

applied separately to the calendar term (Δ = Δt_h) and to each C-rate bin
(Δ = ΔAh_b), with the bins' contributions summed.

Free parameters (14): `A_cal, B_cal, C_cal, Ea_cal, z_cal`,
`{B_cyc, Ea_cyc, z_cyc}` × 3 bins.

Fitting: `scipy.optimize.least_squares` (Trust Region Reflective), residuals on
measured fractional loss `y`, positive-scale parameters optimized in log space,
bounded box, ≥ 24 randomized multi-starts + 1 literature-anchored start, best
final cost retained. Convergence `ftol=xtol=gtol=1e-12`, `max_nfev=20000`.

Identifiability is *expected to be imperfect* (Arrhenius prefactor/activation
energy pairs are near-collinear over a ~20 °C span). Plan: compute the
Jacobian-based parameter correlation matrix and standard errors, and explicitly
flag any parameter with `|corr| > 0.99` or a bound-active solution, per RULE 8.
A reduced variant (shared `Ea_cyc` across bins) is fitted as a sensitivity
check, not as the primary model.

### STEP 6 — Independent validation

**Leave-one-cell-out (LOCO)**: 4 folds; fold *c* fits on the other three cells
and predicts cell *c*, which is never touched during that fit. This is the
strict version of the requested "fit RW9 → validate RW10" protocol. Report per
fold and pooled: R², RMSE, MAE, mean signed bias, residual-vs-fitted and
predicted-vs-measured plots.

Gate: **if pooled LOCO R² < 0.90 or |bias| > 2 %-capacity, STOP and diagnose**
rather than proceeding to optimization.

The model shipped to the optimizer is the **all-cells pooled fit**; LOCO is the
evidence that this fit generalizes across cells. Both are saved.

### STEP 7–8 — Normalization and reward

Session degradation from the calibrated model is a genuine capacity-loss
fraction, ~1e-4 per session, while `ΔSoC ~ 0.5` and `t_h^0.55 ~ 0.7`. Mixing
raw magnitudes would make degradation irrelevant. Each term is therefore
non-dimensionalized by a **reference-protocol anchor fixed before optimization**:

```
e_norm = E_delivered / E_required          (E_required = 40 % of pack energy)
q_norm = Q_total     / Q_ref(cell)         (Q_ref = Q_total of CCCV 1C session)
t_norm = duration    / t_ref(cell)         (t_ref = duration of CCCV 1C session)

R = w_energy·e_norm − w_deg·q_norm − w_time·t_norm
w_energy = 0.5,  w_deg = 2.0,  w_time = 1.0
```

Anchors come from a deterministic simulation of the CCCV 1C protocol, written
once to `results/05_reward_anchors/reward_anchors.json` and read (never
recomputed) by both optimizers. Per RULE 2 they are *not* selected to reproduce
any previous result; they are a declared reference operating point.

Explicitly flagged limitation: because sessions are energy-constrained,
`e_norm ≈ 1` for every feasible profile, so the effective trade-off the
optimizer explores is degradation against time at a 2:1 weight ratio. The
energy term functions as a feasibility/priority statement rather than an active
gradient. This is stated in the methodology and in the results discussion.

Removed relative to the old study (RULE 3 / fairness):

* `qloss_cap` — a soft constraint whose value came from the Random-search winner.
  It made GP-BO's objective depend on Random's outcome; the two searches were
  therefore not optimizing the same function.
* `duration_loss_weight` tie-break — an undeclared second time penalty.

### STEP 9 — Sanity checks

Assert, before any optimization: CCCV really has a CC leg terminating at the
voltage cutoff followed by a tapering CV leg; multi-step profiles show the
commanded staircase and monotone SOC; pulsed profiles show the commanded
on/rest duty; energy and coulomb accounting close; current and voltage stay
inside limits; higher C-rate at matched Ah increases `Q_cyc`; higher temperature
increases both terms; longer rest at high SOC increases `Q_cal`.

### STEP 10–11 — Optimization

Random Search first, then GP-BO, both with:

* identical search space and identical hard constraints (enforced in
  `from_dict`, so both see the same feasible set),
* identical evaluation budget of **80 evaluations per family** (4 families,
  4 cells → 1280 evaluations per method),
* identical objective (no cross-method information of any kind),
* recorded seed, every evaluated point, reward, best-so-far trace, wall-clock.

GP-BO: `skopt.gp_minimize`, EI acquisition, deterministic family seed points
only (no elites from Random Search).

### STEP 12–16 — Analysis and reporting

Comparison tables, per-cell analysis (no pooling assumption), equivalent-cycle
projection reported strictly as a **relative capacity-retention projection**
with the 80 % line labelled an engineering reference, figures 1–10, tables 1–6,
and `validate_experiment.py` re-deriving every reported number from the saved
result files.

---

## 3. Component reuse decision table

See `docs/old_vs_new.md` for the full table. Summary:

* **Reused verbatim (artifacts)**: BDT checkpoints, NASA `.mat` dataset.
* **Reimplemented cleanly, same concept**: `.mat` parsing, OCV fit, BDT
  inference wrapper, charging-profile families, closed-loop simulator, GP-BO and
  Random-Search drivers, figure style.
* **Rebuilt from scratch (depends on degradation)**: degradation model
  coefficients, calibration and validation, normalization, reward, all baseline
  numbers, all optimization runs, all lifetime analysis, all tables/figures.
* **Discarded**: literature Table-7 coefficients, calendar stubs, `qloss_cap`,
  `duration_loss_weight`, the 80 %-at-400-cycles fade scaling, every previous
  numerical result.

## 4. Order of execution and gates

| Step | Script | Gate before proceeding |
|------|--------|------------------------|
| 3 | `02_verify_bdt.py` | max-abs equivalence < 1e-5 for all 4 cells |
| 4 | `01_build_calibration_dataset.py` | ≥ 70 usable reference discharges per cell; monotone-ish fade |
| 5 | `03_fit_degradation.py` | converged, finite Jacobian, no NaN |
| 6 | `04_validate_degradation.py` | **pooled LOCO R² ≥ 0.90 and \|bias\| ≤ 2 %** |
| 7-8 | `05_build_reward_anchors.py` | anchors finite; CCCV 1C feasible on all cells |
| 9 | `06_sanity_checks.py` | all assertions pass |
| 10 | `07_run_random_search.py` | ≥ 1 feasible point per cell |
| 11 | `08_run_gp_bo.py` | ≥ 1 feasible point per cell |
| 12-15 | `09`–`13` | — |
| 16 | `validate_experiment.py` | all checks PASS |
