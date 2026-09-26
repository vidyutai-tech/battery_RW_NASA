# Final Aging-based setup — pre-registered Top-6 re-test

This folder is a **new** run. Existing `results/08_*`, `results/14_litmus_top6/`, and `results/15_zero_shot_rw9_twins/` were not modified. Existing twin checkpoints and original scripts were not modified. New scripts live in `Aging_aware_charging_opt/scripts/final_setup/`.

Criteria C1–C3 were written in `01_criteria.md` before Arm A and were not edited afterwards. Nothing was tuned to make the test pass.

Logged settings: `config_used.json`.

## Phase 0 findings

### Max supported charging current (training windows)

NASA sign: negative current is charge. Charging current is `−I` on samples with `I < 0` in the windows actually used to train each twin.

- RW9 (full author train 60/20/20 seed 42): max = `4.618000030517578` A; p99 = `4.5` A; n>3 A = 132850; n>4 A = 32655; n>4.5 A = 2068; n(V≥4.1 V) = 753461 of 5153177.
- RW10 (train prefix frac 0.60): max = `4.620999813079834` A; p99 = `4.5` A; n>3 A = 111255; n>4 A = 29055; n>4.5 A = 1141; n(V≥4.1 V) = 427789 of 3115130.
- RW11 (train prefix frac 0.60): max = `4.616000175476074` A; p99 = `4.5` A; n>3 A = 154840; n>4 A = 32214; n>4.5 A = 4371; n(V≥4.1 V) = 431393 of 3139894.
- RW12 (train prefix frac 0.60): max = `4.61899995803833` A; p99 = `3.750999927520752` A; n>3 A = 82651; n>4 A = 12700; n>4.5 A = 2111; n(V≥4.1 V) = 416736 of 3130532.
- `twin_source_RW9.pt`: identical to RW9 train (same checkpoint, same windows).

Phase 2 / Arm A / Arm B current upper bound = min of the four cell training maxima = **`4.616000175476074` A** (RW11). Original search used `6.00` A.

Full tables: `00_diagnostics/SUMMARY.md`.

### Fine-tuned twins vs zero-shot `twin_source_RW9.pt` on identical held-out windows

Held-out windows are the author 60/20/20 seed-42 **test** split of each cell. Both models were scored on the same materialised batch per cell. Identical batches: **True**.

RW9’s cell-specific checkpoint **is** `twin_source_RW9.pt`, so the two RMSEs are identical.

| Cell | n_test | Cell-twin V RMSE (V) | Zero-shot V RMSE (V) | Cell-twin better on V? | Cell-twin T RMSE (°C) | Zero-shot T RMSE (°C) | Cell-twin better on T? |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RW9 | 11377 | 0.028346536185207775 | 0.028346536185207775 | False (tie; same file) | 0.30845948572427684 | 0.30845948572427684 | False (tie) |
| RW10 | 11461 | 0.03232914213844151 | 0.02707292935738748 | False | 0.284166795486449 | 0.28916574076525814 | True |
| RW11 | 11553 | 0.04825838049076362 | 0.04896023902663506 | True | 0.15170196855919357 | 0.9600909816459059 | True |
| RW12 | 11519 | 0.06392941710730615 | 0.05467705219013163 | False | 0.26355151510516145 | 1.0347509662567658 | True |

Fine-tuned / cell-specific checkpoints beat zero-shot on **voltage** for **1 / 4** cells (RW11 only). They beat zero-shot on **temperature** for RW10, RW11, and RW12. Fine-tuning is not a uniform voltage improvement over `twin_source_RW9.pt` on these identical windows.

### Original GP-BO initial conditions

All four `results/08_gp_bo` runs already used SOC0=`0.2`, T0=`24.0` °C, age=`0.0`, with cell-specific V0 from OCV(SOC0):

- RW9 V0 = `3.737303728639184` V
- RW10 V0 = `3.735862307051581` V
- RW11 V0 = `3.728857197853136` V
- RW12 V0 = `3.7401394244878614` V

Each matches `start_state_and_vnom`. Phase 2 therefore did **not** change ICs relative to the original runs; it only documented the already-shared (SOC0, T0, age) and kept cell OCV V0.

### Original elite reward-term scales (`14_litmus_top6`)

R = `1.0·ΔSOC − 2.0·Q_loss − 0.1·t_h^0.55`.

Mean |term| across 24 elites: |w_soc·ΔSOC| = `0.37399800228269015`; |w_loss·Q| = `0.0002418507275852371`; |w_time·t^z| = `0.05592749756578195`.

SOC/Q ≈ `1546.4001535859738`; SOC/time ≈ `6.6871935731219425`; time/Q ≈ `231.24800212176763`. The ΔSOC term dominates; Q_loss is ~10³ smaller than the time term. Reconstruction error vs recorded reward: `0.0`.

## Phase 1 — frozen criteria

See `01_criteria.md`. Verdict is PASS only if C1, C2, and C3 all hold. These were not changed after seeing results.

## Arm A verdict (primary litmus)

**FAIL**

Only two changes vs original GP-BO: current cap `4.616000175476074` A, and the (already used) shared SOC0/T0/age with cell OCV V0. Reward, 80 evals/family, families, EI, energy f=`0.40`, V_max=`4.20` V, and seeds are unchanged.

### C1: FAIL

Dominant family must appear in ≥ 4 of that cell’s Top-6.

- RW9: two_step 3, cccv 3 → FAIL
- RW10: two_step 3, three_step 2, cccv 1 → FAIL
- RW11: cccv 6 → PASS
- RW12: two_step 4, cccv 1, three_step 1 → PASS

Pulsed is absent from every Top-6 (original RW9 Top-6 had 3 pulsed).

### C2: FAIL

CV of peak current across 24 elites = `42.6982567798159` % (threshold 15).
mean = `2.9058524393715324` A, std = `1.240748336205401` A, min = `1.7700019097472548` A, max = `4.616000175476074` A.

Original `14_litmus_top6` I_max CV was `46.81444336526601` % (min `1.9851600319247495` A, max `6.0` A). Capping the search at the training maximum removed 6 A elites; it did not bring CV under 15%. RW11’s six elites still sit at or near the new cap (~`4.59`–`4.616` A); RW9/RW10/RW12 elites sit near `1.77`–`2.29` A.

### C3: FAIL

Elites feasible on **all four** twins (existing `score_session` rule: energy target and `peak_v − 4.20 ≤ 1e-3` V):

- RW9: 4 / 6 → PASS
- RW10: 1 / 6 → FAIL
- RW11: 0 / 6 → FAIL
- RW12: 1 / 6 → FAIL

Globally 6 / 24 profiles are feasible on all four twins (original litmus: 4 / 24). C3 is per-cell ≥ 4/6, so this is still a fail.

Tables and figures: `02_armA/litmus/`.

## Phase 3 (reward rescaling)

Run because Arm A failed. Same current cap and ICs as Phase 2. Two variants; neither was tuned after seeing C1–C3.

### normalize_three_terms: FAIL

Each of `w_soc·ΔSOC`, `w_loss·Q`, `w_time·t^z` is divided by the absolute value of that term on that cell’s 0.5C CCCV baseline.

- C1: FAIL (RW9 cccv 3 / two_step 3; RW10 three_step 4 PASS; RW11 cccv 6 PASS; RW12 two_step 6 PASS)
- C2: FAIL, CV = `42.759347063113786` % (mean `2.67043591816536` A, min `1.7812260756258085` A, max `4.616000175476074` A)
- C3: FAIL (RW9 6/6, RW10 2/6, RW11 0/6, RW12 0/6). Globally 8 / 24 pack-feasible.

### normalize_drop_dsoc: FAIL

Same normalization, ΔSOC term removed; energy remains a hard constraint.

- C1: FAIL (RW9 two_step 4 PASS; RW10 two_step 3 / cccv 3 FAIL; RW11 cccv 6 PASS; RW12 two_step 4 PASS)
- C2: FAIL, CV = `42.76420241059796` % (mean `2.652183302230864` A, min `1.773254792844475` A, max `4.616000175476074` A)
- C3: FAIL (RW9 6/6, RW10 3/6, RW11 0/6, RW12 2/6). Globally 11 / 24 pack-feasible.

**No Phase 3 variant passes.** RW11’s Top-6 remains all CCCV at the current cap and none of those six transfer to the other three twins.

## Arm B — shared-surrogate transfer test (not a litmus test)

Because Arm A failed, Arm B used the Phase 3 primary reward (`normalize_three_terms`) on `twin_source_RW9.pt` only: one GP-BO per family, one Top-6, replayed on RW9–RW12 cell twins and on `twin_source_RW9.pt`. RW9’s cell checkpoint is that same file.

Top-6 (shared search): CCCV ~`1.81`–`1.87` A and two-step ~`1.78`–`1.79` A.

All 6 profiles are feasible on RW9, RW10, RW11, RW12, and `twin_source_RW9`. Times are ~27–29 min. V_peak on RW11 is ~`3.96` V; on RW9 it is ~`4.196`–`4.200` V. SRC-P3 native V_peak = `4.200053691864014` V, still marked feasible by the existing `1e-3` V tolerance in `score_session`.

Full table: `04_armB_shared_RW9/TRANSFER.md`.

This is a **shared-surrogate transfer** result, not a litmus result: one twin produced one elite set, then that set was tested on the other twins.

## What produced any consistency

- **Identical ICs** were already the original GP-BO policy. They cannot explain original inconsistency and were not a new intervention.
- **Current cap** (`6.00` A → `4.616000175476074` A) removed 6 A elites and dropped I_max CV from `46.81444336526601` % to `42.6982567798159` %. C2 still fails. RW11 still saturates the (new) upper bound; the other cells do not.
- **Reward rescaling** (and dropping ΔSOC) did not pass C1–C3. I_max CV stayed ~`42.76` %. RW11’s elites still do not transfer. Pack-feasible count rose 4 → 6 → 8 → 11 of 24, which is not a C3 pass.

The only setting that produced a **single** neighbourhood of ~1.8 A profiles feasible on all four cell twins is Arm B’s **shared surrogate** (one search on `twin_source_RW9.pt`). That is transfer of one optimizer’s elites, not independent per-cell Top-6 agreement.

Arm A litmus: **FAIL**. Phase 3: **FAIL** (both variants). Criteria were not changed post hoc.
