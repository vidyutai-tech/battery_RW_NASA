# GP-BO vs Random Search — RW9 Comparison

Comparison of **Gaussian-process Bayesian optimization** (`gp_bo`) against **random search** on the same hybrid Q_loss objective, cell, and profile families.

| | GP-BO | Random search |
|--|-------|---------------|
| Folder | [`RW9/`](RW9/) | [`RW9_random/`](RW9_random/) |
| Method | `gp_bo` (acq = PI) | `random_search` |
| Objective | `hybrid_qloss` | `hybrid_qloss` |
| Weights | \(w_{\mathrm{soc}}=1\), \(w_{\mathrm{qloss}}=1\), \(w_{\mathrm{time}}=0.1\), \(z=0.55\) | same |
| Cell / start | RW9, SoC 20% → 80%, T₀ = 24°C, age = 0 | same |
| Horizon | 150 min | same |
| Seed | 42 | 42 |
| Budgets | 40 and 80 evaluations / family | 40 and 80 samples / family |

**Reward (identical in both stacks):**

```
R = w_soc · ΔSoC − w_qloss · (Q_calendar + Q_cyclic) − w_time · t_h^z
loss = −R + soft SoC / V penalties
```

Lower **loss** is better. All reported winners below are **feasible** (SoC target reached).

**Source files:**

| Budget | GP-BO | Random |
|--------|-------|--------|
| 40 | `RW9/constrained_bo_results_n_calls_40.json` | `RW9_random/constrained_bo_results_n_calls_40.json` |
| 80 | `RW9/constrained_bo_results_n_calls_80.json` | `RW9_random/constrained_bo_results_n_calls_80.json` |

Plots: `best_profiles_n_calls_40.png` / `best_profiles_n_calls_80.png` in each folder.

---

## Headlines

1. **At matched budget 40**, GP-BO finds a better overall profile than random search  
   (Pulsed loss **−0.352** vs random’s best 3-step **−0.345**).
2. **At budget 80**, overall winners are nearly tied  
   (GP Pulsed **−0.3454** vs random CCCV **−0.3452**).
3. GP-BO’s biggest gain is on **pulsed** (harder 4-D space); random stays weak there even at 80 samples.
4. Random **beats GP on 3-step** at both budgets — GP collapses to a nearly flat low current (~0.87 A), while random finds a genuine multi-current schedule.
5. Random **CCCV improves a lot** from 40 → 80 samples; GP-BO CCCV is already strong at 40 and barely moves.

---

## Overall winner by budget

| Budget | Winner | Method | Family | Loss | Reward R | Duration (min) | Q_total |
|--------|--------|--------|--------|------|----------|----------------|---------|
| **40** | **GP-BO** | gp_bo | Pulsed | **−0.3516** | 0.412 | 60.5 | 0.0937 |
| 40 | (baseline) | random | 3-step | −0.3450 | 0.408 | 63.0 | 0.0929 |
| **80** | **GP-BO** (edge) | gp_bo | Pulsed | **−0.3454** | 0.409 | 63.5 | 0.0936 |
| 80 | (baseline) | random | CCCV | −0.3452 | 0.409 | 63.5 | 0.0923 |

Δloss (GP best − random best): **−0.0066** at 40; **−0.0001** at 80.

---

## Per-family comparison (budget = 40)

| Family | GP-BO loss | Random loss | Winner | Δloss (GP − rand) | GP duration | Rand duration |
|--------|------------|-------------|--------|-------------------|-------------|---------------|
| CCCV | −0.3396 | −0.3117 | **GP-BO** | −0.0278 | 66.0 min | 81.0 min |
| 2-step | −0.3416 | −0.3386 | **GP-BO** | −0.0029 | 62.5 min | 66.5 min |
| 3-step | −0.2962 | −0.3450 | **Random** | +0.0488 | 91.0 min | 63.0 min |
| Pulsed | −0.3516 | −0.2522 | **GP-BO** | −0.0994 | 60.5 min | 116.0 min |

**Family wins at 40:** GP-BO **3 / 4**, Random **1 / 4**.

### Best params @ 40

| Family | GP-BO | Random |
|--------|-------|--------|
| CCCV | i_cc=1.22 A, v_cv=4.14 V, i_cut=0.048 A | i_cc=0.98 A, v_cv=4.07 V, i_cut=0.34 A |
| 2-step | i1=1.95 A, i2=0.98 A, soc_sw=0.48 | i1=2.34 A, i2=1.19 A, soc_sw=0.20 |
| 3-step | i1=i2=i3≈0.87 A (flat), soc1=0.54, soc2=0.59 | i1=2.13, i2=1.42, i3=1.21 A; soc1=0.16, soc2=0.36 |
| Pulsed | i=1.98 A, on=1.0 min, rest_frac=0.50, i_floor=0.82 A | i=1.07 A, on=5.6 min, rest_frac=0.59, i_floor=0.86 A |

---

## Per-family comparison (budget = 80)

| Family | GP-BO loss | Random loss | Winner | Δloss (GP − rand) | GP duration | Rand duration |
|--------|------------|-------------|--------|-------------------|-------------|---------------|
| CCCV | −0.3400 | −0.3452 | **Random** | +0.0052 | 66.0 min | 63.5 min |
| 2-step | −0.3452 | −0.3386 | **GP-BO** | −0.0065 | 63.5 min | 66.5 min |
| 3-step | −0.2969 | −0.3450 | **Random** | +0.0481 | 90.5 min | 63.0 min |
| Pulsed | −0.3454 | −0.2882 | **GP-BO** | −0.0572 | 63.5 min | 93.0 min |

**Family wins at 80:** GP-BO **2 / 4**, Random **2 / 4**.

### Best params @ 80

| Family | GP-BO | Random |
|--------|-------|--------|
| CCCV | i_cc=1.23 A, v_cv=4.14 V, i_cut=0.047 A | i_cc=1.26 A, v_cv=4.19 V, i_cut=0.23 A |
| 2-step | i1=1.88 A, i2=1.26 A, soc_sw=0.10 | i1=2.34 A, i2=1.19 A, soc_sw=0.20 *(same as @40)* |
| 3-step | i1=i2=i3≈0.88 A (flat), soc1=0.54, soc2=0.59 | i1=2.13, i2=1.42, i3=1.21 A; soc1=0.16, soc2=0.36 *(same as @40)* |
| Pulsed | i=1.88 A, on=1.0 min, rest_frac=0.50, i_floor=0.75 A | i=1.47 A, on=1.67 min, rest_frac=0.78, i_floor=0.79 A |

---

## Metrics detail (reward, Q_loss, time)

### Budget 40

| Family | Method | Loss | R | Duration (min) | Q_total | Peak V |
|--------|--------|------|---|----------------|---------|--------|
| CCCV | GP-BO | −0.3396 | 0.406 | 66.0 | 0.0921 | 4.145 |
| CCCV | Random | −0.3117 | 0.393 | 81.0 | 0.0907 | 4.022 |
| 2-step | GP-BO | −0.3416 | 0.404 | 62.5 | 0.0957 | 4.198 |
| 2-step | Random | −0.3386 | 0.405 | 66.5 | 0.0921 | 4.147 |
| 3-step | GP-BO | −0.2962 | 0.387 | 91.0 | 0.0900 | 3.970 |
| 3-step | Random | −0.3450 | 0.408 | 63.0 | 0.0929 | 4.188 |
| Pulsed | GP-BO | −0.3516 | 0.412 | 60.5 | 0.0937 | 4.079 |
| Pulsed | Random | −0.2522 | 0.368 | 116.0 | 0.0906 | 3.927 |

### Budget 80

| Family | Method | Loss | R | Duration (min) | Q_total | Peak V |
|--------|--------|------|---|----------------|---------|--------|
| CCCV | GP-BO | −0.3400 | 0.406 | 66.0 | 0.0922 | 4.145 |
| CCCV | Random | −0.3452 | 0.409 | 63.5 | 0.0923 | 4.179 |
| 2-step | GP-BO | −0.3452 | 0.409 | 63.5 | 0.0923 | 4.179 |
| 2-step | Random | −0.3386 | 0.405 | 66.5 | 0.0921 | 4.147 |
| 3-step | GP-BO | −0.2969 | 0.387 | 90.5 | 0.0901 | 3.972 |
| 3-step | Random | −0.3450 | 0.408 | 63.0 | 0.0929 | 4.188 |
| Pulsed | GP-BO | −0.3454 | 0.409 | 63.5 | 0.0936 | 4.067 |
| Pulsed | Random | −0.2882 | 0.381 | 93.0 | 0.0920 | 4.016 |

Q_total is dominated by the **cyclic** term in all runs; calendar fade over one session is ~1e−9 (negligible).

---

## Budget scaling (40 → 80)

| Family | GP Δloss (80−40) | Random Δloss (80−40) | Note |
|--------|------------------|----------------------|------|
| CCCV | −0.0004 (tiny better) | **−0.0335** (large gain) | Random needed more samples to catch up |
| 2-step | **−0.0036** | 0.0000 | Random best unchanged (same point) |
| 3-step | −0.0007 | 0.0000 | GP still stuck near flat CC; random same winner |
| Pulsed | **+0.0062** (slightly worse) | −0.0361 | GP already near a strong mode at 40; PI may re-sample |

---

## Interpretation

- **Use GP-BO as the default search** when evaluation budget is limited (~40): it wins overall and is far better on pulsed.
- **Random search remains a useful baseline**, especially in higher-dimensional families where the GP prior + box bounds + `from_dict` projections can miss structured multi-step currents (3-step here).
- **Do not over-interpret 80-eval overall ranking**: GP and random overall losses differ by ~1e−4; per-family differences (pulsed, 3-step) are the scientifically meaningful story.
- Hybrid ranking rewards **faster feasible charges** with moderate Q_loss; that is why short pulsed / multi-step profiles often beat slow gentle CCCV under these weights.

---

## Reproduce

```powershell
# GP-BO (results → Constrained_BO/results/RW9)
venv\Scripts\python.exe -m Constrained_BO.run --cell RW9 --method gp_bo --acq-func PI --n-calls 40 --n-initial 10 --out-dir Constrained_BO/results/RW9
venv\Scripts\python.exe -m Constrained_BO.run --cell RW9 --method gp_bo --acq-func PI --n-calls 80 --n-initial 10 --out-dir Constrained_BO/results/RW9

# Random search (results → Constrained_BO/results/RW9_random)
venv\Scripts\python.exe -m Constrained_BO.run --cell RW9 --method random_search --n-random 40 --seed 42 --out-dir Constrained_BO/results/RW9_random
venv\Scripts\python.exe -m Constrained_BO.run --cell RW9 --method random_search --n-random 80 --seed 42 --out-dir Constrained_BO/results/RW9_random
```

Rename / keep `*_n_calls_40.json` and `*_n_calls_80.json` as you already do so both budgets coexist in each folder.
