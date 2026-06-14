# NASA RW Battery Digital Twin — Transfer & Charging Optimization

Research codebase for the NASA **Random Walk (RW)** cells (RW9–RW12): train a **Battery Digital Twin (BDT)**, optionally **fine-tune** it to another cell, then search for **lifetime-optimal charging profiles** via Bayesian optimization on the frozen twin.

| Cell | Role |
|------|------|
| RW9 | Source (pretrained twin) |
| RW10–RW12 | Transfer targets |

Raw data: `NASA_RW/dataset/` (`.mat` files, gitignored).  
Generated artifacts: `outputs/` (gitignored).

---

## Theoretical background

### Battery Digital Twin (BDT)

The twin is a sequence model that predicts **voltage** and **temperature** trajectories given:

- **Relative age** (0 = fresh, 1 = end of life in the RW dataset)
- Initial rest voltage **V₀** and temperature **T₀**
- A **current profile** I(t) (charge current is negative in NASA convention)

Training uses random-walk charge / discharge / rest steps from RW9. The twin learns residual dynamics on top of the initial state; conformal **drift margins** (Stage 1b) tighten voltage limits during open-loop rollout.

**Transfer learning:** the same architecture is fine-tuned on a target cell (e.g. RW10) with a small fraction of its data. Only the checkpoint path changes in charging optimization—the BO loop is unchanged.

### State of charge (SoC)

During charging optimization, SoC is **not** inverted from loaded terminal voltage (IR drop would bias it). Instead:

1. An **OCV–SoC curve** is fit from low-current rest steps.
2. SoC evolves by **Coulomb counting**: ΔSoC = ∫(−I dt) / Q(age).
3. **Q(age)** comes from reference 1 A discharge capacity fade.

SoC is age-aware through capacity; the OCV curve is treated as age-invariant for RW9.

### Charging optimization pipeline

Three coupled stages:

```
  Profile parameters  →  BDT rollout (V, T, SoC)  →  Lifetime objective  →  GP Bayesian optimization
       (Stage 1)              (simulator)              (Stage 2)                 (Stage 3)
```

1. **Simulation** — A parametric **profile family** (CCCV, adaptive taper, …) defines I(t). The frozen BDT predicts V(t) and T(t); the simulator re-anchors every few seconds to limit drift.

2. **Objective** — Hard **feasibility** first, then minimize a **composite loss** among feasible profiles:
   - Reach **SoC target** (default 95%) within **max duration** (default 105 min)
   - No temperature violation
   - Among feasible candidates, minimize:

```
Loss = w_sei · (SEI / ΔSoC)
     + w_time · duration_min
     + w_temp · ∫ max(0, T − 35°C)² dt
     + w_vstress · ∫ max(0, V − 4.0 V)² dt
```

Default weights: `w_sei=1`, `w_time=0.02`, `w_temp=0.05`, `w_vstress=0.08`.

**SEI proxy** — Arrhenius-weighted integral of charge current × temperature factor; reported as **SEI/ΔSoC** so profiles are comparable regardless of exactly how much SoC was gained. Lower is better (less degradation per % charged).

**Voltage stress** — Penalizes time spent above 4.0 V (CV hold near the 4.2 V ceiling hurts lifetime in this objective).

At room-temperature starts (~25°C), the temperature penalty is usually zero.

3. **Search** — Gaussian-process Bayesian optimization (scikit-optimize) explores each **profile family** independently (~40 evaluations per family). The best feasible candidate per family is saved.

### Profile families

Each family is a low-dimensional parameterization searched by BO:

| Family | Idea |
|--------|------|
| CCCV | Constant current → constant voltage taper |
| Reduced-CV CCCV | CCCV with CV capped at 4.05–4.20 V |
| Adaptive 2-step / 3-step | Step down current at SoC thresholds |
| Exponential taper | I(SoC) = I₀·exp(−k·SoC) |
| CC-taper | Step down when voltage hits ceiling |
| Multi-step taper | Multiple voltage-triggered steps |
| Pulsed | Charge/rest bursts (rest = fraction of on-time) |

**Interpretation note:** under the current BDT + SEI proxy, **simple CC or CCCV-style profiles** usually beat pulsed or multi-step variants—pulse “recovery” and plating are not modeled, so extra complexity rarely helps.

### Pareto analysis (post-processing)

BO history contains many feasible points, not just the single lowest-loss winner. Pareto analysis extracts **non-dominated** trade-offs on duration, SEI/ΔSoC, voltage stress, and temperature penalty, then tags:

| Tag | Meaning |
|-----|---------|
| **Fastest** | Minimum charge time (may sacrifice SEI) |
| **Lifetime** | Minimum SEI/ΔSoC (may be slower) |
| **Balanced** | Knee point on the Pareto front |

Use these when a single weighted loss does not match your product goal (speed vs battery care).

---

## Setup

```bash
cd battery_RW_NASA
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

---

## Workflow

### 1. Train source twin (RW9)

```bash
venv/bin/python scripts/train_twin.py --config configs/default.yaml
export BDT_CKPT=outputs/twin_source/<TIMESTAMP>/twin_source_RW9.pt
```

Optional: `build_source_registry.py`, `visualize_twin.py`, `train_soc.py` (SOC MLPs—not required for charging BO).

### 2. Fine-tune to another cell (optional)

```bash
venv/bin/python scripts/finetune_twin.py \
  --source_ckpt $BDT_CKPT \
  --out outputs/finetune_two_stage_RW10 \
  --targets RW10

export BDT_CKPT=outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.40.pt
```

Primary transfer metric: **held-out voltage RMSE**.

### 3. Charging profile optimization

One-time prerequisites per BDT checkpoint:

```bash
venv/bin/python scripts/01_fit_ocv_curve.py --cell RW9
venv/bin/python scripts/00_diagnose_drift.py --ckpt $BDT_CKPT --cell RW9
```

**Main benchmark** — 8 families, composite objective, SoC 15% → 95%, ≤105 min:

```bash
venv/bin/python scripts/03_optimize_profile_families.py \
  --bdt_ckpt $BDT_CKPT \
  --out_dir outputs/charging_opt_user/$USER/stage3_optimization \
  --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \
  --n_calls 40 --n_initial 10 \
  --max_duration_min 105 --max_minutes 150
```

Use `--out_dir outputs/charging_opt_user/$USER/stage3_optimization` so outputs are **user-owned** (avoid permission errors if an agent ran as root).

Regenerate plots/CSV without re-running BO:

```bash
venv/bin/python scripts/report_profile_families.py \
  --out_dir outputs/charging_opt_user/$USER/stage3_optimization

venv/bin/python scripts/report_pareto_profiles.py \
  --out_dir outputs/charging_opt_user/$USER/stage3_optimization
```

Legacy single-family CC-taper BO: `scripts/02_optimize_charging_profile.py`.

Objective flags (scripts 02 & 03): `--objective composite|legacy`, `--w_sei`, `--w_time`, `--w_temp`, `--w_vstress`, `--v_ref_stress`, `--t_comfort_c`.

---

## Outputs

User benchmark tree (recommended):

```
outputs/charging_opt_user/<USER>/stage3_optimization/
  models/
    family_optimization_results.json   # best params + BO history per family
    comparison_table.csv
    pareto_analysis.json
    pareto_profiles.csv
  plots/
    profile_families/
      best_<family>.png                # I / V / SoC (separate axes)
      profile_family_comparison.png
    pareto/
      pareto_tradeoffs.png             # duration vs SEI / V-stress / temp
      pareto_reference_profiles.png    # Fastest / Balanced / Lifetime table
```

Canonical shared layout (may be root-owned): `outputs/charging_opt/models|plots/…`

Twin checkpoints: `outputs/twin_source/` or finetune `registry/`.

Fix permissions once if needed: `sudo bash scripts/fix_output_permissions.sh <username>`

---

## Interpreting results (RW9, age=0, composite objective)

Full 8-family benchmark (SoC 15%→95%, ≤105 min, start V=3.711 V, T=24.7°C).  
Source: `outputs/charging_opt_user/hima/stage3_optimization/models/comparison_table.csv`.

| Rank | Family | Loss | Duration (min) | SEI/ΔSoC | V²·min | Feasible | Best parameters |
|------|--------|------|----------------|----------|--------|----------|-----------------|
| 1 | Adaptive 2-step (SoC) | 70.12 | 104.5 | 68.0 | 0.28 | yes | i1=i2=1.05 A, soc_switch=0.80 |
| 2 | CCCV | 70.44 | 100.5 | 68.4 | 0.55 | yes | i_cc=1.09 A, v_cv=4.18 V, i_cutoff=0.50 A |
| 3 | Adaptive 3-step (SoC) | 70.55 | 105.0 | 68.4 | 0.42 | yes | i1=1.13, i2=i3=0.99 A, soc1=0.47, soc2=0.57 |
| 4 | Reduced-CV CCCV | 70.58 | 98.5 | 68.6 | 0.69 | yes | i_cc=1.11 A, v_cv=4.20 V, i_cutoff=0.50 A |
| 5 | Pulsed charge/rest | 70.62 | 100.5 | 68.6 | 0.00 | yes | i=1.26 A, pulse_on=6 min, rest_frac=0.11 |
| 6 | CC-taper (2-level) | 71.42 | 98.7 | 69.3 | 1.47 | yes | i_charge=1.25 A, i_floor=0.75 A |
| 7 | Multi-step taper (voltage) | 74.88 | 91.4 | 72.8 | 2.78 | yes | i_charge=2.0 A, i_floor=0.89 A |
| 8 | Exponential taper I(SoC) | 209.59 | 112.5 | 69.3 | 1.29 | **no** | i0=1.04 A, k=0.15 — exceeds 105 min limit |

**Loss decomposition (rank 1 example):** SEI term 68.0 + time 2.09 + temp 0.00 + V-stress 0.02 ≈ 70.12.

### Pareto reference profiles (same run, from BO history)

These are **not** the same as the single lowest-loss winner—they show explicit trade-offs:

| Tag | Family | Duration (min) | SEI/ΔSoC | V²·min | Loss | Role |
|-----|--------|----------------|----------|--------|------|------|
| Fastest | Pulsed | 50.5 | 81.5 | 0.89 | 82.6 | Minimum time; high degradation |
| Lifetime | Adaptive 2-step | 104.5 | 68.0 | 0.28 | 70.1 | Minimum SEI/ΔSoC among feasible |
| Balanced | Pulsed | 81.0 | 73.1 | 0.16 | 74.7 | Knee on Pareto front |

112 feasible BO evaluations → 37 non-dominated Pareto points. See `models/pareto_profiles.csv` and `plots/pareto/pareto_tradeoffs.png`.

**Takeaways:**

- **Winner (lowest loss)** — Adaptive 2-step; best compromise under default weights, not necessarily fastest.
- **Pareto “Fastest”** — Pulsed at 50.5 min but SEI/ΔSoC ≈ 81.5 vs 68.0 for lifetime mode; do not deploy without accepting that trade-off.
- **Pulsed (rank 5)** — V²·min = 0 (avoids CV hold at ~4.2 V); competitive composite loss.
- **Multi-step taper** — Fastest *feasible family optimum* (91.4 min) but worst SEI among feasible families (72.8).
- **CC-taper** — Good SEI but rank 6 due to V-stress penalty (long hold near 4.2 V).
- **Exponential taper** — Infeasible under the 105 min constraint; excluded from ranking.

Re-run with `--objective legacy` to reproduce SEI-only ranking (CC-taper tended to win under that mode).

---

## Scripts

| Script | Purpose |
|--------|---------|
| `train_twin.py` | Train source BDT on RW9 |
| `finetune_twin.py` | Fine-tune to RW10–RW12 |
| `evaluate_finetune.py` | Re-evaluate finetune checkpoints |
| `01_fit_ocv_curve.py` | OCV–SoC + Q(age) |
| `00_diagnose_drift.py` | Conformal drift margins |
| `03_optimize_profile_families.py` | **Main:** 8-family BO + Pareto export |
| `report_profile_families.py` | Regenerate family plots/CSV from JSON |
| `report_pareto_profiles.py` | Regenerate Pareto plots/CSV from JSON |
| `02_optimize_charging_profile.py` | Legacy single CC-taper BO |
| `sweep_cc_profiles.py` | CC sweep sanity check (Stage 2) |

---

## Code map (charging)

| Module | Role |
|--------|------|
| `charging_opt/profile_simulator.py` | BDT rollout for candidate profiles |
| `charging_opt/charging_profile_family.py` | Profile family definitions |
| `charging_opt/lifetime_reward.py` | Feasibility + composite loss |
| `charging_opt/family_optimizer.py` | Per-family GP-BO |
| `charging_opt/pareto_analysis.py` | Pareto fronts + profile tags |
| `charging_opt/family_reporting.py` | Family plots + CSV |
| `charging_opt/pareto_reporting.py` | Pareto plots + CSV |
| `charging_opt/io_utils.py` | User-writable output paths |

Twin training: `rw_transfer/`, configs in `configs/default.yaml`.

---

## Stage 3 (enhanced): physics degradation, thermal awareness, and figures

This section documents the **updated Stage 3 pipeline** beyond the original SEI-proxy composite benchmark. It adds:

- **Physics-grounded degradation** — Wang et al. (2011) capacity-fade model, calibrated from RW9 measured capacity fade
- **Temperature-aware optimization** — BDT-predicted current derating + optional temperature penalty in the BO objective
- **Ambient sensitivity** — repeat BO at T₀ = 15 / 25 / 35 °C to see how optimal profiles shift with environment
- **Publication figures** — automated plots under `outputs/visualization/`

All commands assume you have already run **§3 prerequisites** (`01_fit_ocv_curve.py`, `00_diagnose_drift.py`) and set `BDT_CKPT`.

### Step 0 — Calibrate the Wang degradation model (one time)

Run after `01_fit_ocv_curve.py` has produced `capacity_fade.npz`:

```bash
venv/bin/python scripts/calibrate_degradation_model.py
```

Writes `outputs/charging_opt/models/stage1_state_estimation/degradation_model.npz`.  
Every feasible BO evaluation now also reports `capacity_fade_pct` and `equiv_cycles_to_eol` in the metrics JSON, even when the objective is still `composite`.

### Step 1 — Baseline enhanced BO (SEI proxy + PI acquisition)

Same 8-family benchmark as §3, but with **Probability of Improvement (PI)** acquisition (recommended over EI):

```bash
venv/bin/python scripts/03_optimize_profile_families.py \
  --bdt_ckpt $BDT_CKPT \
  --acq_func PI \
  --out_dir outputs/charging_opt_user/$USER/stage3_enhanced_20260614 \
  --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \
  --n_calls 40 --n_initial 10 \
  --max_duration_min 105 --max_minutes 150
```

Optional flags (can be combined):

| Flag | Effect |
|------|--------|
| `--objective physics` | Minimize Wang capacity-fade loss instead of SEI/ΔSoC |
| `--thermal_derating` | Reduce current in the simulator when BDT T > comfort threshold |
| `--thermal_loss` | Add temperature penalty on top of the BO loss |
| `--thermal_derate_comfort_c 33` | Start derating above this °C (default 33) |
| `--age_conditioning` | Evaluate each candidate at ages 0 / 0.25 / 0.5 / 0.75 |
| `--chebyshev_omega 0.5` | Directed Pareto scalarization (0=lifetime, 1=fastest) |

Chebyshev sweep across ω (directed Pareto front):

```bash
venv/bin/python scripts/run_chebyshev_pareto_sweep.py \
  --bdt_ckpt $BDT_CKPT \
  --families pulsed cccv adaptive_two_step \
  --n_calls 30 --n_initial 8 \
  --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \
  --max_duration_min 105 --acq_func PI \
  --out_dir outputs/charging_opt_user/$USER/chebyshev_sweep
```

### Step 2 — Physics + thermal combined (recommended “full” single run)

Combines `--objective physics` with thermal derating and thermal loss in **one** BO job:

```bash
venv/bin/python scripts/03_optimize_profile_families.py \
  --objective physics \
  --acq_func PI \
  --thermal_derating \
  --thermal_loss \
  --thermal_derate_comfort_c 33 \
  --bdt_ckpt $BDT_CKPT \
  --out_dir outputs/charging_opt_user/$USER/stage3_physics_thermal \
  --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \
  --n_calls 40 --n_initial 10 \
  --max_duration_min 105 --max_minutes 150
```

Or use the suite script (calibration + baseline BO + ambient sweep):

```bash
venv/bin/python scripts/run_physics_thermal_suite.py \
  --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \
  --out_dir outputs/charging_opt_user/$USER/stage3_physics_thermal
```

Use `--skip_ambient` for baseline only, or `--skip_calibration` if `degradation_model.npz` already exists.

### Step 3 — Ambient temperature sensitivity (Level 2)

Runs BO separately at **T₀ = 15, 25, 35 °C** (3× the work of a single temperature). Use a subset of families for speed:

```bash
venv/bin/python scripts/run_ambient_sensitivity.py \
  --objective physics \
  --thermal_derating --thermal_loss --thermal_derate_comfort_c 33 \
  --bdt_ckpt $BDT_CKPT \
  --families cccv,pulsed,adaptive_two_step \
  --ambient_temps 15,25,35 \
  --n_calls 20 --n_initial 6 \
  --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \
  --max_duration_min 105 \
  --out_dir outputs/charging_opt_user/$USER/stage3_physics_thermal/ambient_sensitivity
```

Outputs:

```
ambient_sensitivity/
  ambient_sensitivity_summary.json       # best family per T0
  ambient_sensitivity_comparison.png     # cross-T comparison plot
  T15/models/ …  T15/plots/profile_families/best_*.png
  T25/ …
  T35/ …
```

Regenerate plots from saved JSON (no BO re-run):

```bash
venv/bin/python scripts/run_ambient_sensitivity.py --plots_only \
  --objective physics --thermal_derating --thermal_loss \
  --out_dir outputs/charging_opt_user/$USER/stage3_physics_thermal/ambient_sensitivity \
  --families cccv,pulsed,adaptive_two_step
```

### Step 4 — Degradation model validation & publication figures

Compare SEI proxy rankings against the calibrated Wang model (requires GPU for BDT re-scoring):

```bash
venv/bin/python scripts/compare_degradation_models.py \
  --enhanced_dir outputs/charging_opt_user/$USER/stage3_enhanced_20260614 \
  --chebyshev_json outputs/charging_opt_user/$USER/chebyshev_sweep/chebyshev_sweep_results.json \
  --out_dir outputs/visualization
```

Generate all meeting figures (fig1–fig5 from enhanced + chebyshev results; add `--with_physics` for fig6):

```bash
venv/bin/python scripts/gen_all_figs.py \
  --out_dir outputs/visualization \
  --enhanced_dir outputs/charging_opt_user/$USER/stage3_enhanced_20260614 \
  --chebyshev_json outputs/charging_opt_user/$USER/chebyshev_sweep/chebyshev_sweep_results.json

# optional: also build fig6 (physics degradation panel)
venv/bin/python scripts/gen_all_figs.py --with_physics --out_dir outputs/visualization
```

Figure index:

| File | Content |
|------|---------|
| `fig1_pareto_front.png` | Chebyshev pulsed sweep + SEI cost vs CCCV |
| `fig2_all_families.png` | 8 families × I / V / SoC |
| `fig3_family_ranking.png` | SEI, duration, composite loss bars |
| `fig4_reference_profiles.png` | Fast / balanced / lifetime reference profiles |
| `fig5_methodology_summary.png` | Family optima + EI vs PI convergence |
| `fig6_physics_degradation.png` | SEI proxy vs Wang model validation |

### Enhanced Stage 3 output layout

```
outputs/charging_opt_user/<USER>/
  stage3_enhanced_20260614/          # PI + composite objective (8 families)
  stage3_physics/                    # physics objective only (optional)
  stage3_physics_thermal/            # physics + thermal baseline
    models/ … plots/pareto/ … plots/profile_families/
    ambient_sensitivity/
      T15/ T25/ T35/
  chebyshev_sweep/
outputs/visualization/               # publication figures + degradation comparison JSON
outputs/charging_opt/models/stage1_state_estimation/
  degradation_model.npz              # calibrated Wang model
```

### New scripts & modules

| Script | Purpose |
|--------|---------|
| `calibrate_degradation_model.py` | Fit Wang model → `degradation_model.npz` |
| `run_chebyshev_pareto_sweep.py` | Directed Pareto via ω sweep |
| `run_physics_thermal_suite.py` | Calibration + physics/thermal BO + ambient sweep |
| `run_ambient_sensitivity.py` | BO at multiple T₀ values |
| `compare_degradation_models.py` | SEI vs Wang validation + fig6 |
| `gen_all_figs.py` | Publication figure bundle |

| Module | Role |
|--------|------|
| `charging_opt/physics_degradation.py` | Wang capacity fade + Paper 1 SEI; `physics_aware_loss()` |
| `charging_opt/thermal_management.py` | Derating controller, ambient states, lumped thermal (standalone) |

### Notes

- **Physics vs SEI:** `--objective physics` changes what BO minimizes; SEI/ΔSoC is still computed for reporting. Spearman rank correlation between SEI proxy and Wang `equiv_cycles_to_eol` is typically strong (see `outputs/visualization/degradation_model_comparison.json`).
- **Thermal derating vs thermal loss:** Derating changes the simulated current profile inside the BDT loop; thermal loss adds a penalty term to the scalar BO objective. Use both for temperature-aware optimization.
- **Lumped thermal model** (`LumpedThermalModel` in `thermal_management.py`) is for **standalone** “what-if cooling” analysis only — do not substitute its temperature into the BDT optimization loop.
- **Runtime:** Full 8-family × 40 evals ≈ 3–4 h GPU; adding ambient sweep (3 temps × 3 families × 20 evals) adds ≈ 1–2 h.
