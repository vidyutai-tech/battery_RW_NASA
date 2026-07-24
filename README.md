# NASA RW Battery Digital Twin — Transfer & Charging Optimization

Research codebase for the NASA **Random Walk (RW)** cells (RW9–RW12): train a **Battery Digital Twin (BDT)**, optionally **fine-tune** it to another cell, then search for **lifetime-aware charging profiles** on the frozen twin.

| Cell | Role |
|------|------|
| RW9 | Source (pretrained twin) |
| RW10–RW12 | Transfer targets |
| LFP | Optional cross-chemistry finetune target |

Raw data: `NASA_RW/dataset/` (`.mat` files, gitignored).  
Generated artifacts: `outputs/` and `Constrained_BO/results/` (gitignored where configured).

---

## Two charging stacks

| Stack | Role | Objective | Search |
|-------|------|-----------|--------|
| **`Constrained_BO/`** (primary) | Closed-loop BDT + hybrid degradation reward | Hybrid Q_loss (calendar + cyclic) | **GP-BO** (default) or random search |
| **`charging_opt/`** (legacy Stage 3) | Open family benchmark + Pareto tools | SEI proxy / Wang physics / Chebyshev | GP-BO (`scripts/03_…`) |

New work should start from **`Constrained_BO`**. The rest of this README documents that path first.

---

## Theoretical background

### Battery Digital Twin (BDT)

The twin is a sequence model that predicts **voltage** and **temperature** trajectories given:

- **Relative age** (0 = fresh, 1 = end of life in the RW dataset)
- Initial rest voltage **V₀** and temperature **T₀**
- A **current profile** I(t) (charge current is **negative** in NASA convention)

Training uses random-walk charge / discharge / rest steps from RW9. The twin learns residual dynamics on top of the initial state; conformal **drift margins** (Stage 1b in `charging_opt`) can tighten voltage limits during open-loop rollout.

**Transfer learning:** the same architecture is fine-tuned on a target cell (e.g. RW10). Only the checkpoint path changes in charging optimization—the BO loop is unchanged.

### State of charge (SoC)

During charging optimization, SoC is **not** inverted from loaded terminal voltage (IR drop would bias it). Instead:

1. An **OCV–SoC curve** is fit from low-current rest steps.
2. SoC evolves by **Coulomb counting**: ΔSoC = ∫(−I dt) / Q(age).
3. **Q(age)** comes from reference 1 A discharge capacity fade.

SoC is age-aware through capacity; the OCV curve is treated as age-invariant for RW9.

### Constrained_BO pipeline (recommended)

```
  Profile parameters  →  Closed-loop BDT rollout (V, T, SoC)
       (families)              (re-anchor ~30 s)
                ↓
       Hybrid Q_loss reward + feasibility
                ↓
       GP Bayesian optimization (per family)
```

1. **Simulation** — A parametric family (CCCV, 2-step, 3-step, pulsed) defines the commanded current. `FrozenBDT` predicts V(t) and T(t) in short decision intervals and re-anchors to limit open-loop drift. SoC is Coulomb-counted.

2. **Hybrid Q_loss reward** (default `reward_mode=hybrid_qloss`):

```
R = w_soc · ΔSoC
  − w_qloss · (Q_calendar + Q_cyclic)
  − w_time · t_h^z
```

| Term | Meaning |
|------|---------|
| **ΔSoC** | SoC gained in the session (reward charge delivered) |
| **Q_calendar** | Calendar loss index: SoC- and T-dependent Arrhenius × t^z (Eq. 2 by default) |
| **Q_cyclic** | Cyclic loss index: Table-7 B(I), Ea(I), z(I) vs C-rate × Ah^z (Eq. 7 by default) |
| **t_h^z** | Power-law time penalty (default z = 0.55) |

Default weights: `w_soc=1`, `w_qloss=1`, `w_time=0.1`, `z=0.55`.

**Q_calendar and Q_cyclic are a *Relative Capacity-Loss Index*, not a calibrated
"Capacity Fade (%)"** — see [Hybrid degradation methodology](#hybrid-degradation-methodology)
below for the full model, units, and limitations.

BO **minimizes** a scalar loss built from this reward:

```
loss = −R
     + soft SoC / energy shortfall penalties (if infeasible)
     + soft V-ceiling overshoot penalty
```

Feasibility (SoC target reached, or energy delivered in energy mode) is preferred when selecting the best candidate per family.

3. **Search** — Gaussian-process BO (`scikit-optimize`, acquisition **PI** by default) explores each family independently (~40 evaluations). Family seeds warm-start the GP; physical ordering constraints (e.g. i₂ ≤ i₁) are enforced by each family’s `from_dict`, identical to random search.

Legacy `--reward-mode legacy_temp_time` keeps the older temperature + time shaped rewards for ablation only.

### Hybrid degradation methodology

Source: Sinha, S. S., Lehman, B., and Bharadwaj, P. (2026). *"Life Extension of
Lithium-Ion Battery: Degradation Comprehension, Modeling, Characterization, and
Mitigation Strategies."* IEEE Open Journal of Power Electronics, vol. 7.
DOI: 10.1109/OJPEL.2025.3639205. Implementation: [`Constrained_BO/hybrid_degradation.py`](Constrained_BO/hybrid_degradation.py).

The hybrid model treats calendar and cyclic stress as two **independent**
degradation components (`Q_calendar` and `Q_cyclic`), summed only at the
top-level reward — never mixed inside either sub-model.

#### Default equations

| Component | Default equation | Formula | Inputs |
|-----------|-------------------|---------|--------|
| Calendar  | **Eq. (2)** (Arrhenius SoC–Temperature–Time) | `Q_cal = A·exp(B·SoC)·exp(-(Ea + C·SoC)/(R·T_K))·t_h^z` | mean SoC, mean T (K), elapsed time (h) |
| Cyclic    | **Eq. (7) / Table 7** (C-rate–Temperature–Ah throughput) | `Q_cyc = B(I)·exp(-Ea(I)/(R·T_K))·Ah^z(I)` | C-rate (via `B`, `Ea`, `z` looked up/interpolated from Table 7 at C/2, 2C, 6C, 10C), mean T (K), Ah throughput |

Both use absolute temperature in **Kelvin** internally; calendar time is in
**hours**; Ah is **ampere-hours**; SoC is a **fraction** in `[0, 1]`.

#### Alternative equations (selectable, never summed with the defaults)

| Equation | Role | How to select |
|----------|------|----------------|
| **Eq. (3)** | Alternative empirical calendar model (storage time in days, T in °C) | `HybridDegradationParameters(calendar_model="eq3")` |
| **Eq. (5)** | Alternative Arrhenius calendar model (`α=2.14e4`, `Ea=36.36 kJ/mol`) | `HybridDegradationParameters(calendar_model="eq5")` |
| **Eq. (8)** | Continuous current-dependent cyclic model (`Ea` linear in `\|I\|` via `alpha_I`) | `HybridDegradationParameters(use_alpha_current=True, alpha_I=...)` — **off by default**: `alpha_I=0.0` is an un-calibrated placeholder, so this path is only meaningful after fitting `alpha_I` to real data |

`calendar_model` is a `Literal["eq2", "eq3", "eq5"]` field on
`HybridDegradationParameters`; pass it through `make_hybrid_params(calendar_model=...)`
in `objective.py`, or construct `HybridDegradationParameters` directly for
custom scripts (e.g. `degradation_report.py`).

#### Interpreting `Q_calendar` / `Q_cyclic` / `Q_total`

**None of the coefficients above have been calibrated against NASA RW9–RW12 or
LFP aging data.** They are semi-empirical fits taken from other cells and test
conditions in the cited literature. Every exported `qloss_*` value is therefore
a **Relative Capacity-Loss Index** — a dimensionless, physically-motivated
stress score useful for *ranking* charging strategies against each other under
the *same* model — and must **not** be reported or read as an absolute
`"Capacity Fade (%)"` for any specific cell. This terminology is used
consistently in the exported JSON (`meta.qloss_terminology_note`), the code
comments in `hybrid_degradation.py`/`objective.py`, and the report figures
below.

A related literature quirk worth knowing when reading Figure 2 / the cyclic
curves: Table 7's published `(B, Ea)` coefficients are **not monotonic across
the full C-rate grid** — at matched Ah and temperature, `Q_cyclic(C/2)` is
slightly *higher* than `Q_cyclic(2C)`, before the expected "higher C-rate is
worse" trend resumes and holds monotonically from 2C through 10C. This is a
property of the published curve fits, not a bug in this implementation —
see the `TABLE7_B`/`TABLE7_EA` note in `hybrid_degradation.py` and the
corresponding test in `Constrained_BO/tests/test_hybrid_degradation.py`.

#### Units of exported metrics

Every metric produced by `evaluate_session()` has a documented unit in
`Constrained_BO.objective.RESULT_METRIC_UNITS` (also embedded in each run's
`constrained_bo_results.json` under `meta.metric_units`). Highlights:

| Metric | Unit |
|--------|------|
| `ah_throughput` | Ah |
| `nominal_c_rate`, `max_c_rate` | dimensionless (multiples of 1C) |
| `efc` | equivalent full cycles (dimensionless; `Ah / (2 × Q_rated_Ah)` — a charge-only session contributes at most 0.5 EFC) |
| `mean_soc`, `soc_start`, `soc_end`, `soc_delta` | fraction `[0, 1]` |
| `energy_delivered_j`, `energy_required_j`, `energy_full_j` | J |
| `peak_voltage` | V |
| `peak_temperature`, `mean_temperature` | °C |
| `qloss_calendar`, `qloss_cyclic`, `qloss_total` | Relative Capacity-Loss Index (dimensionless, **not** % fade) |
| `constraint_margins.voltage_margin_v` | V (`v_max - peak_voltage`; negative = ceiling exceeded) |
| `constraint_margins.energy_margin_j` | J (energy mode only) |
| `constraint_margins.soc_margin` | fraction `[0, 1]` (SoC mode only) |

#### Report figures

```bash
# Fig 1 (calendar contour) + Fig 2 (cyclic curves) — no BDT/checkpoint needed
venv/bin/python -m Constrained_BO.degradation_report --out-dir Constrained_BO/results/degradation_report

# + Fig 3 (cumulative degradation per profile) and Fig 4 (equal-energy table),
# sourced from an existing optimization run
venv/bin/python -m Constrained_BO.degradation_report \
  --results Constrained_BO/results/RW9/constrained_bo_results.json \
  --out-dir Constrained_BO/results/degradation_report
```

Figure 3 re-simulates each family's best profile through the run's original BDT
checkpoint (trajectories aren't persisted in the results JSON); if that
checkpoint isn't available in the current environment, Figure 3 is skipped
with a clear message and Figures 1, 2, and 4 still complete (Figure 4 only
needs the scalar `best_metrics` already stored in the results JSON).

#### Tests

`Constrained_BO/tests/test_hybrid_degradation.py` checks deterministic,
physically-expected monotonic behaviour: calendar loss increases with SoC,
temperature, and time; cyclic loss increases with Ah throughput and (within
the monotonic 2C–10C region) with C-rate. Run with:

```bash
venv/bin/python -m pytest Constrained_BO/tests/test_hybrid_degradation.py -v
```

#### Limitations

- **No chemistry-specific calibration.** All coefficients are literature
  values for other cells; a NASA RW9–RW12 / LFP aging campaign would be
  required before any absolute capacity-fade claim could be made.
- **Eq. 2's `A`, `B`, `C`, `Ea` are illustrative stubs**, not fit to any
  dataset — only Eq. 7/Table 7's coefficients come from a specific cited
  fit ([112]).
- **Eq. 8 (`alpha_I`) is not calibrated** (`alpha_I=0.0` by default), so it
  is provided for future use only, not as an active alternative today.
- Calendar and cyclic terms are **summed, not coupled** — no interaction
  term between simultaneous calendar and cyclic stress is modeled.

### Profile families (`Constrained_BO`)

| Family ID | Label | Idea |
|-----------|-------|------|
| `cccv` | CCCV | Constant current → CV taper to `v_cv` / `i_cutoff` |
| `two_step` | 2-step (SoC) | Step down current at a SoC threshold |
| `three_step` | 3-step (SoC) | Two SoC thresholds, three current levels |
| `pulsed` | Pulsed charge/rest | Charge / rest bursts (`rest_fraction` of on-time) |

---

## Setup

```bash
cd battery_RW_NASA
python -m venv venv

# Linux / macOS
venv/bin/pip install -r requirements.txt

# Windows
venv\Scripts\pip install -r requirements.txt
```

---

## Workflow

### 1. Train source twin (RW9)

```bash
# Linux / macOS
venv/bin/python scripts/train_twin.py --config configs/default.yaml

# Windows
venv\Scripts\python.exe scripts/train_twin.py --config configs/default.yaml
```

Default RW9 checkpoint path used by `Constrained_BO` is set in `Constrained_BO/config.py` (`TWIN_SOURCE`). Update that path after a new train run, or place / symlink the checkpoint accordingly.

Optional: `build_source_registry.py`, `visualize_twin.py`, `train_soc.py` (SOC MLPs—not required for Constrained_BO).

### 2. Fine-tune to another cell (optional)

```bash
venv/bin/python scripts/finetune_twin.py \
  --source_ckpt outputs/twin_source/<TIMESTAMP>/twin_source_RW9.pt \
  --out outputs/finetune_two_stage_RW10 \
  --targets RW10
```

Primary transfer metric: **held-out voltage RMSE**. Finetune fractions per cell are configured in `Constrained_BO/config.py` (`FINETUNE_FRAC_BY_CELL`).

### 3. Constrained_BO charging optimization (primary)

OCV curves for Constrained_BO are fit/cached under `Constrained_BO/data/<CELL>/` (auto on first run; use `--refit-ocv` to force).

**Default: GP-BO + hybrid Q_loss** (RW9, SoC 20% → 80%, ≤150 min):

```bash
# Linux / macOS
venv/bin/python -m Constrained_BO.run --cell RW9 --method gp_bo --acq-func PI --n-calls 40 --n-initial 10

# Windows
venv\Scripts\python.exe -m Constrained_BO.run --cell RW9 --method gp_bo --acq-func PI --n-calls 40 --n-initial 10
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--method gp_bo` | Gaussian-process BO (default) |
| `--method random_search` | Uniform / seed baseline (`--n-random`) |
| `--acq-func PI\|EI\|LCB` | Acquisition (default **PI**) |
| `--n-calls 40` | Evaluations per family (including seeds) |
| `--n-initial 10` | Warm-start budget (seeds + extra random) |
| `--reward-mode hybrid_qloss` | Default hybrid calendar + cyclic reward |
| `--w-soc --w-qloss --w-time --z` | Hybrid reward weights / exponent |
| `--soc-target 0.8` | Absolute SoC stop (classic SoC mode) |
| `--energy-fraction 0.40` | Energy mode: deliver this fraction of pack energy |
| `--max-duration-min 150` | Simulation horizon |
| `--decision-interval 30` | Fixed BDT re-anchor interval (seconds) |
| `--no-auto-decision-interval` | Skip drift-based interval selection |
| `--families cccv two_step …` | Subset of families |
| `--out-dir …` | Output directory (default `Constrained_BO/results/<CELL>`) |
| `--cells RW9 RW10` | Batch multiple cells |

Random-search baseline (same hybrid objective):

```bash
venv/bin/python -m Constrained_BO.run --cell RW9 --method random_search --n-random 80
```

Compare scripts (post-hoc, same reward):

- `Constrained_BO/compare_constant_current.py`
- `Constrained_BO/compare_pulsed.py`

---

## Outputs (`Constrained_BO`)

```
Constrained_BO/results/<CELL>/
  constrained_bo_results.json   # meta + per-family best params, metrics, BO history
  best_profiles.png             # I / V / T / SoC for best profile per family
```

JSON `meta` records `method` (`gp_bo` | `random_search`), `reward_mode`, weights, `acq_func`, `n_calls`, decision interval, profile bounds, and (since the Phase-1 hybrid degradation work) `qloss_terminology_note` and a full `metric_units` table. Per-family `best_metrics` include `qloss_calendar`, `qloss_cyclic`, `qloss_total` (Relative Capacity-Loss Index — see [Hybrid degradation methodology](#hybrid-degradation-methodology)), `efc`, `mean_soc`, `nominal_c_rate`/`max_c_rate`, `ah_throughput`, `energy_delivered_j`/`energy_required_j`, `peak_voltage`, `peak_temperature`, `constraint_margins`, `total_reward`, feasibility, and duration.

Twin checkpoints: `outputs/twin_source/` or finetune `registry/`.

---

## Interpreting results (RW9, GP-BO, hybrid Q_loss)

Source: `Constrained_BO/results/RW9/constrained_bo_results.json`  
Settings: `method=gp_bo`, `acq_func=PI`, `n_calls=40`, `n_initial=10`, SoC 20%→80%, T₀=24°C, age=0, max 150 min.

| Rank | Family | Loss | Reward R | Duration (min) | Q_total | Feasible | Best parameters (approx.) |
|------|--------|------|----------|----------------|---------|----------|---------------------------|
| 1 | Pulsed charge/rest | −0.352 | 0.412 | 60.5 | 0.094 | yes | i≈1.98 A, on=1 min, rest_frac=0.5, i_floor≈0.82 A |
| 2 | 2-step (SoC) | −0.342 | 0.404 | 62.5 | 0.096 | yes | i1≈1.95 A, i2≈0.98 A, soc_switch≈0.48 |
| 3 | CCCV | −0.340 | 0.406 | 66.0 | 0.092 | yes | i_cc≈1.22 A, v_cv≈4.14 V, i_cutoff≈0.05 A |
| 4 | 3-step (SoC) | −0.296 | 0.387 | 91.0 | 0.090 | yes | ~0.87 A flat (steps collapsed), slower |

**Takeaways under hybrid Q_loss (this run):**

- **Lowest loss** favors faster feasible charges that still keep the loss index moderate (pulsed / 2-step win on R).
- **Q_total** is dominated by the **cyclic** term; the calendar loss index over a single session is negligible (calendar aging accumulates over weeks/months, not minutes).
- Rankings **differ** from the older SEI-composite Stage 3 table in `charging_opt`—do not mix the two objectives when comparing papers or plots.
- Re-run with `--method random_search` for a sample-efficiency baseline against GP-BO.

---

## Code map (`Constrained_BO`)

| Module | Role |
|--------|------|
| `Constrained_BO/run.py` | CLI entry: GP-BO or random search, JSON + plots |
| `Constrained_BO/bayesian_optimizer.py` | Per-family `gp_minimize` → `evaluate_session` |
| `Constrained_BO/objective.py` | Session loss / reward aggregation + feasibility |
| `Constrained_BO/hybrid_degradation.py` | Calendar + cyclic Q_loss and hybrid R |
| `Constrained_BO/simulator.py` | Closed-loop BDT rollout + Coulomb SoC |
| `Constrained_BO/profiles.py` | Profile families + BO vector helpers |
| `Constrained_BO/profile_catalog.py` | Per-cell current / V / pulse bounds |
| `Constrained_BO/config.py` | Cell configs, checkpoints, start states |
| `Constrained_BO/ocv.py` | OCV–SoC fit / load for RW cells |
| `Constrained_BO/decision_interval.py` | Re-anchor interval selection |
| `Constrained_BO/bdt_thermal.py` | Thermal metrics from BDT trajectories |
| `Constrained_BO/viz.py` | Best-profile figures |
| `Constrained_BO/degradation_report.py` | Hybrid-degradation report figures (calendar contour, cyclic curves, cumulative degradation, equal-energy table) |
| `Constrained_BO/tests/test_hybrid_degradation.py` | Deterministic monotonicity tests for the hybrid degradation model |

Twin training: `rw_transfer/`, configs in `configs/default.yaml`.

---

## Training & transfer scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_twin.py` | Train source BDT on RW9 |
| `scripts/finetune_twin.py` | Fine-tune to RW10–RW12 (or LFP) |
| `scripts/evaluate_finetune.py` | Re-evaluate finetune checkpoints |
| `scripts/01_fit_ocv_curve.py` | OCV–SoC + Q(age) for `charging_opt` |
| `scripts/00_diagnose_drift.py` | Conformal drift margins (`charging_opt`) |

---

## Legacy Stage 3 (`charging_opt`)

The original eight-family benchmark (SEI composite / Wang physics / Chebyshev Pareto) remains available:

```bash
venv/bin/python scripts/01_fit_ocv_curve.py --cell RW9
venv/bin/python scripts/00_diagnose_drift.py --ckpt $BDT_CKPT --cell RW9

venv/bin/python scripts/03_optimize_profile_families.py \
  --bdt_ckpt $BDT_CKPT \
  --acq_func PI \
  --out_dir outputs/charging_opt_user/$USER/stage3_optimization \
  --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \
  --n_calls 40 --n_initial 10 \
  --max_duration_min 105 --max_minutes 150
```

| Module / script | Role |
|-----------------|------|
| `charging_opt/family_optimizer.py` | Per-family GP-BO |
| `charging_opt/lifetime_reward.py` | SEI / composite / physics / Chebyshev loss |
| `charging_opt/physics_degradation.py` | Wang capacity-fade model |
| `charging_opt/pareto_analysis.py` | Fastest / Lifetime / Balanced tags |
| `scripts/run_chebyshev_pareto_sweep.py` | Directed Pareto via ω |
| `scripts/run_physics_thermal_suite.py` | Physics + thermal + ambient suite |
| `scripts/gen_all_figs.py` | Publication figure bundle |

Example SEI-composite RW9 ranking (historical, SoC 15%→95%, ≤105 min) lived under `outputs/charging_opt_user/…/stage3_optimization/`—use that tree for SEI/Pareto comparisons, not `Constrained_BO/results/`.

### Notes (legacy Stage 3)

- **`--objective physics`** changes what BO minimizes; SEI/ΔSoC may still be reported.
- **Thermal derating** changes simulated current; **thermal loss** adds a penalty to the scalar objective.
- Full 8-family × 40 evals is typically multi-hour on GPU.

---

## Quick reference

```bash
# Primary path — hybrid Q_loss + GP-BO
python -m Constrained_BO.run --cell RW9 --method gp_bo --acq-func PI --n-calls 40 --n-initial 10

# Same objective, random baseline
python -m Constrained_BO.run --cell RW9 --method random_search --n-random 80

# Transfer cell (uses finetune checkpoint from config)
python -m Constrained_BO.run --cell RW10 --method gp_bo --n-calls 40 --n-initial 10
```
