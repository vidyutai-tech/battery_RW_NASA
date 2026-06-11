# NASA RW Battery Digital Twin — Transfer Learning & Charging Optimization

Research codebase for cross-battery transfer learning on the NASA **Random Walk (RW)** dataset (RW9–RW12), plus **lifetime-focused charging profile optimization** via Bayesian search on the frozen digital twin.

## Mapping

| Cell ID | Alias | Role |
|---------|-------|------|
| RW9 | A | Source (pretrained twin) |
| RW10 | B | Target |
| RW11 | C | Target |
| RW12 | D | Target |

## Default data filter (Option A)

- `charge (random walk)`
- `discharge (random walk)`
- `rest (random walk)`

Options B/C are configurable via `data.step_mode` in `configs/default.yaml`.

## Model I/O (digital twin)

Same v9 layout as the main `batt_prof_optimization` repository:

- **Inputs:** relative age, V₀, T₀, current sequence (I/5 + ΔI/5)
- **Outputs:** V(t), T(t) (residual from initial state)

## SOC (separate MLPs)

- **Inputs (training):** always **measured** voltage, temperature, per-sample age (and |I| for `vta_i`) — never twin predictions
- **Labels:** per-step Coulomb counting on RW charge/discharge steps
- Variants: **`v_only`**, **`vta`**, **`vta_i`**
- Not required for charging BO (which uses OCV–SoC curves from coulomb counting on reference discharges)

---

## Output directory layout

All generated artifacts live under `outputs/` (gitignored).

### Digital twin (source)

```
outputs/twin_source/<YYYYMMDD_HHMMSS>/
  twin_source_RW9.pt           # source BDT checkpoint
  twin_train_log.jsonl
  twin_train_curves.png
  twin_train_predictions.png
  twin_train_summary.json
  source_registry.json       # after build_source_registry.py
  plots/                     # after visualize_twin.py (optional)
```

### Fine-tuning (target cells)

Default run folder if `--out` is omitted: `outputs/finetune_percent/<timestamp>/`.

Recommended explicit name, e.g. `outputs/finetune_two_stage_RW10/`:

```
outputs/finetune_two_stage_RW10/
  plots/
    twin_finetune_percent_RW10.png
    finetune_curves_RW10_frac0.20_stage1.png
    finetune_curves_RW10_frac0.20_stage2.png
    actual_vs_pred_RW10_frac0.20.png
    ...                        # same for frac 0.40, 0.60
  registry/
    finetune_registry.json     # metrics: RMSE, MAPE, R², train time, …
    finetune_RW10_frac0.20.pt  # fine-tuned checkpoints
    finetune_RW10_frac0.40.pt
    finetune_RW10_frac0.60.pt
    train_log_RW10_frac0.20_stage1.jsonl
    train_log_RW10_frac0.20_stage2.jsonl
    ...
  finetune_percent_results.csv
  finetune_percent_summary.json
```

Hours study (optional): `outputs/finetune_hours/<timestamp>/`

### Charging profile optimization

Three top-level folders under `outputs/charging_opt/`:

```
outputs/charging_opt/
  models/                              # .npz, JSON sessions, per-stage registry.json
    stage1_state_estimation/           # ocv_soc_curve.npz, capacity_fade.npz
    stage1_drift/                      # conformal_margins.npz
    stage3_optimization/               # optimization_result.json, best_session.json
  plots/                               # PNG + tabular JSON per stage
    stage1_state_estimation/
    stage1_drift/
    stage2_reward_diagnostic/          # cc_sweep.json, sweep plots
    stage3_optimization/               # best_profile.png, bo_convergence.png
  registry/
    charging_opt_registry.json         # master index of all stage metrics + paths
    artifacts_manifest.json
  snapshots/                           # dated backups (save_charging_opt_artifacts.py)
```

BDT checkpoints stay in `outputs/twin_source/` or finetune `registry/`; they are referenced from `charging_opt/registry/charging_opt_registry.json` under `external_models`.

---

## Scripts reference

| Script | Output | Description |
|--------|--------|-------------|
| `scripts/train_twin.py` | `outputs/twin_source/<ts>/` | Train source BDT on RW9 |
| `scripts/train_soc.py` | same run dir | SOC MLPs (optional) |
| `scripts/build_source_registry.py` | `<run>/source_registry.json` | Profile checkpoint + metrics |
| `scripts/visualize_twin.py` | `<run>/plots/` | Validation figures |
| `scripts/finetune_twin.py` | `<out>/plots/` + `<out>/registry/` | Two-stage % fine-tune (20/40/60%) |
| `scripts/evaluate_finetune.py` | `<run>/plots/` + `registry/` | Re-evaluate saved finetune ckpts |
| `scripts/hours_study.py` | `outputs/finetune_hours/<ts>/` | Hours sweep + recommendation |
| `scripts/01_fit_ocv_curve.py` | `charging_opt/models/stage1_state_estimation/` | OCV–SoC + Q(age) |
| `scripts/00_diagnose_drift.py` | `charging_opt/models/stage1_drift/` | Conformal drift margins |
| `scripts/sweep_cc_profiles.py` | `charging_opt/plots/stage2_reward_diagnostic/` | CC sweep (sanity check) |
| `scripts/diagnose_bo_objective.py` | stdout | Loss table for fixed currents |
| `scripts/02_optimize_charging_profile.py` | `charging_opt/models|plots/stage3_optimization/` | Bayesian lifetime optimizer |
| `scripts/save_charging_opt_artifacts.py` | `charging_opt/registry/` | Verify / snapshot / replot |
| `scripts/migrate_charging_opt_layout.py` | — | One-time move from legacy flat layout |

---

## Full pipeline (from scratch)

Prerequisites: NASA `.mat` files under `NASA_RW/dataset/`, Python venv with dependencies.

```bash
cd battery_RW_NASA

python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

### Step 1 — Train source digital twin (RW9)

Long-running; early-stops on validation loss.

```bash
venv/bin/python scripts/train_twin.py --config configs/default.yaml
```

Note the checkpoint path printed at the end, then:

```bash
export BDT_RUN=outputs/twin_source/<TIMESTAMP>
export BDT_CKPT=$BDT_RUN/twin_source_RW9.pt
```

Optional:

```bash
venv/bin/python scripts/build_source_registry.py --run_dir $BDT_RUN --evaluate
venv/bin/python scripts/visualize_twin.py --ckpt_dir $BDT_RUN
venv/bin/python scripts/train_soc.py --run-dir $BDT_RUN   # optional SOC MLPs
```

### Step 2 — Fine-tune on target cell(s) (optional)

```bash
venv/bin/python scripts/finetune_twin.py \
  --source_ckpt $BDT_CKPT \
  --out outputs/finetune_two_stage_RW10 \
  --targets RW10
```

Checkpoints: `outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac{0.20,0.40,0.60}.pt`

Use a fine-tuned twin in charging optimization:

```bash
export BDT_CKPT=outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.40.pt
```

### Step 3 — Charging profile optimization

Always pass `--bdt_ckpt` (or `--ckpt` for drift) after a fresh train — default paths in `charging_opt/artifacts.py` may point at an older run.

```bash
# Stage 1a — OCV–SoC curve + capacity fade
venv/bin/python scripts/01_fit_ocv_curve.py --cell RW9

# Stage 1b — BDT drift margins
venv/bin/python scripts/00_diagnose_drift.py --ckpt $BDT_CKPT --cell RW9

# Stage 2 — objective sanity check (optional)
venv/bin/python scripts/sweep_cc_profiles.py \
  --bdt_ckpt $BDT_CKPT --max_duration_min 105

# Stage 3 — Bayesian optimization (default: 95% SoC within 105 min)
venv/bin/python scripts/02_optimize_charging_profile.py \
  --bdt_ckpt $BDT_CKPT \
  --n_calls 40 \
  --max_duration_min 105 \
  --max_minutes 150

# Verify artifacts
venv/bin/python scripts/save_charging_opt_artifacts.py --verify
```

**Charging objective:** minimize SEI/ΔSoC among profiles that reach the SoC target within `max_duration_min`. Use `--no_time_limit` for unconstrained lifetime-only (may select ~0.75 A / ~140 min).

**Three-stage pipeline (code):**

| Stage | Module | Role |
|-------|--------|------|
| 1 | `charging_opt/profile_simulator.py` | Frozen BDT simulates V/T for candidate profiles |
| 2 | `charging_opt/lifetime_reward.py` | SEI proxy + SoC/time constraints → scalar loss |
| 3 | `charging_opt/bayesian_optimizer.py` | GP Bayesian optimization over profile parameters |

---

## Quick start (twin + transfer only)

```bash
cd battery_RW_NASA
venv/bin/pip install -r requirements.txt

# Train source twin on RW9
venv/bin/python scripts/train_twin.py

# Fine-tune RW10 (replace source ckpt path)
venv/bin/python scripts/finetune_twin.py \
  --source_ckpt outputs/twin_source/<TIMESTAMP>/twin_source_RW9.pt \
  --out outputs/finetune_two_stage_RW10 \
  --targets RW10

# Re-evaluate finetuned checkpoints
venv/bin/python scripts/evaluate_finetune.py \
  --run_dir outputs/finetune_two_stage_RW10 \
  --source_ckpt outputs/twin_source/<TIMESTAMP>/twin_source_RW9.pt \
  --target RW10
```

---

## Metric definitions

| Metric | Formula | Unit |
|--------|---------|------|
| **MSE** | mean((ŷ − y)²) | V² or °C² |
| **RMSE** | √MSE | V or °C |
| **MAE** | mean(\|ŷ − y\|) | V or °C |
| **MAPE_pct** | mean(\|ŷ − y\| / \|y\|) × 100 | % |
| **R²** | 1 − SS_res / SS_tot | dimensionless |

**Primary transfer metric:** held-out voltage RMSE.

### Fine-tuning (two-stage, `configs/default.yaml`)

| Stage | Layers | Train loss | LR | Max epochs |
|-------|--------|------------|-----|------------|
| 1 — head warmup | Output head only | `1·MSE_V + 100·MSE_T + 5·Pearson` | 5e-4 | 150 |
| 2 — full fine-tune | All layers | `50·MSE_V + 50·MSE_T + 5·Pearson` | 5e-7 | 500 |

Default data fractions: **20%, 40%, 60%** (`phase2.data_fractions`).

### Author vs RW-window twin training

| Setting | `configs/default.yaml` | `configs/rw_transfer.yaml` |
|--------|-------------------------|----------------------------|
| Data | All steps, `decimation: 1` | RW steps, `decimation: 10` |
| Windows | Chunks of 150 | Sliding, stride 50 |
| Split | Random 60/20/20 | Chronological 80/10/10 |

## Data volume note

Each cell has **~7.7M** samples at full resolution. Default `data.decimation: 1` in `configs/default.yaml` uses full fidelity (more RAM/time). Use `decimation: 10` in `configs/rw_transfer.yaml` for faster window-based training.

---

## Project layout

```
battery_RW_NASA/
  configs/default.yaml
  charging_opt/
    paths.py              # models/ | plots/ | registry/ layout
    artifacts.py          # canonical paths + master registry
    profile_simulator.py  # Stage 1 — BDT rollout
    lifetime_reward.py    # Stage 2 — objective
    bayesian_optimizer.py # Stage 3 — BO
    bdt_rollout.py        # frozen twin wrapper
    soc_utils.py          # OCV–SoC, Q(age)
  rw_transfer/
    registry.py           # FinetuneRegistry + SourceModelRegistry
    training/twin_trainer.py
    viz/plots.py
  scripts/
    train_twin.py
    finetune_twin.py
    00_diagnose_drift.py
    01_fit_ocv_curve.py
    02_optimize_charging_profile.py
    sweep_cc_profiles.py
    save_charging_opt_artifacts.py
  outputs/                # gitignored — see layout above
  NASA_RW/dataset/        # raw .mat files (gitignored)
```

## Notebook

`notebooks/01_dataset_exploration.ipynb` — step counts, duration, V/I/T previews per cell.

## Plot style

All plots follow the main repo theme: DejaVu Sans, 150 dpi, no top/right spines, dashed grid α 0.4.  
Colors: `ACCENT #2563EB`, `ORANGE #EA580C`, `GREEN #16A34A`, `PURPLE #7C3AED`, `GREY #6B7280`
