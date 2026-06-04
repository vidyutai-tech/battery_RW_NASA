# NASA RW Battery Digital Twin — Transfer Learning

Research codebase for cross-battery transfer learning and minimum adaptation data studies on the NASA **Random Walk (RW)** dataset (RW9–RW12).

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
- **Labels:** per-step Coulomb counting on RW charge/discharge steps (`coulomb_soc_stitched_operational`)
- Variants: **`v_only`**, **`vta`**, **`vta_i`**
- Transfer experiments on target cells also use **measured** V/T

## Scripts and output naming

| Script | Experiment module | Output dir | Description |
|--------|------------------|-----------|-------------|
| `scripts/train_twin.py` | `twin_train` | `outputs/twin_source/` | Train twin + SOC on RW9 |
| `scripts/train_soc.py` | `soc_train` | *(same run dir)* | SOC only — measured V/T, per-step Coulomb labels |
| `scripts/finetune_twin.py` | `twin_finetune_percent` | `outputs/finetune_percent/` | % sweep: fine-tune vs scratch |
| `scripts/hours_study.py` | `twin_finetune_hours` | `outputs/finetune_hours/` | Hours sweep + threshold report |

### Key output files

| File | Contents |
|------|----------|
| `twin_train_curves.png` | Train loss + val RMSE curves |
| `twin_train_predictions.png` | Held-out V/T window predictions |
| `soc_train_log.jsonl` | Per-epoch train loss + val RMSE/MAPE/R² (all variants) |
| `soc_train_curves.png` | Training curves from the log |
| `soc_train_comparison.png` | v_only / vta / vta_i bar chart |
| `soc_train_series.png` | Coulomb label vs SOC prediction time series |
| `twin_finetune_percent_{target}.png` | RMSE vs % target data |
| `twin_finetune_gain_all_targets.png` | Transfer gain across all targets |
| `twin_finetune_hours_{target}.png` | RMSE vs adaptation hours |
| `twin_finetune_gap_closed_all.png` | Gap-closed fraction (0–1) vs hours |
| `twin_finetune_threshold_summary.png` | Min hours for 90/95/98/99% |
| `practical_recommendation.json` | Human-readable hours recommendation |

Target evaluation depends on `twin.pipeline`:

- **`author`** (default): random **60/20/20** chunk split on the target cell (same as source twin training); metrics on held-out **test chunks**; adaptation uses a fraction of **train chunks** only.
- **`rw_windows`**: fixed chronological **20% tail** (`transfer.eval_tail_frac`); adaptation from the prefix before that tail.

**Primary metric:** held-out voltage RMSE (transfer studies).  
**Twin training (default author config):** early-stops on validation `MSE(V,T)` (`early_stop_metric: val_loss`, patience 50).  
**Twin training (`high_mape.yaml`):** early-stops on `MAPE_V + MAPE_T`.  
**Gap-closed threshold:** `(RMSE_scratch − RMSE_ft) / (RMSE_scratch − RMSE_full)`.

### Author vs RW-window twin training

| Setting | `configs/default.yaml` (author) | `configs/rw_transfer.yaml` |
|--------|-----------------------------------|----------------------------|
| Data | All steps stitched, `decimation: 1` | RW operational steps, `decimation: 10` |
| Windows | Non-overlapping chunks of 150 | Sliding windows, stride 50 |
| Split | Random 60/20/20, seed 42 | Chronological 80/10/10 |
| Loss | `100·MSE_V + MSE_T` | Same (author-style weights) |
| Model | v9 decoder, `T += ΔT/10`, `nhead=20` | Same |

Fine-tune / hours scripts follow `twin.pipeline` in the config you pass:

```bash
# Author pipeline (matches source twin training)
python scripts/finetune_twin.py --config configs/default.yaml \
  --source_ckpt outputs/twin_source/20260601_182816/twin_source_RW9.pt \
  --out outputs/finetune_percent

# RW-window transfer study (legacy chronological protocol)
python scripts/finetune_twin.py --config configs/rw_transfer.yaml \
  --source_ckpt outputs/twin_source/.../twin_source_RW9.pt \
  --out outputs/finetune_percent_rw
```

## Data volume note

Each cell has **~7.7M** samples at full resolution. Default `data.decimation: 10` keeps every 10th point (~770k samples, ~15k windows) for practical training. Set `decimation: 1` in `configs/default.yaml` for full fidelity (requires more RAM/time).

## Quick start

Use the project venv at `../venv` (or install `requirements.txt`):

```bash
cd battery_RW_NASA
../venv/bin/pip install PyYAML  # if needed

# 1) Train source twin on RW9 (author pipeline — default)
../venv/bin/python scripts/train_twin.py
#    → writes outputs/twin_source/<timestamp>/ (checkpoint, logs, twin_train_*.png)

# 2) SOC only (measured V/T — does not use twin predictions; twin ckpt optional)
../venv/bin/python scripts/train_soc.py --run-dir outputs/twin_source/<timestamp>

# 3) Optional: main-repo-style twin validation figures
../venv/bin/python scripts/visualize_twin.py --ckpt_dir outputs/twin_source/<timestamp>

# RW-window pipeline for transfer studies (chronological split, decimation 10)
../venv/bin/python scripts/train_twin.py --config configs/rw_transfer.yaml

# Percentage fine-tune study
../venv/bin/python scripts/finetune_twin.py \
    --source_ckpt outputs/twin_source/<timestamp>/twin_source_RW9.pt

# Hours-based study + practical recommendation
../venv/bin/python scripts/hours_study.py \
    --source_ckpt outputs/twin_source/<timestamp>/twin_source_RW9.pt
```

Outputs: `outputs/twin_source/`, `outputs/finetune_percent/`, `outputs/finetune_hours/`.

### Visualization (after twin training)

`scripts/train_twin.py` already saves basic training plots in the run folder (`twin_train_curves.png`, `twin_train_predictions.png` via `rw_transfer/viz/plots.py`).

For **main-repo-style** validation figures (2×3 measured vs predicted V/T, SOC vs Coulomb, best test chunks), run:

```bash
../venv/bin/python scripts/visualize_twin.py --ckpt_dir outputs/twin_source/<timestamp>
```

| Component | Path |
|-----------|------|
| Plotting logic | `rw_transfer/viz/twin_validation_plots.py` |
| CLI entry point | `scripts/visualize_twin.py` |

**Outputs** (default: `<run>/plots/`, or `--out_dir`):

- `digital_twin_validation.png` — measured vs predicted V/T on best held-out test chunks  
- `digital_twin_validation_val_mean.png` — mean trajectories over validation chunks  
- `soc_estimation.png`, `soc_variant_comparison.png`, and related SOC charts  

**Examples:**

```bash
../venv/bin/python scripts/visualize_twin.py --ckpt_dir outputs/twin_source/<timestamp>
../venv/bin/python scripts/visualize_twin.py \
    --ckpt outputs/twin_source/<timestamp>/twin_source_RW9.pt \
    --out_dir outputs/twin_source/<timestamp>/plots
```

## Project layout

```
battery_RW_NASA/
  configs/default.yaml
  rw_transfer/          # library (models, data, viz/twin_validation_plots.py, …)
  scripts/              # CLI (train_twin.py, visualize_twin.py, …)
  notebooks/            # EDA
  outputs/              # experiments (gitignored)
  NASA_RW/dataset/      # raw .mat files
```

## Notebook

`notebooks/01_dataset_exploration.ipynb` — step counts, duration, V/I/T previews per cell.

## Plot style

All plots follow the main repo (`visualize.py`) theme:  
- DejaVu Sans, 150 dpi, no top/right spines, dashed grid α 0.4  
- Colors: `ACCENT #2563EB`, `ORANGE #EA580C`, `GREEN #16A34A`, `PURPLE #7C3AED`, `GREY #6B7280`
