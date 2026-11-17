# Aging-aware discrete charging optimization

Search for discrete charging profiles on frozen NASA Randomized Battery Usage digital twins (RW9–RW12). Commanded charge current is restricted to **1.0, 1.5, 2.0, 3.0, and 4.5 A**. Saved search output is in `Final_Aging_aware_charging_opt/results_discrete_current/`.

Charge current follows the NASA sign (negative while charging). The reported reward is the advantage over that cell’s own 0.5C CCCV charge, multiplied by 1000.

| Path | Role |
|------|------|
| `outputs/twin_source/20260610_111409/twin_source_RW9.pt` | RW9 source twin |
| `outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.60.pt` | RW10 twin, fine-tuned on 60% of its training windows |
| `outputs/finetune_two_stage_RW11/registry/finetune_RW11_frac0.60.pt` | RW11 twin, same fine-tune |
| `outputs/finetune_two_stage_RW12/registry/finetune_RW12_frac0.60.pt` | RW12 twin, same fine-tune |
| `Final_Aging_aware_charging_opt/` | Frozen-twin inference, GP-BO, random search, figures |
| `Final_Aging_aware_charging_opt/ocv_curves.npz` | OCV–SOC curves used by the simulator |
| `Final_Aging_aware_charging_opt/results_discrete_current/` | Discrete-current results |
| `NASA_RW/dataset/` | NASA Matlab files (local only; not in git) |

## 1. Twins

The charging search loads the four checkpoints above. The commands to retrain them are in section 2. `scripts/train_twin.py` also fits the SOC models after the twin. A new RW9 file is timestamped, so set the RW9 entry in `CHECKPOINTS` (`Final_Aging_aware_charging_opt/config.py`) to that file before searching again.

**RW9 source twin.** Windows come from the NASA RW Matlab files (`rw_operational` steps, decimation 10, length 150, stride 50), split chronologically 80% / 10% / 10%. The model is a one-layer Transformer decoder (`d_model` 150, 20 heads, dropout 0.1). Each window is conditioned on relative age, initial voltage, and initial temperature, and on the current sequence. It predicts voltage and temperature. Training used AdamW at `2e-4`, batch size 64, gradient clip 1.0, and early stopping on validation loss (patience 80, cap 10000 epochs). The loss was `100 * MSE(voltage) + MSE(temperature)`. The kept file is `outputs/twin_source/20260610_111409/twin_source_RW9.pt`.

**RW10–RW12 fine-tunes.** Each target cell starts from that RW9 checkpoint. Adaptation uses 60% of that cell’s training windows (the sweep also fit 20% and 40%; the optimizer uses 60%). Fine-tuning is two-stage, learning rate `5e-7`, Pearson temperature weight 5:

- Stage 1, 150 epochs: voltage weight 1, temperature weight 100.
- Stage 2, 500 epochs: voltage weight 10, temperature weight 50.

Those three files are `outputs/finetune_two_stage_RW1x/registry/finetune_RW1x_frac0.60.pt`. Settings are in `configs/rw_transfer.yaml`. The trainer is `scripts/train_twin.py`, `scripts/finetune_twin.py`, and `rw_transfer/`.

## 2. Setup

From the repository root, once per machine:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd Final_Aging_aware_charging_opt
```

Later commands assume that shell. Place the NASA PCoE “Battery Uniform Distribution Charge Discharge DataSet 2Post” Matlab files at the path in `config.py` (`MATLAB_DIR`). Those `.mat` files are gitignored. The four `.pt` checkpoints and `ocv_curves.npz` are already in the tree.

Check that each frozen twin loads:

```bash
python twin.py
```

To retrain, go back to the repository root in the same shell. A new RW9 file lands in `outputs/twin_source/<timestamp>/`. Pass that path as `--source_ckpt` if you want the fine-tunes to start from it. These `--out` folders are the ones `config.py` already loads, so a retrain replaces those checkpoints.

```bash
cd ..
python scripts/train_twin.py --config configs/rw_transfer.yaml
python scripts/finetune_twin.py --config configs/rw_transfer.yaml --source_ckpt outputs/twin_source/20260610_111409/twin_source_RW9.pt --targets RW10 --out outputs/finetune_two_stage_RW10
python scripts/finetune_twin.py --config configs/rw_transfer.yaml --source_ckpt outputs/twin_source/20260610_111409/twin_source_RW9.pt --targets RW11 --out outputs/finetune_two_stage_RW11
python scripts/finetune_twin.py --config configs/rw_transfer.yaml --source_ckpt outputs/twin_source/20260610_111409/twin_source_RW9.pt --targets RW12 --out outputs/finetune_two_stage_RW12
cd Final_Aging_aware_charging_opt
```

## 3. Discrete charging search

From `Final_Aging_aware_charging_opt/`. Discrete current is on by default. Existing JSON files are skipped unless you pass `--no-resume`.

Default session: start at 20% SOC and 24 °C, stop at 4.2 V, deliver 40% of nominal energy, hard cap 150 min, decision every 30 s. Families are CCCV, two-step, three-step, and pulsed. Each family gets 80 evaluations. GP-BO uses seed 42, 15 initial points, and expected improvement. Baselines are CCCV at 0.5C, 1C, and 2C. Current is capped at 4.616 A, the smallest training-window maximum across the four cells.

```bash
python optimization.py --methods gp_bo random_search
```

That writes, for each cell and method:

- `results_discrete_current/gp_bo_RW9_cccv.json` (and `two_step`, `three_step`, `pulsed`)
- `results_discrete_current/gp_bo_RW9_baselines.json`
- `results_discrete_current/gp_bo_RW9_summary.json`
- the same names with `random_search_`

## 4. Energy and time limits

GP-BO again, 40 evaluations per family, for five tighter cases. A profile counts only if it delivers the energy before the time limit. Output goes to `results_discrete_current/requirements/`.

| Case | Energy | Time limit |
|------|--------|------------|
| `e30_t20` | 30% | 20 min |
| `e40_t30` | 40% | 30 min |
| `e50_t30` | 50% | 30 min |
| `e50_t20` | 50% | 20 min |
| `e60_t30` | 60% | 30 min |

```bash
python optimization.py --requirements
python optimization.py --top6
```

`--top6` writes `requirements/top6_summary.md`, `requirements/top6_summary.json`, and the current-profile figures for the 30%/20 min, 40%/30 min, 50%/30 min, and 60%/45 min cases.

## 5. Figures

```bash
python visualization.py --only figures
```

Figures land in `results_discrete_current/12_figures/`, with one folder per cell (`RW9`–`RW12`) plus the shared comparison plots. `--only litmus` writes the cross-cell elite comparison into the same results directory. `--only all` does both.
