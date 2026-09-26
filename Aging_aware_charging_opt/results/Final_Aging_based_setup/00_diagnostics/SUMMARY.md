# Phase 0 diagnostics

No optimization was run. Numbers are as computed (Python `repr` / full JSON).

## 0.1 Training-window charging currents

NASA sign: negative current is charge. Charging current reported here is `−I` on samples with `I < 0`.

Windows used:
- RW9 / `twin_source_RW9.pt`: author random 60/20/20 seed-42 **train** split.
- RW10–RW12 frac0.60 twins: deterministic **prefix** of that train split at fraction 0.6.

- **RW9** (source, author_random_60_20_20_seed42_train): n_windows=34127, n_charging=2447723, max_charging=4.618000030517578 A, p99=4.5 A, n>3A=132850, n>4A=32655, n>4.5A=2068, n(V≥4.1 V)=753461 of 5153177 voltage samples (charging-only V≥4.1: 729609 of 2447723).
- **RW10** (finetune_frac0.60, author_train_prefix_frac0.60): n_windows=20630, n_charging=1486590, max_charging=4.620999813079834 A, p99=4.5 A, n>3A=111255, n>4A=29055, n>4.5A=1141, n(V≥4.1 V)=427789 of 3115130 voltage samples (charging-only V≥4.1: 412645 of 1486590).
- **RW11** (finetune_frac0.60, author_train_prefix_frac0.60): n_windows=20794, n_charging=1488780, max_charging=4.616000175476074 A, p99=4.5 A, n>3A=154840, n>4A=32214, n>4.5A=4371, n(V≥4.1 V)=431393 of 3139894 voltage samples (charging-only V≥4.1: 418812 of 1488780).
- **RW12** (finetune_frac0.60, author_train_prefix_frac0.60): n_windows=20732, n_charging=1495027, max_charging=4.61899995803833 A, p99=3.750999927520752 A, n>3A=82651, n>4A=12700, n>4.5A=2111, n(V≥4.1 V)=416736 of 3130532 voltage samples (charging-only V≥4.1: 402502 of 1495027).
- **twin_source_RW9.pt** (source_RW9, author_random_60_20_20_seed42_train): n_windows=34127, n_charging=2447723, max_charging=4.618000030517578 A, p99=4.5 A, n>3A=132850, n>4A=32655, n>4.5A=2068, n(V≥4.1 V)=753461 of 5153177 voltage samples (charging-only V≥4.1: 729609 of 2447723).

**Phase 2 current upper bound** = min of the four cell training maxima = **4.616000175476074 A** (from RW11).

JSON: `training_currents.json`.

## 0.2 Identical held-out windows

Fine-tuned twins (frac 0.60) and twin_source_RW9.pt are both scored on the author 60/20/20 seed-42 TEST split of that cell. Training subsets differ (full train vs 60% prefix), but the held-out test windows are the same function of (cell, seed=42). Re-evaluation below uses one materialised test batch per cell for both models.

Identical test batches for both models on every cell: **True**.

- **RW9**: n_test=11377, identical_batch=True. Voltage RMSE finetuned/source-cell twin = 0.028346536185207775 V (28.346536185207775 mV); zero-shot twin_source_RW9.pt = 0.028346536185207775 V (28.346536185207775 mV); finetuned better on V? False. Temp RMSE finetuned = 0.30845948572427684 °C; zero-shot = 0.30845948572427684 °C; finetuned better on T? False.
- **RW10**: n_test=11461, identical_batch=True. Voltage RMSE finetuned/source-cell twin = 0.03232914213844151 V (32.32914213844151 mV); zero-shot twin_source_RW9.pt = 0.02707292935738748 V (27.07292935738748 mV); finetuned better on V? False. Temp RMSE finetuned = 0.284166795486449 °C; zero-shot = 0.28916574076525814 °C; finetuned better on T? True.
- **RW11**: n_test=11553, identical_batch=True. Voltage RMSE finetuned/source-cell twin = 0.04825838049076362 V (48.25838049076362 mV); zero-shot twin_source_RW9.pt = 0.04896023902663506 V (48.96023902663506 mV); finetuned better on V? True. Temp RMSE finetuned = 0.15170196855919357 °C; zero-shot = 0.9600909816459059 °C; finetuned better on T? True.
- **RW12**: n_test=11519, identical_batch=True. Voltage RMSE finetuned/source-cell twin = 0.06392941710730615 V (63.92941710730615 mV); zero-shot twin_source_RW9.pt = 0.05467705219013163 V (54.67705219013163 mV); finetuned better on V? False. Temp RMSE finetuned = 0.26355151510516145 °C; zero-shot = 1.0347509662567658 °C; finetuned better on T? True.

Cells where the cell-specific checkpoint has lower voltage RMSE than zero-shot `twin_source_RW9.pt`: 1 / 4.

JSON: `identical_windows_rmse.json`.

## 0.3 Original GP-BO initial conditions (`results/08_gp_bo/<CELL>/summary.json`)

- **RW9**: SOC0=0.2, V0=3.737303728639184 V, T0=24.0 °C, age=0.0, prev_i=0.0. Matches start_state_and_vnom: True.
- **RW10**: SOC0=0.2, V0=3.735862307051581 V, T0=24.0 °C, age=0.0, prev_i=0.0. Matches start_state_and_vnom: True.
- **RW11**: SOC0=0.2, V0=3.728857197853136 V, T0=24.0 °C, age=0.0, prev_i=0.0. Matches start_state_and_vnom: True.
- **RW12**: SOC0=0.2, V0=3.7401394244878614 V, T0=24.0 °C, age=0.0, prev_i=0.0. Matches start_state_and_vnom: True.

Shared (SOC0, T0, age) already used: SOC0=0.2, T0=24.0 °C, age=0.0.
V0 differs by cell because it is taken from that cell's OCV–SOC curve at SOC0 (`start_state_and_vnom`). Phase 2 will keep this policy.

JSON: `original_initial_conditions.json`.

## 0.4 Reward-term decomposition of existing 14_litmus_top6 elites

R = w_soc·ΔSOC − w_loss·Q_loss − w_time·t_h^z_time
with w_soc=1.0, w_loss=2.0, w_time=0.1, z_time=0.55.

Mean |term| across 24 elites:
- |w_soc·ΔSOC| = 0.37399800228269015
- |w_loss·Q_loss| = 0.0002418507275852371
- |w_time·t^z| = 0.05592749756578195

Ratios of those means: SOC/Q = 1546.4001535859738; SOC/time = 6.6871935731219425; time/Q = 231.24800212176763.

Spread (mean, std, CV%, min, max):
- w_soc·ΔSOC: 0.37399800228269015, 0.003696864982726626, 0.9884718528342067 %, 0.36881649420925144, 0.3799778297459108
- w_loss·Q: 0.0002418507275852371, 1.0559178538729631e-05, 4.365989982398622 %, 0.000231860471390405, 0.00026482186769191386
- w_time·t^z: 0.05592749756578195, 0.008136012358373515, 14.54742785300546 %, 0.041890068489948605, 0.06650876645229654

Max |reconstructed R − recorded R| = 0.0.

CSV: `elite_reward_terms.csv`.
