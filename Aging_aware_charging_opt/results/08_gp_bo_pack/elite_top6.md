# Pack GP-BO elite set (minimax / worst-cell loss)

One shared profile per row. Pack-feasible means **every** cell met the
energy target and stayed at or below 4.20 V. Time and Q_loss are
shown per cell so ranking inside the set can shuffle without changing
the waveform.

| Rank | Family | Pack loss | Worst cell | RW9 t (min) | RW10 t (min) | RW11 t (min) | RW12 t (min) | RW9 Q | RW10 Q | RW11 Q | RW12 Q | Feasible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Three-step | -0.2822 | RW9 | 27.87 | 28.03 | 29.47 | 28.10 | 1.178e-04 | 1.186e-04 | 1.121e-04 | 1.200e-04 | yes |
| 2 | Three-step | -0.2818 | RW9 | 28.27 | 28.40 | 29.87 | 28.52 | 1.180e-04 | 1.187e-04 | 1.123e-04 | 1.201e-04 | yes |
| 3 | Three-step | -0.2811 | RW9 | 28.83 | 28.95 | 30.45 | 29.12 | 1.184e-04 | 1.191e-04 | 1.127e-04 | 1.201e-04 | yes |
| 4 | CCCV | -0.2810 | RW9 | 28.98 | 29.07 | 30.37 | 29.27 | 1.180e-04 | 1.186e-04 | 1.121e-04 | 1.196e-04 | yes |
| 5 | Three-step | -0.2803 | RW9 | 28.88 | 29.12 | 31.38 | 29.22 | 1.202e-04 | 1.211e-04 | 1.154e-04 | 1.224e-04 | yes |
| 6 | Three-step | -0.2798 | RW9 | 29.53 | 29.65 | 31.20 | 29.92 | 1.189e-04 | 1.195e-04 | 1.132e-04 | 1.204e-04 | yes |

## Parameters

- **#1** `three_step`: `{'family_id': 'three_step', 'i1': 1.7984469107403447, 'i2': 1.5484469107403447, 'i3': 0.75, 'soc1': 0.55, 'soc2': 0.9}`
- **#2** `three_step`: `{'family_id': 'three_step', 'i1': 1.7745734582449346, 'i2': 1.5245734582449346, 'i3': 0.75, 'soc1': 0.55, 'soc2': 0.67}`
- **#3** `three_step`: `{'family_id': 'three_step', 'i1': 1.742679591259236, 'i2': 1.492679591259236, 'i3': 1.242679591259236, 'soc1': 0.55, 'soc2': 0.9}`
- **#4** `cccv`: `{'family_id': 'cccv', 'i_cc': 1.719038745707354, 'v_cv': 4.2, 'i_cutoff': 0.5}`
- **#5** `three_step`: `{'family_id': 'three_step', 'i1': 1.8007135814312711, 'i2': 1.0, 'i3': 0.75, 'soc1': 0.55, 'soc2': 0.67}`
- **#6** `three_step`: `{'family_id': 'three_step', 'i1': 1.7044522305231344, 'i2': 1.4544522305231344, 'i3': 1.2044522305231344, 'soc1': 0.55, 'soc2': 0.67}`
