# Zero-shot RW9 twin comparison (no fine-tuning)

Both models were trained on **RW9 only**. Weights are frozen. Each cell uses the
same author test split (60/20/20, seed 42, chunk length 150). RW9 is in-distribution;
RW10–RW12 are **zero-shot transfer**.

## Voltage — held-out test

| Cell | legacy_cpu_pth V RMSE | legacy_cpu_pth V MAE | legacy_cpu_pth V MAPE % | legacy_cpu_pth V R² | source_RW9_pt V RMSE | source_RW9_pt V MAE | source_RW9_pt V MAPE % | source_RW9_pt V R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RW9 | 0.0373 | 0.0239 | 0.650 | 0.9846 | 0.0283 | 0.0151 | 0.415 | 0.9911 |
| RW10 | 0.0367 | 0.0245 | 0.664 | 0.9848 | 0.0271 | 0.0149 | 0.408 | 0.9917 |
| RW11 | 0.3802 | 0.1363 | 3.502 | -0.6319 | 0.0490 | 0.0307 | 0.830 | 0.9729 |
| RW12 | 0.3843 | 0.1447 | 3.719 | -0.6291 | 0.0547 | 0.0347 | 0.935 | 0.9670 |

## Temperature — held-out test

| Cell | legacy_cpu_pth T RMSE | legacy_cpu_pth T MAE | legacy_cpu_pth T MAPE % | legacy_cpu_pth T R² | source_RW9_pt T RMSE | source_RW9_pt T MAE | source_RW9_pt T MAPE % | source_RW9_pt T R² |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RW9 | 0.2993 | 0.1222 | 0.381 | 0.9936 | 0.3085 | 0.1353 | 0.418 | 0.9932 |
| RW10 | 0.2908 | 0.1315 | 0.417 | 0.9939 | 0.2892 | 0.1285 | 0.405 | 0.9939 |
| RW11 | 1.1370 | 0.8736 | 3.613 | 0.8418 | 0.9601 | 0.7134 | 2.993 | 0.8872 |
| RW12 | 1.1972 | 0.9161 | 3.759 | 0.8525 | 1.0348 | 0.7809 | 3.232 | 0.8898 |

## In-distribution vs transfer (mean RMSE)

| Model | RW9 V RMSE | mean RW10–12 V RMSE | RW9 T RMSE | mean RW10–12 T RMSE |
| --- | ---: | ---: | ---: | ---: |
| legacy_cpu_pth | 0.0373 | 0.2671 | 0.2993 | 0.8750 |
| source_RW9_pt | 0.0283 | 0.0436 | 0.3085 | 0.7613 |

## Recommendation (this metric, this split)

- Lower mean **voltage** RMSE on RW10–RW12: **source_RW9_pt**
- Lower mean **temperature** RMSE on RW10–RW12: **source_RW9_pt**
- Lower mean **V+T** RMSE on RW10–RW12: **source_RW9_pt**

Pick the generalized model from the transfer columns (RW10–RW12), not from RW9
test error. RW9 test error only says which model fit its training cell.

If voltage error on transfer cells is the charging-constraint quantity (4.20 V),
prefer the voltage winner.
