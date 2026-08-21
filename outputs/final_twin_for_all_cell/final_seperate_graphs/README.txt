Paper-ready Digital Twin figures (NASA RW9–RW12)
================================================

RW9/
  Source twin (no finetune fractions).
  digital_twin_validation.png, val_mean, residual, metrics.json, SOC/train plots

RW10|RW11|RW12/
  frac0.20/  frac0.40/  frac0.60/   — full figure set + metrics.json each
  Cell root mirrors frac0.60 (paper default).

Root summaries
  summary_mape_across_cells.png   — RW9 + cells @ 60%
  summary_mape_by_fraction.png    — V/T MAPE vs 20/40/60% (RW10–12)
  summary_mape_all_fractions.png  — flat bar view of all frac runs
  summary_metrics.json / .csv     — pooled val MAPE + registry RMSE/MAE/R²

Display note (senior review: noisy voltage predictions)
-------------------------------------------------------
Voltage/temperature predicted traces use a light Savitzky–Golay
overlay for paper readability; the faint dotted line is the raw
twin output. Reported MAPE is always computed on raw predictions.

Checkpoints
  RW9  : outputs/twin_source/20260610_111409/twin_source_RW9.pt
  RW10–12 : outputs/finetune_two_stage_RW*/registry/finetune_*_frac{0.20,0.40,0.60}.pt

Regenerate:
  python scripts/export_final_twin_paper_figs.py --device cpu
  python scripts/export_final_twin_paper_figs.py --device cuda --fractions 0.20 0.40 0.60
