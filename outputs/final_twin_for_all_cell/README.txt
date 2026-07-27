Paper-ready Digital Twin figures (NASA RW9–RW12)
================================================

Per cell/
  digital_twin_validation.png           — held-out test chunks (V/T)
  digital_twin_validation_val_mean.png  — mean over validation windows
  voltage_residual_test_chunks.png      — pred−meas residual in mV
  metrics.json

Root
  summary_mape_across_cells.png
  summary_metrics.json

Display note (senior review: noisy voltage predictions)
-------------------------------------------------------
Voltage/temperature predicted traces use a light Savitzky–Golay
overlay for paper readability; the faint dotted line is the raw
twin output. Reported MAPE is always computed on raw predictions.
Absolute voltage residuals are typically a few millivolts.

Checkpoints
  RW9  : outputs/twin_source/20260610_111409/twin_source_RW9.pt
  RW10–12 : finetune_*_frac0.60.pt (best registry checkpoints)

Regenerate:
  python scripts/export_final_twin_paper_figs.py
