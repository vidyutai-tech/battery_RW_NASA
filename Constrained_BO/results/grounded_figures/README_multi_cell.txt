Per-cell grounded BO/CC/lifetime figures
========================================
Each RW10/RW11/RW12 folder uses that cell's best finetuned BDT
(frac 0.60) and default energy window (RW10=0.55, RW11/12=0.40).

Contents per cell/: fig8*, fig9*, gp_bo_results.json,
random_search_results.json, CSVs.

Regenerate:
  python -m Constrained_BO.run_grounded_multi_cell --device cuda
