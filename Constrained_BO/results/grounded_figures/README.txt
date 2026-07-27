Grounded NASA RW figures
========================
All capacity points come from measured 1 A 'reference discharge' steps
in RW9–RW12 .mat files (OCV-corrected full-window Ah when an OCV curve
is available). Temperatures are onboard cell thermocouple readings.

Humidity is not recorded in NASA RW — no RH plots are included.

CSV: capacity_fade_measured.csv (one row per reference discharge).

BO vs CC degradation (fig8*)
----------------------------
Same-session comparison under the UI run constraint (40% energy):
  CC ½C / 1C / 2C  vs  Random best  vs  GP-BO best.

  fig8_bo_vs_cc_degradation.png        — one-plot Q_loss bars
  fig8b_bo_vs_cc_degradation_detail.png — calendar/cyclic split
  fig8c_bo_vs_cc_pareto.png            — duration vs Q_loss cloud
  bo_vs_cc_degradation.csv

Lifetime fade projections (fig9*)
---------------------------------
Line graphs: remaining capacity vs cycles / Ah under repeated equal-energy
sessions, using the hybrid calendar+cyclic model (policy ranking grounded;
% SoH scale illustrative, anchored so CC ½C = 80% at cycle 400).

  fig9_lifetime_fade_vs_cycles.png
  fig9b_lifetime_fade_vs_throughput.png   — + measured cell on Ah axis
  fig9c_lifetime_capacity_vs_cycle_index.png  — fig1-style Ah vs cycle
  fig9d_lifetime_delta_vs_halfC.png       — ΔSoH vs CC ½C (clearest)
  lifetime_fade_projection.csv

Per-cell folders (finetuned BDT frac0.60)
-----------------------------------------
  grounded_figures/RW10/  energy=0.55  finetune_RW10_frac0.60.pt
  grounded_figures/RW11/  energy=0.40  finetune_RW11_frac0.60.pt
  grounded_figures/RW12/  energy=0.40  finetune_RW12_frac0.60.pt

Each contains the same fig8*/fig9* suite + gp_bo_results.json /
random_search_results.json from a fresh hybrid_qloss run.

Regenerate:
  python -m Constrained_BO.bo_degradation_comparison
  python -m Constrained_BO.lifetime_fade_projection
  python -m Constrained_BO.run_grounded_multi_cell --device cuda
