Final charging optimization results (paper + UI)
================================================
Per-cell folders RW9–RW12 contain fig8*/fig9*, comparison_table.*,
GP-BO/random JSON+PNG, UI-style bar charts (time/temp/reward), and lifetime CSVs.

* Same energy target for all cells (default ``--energy-fraction 0.40``).
* w_qloss=2.0, w_time=0.1.
* Soft qloss_cap = Random reward-best Q (GP-BO must match/beat Random on Q).
* Baselines = classic CCCV (CC→CV at Vmax) at **½C and 1C** (1.1 A / 2.2 A).
* Paper % columns use **CCCV ½C** as the reference baseline.
* Hitting 4.2 V enters CV (not infeasible); energy target still applies.
* GP-BO / Random best-profile PNGs are regenerated from saved JSON.
* Also writes time_comparison.png, temperature_comparison.png, reward_comparison.png.

UI mirrors: Constrained_BO/results/ui_runs/{cell}/
Regenerate figures (reuse existing JSON):
  python -m Constrained_BO.export_final_charging_opt_results --device cpu --figures-only --energy-fraction 0.40
Full re-optimize:
  python -m Constrained_BO.export_final_charging_opt_results --device cuda --energy-fraction 0.40
