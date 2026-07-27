Final charging optimization results (paper + UI)
================================================
Per-cell folders RW9–RW12 contain fig8*/fig9*, comparison_table.*,
GP-BO/random JSON+PNG, baseline bar charts, and lifetime CSVs.

* Same energy target for all cells (default ``--energy-fraction 0.40``).
* w_qloss=2.0, w_time=0.1.
* Soft qloss_cap = Random reward-best Q (GP-BO must match/beat Random on Q).
* Tiny duration_loss_weight keeps a speed preference among equal-Q profiles.
* Profile families keep structural constraints (2-/3-step ΔI, pulsed pulses).
* Lifetime fig9 uses reward-best GP-BO (aligned with the constrained objective).

UI mirrors: Constrained_BO/results/ui_runs/{cell}/
Regenerate:
  python -m Constrained_BO.export_final_charging_opt_results --device cuda --energy-fraction 0.40
