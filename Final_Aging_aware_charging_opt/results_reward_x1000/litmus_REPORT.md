# Top-6 cross-cell litmus test

Frozen per-cell digital twins. Frozen stage-8 GP-BO (80 evals / family / cell).
Identical operating box for every cell: SOC₀ = 20 %, T₀ = 24 °C, energy target = 40 % of
nominal pack energy, V_max = 4.20 V, I ∈ [0.75, 6.00] A, same reward, same GP-BO budget.
The only factor that changes is the cell twin.

This analysis does **not** force the four elite sets to match. It measures whether they do.

## Level 1 — charging family in each Top-6

| Family | RW9 | RW10 | RW11 | RW12 |
| --- | ---: | ---: | ---: | ---: |
| CCCV | 6 | 6 | 6 | 5 |
| Two-step | 0 | 0 | 0 | 1 |
| Three-step | 0 | 0 | 0 | 0 |
| Pulsed | 0 | 0 | 0 | 0 |

CCCV is the only family that appears in every cell's Top-6. Pulsed appears only for RW9.
RW11's entire Top-6 is CCCV.

## Level 2 — 24 elite profiles (native-twin replay)

| Cell | Rank | Family | t (min) | Q_loss | Reward | V_peak | I_mean (A) | I_max (A) | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RW9 | 1 | CCCV | 27.07 | 1.1736e-04 | -652.3515 | 4.201 | 1.83 | 1.85 | I=1.85A Vcv=4.200 |
| RW9 | 2 | CCCV | 27.05 | 1.1741e-04 | -652.7570 | 4.200 | 1.83 | 1.85 | I=1.85A Vcv=4.200 |
| RW9 | 3 | CCCV | 27.12 | 1.1734e-04 | -652.8328 | 4.200 | 1.83 | 1.84 | I=1.84A Vcv=4.200 |
| RW9 | 4 | CCCV | 27.17 | 1.1725e-04 | -652.9342 | 4.201 | 1.82 | 1.84 | I=1.84A Vcv=4.200 |
| RW9 | 5 | CCCV | 27.13 | 1.1731e-04 | -653.0073 | 4.200 | 1.83 | 1.84 | I=1.84A Vcv=4.199 |
| RW9 | 6 | CCCV | 27.17 | 1.1729e-04 | -652.9645 | 4.200 | 1.82 | 1.84 | I=1.84A Vcv=4.200 |
| RW10 | 1 | CCCV | 23.87 | 1.1652e-04 | -604.1032 | 4.200 | 2.06 | 2.07 | I=2.07A Vcv=4.200 |
| RW10 | 2 | CCCV | 23.98 | 1.1652e-04 | -605.3275 | 4.199 | 2.06 | 2.06 | I=2.06A Vcv=4.200 |
| RW10 | 3 | CCCV | 24.03 | 1.1650e-04 | -606.0614 | 4.197 | 2.05 | 2.05 | I=2.05A Vcv=4.200 |
| RW10 | 4 | CCCV | 24.08 | 1.1654e-04 | -606.6445 | 4.196 | 2.05 | 2.05 | I=2.05A Vcv=4.200 |
| RW10 | 5 | CCCV | 24.22 | 1.1657e-04 | -608.4114 | 4.191 | 2.04 | 2.04 | I=2.04A Vcv=4.200 |
| RW10 | 6 | CCCV | 24.25 | 1.1659e-04 | -608.8080 | 4.191 | 2.03 | 2.03 | I=2.03A Vcv=4.200 |
| RW11 | 1 | CCCV | 15.00 | 1.1127e-04 | -399.9859 | 4.158 | 3.34 | 4.62 | I=4.62A Vcv=4.081 |
| RW11 | 2 | CCCV | 15.15 | 1.1093e-04 | -399.9970 | 4.159 | 3.31 | 4.59 | I=4.59A Vcv=4.079 |
| RW11 | 3 | CCCV | 14.98 | 1.1119e-04 | -400.0699 | 4.158 | 3.34 | 4.62 | I=4.62A Vcv=4.086 |
| RW11 | 4 | CCCV | 15.00 | 1.1128e-04 | -400.1059 | 4.158 | 3.34 | 4.62 | I=4.62A Vcv=4.084 |
| RW11 | 5 | CCCV | 14.98 | 1.1123e-04 | -400.3742 | 4.158 | 3.34 | 4.62 | I=4.62A Vcv=4.094 |
| RW11 | 6 | CCCV | 15.37 | 1.1056e-04 | -400.4041 | 4.159 | 3.26 | 4.55 | I=4.55A Vcv=4.082 |
| RW12 | 1 | CCCV | 21.37 | 1.1910e-04 | -595.0453 | 4.196 | 2.28 | 2.28 | I=2.28A Vcv=4.200 |
| RW12 | 2 | CCCV | 22.10 | 1.1956e-04 | -606.9875 | 4.180 | 2.21 | 2.21 | I=2.21A Vcv=4.200 |
| RW12 | 3 | CCCV | 21.57 | 1.2026e-04 | -608.3402 | 4.201 | 2.26 | 2.73 | I=2.73A Vcv=4.200 |
| RW12 | 4 | CCCV | 22.73 | 1.1946e-04 | -613.3028 | 4.169 | 2.16 | 2.16 | I=2.16A Vcv=4.200 |
| RW12 | 5 | CCCV | 23.48 | 1.1925e-04 | -619.8622 | 4.158 | 2.10 | 2.10 | I=2.10A Vcv=4.200 |
| RW12 | 6 | Two-step | 24.43 | 1.1906e-04 | -629.2966 | 4.145 | 2.02 | 2.02 | I1=2.02 I2=0.75 |

## Parameter / outcome spread across the 24 native elites

| Quantity | Mean | Std | CV % | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| I_mean (A) | 2.342 | 0.594 | 25.4 | 1.824 | 3.341 |
| I_max (A) | 2.685 | 1.145 | 42.6 | 1.836 | 4.616 |
| Time (min) | 22.22 | 4.561 | 20.5 | 14.98 | 27.17 |
| Q_loss | 0.0001161 | 3.164e-06 | 2.7 | 0.0001106 | 0.0001203 |
| Reward | -567.9 | 100.8 | 17.7 | -653 | -400 |
| V_peak (V) | 4.182 | 0.02 | 0.5 | 4.145 | 4.201 |

A small CV means the elite set is a shared neighbourhood. A large CV on current or time
means the independently optimized elites are not the same protocol.

## Trajectory similarity (time-normalized RMSE of I(t))

Mean pairwise RMSE **within** a cell's own Top-6:

| Cell | Mean within-cell RMSE (A) |
| --- | ---: |
| RW9 | 0.029 |
| RW10 | 0.018 |
| RW11 | 0.036 |
| RW12 | 0.139 |

Mean pairwise RMSE **between different cells**: 0.816 A.

If within-cell RMSE is small and between-cell RMSE is comparable, the four Top-6 clouds
overlap. If between-cell RMSE is much larger, the elite sets occupy different current
trajectories.

## Cross-cell validation (24 profiles × 4 twins)

A profile is counted as pack-feasible if it meets the energy target and stays ≤ 4.20 V
on **every** twin.

Profiles feasible on all four twins: **16 / 24**
— RW9-P1, RW9-P2, RW9-P3, RW9-P4, RW9-P5, RW9-P6, RW10-P1, RW10-P2, RW10-P3, RW10-P4, RW10-P5, RW10-P6, RW12-P1, RW12-P2, RW12-P4, RW12-P5

## Voltage behaviour of the twins (from frozen stage-8 baselines + stage-2 RMSE)

These numbers describe **predicted** restrictiveness, not a claim about physical impedance.

| Cell | CCCV 1C V_peak | Headroom to 4.20 V (mV) | CCCV 2C feasible? | Twin V RMSE (mV) |
| --- | ---: | ---: | ---: | ---: |
| RW9 | 4.200 | -0 | False | nan |
| RW10 | 4.201 | -1 | False | nan |
| RW11 | 4.000 | 200 | False | nan |
| RW12 | 4.178 | 22 | False | nan |

Headroom of tens of millivolts is comparable to twin voltage RMSE on RW11/RW12
(48 mV / 64 mV). A predicted V = 4.05 V is therefore **not** 150 mV of demonstrated
safety margin.

## How to read the result

- **Consistent strategy:** same families dominate, currents in a tight band, many of the
  24 profiles remain feasible on the other twins.
- **Cell-specific / twin-specific:** one cell's elite set is pulsed or 6 A CCCV while
  the others sit near 2 A, and those profiles fail the 4.20 V (or energy) check on
  other twins.

The figures `family_counts.png`, `current_overlay.png`, `similarity_24x24.png`, and
`cross_feasible.png` are the visual version of this test.
