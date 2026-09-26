# Top-6 cross-cell litmus test

Frozen per-cell digital twins. Frozen stage-8 GP-BO (80 evals / family / cell).
Identical operating box for every cell: SOC₀ = 20 %, T₀ = 24 °C, energy target = 40 % of
nominal pack energy, V_max = 4.20 V, I ∈ [0.75, 6.00] A, same reward, same GP-BO budget.
The only factor that changes is the cell twin.

This analysis does **not** force the four elite sets to match. It measures whether they do.

## Level 1 — charging family in each Top-6

| Family | RW9 | RW10 | RW11 | RW12 |
| --- | ---: | ---: | ---: | ---: |
| CCCV | 5 | 5 | 6 | 2 |
| Two-step | 1 | 1 | 0 | 3 |
| Three-step | 0 | 0 | 0 | 1 |
| Pulsed | 0 | 0 | 0 | 0 |

CCCV is the only family that appears in every cell's Top-6. Pulsed appears only for RW9.
RW11's entire Top-6 is CCCV.

## Level 2 — 24 elite profiles (native-twin replay)

| Cell | Rank | Family | t (min) | Q_loss | Reward | V_peak | I_mean (A) | I_max (A) | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RW9 | 1 | CCCV | 27.15 | 1.1731e-04 | -0.6529 | 4.200 | 1.83 | 1.84 | I=1.84A Vcv=4.200 |
| RW9 | 2 | CCCV | 27.82 | 1.1728e-04 | -0.6600 | 4.197 | 1.79 | 1.79 | I=1.79A Vcv=4.200 |
| RW9 | 3 | CCCV | 27.57 | 1.1768e-04 | -0.6607 | 4.190 | 1.80 | 1.82 | I=1.82A Vcv=4.188 |
| RW9 | 4 | CCCV | 28.22 | 1.1755e-04 | -0.6664 | 4.192 | 1.76 | 1.76 | I=1.76A Vcv=4.191 |
| RW9 | 5 | CCCV | 26.97 | 1.1835e-04 | -0.6606 | 4.200 | 1.83 | 1.89 | I=1.89A Vcv=4.200 |
| RW9 | 6 | Two-step | 28.60 | 1.1779e-04 | -0.6728 | 4.188 | 1.74 | 1.74 | I1=1.74 I2=0.75 |
| RW10 | 1 | CCCV | 24.38 | 1.1662e-04 | -0.6106 | 4.187 | 2.02 | 2.02 | I=2.02A Vcv=4.191 |
| RW10 | 2 | CCCV | 24.45 | 1.1664e-04 | -0.6114 | 4.185 | 2.02 | 2.02 | I=2.02A Vcv=4.200 |
| RW10 | 3 | CCCV | 24.53 | 1.1664e-04 | -0.6126 | 4.182 | 2.01 | 2.01 | I=2.01A Vcv=4.194 |
| RW10 | 4 | CCCV | 24.72 | 1.1670e-04 | -0.6151 | 4.176 | 2.00 | 2.00 | I=2.00A Vcv=4.200 |
| RW10 | 5 | Two-step | 24.72 | 1.1670e-04 | -0.6151 | 4.176 | 2.00 | 2.00 | I1=2.00 I2=1.00 |
| RW10 | 6 | CCCV | 24.87 | 1.1675e-04 | -0.6171 | 4.173 | 1.99 | 1.99 | I=1.99A Vcv=4.189 |
| RW11 | 1 | CCCV | 14.98 | 1.1125e-04 | -0.4005 | 4.158 | 3.34 | 4.62 | I=4.62A Vcv=4.097 |
| RW11 | 2 | CCCV | 14.98 | 1.1125e-04 | -0.4012 | 4.158 | 3.34 | 4.61 | I=4.61A Vcv=4.101 |
| RW11 | 3 | CCCV | 15.05 | 1.1119e-04 | -0.4028 | 4.158 | 3.32 | 4.60 | I=4.60A Vcv=4.120 |
| RW11 | 4 | CCCV | 15.52 | 1.1039e-04 | -0.4026 | 4.160 | 3.23 | 4.51 | I=4.51A Vcv=4.096 |
| RW11 | 5 | CCCV | 16.08 | 1.0978e-04 | -0.4021 | 4.158 | 3.14 | 4.62 | I=4.62A Vcv=4.050 |
| RW11 | 6 | CCCV | 16.08 | 1.0978e-04 | -0.4021 | 4.158 | 3.14 | 4.62 | I=4.62A Vcv=4.050 |
| RW12 | 1 | CCCV | 21.33 | 1.1903e-04 | -0.5947 | 4.196 | 2.28 | 2.28 | I=2.28A Vcv=4.200 |
| RW12 | 2 | CCCV | 21.48 | 1.1913e-04 | -0.5970 | 4.193 | 2.27 | 2.27 | I=2.27A Vcv=4.200 |
| RW12 | 3 | Two-step | 21.73 | 1.1928e-04 | -0.6010 | 4.188 | 2.24 | 2.24 | I1=2.24 I2=0.75 |
| RW12 | 4 | Three-step | 21.82 | 1.1965e-04 | -0.6046 | 4.188 | 2.24 | 2.25 | I1=2.25 I2=2.00 I3=1.75 |
| RW12 | 5 | Two-step | 22.23 | 1.1968e-04 | -0.6093 | 4.178 | 2.20 | 2.20 | I1=2.20 I2=0.75 |
| RW12 | 6 | Two-step | 22.27 | 1.1965e-04 | -0.6096 | 4.177 | 2.20 | 2.20 | I1=2.20 I2=1.95 |

## Parameter / outcome spread across the 24 native elites

| Quantity | Mean | Std | CV % | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| I_mean (A) | 2.322 | 0.5736 | 24.7 | 1.741 | 3.341 |
| I_max (A) | 2.662 | 1.151 | 43.2 | 1.741 | 4.616 |
| Time (min) | 22.4 | 4.64 | 20.7 | 14.98 | 28.6 |
| Q_loss | 0.0001161 | 3.406e-06 | 2.9 | 0.0001098 | 0.0001197 |
| Reward | -0.5701 | 0.1019 | 17.9 | -0.6728 | -0.4005 |
| V_peak (V) | 4.18 | 0.01464 | 0.4 | 4.158 | 4.2 |

A small CV means the elite set is a shared neighbourhood. A large CV on current or time
means the independently optimized elites are not the same protocol.

## Trajectory similarity (time-normalized RMSE of I(t))

Mean pairwise RMSE **within** a cell's own Top-6:

| Cell | Mean within-cell RMSE (A) |
| --- | ---: |
| RW9 | 0.092 |
| RW10 | 0.017 |
| RW11 | 0.131 |
| RW12 | 0.052 |

Mean pairwise RMSE **between different cells**: 0.824 A.

If within-cell RMSE is small and between-cell RMSE is comparable, the four Top-6 clouds
overlap. If between-cell RMSE is much larger, the elite sets occupy different current
trajectories.

## Cross-cell validation (24 profiles × 4 twins)

A profile is counted as pack-feasible if it meets the energy target and stays ≤ 4.20 V
on **every** twin.

Profiles feasible on all four twins: **12 / 24**
— RW9-P1, RW9-P2, RW9-P3, RW9-P4, RW9-P5, RW9-P6, RW10-P1, RW10-P2, RW10-P3, RW10-P4, RW10-P6, RW12-P1

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
