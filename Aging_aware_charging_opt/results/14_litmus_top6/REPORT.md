# Top-6 cross-cell litmus test

Frozen per-cell digital twins. Frozen stage-8 GP-BO (80 evals / family / cell).
Identical operating box for every cell: SOC₀ = 20 %, T₀ = 24 °C, energy target = 40 % of
nominal pack energy, V_max = 4.20 V, I ∈ [0.75, 6.00] A, same reward, same GP-BO budget.
The only factor that changes is the cell twin.

This analysis does **not** force the four elite sets to match. It measures whether they do.

## Level 1 — charging family in each Top-6

| Family | RW9 | RW10 | RW11 | RW12 |
| --- | ---: | ---: | ---: | ---: |
| CCCV | 3 | 1 | 6 | 3 |
| Two-step | 0 | 3 | 0 | 0 |
| Three-step | 0 | 2 | 0 | 3 |
| Pulsed | 3 | 0 | 0 | 0 |

CCCV is the only family that appears in every cell's Top-6. Pulsed appears only for RW9.
RW11's entire Top-6 is CCCV.

## Level 2 — 24 elite profiles (native-twin replay)

| Cell | Rank | Family | t (min) | Q_loss | Reward | V_peak | I_mean (A) | I_max (A) | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RW9 | 1 | Pulsed | 21.60 | 1.3241e-04 | 0.3131 | 4.192 | 2.27 | 3.35 | Ion=3.35 on=1.0 rest=0.5 |
| RW9 | 2 | Pulsed | 22.00 | 1.3106e-04 | 0.3128 | 4.185 | 2.23 | 3.27 | Ion=3.27 on=1.0 rest=0.5 |
| RW9 | 3 | Pulsed | 23.10 | 1.3138e-04 | 0.3118 | 4.179 | 2.12 | 3.14 | Ion=3.14 on=1.0 rest=0.5 |
| RW9 | 4 | CCCV | 28.58 | 1.2747e-04 | 0.3128 | 4.199 | 1.76 | 4.75 | I=4.75A Vcv=4.051 |
| RW9 | 5 | CCCV | 28.37 | 1.2705e-04 | 0.3118 | 4.197 | 1.76 | 4.72 | I=4.72A Vcv=4.065 |
| RW9 | 6 | CCCV | 27.98 | 1.2739e-04 | 0.3111 | 4.194 | 1.78 | 4.62 | I=4.62A Vcv=4.080 |
| RW10 | 1 | Three-step | 23.95 | 1.1688e-04 | 0.3122 | 4.200 | 2.06 | 2.07 | I1=2.07 I2=1.82 I3=0.75 |
| RW10 | 2 | Three-step | 24.15 | 1.1829e-04 | 0.3121 | 4.173 | 2.04 | 2.22 | I1=2.22 I2=1.97 I3=1.72 |
| RW10 | 3 | CCCV | 24.72 | 1.1670e-04 | 0.3126 | 4.176 | 2.00 | 2.00 | I=2.00A Vcv=4.200 |
| RW10 | 4 | Two-step | 24.72 | 1.1670e-04 | 0.3126 | 4.176 | 2.00 | 2.00 | I1=2.00 I2=1.00 |
| RW10 | 5 | Two-step | 24.93 | 1.1681e-04 | 0.3128 | 4.171 | 1.99 | 1.99 | I1=1.99 I2=0.75 |
| RW10 | 6 | Two-step | 24.80 | 1.1674e-04 | 0.3127 | 4.175 | 1.99 | 1.99 | I1=1.99 I2=1.74 |
| RW11 | 1 | CCCV | 12.95 | 1.1772e-04 | 0.3355 | 4.166 | 3.87 | 6.00 | I=6.00A Vcv=4.064 |
| RW11 | 2 | CCCV | 12.93 | 1.1761e-04 | 0.3352 | 4.166 | 3.87 | 6.00 | I=6.00A Vcv=4.074 |
| RW11 | 3 | CCCV | 13.70 | 1.1593e-04 | 0.3354 | 4.166 | 3.67 | 5.97 | I=5.97A Vcv=4.054 |
| RW11 | 4 | CCCV | 12.90 | 1.1750e-04 | 0.3345 | 4.166 | 3.87 | 6.00 | I=6.00A Vcv=4.085 |
| RW11 | 5 | CCCV | 12.70 | 1.1782e-04 | 0.3341 | 4.166 | 3.93 | 5.92 | I=5.92A Vcv=4.093 |
| RW11 | 6 | CCCV | 12.33 | 1.1916e-04 | 0.3337 | 4.167 | 4.03 | 6.00 | I=6.00A Vcv=4.106 |
| RW12 | 1 | CCCV | 21.53 | 1.1916e-04 | 0.3117 | 4.192 | 2.26 | 2.26 | I=2.26A Vcv=4.200 |
| RW12 | 2 | Three-step | 21.68 | 1.1966e-04 | 0.3116 | 4.192 | 2.25 | 2.26 | I1=2.26 I2=2.01 I3=0.75 |
| RW12 | 3 | Three-step | 21.87 | 1.1969e-04 | 0.3117 | 4.187 | 2.23 | 2.24 | I1=2.24 I2=1.99 I3=1.74 |
| RW12 | 4 | CCCV | 22.38 | 1.1965e-04 | 0.3122 | 4.175 | 2.19 | 2.19 | I=2.19A Vcv=4.200 |
| RW12 | 5 | CCCV | 22.38 | 1.1965e-04 | 0.3122 | 4.175 | 2.19 | 2.19 | I=2.19A Vcv=4.200 |
| RW12 | 6 | Three-step | 21.95 | 1.1977e-04 | 0.3118 | 4.186 | 2.22 | 2.23 | I1=2.23 I2=1.98 I3=0.75 |

## Parameter / outcome spread across the 24 native elites

| Quantity | Mean | Std | CV % | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| I_mean (A) | 2.524 | 0.8113 | 32.1 | 1.756 | 4.03 |
| I_max (A) | 3.558 | 1.666 | 46.8 | 1.985 | 6 |
| Time (min) | 21.18 | 5.278 | 24.9 | 12.33 | 28.58 |
| Q_loss | 0.0001209 | 5.28e-06 | 4.4 | 0.0001159 | 0.0001324 |
| Reward | 0.3178 | 0.009984 | 3.1 | 0.3111 | 0.3355 |
| V_peak (V) | 4.18 | 0.01169 | 0.3 | 4.166 | 4.2 |

A small CV means the elite set is a shared neighbourhood. A large CV on current or time
means the independently optimized elites are not the same protocol.

## Trajectory similarity (time-normalized RMSE of I(t))

Mean pairwise RMSE **within** a cell's own Top-6:

| Cell | Mean within-cell RMSE (A) |
| --- | ---: |
| RW9 | 1.508 |
| RW10 | 0.066 |
| RW11 | 0.176 |
| RW12 | 0.058 |

Mean pairwise RMSE **between different cells**: 1.534 A.

If within-cell RMSE is small and between-cell RMSE is comparable, the four Top-6 clouds
overlap. If between-cell RMSE is much larger, the elite sets occupy different current
trajectories.

## Cross-cell validation (24 profiles × 4 twins)

A profile is counted as pack-feasible if it meets the energy target and stays ≤ 4.20 V
on **every** twin.

Profiles feasible on all four twins: **4 / 24**
— RW10-P3, RW12-P1, RW12-P4, RW12-P5

## Voltage behaviour of the twins (from frozen stage-8 baselines + stage-2 RMSE)

These numbers describe **predicted** restrictiveness, not a claim about physical impedance.

| Cell | CCCV 1C V_peak | Headroom to 4.20 V (mV) | CCCV 2C feasible? | Twin V RMSE (mV) |
| --- | ---: | ---: | ---: | ---: |
| RW9 | 4.200 | -0 | False | 28.3 |
| RW10 | 4.201 | -1 | False | 32.3 |
| RW11 | 4.000 | 200 | False | 48.3 |
| RW12 | 4.178 | 22 | False | 63.9 |

Headroom of tens of millivolts is comparable to twin voltage RMSE on RW11/RW12
(48 mV / 64 mV). A predicted V = 4.05 V is therefore **not** 150 mV of demonstrated
safety margin.

## Verdict (do not force “same Top-6”)

The litmus test **does not support** the claim that RW9–RW12 share one Top-6 list
with only ranking shuffle.

What *is* consistent:

- Every elite profile rides the 4.20 V neighbourhood (V_peak CV = 0.3 %).
- Q_loss across the 24 native elites is tight (CV = 4.4 %).
- RW10 and RW12 occupy the same neighbourhood: ~2.0–2.3 A CC / multi-step, ~22–25 min
  (within-cell I(t) RMSE 0.06 A).

What is **not** a ranking shuffle:

- RW9 Top-6 mixes 3.1–3.4 A pulsed and 4.6–4.8 A CCCV tapers (within-cell RMSE 1.51 A).
- RW11 Top-6 is 5.9–6.0 A CCCV finishing in ~13 min.
- Peak current across the 24 elites spans 2.0–6.0 A (CV = 47 %).

Cross-cell transfer:

- **4 / 24** profiles are feasible on all four twins:
  `RW10-P3` (2.00 A CCCV) and `RW12-P1`, `P4`, `P5` (~2.19–2.26 A CCCV).
- All six RW11 elites fail on RW9, RW10, and RW12 (V > 4.20 V).
- RW9 pulsed elites fail on RW12; RW9 high-current CCCV fails on RW10 and RW12.

Honest paper sentence:

> Independently optimized elite sets are **not interchangeable**. RW10 and RW12 agree
> on a ~2 A CC-like protocol; RW9’s elites include pulsing; RW11’s elites are 6 A CCCV
> that violate 4.20 V on the other twins. The only profiles that remain feasible on
> every twin are ~2.0–2.3 A CCCV. Cell-to-cell differences in predicted voltage
> headroom (0 mV on RW9/RW10 vs 200 mV on RW11 at 1C) are comparable to twin voltage
> RMSE on RW11/RW12 (48–64 mV), so the 6 A RW11 result should not be treated as a
> demonstrated physical headroom.

The figures `family_counts.png`, `current_overlay.png`, `similarity_24x24.png`, and
`cross_feasible.png` are the visual version of this test.
