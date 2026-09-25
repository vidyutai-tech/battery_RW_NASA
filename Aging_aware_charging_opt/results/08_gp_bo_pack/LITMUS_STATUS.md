# Litmus status — four cells, same operating box

**Frozen reference. Do not treat this as a new optimization.**

This note records the **independent per-cell GP-BO Top-6** (stage 8) under
identical operating conditions. Pack GP-BO outputs already in this folder
(`elite_top6.md`, `summary.json`) are a **different experiment** (one shared
current, feasible on every twin). Both are kept here so they can be compared.

Source of the litmus numbers: `litmus_reference/` (copied from
`results/14_litmus_top6/`).

---

## Same operating conditions (the only thing that does *not* change)

| Setting | Value |
|---|---|
| Start SOC | 0.20 |
| Start T | 24 °C |
| Age | 0 |
| Energy target | 40 % of nominal pack energy |
| Voltage ceiling | 4.20 V |
| Current box | 0.75 – 6.00 A |
| Families | CCCV, two-step, three-step, pulsed |
| Budget | 80 GP-BO evals / family / cell |
| Reward | \(R = \Delta\mathrm{SOC} - 2\,Q_{\mathrm{loss}} - 0.1\, t_h^{0.55}\) |
| Decision interval | 30 s |

What **does** change: the **cell digital twin** (RW9 source; RW10/11/12 fine-tunes).

---

## Litmus result in one sentence

Under that shared box, the four Top-6 lists are **not** the same strategy with
a shuffled order. RW10 ≈ RW12 (~2 A CC). RW9 mixes pulse + high-current CCCV.
RW11 is 6 A CCCV that **fails 4.20 V** on the other twins.

Only **4 of 24** independently optimized elites stay legal on all four cells:
~2.0–2.3 A CCCV.

---

## Level 1 — family counts in each Top-6

| Family | RW9 | RW10 | RW11 | RW12 |
|---|--:|--:|--:|--:|
| CCCV | 3 | 1 | 6 | 3 |
| Two-step | 0 | 3 | 0 | 0 |
| Three-step | 0 | 2 | 0 | 3 |
| Pulsed | 3 | 0 | 0 | 0 |

CCCV is the only family in every list. Pulsed is RW9-only. RW11 is CCCV-only.

---

## Level 2 — Top-6 optimized profiles (native twin)

Rank = lowest optimizer **loss** among feasible trials on that cell.

### RW9

| Rank | Family | t (min) | Q_loss | Reward | V_peak | I_max (A) | Parameters |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Pulsed | 21.60 | 1.324e-04 | 0.3131 | 4.192 | 3.35 | Ion=3.35 A, on=1.0 min, rest=0.5 min, floor=3.30 A |
| 2 | Pulsed | 22.00 | 1.311e-04 | 0.3128 | 4.185 | 3.27 | Ion=3.27 A, on=1.0, rest=0.5, floor=0.75 A |
| 3 | Pulsed | 23.10 | 1.314e-04 | 0.3118 | 4.179 | 3.14 | Ion=3.14 A, on=1.0, rest=0.5, floor=0.75 A |
| 4 | CCCV | 28.58 | 1.275e-04 | 0.3128 | 4.199 | 4.75 | I=4.75 A, Vcv=4.051 V, icut=0.43 A |
| 5 | CCCV | 28.37 | 1.271e-04 | 0.3118 | 4.197 | 4.72 | I=4.72 A, Vcv=4.065 V, icut=0.49 A |
| 6 | CCCV | 27.98 | 1.274e-04 | 0.3111 | 4.194 | 4.62 | I=4.62 A, Vcv=4.080 V, icut=0.02 A |

### RW10

| Rank | Family | t (min) | Q_loss | Reward | V_peak | I_max (A) | Parameters |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Three-step | 23.95 | 1.169e-04 | 0.3122 | 4.200 | 2.07 | I1=2.07, I2=1.82, I3=0.75 A @ SOC 0.55 / 0.67 |
| 2 | Three-step | 24.15 | 1.183e-04 | 0.3121 | 4.173 | 2.22 | I1=2.22, I2=1.97, I3=1.72 A |
| 3 | CCCV | 24.72 | 1.167e-04 | 0.3126 | 4.176 | 2.00 | **I=2.00 A, Vcv=4.200 V** (pack-feasible) |
| 4 | Two-step | 24.72 | 1.167e-04 | 0.3126 | 4.176 | 2.00 | I1=2.00 → I2=1.00 A @ SOC 0.61 |
| 5 | Two-step | 24.93 | 1.168e-04 | 0.3128 | 4.171 | 1.99 | I1=1.99 → I2=0.75 A @ SOC 0.90 |
| 6 | Two-step | 24.80 | 1.167e-04 | 0.3127 | 4.175 | 1.99 | I1=1.99 → I2=1.74 A @ SOC 0.90 |

### RW11

| Rank | Family | t (min) | Q_loss | Reward | V_peak | I_max (A) | Parameters |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | CCCV | 12.95 | 1.177e-04 | 0.3355 | 4.166 | 6.00 | I=6.00 A, Vcv=4.064 V |
| 2 | CCCV | 12.93 | 1.176e-04 | 0.3352 | 4.166 | 6.00 | I=6.00 A, Vcv=4.074 V |
| 3 | CCCV | 13.70 | 1.159e-04 | 0.3354 | 4.166 | 5.97 | I=5.97 A, Vcv=4.054 V |
| 4 | CCCV | 12.90 | 1.175e-04 | 0.3345 | 4.166 | 6.00 | I=6.00 A, Vcv=4.085 V |
| 5 | CCCV | 12.70 | 1.178e-04 | 0.3341 | 4.166 | 5.92 | I=5.92 A, Vcv=4.093 V |
| 6 | CCCV | 12.33 | 1.192e-04 | 0.3337 | 4.167 | 6.00 | I=6.00 A, Vcv=4.106 V |

All six fail 4.20 V on RW9, RW10, and RW12.

### RW12

| Rank | Family | t (min) | Q_loss | Reward | V_peak | I_max (A) | Parameters |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | CCCV | 21.53 | 1.192e-04 | 0.3117 | 4.192 | 2.26 | **I=2.26 A, Vcv=4.200 V** (pack-feasible) |
| 2 | Three-step | 21.68 | 1.197e-04 | 0.3116 | 4.192 | 2.26 | I1=2.26, I2=2.01, I3=0.75 A |
| 3 | Three-step | 21.87 | 1.197e-04 | 0.3117 | 4.187 | 2.24 | I1=2.24, I2=1.99, I3=1.74 A |
| 4 | CCCV | 22.38 | 1.196e-04 | 0.3122 | 4.175 | 2.19 | **I=2.19 A, Vcv=4.200 V** (pack-feasible) |
| 5 | CCCV | 22.38 | 1.196e-04 | 0.3122 | 4.175 | 2.19 | **I=2.19 A, Vcv=4.200 V** (pack-feasible) |
| 6 | Three-step | 21.95 | 1.198e-04 | 0.3118 | 4.186 | 2.23 | I1=2.23, I2=1.98, I3=0.75 A |

Exact numeric rows: `litmus_reference/elite_master.csv`.

---

## Cross-cell transfer (24 profiles × 4 twins)

A profile is pack-feasible only if energy target is met **and** V ≤ 4.20 V on
**every** twin.

**Pack-feasible independently found elites: 4 / 24**

| Profile | Waveform | Feasible on RW9 | RW10 | RW11 | RW12 |
|---|---|---|---|---|---|
| RW10-P3 | 2.00 A CCCV | yes | yes | yes | yes |
| RW12-P1 | 2.26 A CCCV | yes | yes | yes | yes |
| RW12-P4 | 2.19 A CCCV | yes | yes | yes | yes |
| RW12-P5 | 2.19 A CCCV | yes | yes | yes | yes |
| All RW11 P1–P6 | ~6 A CCCV | **no** | **no** | yes | **no** |
| RW9 P1–P3 | pulsed ~3.3 A | yes | yes | yes | **no** |
| RW9 P4–P6 | CCCV ~4.7 A | yes | **no** | yes | **no** |

Heatmap: `litmus_reference/figures/cross_feasible.png`.

---

## Why they differ (same chemistry, same box)

The twins **predict different voltage headroom** at CCCV 1C (from stage-8 baselines).
This is predicted restrictiveness, **not** a measured-impedance claim.

| Cell | CCCV 1C V_peak | Headroom to 4.20 V | Twin V RMSE |
|---|---:|---:|---:|
| RW9 | 4.200 V | ~0 mV | 28 mV |
| RW10 | 4.201 V | ~0 mV | 32 mV |
| RW11 | 4.000 V | ~200 mV | 48 mV |
| RW12 | 4.178 V | ~22 mV | 64 mV |

GP-BO rides that predicted 4.20 V surface. On RW11 the surface looks slack, so
6 A wins. On RW9/RW10 it is already binding at 1C, so ~2 A or pulsing wins.
RW11’s 200 mV predicted headroom is **not** large compared with 48 mV twin
voltage RMSE, so 6 A is “optimal for that twin,” not a demonstrated physical
margin.

What **is** consistent: every elite sits near 4.20 V (V_peak CV 0.3 %) and
Q_loss CV is only 4.4 %. The **strategy** is “ride Vmax.” The **current used
to do that** is not shared (I_max CV 47 %).

---

## What needs to be changed (if you want one Top-6 for a pack)

Do **not** merge the four twins or force seeds. That would fake similarity.

| Goal | Change | Status in this folder |
|---|---|---|
| Measure whether per-cell Top-6 match | Keep four twins + four GP-BOs + this litmus | **Done** (`litmus_reference/`) |
| One protocol that is legal on every cell | One GP-BO, same \(I(t)\), feasible iff **all** twins stay ≤ 4.20 V; minimise worst-cell loss | **Already run** — see `elite_top6.md` |
| One chemistry-level twin | Freeze RW9 `twin_source_RW9.pt` for all cells (zero-shot). Do not use `simulator_cpu.pth` on RW11/12 | Separate experiment (`results/15_zero_shot_rw9_twins/`) |

### Pack GP-BO Top-6 already in this folder (shared current)

These are **one waveform each**, scored on all four twins. Worst cell is always RW9.
Times ~28–31 min, currents ~1.7–1.8 A — slower than RW11’s personal 6 A, legal
everywhere.

| Rank | Family | Parameters | Worst cell |
|---:|---|---|---|
| 1 | Three-step | I1=1.80, I2=1.55, I3=0.75 A, SOC 0.55 / 0.90 | RW9 |
| 2 | Three-step | I1=1.77, I2=1.52, I3=0.75 A, SOC 0.55 / 0.67 | RW9 |
| 3 | Three-step | I1=1.74, I2=1.49, I3=1.24 A, SOC 0.55 / 0.90 | RW9 |
| 4 | CCCV | I=1.72 A, Vcv=4.20 V, icut=0.50 A | RW9 |
| 5 | Three-step | I1=1.80, I2=1.00, I3=0.75 A, SOC 0.55 / 0.67 | RW9 |
| 6 | Three-step | I1=1.70, I2=1.45, I3=1.20 A, SOC 0.55 / 0.67 | RW9 |

Full table: `elite_top6.md`. Plot: `pack_family_best_profiles.png`.

---

## Honest paper sentences

**Litmus (this file):** Independently optimized Top-6 sets under identical
operating conditions are **not interchangeable**. RW10 and RW12 agree on ~2 A
CC-like charging. RW9’s elite set includes pulsing. RW11’s 6 A CCCV elites
violate 4.20 V on the other twins. The only independently found profiles that
remain feasible on every twin are ~2.0–2.3 A CCCV.

**Pack (already computed here):** A shared ~1.7–1.8 A three-step / CCCV
protocol is pack-feasible; RW9 is the binding (worst) cell in the minimax
search.

Do not write that all four cells have the same Top-6 with only ranking shuffle.

---

## Files in this folder

| Path | What it is |
|---|---|
| `LITMUS_STATUS.md` | This note |
| `litmus_reference/elite_master.csv` | 24 per-cell Top-6 rows + parameters |
| `litmus_reference/cross_cell.csv` | 24 × 4 replay |
| `litmus_reference/figures/` | Family bars, I(t) overlay, 24×24 RMSE, feasibility heatmap |
| `elite_top6.md` / `elite_top6.json` | Pack GP-BO shared Top-6 |
| `cccv.json` … `pulsed.json` | Pack family search traces (do not overwrite) |
