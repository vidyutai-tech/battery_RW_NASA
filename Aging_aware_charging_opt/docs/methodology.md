# Methodology

Aging-aware charging-profile optimization for NASA RW9–RW12 lithium-ion cells.

Placeholders written `<<filled by scripts>>` are replaced with values read from
saved result files by `scripts/13_make_tables.py`; they are never typed in by
hand. Until the corresponding stage has been run they remain placeholders.

---

## 1. Notation and constants

| Symbol | Meaning | Unit |
|---|---|---|
| `R` | universal gas constant, 8.314 | J mol⁻¹ K⁻¹ |
| `T_K` | cell temperature | K |
| `SOC` | state of charge | fraction, [0, 1] |
| `t_h` | calendar (wall-clock) time | h |
| `Ah` | charge throughput | A h |
| `C` | C-rate, `\|I\| / Q_nom` | – |
| `Q_nom` | nominal capacity, 2.2 | A h |
| `y` | fractional capacity loss, `1 − Q/Q₀` | fraction |

Sign convention inherited from the NASA dataset: **negative current = charge**,
positive current = discharge.

---

## 2. Degradation model

The functional form is unchanged from the study's stated methodology; only the
coefficients are re-derived. Calendar and cyclic mechanisms are treated as
independent contributions summed at the top level, with no interaction term.

### 2.1 Calendar term

```
Q_cal(SOC, T, t_h) = A_cal · exp(B_cal · SOC)
                          · exp( −(E_a,cal + C_cal · SOC) / (R · T_K) )
                          · t_h^{z_cal}
```

`A_cal` sets the scale, `B_cal` the SOC sensitivity of the pre-exponential term,
`E_a,cal` the baseline activation energy, `C_cal` the SOC-dependence of the
activation energy (a higher-SOC cell has a different effective barrier), and
`z_cal` the sub-linear time exponent characteristic of SEI-growth-limited
calendar aging.

### 2.2 Cyclic term

```
Q_cyc(Ah, T, C) = B_cyc(C) · exp( −E_a,cyc(C) / (R · T_K) ) · Ah^{z_cyc(C)}
```

`B_cyc`, `E_a,cyc` and `z_cyc` are defined on a **C-rate grid of {0.5, 1.0, 2.0}C**
and linearly interpolated in `C`, with clamping outside the grid. The grid is
chosen to span the measured duty cycle: over RW9–RW12 the random-walk currents
occupy `\|I\| ∈ [0.75, 4.8] A`, i.e. `0.34C – 2.18C` on a 2.2 Ah cell. Anchoring
the grid inside the observed range means the model interpolates rather than
extrapolates for every point it is fitted on and every charging profile it is
later asked to score.

### 2.3 Total

```
Q_total = Q_cal + Q_cyc
```

`Q_total` is a **fractional capacity loss** in the same units as the measured
target `y`, because the coefficients are fitted directly against `y`. It is not
a dimensionless ranking index.

### 2.4 Accumulation under time-varying stress

Both terms are sub-linear power laws, so their increments cannot be summed
across intervals with different stress: `k·(Δ₁ + Δ₂)^z ≠ k₁Δ₁^z + k₂Δ₂^z`.
The model is therefore integrated in **equivalent-time (state-shift)** form,
the standard treatment for power-law aging under varying stress.

Given an accumulated loss `Q` and a stress-dependent rate coefficient `k` for
the current interval, the equivalent exposure that would have produced `Q`
under that stress is

```
t_eq = (Q / k)^{1/z}
```

and after advancing by `Δ`,

```
Q ← k · (t_eq + Δ)^z
```

This is applied with
* `k = A_cal·exp(B_cal·SOC)·exp(−(E_a,cal + C_cal·SOC)/(R·T_K))`, `z = z_cal`,
  `Δ = Δt_h` for the calendar state, and
* `k = B_cyc(C_b)·exp(−E_a,cyc(C_b)/(R·T_K))`, `z = z_cyc(C_b)`, `Δ = ΔAh_b`
  for each C-rate bin `b`, each bin carrying its own state.

The prediction at reference discharge *k* is the calendar state plus the sum of
the three bin states. Consequences: the model is path-dependent (order of
stresses matters), reduces exactly to the closed form under constant stress,
and is monotonically non-decreasing.

---

## 3. Calibration data

### 3.1 Measured capacity

The NASA RW protocol interleaves periodic **reference discharge** steps
(nominally 1 A to a 3.2 V cutoff) with the random-walk duty cycle. For each
reference discharge step the removed charge is

```
q_meas = ∫ |I| dt / 3600      [A h]
```

Because a reference discharge does not always start from a full cell or end at
exactly 0 % SOC, `q_meas` is corrected to a full 0–100 % window using the
cell's own low-current OCV–SOC curve:

```
q_full = q_meas / ( soc_ocv(V_start) − soc_ocv(V_end) )
```

applied only when the traversed SOC window exceeds 0.05, and otherwise the
reference is dropped. The OCV–SOC curve is re-fitted in this project from each
cell's `low current discharge at 0.04A` step (5 mV voltage bins, monotone PCHIP
interpolation), which is a near-equilibrium sweep.

The regression target is the fractional capacity loss relative to that cell's
first accepted reference:

```
y_k = 1 − q_full,k / q_full,0
```

### 3.2 Interval stress descriptors

Between consecutive accepted references the following are computed from the raw
samples, using the absolute `time` field so that rests are counted:

| Field | Definition |
|---|---|
| `duration_h` | `(t_end − t_start)/3600`, all steps in the interval |
| `rest_hours` | duration of samples with `\|I\| < 0.01 A` |
| `mean_soc` | mean of the coulomb-counted SOC trace over the interval |
| `mean_temperature_c` | sample mean of measured temperature |
| `dah_b` | Σ `\|I\|·Δt/3600` over samples whose `C = \|I\|/Q_nom` is nearest bin `b` |

SOC is coulomb-counted on the stitched timeline and re-anchored to 100 % at the
end of each `reference charge` step and to 0 % at the end of each accepted
reference discharge, which bounds integration drift over the ~3500 h record.

---

## 4. Parameter estimation

### 4.1 Objective

Weighted nonlinear least squares on the measured fractional loss:

```
minimize_θ  Σ_c Σ_k  w_ck · ( ŷ(θ; interval history up to k) − y_ck )²
```

Weights `w_ck = 1` (unweighted): the target `y` is a capacity fraction measured
by the same procedure at every reference, so the measurement variance is
approximately homoscedastic. No weighting scheme was searched over.

### 4.2 Parameterization and bounds

Positive-scale parameters (`A_cal`, `B_cyc(·)`) are estimated as
`log₁₀`-transformed variables so the optimizer explores them multiplicatively
and cannot leave the positive domain. Bounds:

| Parameter | Lower | Upper | Basis |
|---|---|---|---|
| `log₁₀ A_cal` | −12 | 6 | wide, non-informative |
| `B_cal` | −10 | 10 | allows either sign of SOC sensitivity |
| `C_cal` | −2×10⁴ | 2×10⁴ | J mol⁻¹ per unit SOC |
| `E_a,cal` | 1×10⁴ | 1×10⁵ | J mol⁻¹; spans reported Li-ion calendar-aging barriers |
| `z_cal` | 0.20 | 1.00 | sub-linear to linear |
| `log₁₀ B_cyc(C_b)` | −12 | 12 | wide, non-informative |
| `E_a,cyc(C_b)` | 1×10⁴ | 1×10⁵ | J mol⁻¹ |
| `z_cyc(C_b)` | 0.20 | 1.20 | sub-linear through mildly super-linear |

### 4.3 Algorithm

`scipy.optimize.least_squares`, method `trf` (trust-region reflective, handles
box bounds), analytic-free 2-point Jacobian, `ftol = xtol = gtol = 1e-12`,
`max_nfev = 20000`, `x_scale='jac'`.

Initialization: one literature-anchored start (`z ≈ 0.55`, `E_a ≈ 3×10⁴` J mol⁻¹,
scale coefficients set so the predicted end-of-life loss is order 0.5) plus
`n_starts − 1` Latin-hypercube starts drawn inside the bounds with a fixed seed.
The start with the lowest final cost is retained; the spread of final costs
across starts is reported as evidence for or against a unique optimum.

### 4.4 Uncertainty and identifiability

From the final Jacobian `J` and residuals `r` with `n` observations and `p`
parameters:

```
s² = rᵀr / (n − p)
Cov(θ̂) = s² (JᵀJ)⁻¹      (Moore–Penrose pseudo-inverse if ill-conditioned)
SE_i = sqrt(Cov_ii)
Corr_ij = Cov_ij / (SE_i · SE_j)
```

Reported alongside the fit: standard errors, the full correlation matrix, the
condition number of `JᵀJ`, and an explicit flag list containing every parameter
that is (a) within 1 % of a bound, (b) correlated `|ρ| > 0.99` with another
parameter, or (c) has `SE_i / |θ̂_i| > 1`. Arrhenius prefactor/activation-energy
pairs are expected to appear here: over the ~19–40 °C span present in these
cells, `A` and `E_a` are near-collinear, so their individual values are weakly
determined even where their *combination* is well determined. This is a
limitation of the data, reported rather than hidden, and it is the reason
predictive metrics (Section 5) rather than individual coefficient values are
the basis for accepting the model.

A reduced model with a single shared `E_a,cyc` across the three C-rate bins is
fitted as a sensitivity check to show how much of the fit quality depends on
the extra freedom.

---

## 5. Validation

Validation is **out-of-sample by cell** and never used to tune anything.

**Leave-one-cell-out (LOCO)**: for each `c ∈ {RW9, RW10, RW11, RW12}`, fit on
the other three cells and predict cell `c`. The held-out cell's data enters no
residual, no starting point and no stopping rule of that fold's fit.

Reported per fold and pooled over folds:

| Metric | Definition |
|---|---|
| R² | `1 − Σ(ŷ−y)² / Σ(y−ȳ)²` on held-out data |
| RMSE | `sqrt(mean((ŷ−y)²))`, in capacity fraction and %-capacity |
| MAE | `mean(\|ŷ−y\|)` |
| Bias | `mean(ŷ−y)`, signed |
| Residual structure | residual vs fitted, residual vs cumulative Ah, residual vs mean T |

**Acceptance gate**: pooled LOCO `R² ≥ 0.90` and `|bias| ≤ 0.02` (2 %-capacity).
Failing the gate stops the pipeline for diagnosis; it does not trigger a search
for a different validation split.

The model handed to the optimizer is the **pooled all-cells fit**. LOCO is the
evidence that this functional form and this fitting procedure generalize to a
cell they have not seen; the pooled fit uses all available information for the
coefficients actually deployed. Both parameter sets are saved.

---

## 6. Charging-profile representation

Every profile is a parametric current policy evaluated in closed loop against
the BDT at a fixed decision interval.

| Family | Parameters | Behaviour |
|---|---|---|
| CCCV | `i_cc`, `v_cv`, `i_cutoff` | Constant current `i_cc` until the terminal voltage reaches `v_cv`; then constant voltage, realized as a monotone current taper that holds `V ≈ v_cv`, until the current reaches `i_cutoff`. |
| 2-step | `i₁`, `i₂`, `soc_switch` | `i₁` below `soc_switch`, `i₂` above, with the hard constraint `i₂ ≤ i₁ − 0.25 A` and a minimum stage-1 SOC span. |
| 3-step | `i₁`, `i₂`, `i₃`, `soc₁`, `soc₂` | Descending staircase, `i₃ ≤ i₂ − 0.25 ≤ i₁ − 0.50 A`, `soc₂ ≥ soc₁ + 0.12`. |
| Pulsed | `i_charge`, `pulse_on_min`, `rest_fraction`, `i_floor` | Charge at `i_charge` for `pulse_on_min` minutes, rest for `rest_fraction × pulse_on_min` minutes, repeat; current tapers toward `i_floor` if the voltage ceiling is reached. |

Common hard constraints for all families: `\|I\| ∈ [0.75, 6.0] A`,
terminal voltage `≤ 4.2 V` (enforced by truncating the step and tapering, never
by penalty alone), and a 150 min time budget. Ordering constraints are enforced
inside each family's parameter constructor, so the random sampler and the GP
proposal map onto exactly the same feasible set.

Charging profiles are generated per family as **80 evaluations per family per
cell**, for both optimizers.

---

## 7. Session evaluation

A session starts from an OCV-consistent rest state at 20 % SOC at 24 °C and is
**energy-constrained**: it terminates when the delivered energy

```
E_delivered = ∫ (−V·I) dt
```

reaches `E_required = 0.40 × Q_nom,As × V_nom`, i.e. 40 % of the cell's nominal
energy — the same delivered-energy target for every profile and every cell, so
that time and degradation are compared at equal useful work. Secondary
terminations: the 150 min budget, SOC = 100 %, or a family-specific end
condition (CV cutoff current reached, pulsed floor at the voltage ceiling).

From the resulting `(t, I, V, T, SOC)` trajectory:

* `Ah` throughput split into the same three C-rate bins used in calibration,
* mean SOC and mean temperature,
* `Q_cal`, `Q_cyc` and `Q_total` from the calibrated model with the
  equivalent-time integrator applied along the trajectory,
* charging duration, peak voltage, peak and mean temperature,
* feasibility = delivered energy met the target without exceeding the
  voltage ceiling.

---

## 8. Reward

The published objective is restored verbatim. Only the numerical values inside
`Q_loss` change: they now come from the NASA-calibrated hybrid coefficients
rather than the literature Table-7 / calendar-stub index used in the draft
tables.

```
R = w_soc · ΔSOC − w_loss · Q_loss − w_time · t_h^{z_time}

w_soc = 1,    w_loss = 2,    w_time = 0.1,    z_time = 0.55
```

`Q_loss = Q_calendar + Q_cyclic` is the paper session-mean closed form
(mean SOC, mean T, elapsed hours, charge Ah, mean charge C-rate), evaluated
with `configs/degradation_fitted.yaml`. Energy remains a hard constraint
(`E_delivered ≥ E_required`, `f = 0.40`); ΔSOC is the continuous progress
term, not the feasibility test.

Because a calibrated session fade is ~10⁻⁴ capacity fraction, `w_loss Q_loss`
is ~2×10⁻⁴ — much smaller than the draft-table index (~0.07). Reward values
are therefore ~0.2–0.4 rather than ~0.14–0.19. The ranking still uses the
same equation. The draft numbers are not retargeted.

### 8.3 Loss and constraints

```
L = −R + ε t_min + 300 · g_E + 100 · g_V
ε = 10^{-3}     (tie-break only; not a second physical time weight)
g_E = max(0, 1 − E_delivered/E_required)
g_V = max(0, V_peak − 4.2)
```

No `qloss_cap`, no Random-search warm start. Both optimizers minimize this
identical function.

---

## 9. Optimization

| Setting | Value |
|---|---|
| Families | CCCV, 2-step, 3-step, pulsed |
| Evaluations per family per cell | 80 (identical for both methods) |
| Random Search | i.i.d. uniform over the family's box, projected through the family's constraint constructor |
| GP-BO | `skopt.gp_minimize`, Matérn GP, Expected Improvement (`xi = 0.01`), observation noise `1e-6` |
| GP-BO initial design | family seed points (deterministic, physically motivated) + random points to 15 total |
| Seed | 20260904, offset deterministically per cell and family |
| Cross-method information | none (no elite warm-start, no shared caps) |

Recorded for every run: seed, every evaluated parameter vector, its reward and
all raw metrics, feasibility, best-so-far trace, and wall-clock cost.

---

## 10. Baselines

Defined independently of the optimizers, as conventional CCCV protocols on the
same cell, same start state, same energy target:

| Baseline | `i_cc` | `v_cv` | `i_cutoff` |
|---|---|---|---|
| CCCV 0.5C | 1.10 A | 4.20 V | 0.05 A |
| CCCV 1C | 2.20 A | 4.20 V | 0.05 A |
| CCCV 2C | 4.40 A | 4.20 V | 0.05 A |

CCCV 1C additionally supplies the reward normalization anchors (Section 8.1).
For each baseline the same session metrics and reward are computed and reported.

---

## 11. Equivalent-cycle analysis — scope and limits

The calibrated model is applied forward over repeated identical equal-energy
charging sessions to obtain a **relative capacity-retention projection**. For a
policy with per-session `(ΔAh_b, T̄, SOC̄, Δt_h)`, the equivalent-time integrator
is advanced `N` times and the retention reported as `100·(1 − Q_total(N))`.

What this supports:
* Ranking of charging policies by projected retention at a matched number of
  equal-energy sessions.
* Statements of the form "policy X is projected to retain `d` percentage points
  more capacity than CCCV 1C after N equal-energy charge sessions, under the
  calibrated model".

What this does **not** support:
* A calendar-life or cycle-life figure for these cells under these protocols.
  The coefficients were fitted to a random-walk duty cycle, not to any of the
  optimized protocols; applying them to a different duty cycle is an
  extrapolation in the load pattern even though it interpolates in C-rate.
* Any claim about behaviour beyond the observed fade range.
* Any claim resting on the 80 % retention line. Where an 80 % line is drawn it
  is an **engineering reference level**, not an experimentally demonstrated
  end-of-life for these cells under these protocols, and is labelled as such on
  every figure.

Terminology used throughout the results: *relative degradation index*,
*normalized degradation*, *equivalent-cycle degradation comparison*, *relative
capacity-retention projection*. The phrase "capacity fade (%)" is used only for
the **measured** reference-discharge data of Section 3.

---

## 12. Reproducibility

Every stage writes a JSON with the git commit, the resolved configuration hash,
the random seed, package versions and the input file paths it read. Stages are
pure functions of their configuration plus upstream artifacts, executed in the
fixed order given in `docs/experiment_protocol.md`.
`scripts/validate_experiment.py` re-reads the saved artifacts and re-checks that
the reported tables and figures are consistent with them.
