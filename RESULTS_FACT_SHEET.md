# RW9 Physics + Thermal BO — Results Fact Sheet

**Run:** `stage3_physics_thermal_R1`  
**Config:** SoC 15%→95%, T₀=24.7°C, age=0, max 105 min, PI-BO, 40 evals/family, Wang objective, thermal derating+loss @ 33°C

---

## Table 1 — All 8 profile families (corrected slide 11)

| Family | Feasible | Duration (min) | ΔQ/Q₀ (%) | Composite loss | Peak V (V) | Peak T (°C) | Thermal suspect |
|--------|----------|----------------|-----------|----------------|------------|-------------|-----------------|
| Pulsed charge/rest | Yes | 48.5 | 1.025 | 1.56 | 4.19 | 25.5 | ⚠ Yes |
| Adaptive 3-step (SoC) | Yes | 95.0 | 1.043 | 2.51 | 4.20 | 24.7 | No |
| Multi-step taper (voltage) | Yes | 91.3 | 1.071 | 2.58 | 4.20 | 24.7 | ⚠ Yes |
| Reduced-CV CCCV | Yes | 103.0 | 1.022 | 2.60 | 4.15 | 24.6 | No |
| CC-taper (2-level) | Yes | 98.7 | 1.038 | 2.61 | 4.20 | 24.6 | No |
| Adaptive 2-step (SoC) | Yes | 99.1 | 1.039 | 2.62 | 4.20 | 24.6 | No |
| CCCV | Yes | 104.5 | 1.021 | 2.62 | 4.14 | 24.6 | No |
| Exponential taper I(SoC) | No | 118.2 | — | — | 4.20 | 24.6 | No |

**Footnotes**
- **Lowest composite loss:** Pulsed (1.56) — wins mainly on charge time, not lowest ΔQ/Q₀.
- **Lowest ΔQ/Q₀ among family bests:** CCCV (1.021%), not Pulsed (1.025%).
- **Infeasible:** Exponential taper did not reach 95% SoC within 105 min (118.2 min, time budget).
- **Thermal suspect (⚠):** BDT peak T vs lumped thermal model differs by >5°C — pulsed and multi-step taper.

## Table 2 — Thermal cross-check (BDT vs lumped model)

| Family | BDT peak (°C) | Lumped peak (°C) | Δ (°C) | Suspect |
|--------|---------------|------------------|--------|---------|
| Pulsed charge/rest | 25.5 | 37.1 | 11.6 | ⚠ |
| Adaptive 3-step (SoC) | 24.7 | 27.4 | 2.8 |  |
| Multi-step taper (voltage) | 24.7 | 30.1 | 5.5 | ⚠ |
| Reduced-CV CCCV | 24.6 | 26.8 | 2.2 |  |
| CC-taper (2-level) | 24.6 | 27.6 | 3.0 |  |
| Adaptive 2-step (SoC) | 24.6 | 27.7 | 3.1 |  |
| CCCV | 24.6 | 26.8 | 2.1 |  |
| Exponential taper I(SoC) | 24.6 | 26.4 | 1.8 |  |

## Table 3 — Pareto reference profiles (from BO history)

| Tag | Family | Duration (min) | ΔQ/Q₀ (%) | Composite loss |
|-----|--------|----------------|-----------|----------------|
| Fastest | Pulsed charge/rest | 48.5 | 1.025 | 1.56 |
| Lifetime | Pulsed charge/rest | 54.5 | 1.007 | 1.65 |
| Balanced | Pulsed charge/rest | 54.5 | 1.007 | 1.65 |

*Lifetime/Balanced pulsed (54.5 min, 1.007%) ≠ pulsed family-best in Table 1 (48.5 min, 1.025%).*

## Chebyshev sweep (R1) — best degradation point

- ω=0.4, Pulsed charge/rest, 54.5 min, ΔQ/Q₀=1.007%
- Front is **non-monotone** (ω=0.8–0.9 faster but higher fade than ω=0.4–0.6). Consider n_calls≥60.

## Methodology caveats (for slides)

1. **BDT accuracy:** Random-walk V RMSE ≈ 0.028 V; reference-charge (CC-like) RMSE ≈ 0.12 V (~4× worse). BO on smooth profiles is approximate.
2. **Thermal penalty:** All profiles have ∫(T−33°C)² dt = 0 at T₀=24.7°C. Derating inactive; thermal trade-off visible at higher T₀ (ambient T35: peak 34.8°C).
3. **SEI vs Wang:** Spearman ρ = -0.06 (p = 0.81) — rankings disagree.
4. **Aging calibration:** Wang model R² = 0.97, γ = 0.55.
5. **Acquisition:** PI (Probability of Improvement) — confirmed in R1 run.

## Suggested slide 11 replacement text

> At room temperature (24.7°C), **Pulsed** achieves the lowest physics-aware composite loss (1.56) and fastest charge (48.5 min). **CCCV** achieves the lowest single-session ΔQ/Q₀ among family optima (1.021% vs 1.025% for Pulsed). All feasible profiles stay below 33°C; thermal penalties are inactive at this T₀. Exponential taper exceeded the 105 min deadline.