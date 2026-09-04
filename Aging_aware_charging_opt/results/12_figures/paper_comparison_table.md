# GP-BO comparison table (calibrated coefficients, paper Eq. 10)

Percent columns use **CCCV 0.5C** as the reference. Positive Time ↓ / Deg. ↓ means GP-BO is faster / lower $Q_{loss}$.

| Cell | GP-BO family | Time (min) | Time ↓ vs 0.5C | Deg. ↓ vs 0.5C | Time ↓ vs Random | Deg. ↓ vs Random | Reward |
|---|---|---|---|---|---|---|---|
| RW9 | Three-step | 21.60 | +53.7% | +12.2% | +23.9% | -0.0% | 0.3131 |
| RW10 | CCCV | 23.95 | +48.3% | +14.2% | -2.6% | +3.1% | 0.3122 |
| RW11 | CCCV | 12.95 | +73.3% | +17.6% | +2.3% | +8.7% | 0.3355 |
| RW12 | Three-step | 21.53 | +56.2% | +9.8% | +2.1% | +1.3% | 0.3117 |

Lifetime curves accumulate the session $Q_{\mathrm{loss}}$ index and anchor CCCV 0.5C to 80% remaining capacity at 600 cycles (paper ranking projection). The GP-BO lifetime line is the lowest-Q feasible GP-BO trial (health-first). GP-BO (max R) is the Eq.~10 reward maximiser (the profile the proposed algorithm selects).
