# GP-BO comparison table (calibrated coefficients, paper Eq. 10)

Percent columns use **CCCV 0.5C** as the reference. Positive Time ↓ / Deg. ↓ means GP-BO is faster / lower $Q_{loss}$.

| Cell | GP-BO family | Time (min) | Time ↓ vs 0.5C | Deg. ↓ vs 0.5C | Time ↓ vs Random | Deg. ↓ vs Random | Reward |
|---|---|---|---|---|---|---|---|
| RW9 | CCCV | 27.07 | +42.0% | +12.7% | -0.1% | -0.2% | 339.5329 |
| RW10 | CCCV | 23.87 | +48.5% | +14.3% | +2.5% | +0.1% | 404.9157 |
| RW11 | Two-step | 15.00 | +69.1% | +17.5% | +5.7% | +3.2% | 535.3248 |
| RW12 | CCCV | 21.37 | +56.6% | +9.0% | -0.2% | +0.0% | 347.1644 |

Lifetime curves accumulate the session $Q_{\mathrm{loss}}$ index and anchor CCCV 0.5C to 80% remaining capacity at 600 cycles (paper ranking projection). GP-BO (min Q) is the lowest-Q feasible GP-BO trial (health-first). GP-BO (max R) is the Eq.~10 reward maximiser (the profile the proposed algorithm selects).
