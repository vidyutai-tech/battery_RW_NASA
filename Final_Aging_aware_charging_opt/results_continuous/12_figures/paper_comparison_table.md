# GP-BO comparison table (calibrated coefficients, paper Eq. 10)

Percent columns use **CCCV 0.5C** as the reference. Positive Time ↓ / Deg. ↓ means GP-BO is faster / lower $Q_{loss}$.

| Cell | GP-BO family | Time (min) | Time ↓ vs 0.5C | Deg. ↓ vs 0.5C | Time ↓ vs Random | Deg. ↓ vs Random | Reward |
|---|---|---|---|---|---|---|---|
| RW9 | CCCV | 27.15 | +41.8% | +12.5% | +2.9% | +0.1% | 0.3390 |
| RW10 | CCCV | 24.38 | +47.4% | +14.4% | +0.4% | +0.2% | 0.3985 |
| RW11 | CCCV | 14.98 | +69.1% | +17.5% | -3.2% | +5.2% | 0.5348 |
| RW12 | Two-step | 21.33 | +56.7% | +10.0% | +0.0% | +1.2% | 0.3475 |

Lifetime curves accumulate the session $Q_{\mathrm{loss}}$ index and anchor CCCV 0.5C to 80% remaining capacity at 600 cycles (paper ranking projection). GP-BO (min Q) is the lowest-Q feasible GP-BO trial (health-first). GP-BO (max R) is the Eq.~10 reward maximiser (the profile the proposed algorithm selects).
