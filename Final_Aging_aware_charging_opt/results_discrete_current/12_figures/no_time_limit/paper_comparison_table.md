# GP-BO comparison table (calibrated coefficients, paper Eq. 10)

Percent columns use **CCCV 0.5C** as the reference. Positive Time ↓ / Deg. ↓ means GP-BO is faster / lower $Q_{loss}$.

| Cell | GP-BO family | Time (min) | Time ↓ vs 0.5C | Deg. ↓ vs 0.5C | Time ↓ vs Random | Deg. ↓ vs Random | Reward |
|---|---|---|---|---|---|---|---|
| RW9 | CCCV | 28.00 | +45.8% | +13.4% | +2.2% | +0.5% | 371.2794 |
| RW10 | CCCV | 24.72 | +51.9% | +17.4% | +0.0% | +0.0% | 482.0522 |
| RW11 | CCCV | 16.82 | +68.6% | +20.4% | -7.5% | +3.0% | 622.2200 |
| RW12 | CCCV | 24.72 | +56.8% | +15.3% | +0.0% | +0.0% | 410.4469 |

Lifetime curves accumulate the session $Q_{\mathrm{loss}}$ index and anchor CCCV 0.5C to 80% remaining capacity at 600 cycles (paper ranking projection). GP-BO (min Q) is the lowest-Q feasible GP-BO trial (health-first). GP-BO (max R) is the Eq.~10 reward maximiser (the profile the proposed algorithm selects).
