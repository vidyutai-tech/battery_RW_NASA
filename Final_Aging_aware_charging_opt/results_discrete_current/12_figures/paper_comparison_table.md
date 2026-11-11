# Discrete GP-BO, 40% energy in 30 min

GP-BO is the best two-step whose second current is reached before the charge ends.
Time and degradation are compared with the finished CCCV 0.5C charge of the same 40% energy.
That 0.5C charge takes 51–57 min, so it does not itself meet the 30-minute limit.
Positive Time ↓ / Deg. ↓ means the two-step is faster or has lower Q_loss.
Reward is the objective inside the 30-minute search. It is not comparable with the no-limit table.

| Cell | GP-BO family | Time (min) | Time ↓ vs 0.5C | Deg. ↓ vs 0.5C | Time ↓ vs Random | Deg. ↓ vs Random | Reward |
|---|---|---:|---:|---:|---:|---:|---:|
| RW9 | Two-step | 28.45 | +45.0% | +13.0% | -5.2% | -1.0% | 225.4 |
| RW10 | Two-step | 25.48 | +50.4% | +16.1% | -3.1% | -1.5% | 302.5 |
| RW11 | Two-step | 22.58 | +57.9% | +15.9% | -44.3% | -2.5% | 392.7 |
| RW12 | Two-step | 28.83 | +49.6% | +11.6% | -16.7% | -4.4% | 189.7 |

Mean across RW9–RW12: time 50.7% faster, degradation 14.1% lower.
