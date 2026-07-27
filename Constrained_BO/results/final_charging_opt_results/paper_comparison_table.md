# GP-BO comparison table

Baselines are classic **CCCV (CC→CV at Vmax)** at ½C / 1C / 2C (all shown).
Percent columns use the **best feasible CCCV** per cell (lowest session Q, then shortest time).

| Cell | Energy | GP-BO time (min) | Time ↓ vs best CCCV | Deg. ↓ vs best CCCV | Time ↓ vs Random | Deg. ↓ vs Random | Best CCCV baseline |
|------|--------|------------------|---------------------|---------------------|------------------|------------------|--------------------|
| RW9 | 40% | 34.2 | -78.1% | 4.9% | -6.0% | 1.2% | CCCV 2C |
| RW10 | 40% | 24.8 | 46.3% | 3.2% | -0.7% | 3.7% | CCCV ½C |
| RW11 | 40% | 11.6 | 12.4% | 3.9% | 3.3% | 1.3% | CCCV 2C |
| RW12 | 40% | 29.8 | 36.5% | -4.6% | 0.0% | 0.0% | CCCV ½C |

**Reading guide**

- **Energy**: delivered fraction of full pack (same 40% target on all cells).
- **Time ↓**: \((t_{\mathrm{base}}-t_{\mathrm{GPBO}})/t_{\mathrm{base}}\). Positive = faster than baseline.
- **Deg. ↓**: \((Q_{\mathrm{base}}-Q_{\mathrm{GPBO}})/Q_{\mathrm{base}}\). Positive = *less* session degradation than baseline.
- Hitting 4.2 V enters the CV phase; it does not mark the baseline infeasible.
