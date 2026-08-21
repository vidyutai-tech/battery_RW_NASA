# GP-BO comparison table

Baselines are classic **CCCV (CC→CV at Vmax)** at **½C and 1C**.
Percent columns use **CCCV ½C** as the reference (falls back to 1C if ½C is infeasible).

| Cell | Energy | GP-BO time (min) | Time ↓ vs CCCV ½C | Deg. ↓ vs CCCV ½C | Time ↓ vs Random | Deg. ↓ vs Random | Baseline |
|------|--------|------------------|-------------------|-------------------|------------------|------------------|----------|
| RW10 | 40% | 24.75 | 46.3% | 3.2% | -0.7% | 3.7% | CCCV ½C |
| RW11 | 40% | 11.58 | 76.3% | 8.6% | 3.3% | 1.3% | CCCV ½C |
| RW12 | 40% | 29.75 | 36.5% | -4.6% | 0.0% | 0.0% | CCCV ½C |
| RW9 | 40% | 34.22 | 25.3% | 5.2% | -6.0% | 1.2% | CCCV ½C |

**Reading guide**

- **Energy**: delivered fraction of full pack (same 40% target on all cells).
- **Time ↓**: \((t_{\mathrm{base}}-t_{\mathrm{GPBO}})/t_{\mathrm{base}}\). Positive = faster than baseline.
- **Deg. ↓**: \((Q_{\mathrm{base}}-Q_{\mathrm{GPBO}})/Q_{\mathrm{base}}\). Positive = *less* session degradation than baseline.
- Hitting 4.2 V enters the CV phase; it does not mark the baseline infeasible.
