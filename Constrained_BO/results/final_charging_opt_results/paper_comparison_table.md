# GP-BO comparison table

| Cell | Energy charged | GP-BO time (min) | Time ↓ vs CC | Deg. ↓ vs CC | Time ↓ vs Random | Deg. ↓ vs Random | CC baseline |
|------|----------------|------------------|--------------|--------------|------------------|------------------|-------------|
| RW9 | 40% | 40.3 | 12.0% | -1.2% | -24.9% | -5.5% | CC ½C |
| RW10 | 55% | 34.2 | — | — | 10.2% | 1.2% | none (all CC infeasible) |
| RW11 | 40% | 11.8 | 75.8% | 7.8% | 2.6% | 0.9% | CC ½C |
| RW12 | 40% | 28.6 | 38.9% | -5.0% | 3.9% | -0.4% | CC ½C |

**Reading guide**

- **Energy charged**: fraction of full pack energy delivered in the session (vehicle battery %).
- **Time ↓**: \((t_{\mathrm{base}}-t_{\mathrm{GPBO}})/t_{\mathrm{base}}\). Positive = faster than baseline.
- **Deg. ↓**: \((Q_{\mathrm{base}}-Q_{\mathrm{GPBO}})/Q_{\mathrm{base}}\). Positive = *less* session degradation than baseline.
- Fig. 8/9 do **not** imply GP-BO always has the most degradation. On RW9, reward-best GP-BO is close to CC ½C; Random is slightly better on Q. On RW11, GP-BO improves both time and degradation vs CC ½C. CC 1C/2C can show lower Q only when they fail the energy target (infeasible).
