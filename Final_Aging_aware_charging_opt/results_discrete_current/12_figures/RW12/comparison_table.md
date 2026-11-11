# RW12 — 40% energy in 30 min

GP-BO is the best discrete two-step whose second current is actually used.
CCCV 0.5C is the finished equal-energy charge (it needs more than 30 min).
Random is the best trial from the main random search that already finishes within 30 min.

| Method | Family | Time (min) | Peak T (°C) | Q_loss | Retention @ 600 |
|---|---|---:|---:|---:|---:|
| CCCV 0.5C | CCCV | 57.17 | 24.00 | 1.4055e-04 | 80.00% |
| CCCV 1C | CCCV | 24.72 | 24.32 | 1.1902e-04 | 83.06% |
| Random | CCCV | 24.72 | 24.32 | 1.1902e-04 | 83.06% |
| GP-BO | Two-step | 28.83 | 24.32 | 1.2429e-04 | 82.31% |
