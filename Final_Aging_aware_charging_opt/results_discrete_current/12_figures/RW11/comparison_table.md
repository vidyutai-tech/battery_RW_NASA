# RW11 — 40% energy in 30 min

GP-BO is the best discrete two-step whose second current is actually used.
CCCV 0.5C is the finished equal-energy charge (it needs more than 30 min).
Random is the best trial from the main random search that already finishes within 30 min.

| Method | Family | Time (min) | Peak T (°C) | Q_loss | Retention @ 600 |
|---|---|---:|---:|---:|---:|
| CCCV 0.5C | CCCV | 53.62 | 23.99 | 1.3437e-04 | 80.00% |
| CCCV 1C | CCCV | 25.95 | 24.05 | 1.0968e-04 | 83.68% |
| Random | CCCV | 15.65 | 24.22 | 1.1027e-04 | 83.59% |
| GP-BO | Two-step | 22.58 | 24.06 | 1.1304e-04 | 83.18% |
