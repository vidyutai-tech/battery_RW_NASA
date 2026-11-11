# RW9 — 40% energy in 30 min

GP-BO is the best discrete two-step whose second current is actually used.
CCCV 0.5C is the finished equal-energy charge (it needs more than 30 min).
Random is the best trial from the main random search that already finishes within 30 min.

| Method | Family | Time (min) | Peak T (°C) | Q_loss | Retention @ 600 |
|---|---|---:|---:|---:|---:|
| CCCV 0.5C | CCCV | 51.70 | 23.93 | 1.3965e-04 | 80.00% |
| CCCV 1C | CCCV | 27.53 | 23.99 | 1.2171e-04 | 82.57% |
| Random | Two-step | 27.03 | 23.99 | 1.2029e-04 | 82.77% |
| GP-BO | Two-step | 28.45 | 23.99 | 1.2149e-04 | 82.60% |
