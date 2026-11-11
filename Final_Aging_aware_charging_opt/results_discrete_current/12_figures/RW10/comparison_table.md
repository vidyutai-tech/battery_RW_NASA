# RW10 — 40% energy in 30 min

GP-BO is the best discrete two-step whose second current is actually used.
CCCV 0.5C is the finished equal-energy charge (it needs more than 30 min).
Random is the best trial from the main random search that already finishes within 30 min.

| Method | Family | Time (min) | Peak T (°C) | Q_loss | Retention @ 600 |
|---|---|---:|---:|---:|---:|
| CCCV 0.5C | CCCV | 51.33 | 23.98 | 1.4126e-04 | 80.00% |
| CCCV 1C | CCCV | 24.72 | 24.02 | 1.1670e-04 | 83.48% |
| Random | CCCV | 24.72 | 24.02 | 1.1670e-04 | 83.48% |
| GP-BO | Two-step | 25.48 | 24.02 | 1.1848e-04 | 83.23% |
