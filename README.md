# Aging-aware charging optimization

Discrete-current search for charging profiles on frozen NASA RW digital twins (RW9–RW12).

Commanded charge current is restricted to 1.0, 1.5, 2.0, 3.0, and 4.5 A. Saved search output is in `Final_Aging_aware_charging_opt/results_discrete_current/`.

| Path | Role |
|------|------|
| `Final_Aging_aware_charging_opt/` | Config, twin, GP-BO / random search, figures |
| `Final_Aging_aware_charging_opt/results_discrete_current/` | Discrete-current GP-BO and random-search results |
| `outputs/.../*.pt` | Frozen twin checkpoints (RW9 source, RW10–RW12 fine-tunes) |
| `NASA_RW/dataset/` | NASA Matlab files for RW9–RW12 (local only; too large for git) |
| `requirements.txt` | Runtime packages |

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
cd Final_Aging_aware_charging_opt
PYTHONDONTWRITEBYTECODE=1 ../venv/bin/python optimization.py --methods gp_bo random_search
PYTHONDONTWRITEBYTECODE=1 ../venv/bin/python visualization.py --only figures
```

Charge current follows the NASA sign (negative while charging). Reported reward is the advantage over that cell’s own 0.5C CCCV charge.
