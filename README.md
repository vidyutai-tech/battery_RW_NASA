# Aging-aware charging optimization

Self-contained search for charging profiles on frozen NASA RW digital twins (RW9–RW12).

Code and saved results live in `Final_Aging_aware_charging_opt/`.

| Path | Role |
|------|------|
| `Final_Aging_aware_charging_opt/` | Config, twin, GP-BO / random search, figures |
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
