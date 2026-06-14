#!/usr/bin/env python3
"""
Calibrate SEI Arrhenius k from RW9 capacity fade table (Enhancement 6).

Usage
-----
    venv/bin/python scripts/calibrate_sei_k.py
    venv/bin/python scripts/calibrate_sei_k.py --cell RW9
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rw_transfer.config import load_config

from charging_opt.artifacts import CANONICAL
from charging_opt.reward import DEFAULT_K_ARRHENIUS
from charging_opt.sei_calibration import calibrate_arrhenius_k
from charging_opt.soc_utils import capacity_fade_table, load_ocv_curve, load_steps_with_age


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate SEI Arrhenius k from capacity fade.")
    p.add_argument("--config", default=None)
    p.add_argument("--cell", default="RW9")
    p.add_argument("--ocv_curve", default=CANONICAL["ocv_curve"])
    p.add_argument("--out", default=None, help="JSON output path")
    args = p.parse_args()

    cfg = load_config(args.config)
    matlab_dir = cfg["data"]["matlab_dir"]
    steps, step_age = load_steps_with_age(matlab_dir, args.cell)
    ocv = load_ocv_curve(ROOT / args.ocv_curve)
    table = capacity_fade_table(steps, step_age, ocv_spline=ocv)

    k_opt, info = calibrate_arrhenius_k(table)

    print(f"Default k     : {DEFAULT_K_ARRHENIUS}")
    print(f"Calibrated k  : {k_opt:.5f}")
    print(f"Fit residual  : {info.get('fit_residual')}")
    print(f"Samples       : {info.get('n_samples')}")

    out_path = (
        Path(args.out)
        if args.out
        else ROOT / "outputs" / "charging_opt_user" / "sei_k_calibration.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
