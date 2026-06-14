#!/usr/bin/env python3
"""Calibrate Wang capacity-fade model from RW9 capacity_fade.npz."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig_{os.getenv('USER', 'default')}")

from charging_opt.artifacts import CANONICAL
from charging_opt.physics_degradation import calibrate_and_save


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capacity", default=CANONICAL["capacity_fade"])
    p.add_argument("--out", default=CANONICAL["degradation_model"])
    args = p.parse_args()
    calibrate_and_save(capacity_fade_path=ROOT / args.capacity, out_path=ROOT / args.out)


if __name__ == "__main__":
    main()
