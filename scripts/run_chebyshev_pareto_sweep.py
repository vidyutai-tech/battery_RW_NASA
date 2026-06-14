#!/usr/bin/env python3
"""Entry point for Chebyshev Pareto sweep (repo root on sys.path)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig_{os.getenv('USER', 'default')}")

from charging_opt.run_chebyshev_pareto_sweep import main

if __name__ == "__main__":
    main()
