#!/usr/bin/env python3
"""Wrapper for charging_opt.replay_nasa_step."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "charging_opt" / "replay_nasa_step.py"),
        run_name="__main__",
    )
