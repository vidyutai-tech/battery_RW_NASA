#!/usr/bin/env python
"""STAGE 0 — environment and inherited-artifact check."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aacopt.config import Paths, package_versions, provenance, stage_dir, write_json

STAGE = "00_environment"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    paths = Paths.load()
    mods = package_versions()
    missing_mods = [k for k, v in mods.items() if v == "unavailable" and k != "skopt"]
    try:
        import skopt  # noqa: F401
        mods["skopt"] = getattr(skopt, "__version__", "present")
    except Exception:
        missing_mods.append("skopt")
        mods["skopt"] = "unavailable"

    checkpoints = {}
    for cell in paths.cells:
        p = paths.bdt_checkpoints.get(cell)
        checkpoints[cell] = {
            "path": str(p) if p else None,
            "exists": bool(p and p.is_file()),
            "bytes": int(p.stat().st_size) if p and p.is_file() else 0,
        }
    mats = {}
    for cell in paths.cells:
        try:
            mp = paths.mat_path(cell)
            mats[cell] = {"path": str(mp), "exists": True, "bytes": int(mp.stat().st_size)}
        except FileNotFoundError as exc:
            mats[cell] = {"exists": False, "error": str(exc)}

    ocv = Path(__file__).resolve().parents[1] / "results" / "01_calibration_dataset" / "ocv_curves.npz"
    fitted = Path(__file__).resolve().parents[1] / "configs" / "degradation_fitted.yaml"
    payload = {
        "packages": mods,
        "missing_packages": missing_mods,
        "checkpoints": checkpoints,
        "matlab": mats,
        "ocv_curves": {"path": str(ocv), "exists": ocv.is_file()},
        "degradation_fitted": {"path": str(fitted), "exists": fitted.is_file()},
        "provenance": provenance(STAGE, configs=["paths"]),
    }
    out = stage_dir(STAGE) / "environment.json"
    write_json(out, payload)
    ok = (
        not missing_mods
        and all(v["exists"] for v in checkpoints.values())
        and all(v.get("exists") for v in mats.values())
    )
    print(f"Wrote {out}")
    print("packages:", mods)
    print("checkpoints:", {k: v["exists"] for k, v in checkpoints.items()})
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
