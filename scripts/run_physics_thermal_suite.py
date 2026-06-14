#!/usr/bin/env python3
"""
Run physics-grounded + temperature-aware optimization in one go.

Steps:
  1. (optional) Calibrate Wang degradation model from capacity_fade.npz
  2. Multi-family BO at baseline T0 with --objective physics + thermal derating/loss
  3. (optional) Ambient T0 sweep at 15 / 25 / 35 °C with the same settings

Example:
  venv/bin/python scripts/run_physics_thermal_suite.py \\
    --soc 0.15 --v0 3.711 --t0 24.7 --age 0.0 \\
    --out_dir outputs/charging_opt_user/hima/stage3_physics_thermal
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig_{os.getenv('USER', 'default')}")

from charging_opt.artifacts import CANONICAL


def _run(cmd: list[str], *, label: str) -> None:
    print(f"\n{'=' * 72}\n  {label}\n{'=' * 72}")
    print(" ", " ".join(cmd), "\n")
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def main() -> None:
    py = sys.executable
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="outputs/charging_opt_user/hima/stage3_physics_thermal")
    p.add_argument("--bdt_ckpt", default=CANONICAL["bdt_source"])
    p.add_argument(
        "--families",
        default="cccv,reduced_cv_cccv,pulsed,cc_taper,adaptive_two_step,adaptive_three_step,multi_step_taper,exponential_taper",
    )
    p.add_argument("--n_calls", type=int, default=40)
    p.add_argument("--n_initial", type=int, default=10)
    p.add_argument("--acq_func", default="PI", choices=["EI", "PI", "LCB"])
    p.add_argument("--soc", type=float, default=0.15)
    p.add_argument("--v0", type=float, default=3.711)
    p.add_argument("--t0", type=float, default=24.7)
    p.add_argument("--age", type=float, default=0.0)
    p.add_argument("--max_duration_min", type=float, default=105.0)
    p.add_argument("--thermal_derate_comfort_c", type=float, default=33.0)
    p.add_argument("--skip_calibration", action="store_true")
    p.add_argument("--skip_ambient", action="store_true", help="Skip 15/25/35°C ambient sweep")
    p.add_argument("--ambient_n_calls", type=int, default=20, help="BO evals/family in ambient sweep")
    p.add_argument("--ambient_families", default="cccv,pulsed,adaptive_two_step")
    args = p.parse_args()

    out_dir = (ROOT / args.out_dir).resolve()
    ambient_dir = out_dir / "ambient_sensitivity"

    if not args.skip_calibration:
        deg_path = ROOT / CANONICAL["degradation_model"]
        if not deg_path.is_file():
            _run(
                [py, "scripts/calibrate_degradation_model.py"],
                label="Step 1 — Calibrate Wang degradation model",
            )
        else:
            print(f"Using existing degradation model: {deg_path}")

    common = [
        py, "scripts/03_optimize_profile_families.py",
        "--objective", "physics",
        "--acq_func", args.acq_func,
        "--thermal_derating",
        "--thermal_loss",
        "--thermal_derate_comfort_c", str(args.thermal_derate_comfort_c),
        "--bdt_ckpt", args.bdt_ckpt,
        "--soc", str(args.soc),
        "--v0", str(args.v0),
        "--t0", str(args.t0),
        "--age", str(args.age),
        "--max_duration_min", str(args.max_duration_min),
        "--n_calls", str(args.n_calls),
        "--n_initial", str(args.n_initial),
    ]

    _run(
        common + ["--families", args.families, "--out_dir", str(out_dir)],
        label="Step 2 — Physics + thermal BO (baseline T0)",
    )

    if not args.skip_ambient:
        _run(
            [
                py, "scripts/run_ambient_sensitivity.py",
                "--objective", "physics",
                "--thermal_derating",
                "--thermal_loss",
                "--thermal_derate_comfort_c", str(args.thermal_derate_comfort_c),
                "--bdt_ckpt", args.bdt_ckpt,
                "--soc", str(args.soc),
                "--v0", str(args.v0),
                "--t0", str(args.t0),
                "--age", str(args.age),
                "--max_duration_min", str(args.max_duration_min),
                "--n_calls", str(args.ambient_n_calls),
                "--n_initial", str(max(6, args.ambient_n_calls // 4)),
                "--families", args.ambient_families,
                "--ambient_temps", "15,25,35",
                "--out_dir", str(ambient_dir),
            ],
            label="Step 3 — Ambient T0 sweep (physics + thermal)",
        )

    manifest = {
        "baseline_run": str(out_dir),
        "ambient_run": None if args.skip_ambient else str(ambient_dir),
        "settings": {
            "objective": "physics",
            "thermal_derating": True,
            "thermal_loss": True,
            "thermal_derate_comfort_c": args.thermal_derate_comfort_c,
            "acq_func": args.acq_func,
            "start_state": {"soc": args.soc, "v0": args.v0, "t0": args.t0, "age": args.age},
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "suite_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"\n{'=' * 72}")
    print("  SUITE COMPLETE")
    print(f"  Baseline results : {out_dir}/")
    if not args.skip_ambient:
        print(f"  Ambient results  : {ambient_dir}/ambient_sensitivity_summary.json")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
