#!/usr/bin/env python3
"""
Full Stage 3 pipeline for a target cell (e.g. RW10 fine-tuned BDT).

Runs per-cell prerequisites without overwriting RW9 canonical artifacts, then:
  1. OCV + capacity fade fit
  2. Wang degradation model calibration
  3. BDT drift / conformal margins
  4. Physics + thermal multi-family BO (03)
  5. Ambient T0 sweep (optional)
  6. Chebyshev Pareto sweep (optional)
  7. Publication figures (optional)

Example (RW10, 40% fine-tune checkpoint):
  venv/bin/python scripts/run_stage3_cell.py \\
    --cell RW10 \\
    --bdt_ckpt outputs/finetune_two_stage_RW10/registry/finetune_RW10_frac0.40.pt \\
    --user hima

Skip long steps while iterating:
  venv/bin/python scripts/run_stage3_cell.py --cell RW10 --skip_bo --skip_chebyshev
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


def cell_artifact_root(user: str, cell: str) -> Path:
    return ROOT / "outputs" / "charging_opt_user" / user / "cells" / cell


def cell_stage3_root(user: str, cell: str) -> Path:
    return ROOT / "outputs" / "charging_opt_user" / user / f"stage3_physics_thermal_{cell}"


def default_bdt_ckpt(cell: str) -> str:
    if cell == "RW10":
        return CANONICAL["bdt_finetune_rw10_40"]
    if cell == "RW9":
        return CANONICAL["bdt_source"]
    raise ValueError(f"No default BDT checkpoint for {cell}; pass --bdt_ckpt")


def main() -> None:
    py = sys.executable
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cell", default="RW10", choices=["RW9", "RW10", "RW11", "RW12"])
    p.add_argument("--user", default=os.getenv("USER", "hima"))
    p.add_argument("--bdt_ckpt", default=None)
    p.add_argument("--soc", type=float, default=0.15)
    p.add_argument("--v0", type=float, default=3.711)
    p.add_argument("--t0", type=float, default=24.7)
    p.add_argument("--age", type=float, default=0.0)
    p.add_argument("--n_calls", type=int, default=40)
    p.add_argument("--n_initial", type=int, default=10)
    p.add_argument("--acq_func", default="PI", choices=["EI", "PI", "LCB"])
    p.add_argument("--skip_prereqs", action="store_true")
    p.add_argument("--skip_bo", action="store_true")
    p.add_argument("--skip_ambient", action="store_true")
    p.add_argument("--skip_chebyshev", action="store_true")
    p.add_argument("--skip_figures", action="store_true")
    p.add_argument("--ambient_n_calls", type=int, default=20)
    args = p.parse_args()

    bdt_ckpt = args.bdt_ckpt or default_bdt_ckpt(args.cell)
    art_root = cell_artifact_root(args.user, args.cell)
    stage1 = art_root / "stage1_state_estimation"
    drift = art_root / "stage1_drift"
    ocv_path = stage1 / "ocv_soc_curve.npz"
    cap_path = stage1 / "capacity_fade.npz"
    deg_path = stage1 / "degradation_model.npz"
    margins_path = drift / "conformal_margins.npz"
    out_dir = cell_stage3_root(args.user, args.cell)
    cheb_dir = ROOT / "outputs" / "charging_opt_user" / args.user / f"chebyshev_sweep_{args.cell}"
    viz_dir = ROOT / "outputs" / "charging_opt_user" / args.user / f"visualization_{args.cell}"

    state_args = [
        "--soc", str(args.soc),
        "--v0", str(args.v0),
        "--t0", str(args.t0),
        "--age", str(args.age),
    ]

    if not args.skip_prereqs:
        _run(
            [py, "scripts/01_fit_ocv_curve.py", "--cell", args.cell,
             "--artifact_root", str(art_root)],
            label=f"Prereq 1/3 — OCV + capacity fade ({args.cell})",
        )
        _run(
            [py, "scripts/calibrate_degradation_model.py",
             "--capacity", str(cap_path),
             "--out", str(deg_path)],
            label=f"Prereq 2/3 — Wang degradation calibration ({args.cell})",
        )
        _run(
            [py, "scripts/00_diagnose_drift.py",
             "--cell", args.cell,
             "--ckpt", bdt_ckpt,
             "--artifact_root", str(art_root)],
            label=f"Prereq 3/3 — Drift margins ({args.cell} BDT)",
        )
    else:
        for path, name in [
            (ocv_path, "OCV curve"),
            (cap_path, "capacity fade"),
            (deg_path, "degradation model"),
            (margins_path, "conformal margins"),
        ]:
            if not path.is_file():
                raise SystemExit(f"Missing {name}: {path} (run without --skip_prereqs)")

    bo_common = [
        py, "scripts/03_optimize_profile_families.py",
        "--cell", args.cell,
        "--objective", "physics",
        "--acq_func", args.acq_func,
        "--thermal_derating", "--thermal_loss",
        "--thermal_derate_comfort_c", "33",
        "--bdt_ckpt", bdt_ckpt,
        "--ocv_curve", str(ocv_path),
        "--capacity", str(cap_path),
        "--margins", str(margins_path),
        "--n_calls", str(args.n_calls),
        "--n_initial", str(args.n_initial),
        "--max_duration_min", "105",
        "--max_minutes", "150",
        *state_args,
    ]

    if not args.skip_bo:
        _run(
            bo_common + ["--out_dir", str(out_dir)],
            label=f"Stage 3 — Physics + thermal BO ({args.cell})",
        )

    if not args.skip_ambient:
        _run(
            [
                py, "scripts/run_ambient_sensitivity.py",
                "--objective", "physics",
                "--thermal_derating", "--thermal_loss",
                "--thermal_derate_comfort_c", "33",
                "--acq_func", args.acq_func,
                "--bdt_ckpt", bdt_ckpt,
                "--capacity", str(cap_path),
                "--margins", str(margins_path),
                "--families", "cccv,pulsed,adaptive_two_step",
                "--ambient_temps", "15,25,35",
                "--n_calls", str(args.ambient_n_calls),
                "--n_initial", str(max(6, args.ambient_n_calls // 4)),
                "--max_duration_min", "105",
                *state_args,
                "--out_dir", str(out_dir / "ambient_sensitivity"),
            ],
            label=f"Stage 3 — Ambient sensitivity ({args.cell})",
        )

    if not args.skip_chebyshev:
        cheb_json = cheb_dir / "chebyshev_sweep_results.json"
        _run(
            [
                py, "scripts/run_chebyshev_pareto_sweep.py",
                "--objective", "physics",
                "--thermal_derating", "--thermal_loss",
                "--thermal_derate_comfort_c", "33",
                "--acq_func", args.acq_func,
                "--bdt_ckpt", bdt_ckpt,
                "--capacity", str(cap_path),
                "--margins", str(margins_path),
                "--families", "pulsed", "cccv", "adaptive_two_step",
                "--n_calls", "30", "--n_initial", "8",
                *state_args,
                "--max_duration_min", "105",
                "--max_minutes", "150",
                "--out_dir", str(cheb_dir),
            ],
            label=f"Stage 3 — Chebyshev sweep ({args.cell})",
        )

    if not args.skip_figures and not args.skip_bo:
        cheb_json = cheb_dir / "chebyshev_sweep_results.json"
        fig_cmd = [
            py, "scripts/gen_all_figs.py",
            "--run_dir", str(out_dir),
            "--out_dir", str(viz_dir),
            "--with_physics",
        ]
        if cheb_json.is_file():
            fig_cmd += ["--chebyshev_json", str(cheb_json)]
        _run(fig_cmd, label=f"Stage 3 — Publication figures ({args.cell})")

    manifest = {
        "cell": args.cell,
        "bdt_ckpt": bdt_ckpt,
        "artifact_root": str(art_root),
        "stage3_out_dir": str(out_dir),
        "chebyshev_dir": str(cheb_dir),
        "visualization_dir": str(viz_dir),
        "paths": {
            "ocv_curve": str(ocv_path),
            "capacity_fade": str(cap_path),
            "degradation_model": str(deg_path),
            "conformal_margins": str(margins_path),
        },
        "start_state": {
            "soc": args.soc, "v0": args.v0, "t0": args.t0, "age": args.age,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage3_cell_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print(f"\n{'=' * 72}")
    print(f"  STAGE 3 PIPELINE COMPLETE — {args.cell}")
    print(f"  Artifacts : {art_root}/")
    print(f"  BO results: {out_dir}/")
    if not args.skip_chebyshev:
        print(f"  Chebyshev : {cheb_dir}/")
    if not args.skip_figures:
        print(f"  Figures   : {viz_dir}/")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
