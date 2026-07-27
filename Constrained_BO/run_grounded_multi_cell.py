"""Run GP-BO + Random (hybrid_qloss) for RW10/11/12 with finetuned BDTs,
then write the same fig8/fig9 grounded comparison suite per cell.

Uses ``get_cell_config`` / ``energy_fraction_for`` so each cell gets its
best finetune fraction (0.60) and energy window (RW10=0.55, others=0.40).

Usage
-----
    python -m Constrained_BO.run_grounded_multi_cell \\
        --cells RW10 RW11 RW12 \\
        --n-calls 40 --n-random 40 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from Constrained_BO.bo_degradation_comparison import (
    build_comparison_rows,
    plot_degradation_comparison,
    plot_pareto_cloud,
    plot_simple_one_axis,
    _save_csv,
)
from Constrained_BO.config import energy_fraction_for, finetune_frac_for
from Constrained_BO.lifetime_fade_projection import (
    _collect_policies,
    _load_measured_cell,
    plot_delta_vs_halfc,
    plot_lifetime_vs_ah,
    plot_lifetime_vs_cycles,
    plot_lifetime_vs_ref_style,
    project_fade,
    save_projection_csv,
)
from Constrained_BO.optimize_api import build_cell, run_optimization
from Constrained_BO.simulator import ChargingSimulator
from Constrained_BO.viz import plot_best_profiles

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "Constrained_BO" / "results" / "grounded_figures"


def _try_plot_profiles(results, cell, out_path: Path, title_suffix: str) -> None:
    try:
        plot_best_profiles(
            results,
            cell_id=cell.cell_id,
            soc_target=float(cell.soc_target),
            soc_start=float(cell.start_state.get("soc", 0.2)),
            out_path=out_path,
            title_suffix=title_suffix,
        )
    except Exception as exc:
        print(f"  (skip profile plot {out_path.name}: {exc})")


def _run_pair(
    cell_id: str,
    out_dir: Path,
    *,
    n_calls: int,
    n_initial: int,
    n_random: int,
    device: str,
    seed: int,
    energy_fraction: Optional[float],
    resume: bool = True,
) -> tuple[Path, Path]:
    cell = build_cell(cell_id, energy_fraction=energy_fraction, soc_mode=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"\n=== {cell_id}  ckpt={cell.bdt_ckpt.name}  "
        f"finetune_frac={finetune_frac_for(cell_id)}  "
        f"energy_fraction={cell.energy_fraction}  "
        f"constraint={cell.constraint_mode}"
    )

    rnd_path = out_dir / "random_search_results.json"
    bo_path = out_dir / "gp_bo_results.json"

    if resume and rnd_path.is_file() and bo_path.is_file():
        print("  Resume: both result JSONs present → skip optimization")
        return bo_path, rnd_path

    sim = ChargingSimulator.from_cell(cell, device=device)

    if resume and rnd_path.is_file():
        print(f"  Resume: reusing {rnd_path.name}")
        rnd_payload = json.loads(rnd_path.read_text())
    else:
        t0 = time.time()
        print(f"  Random search (n_random={n_random}/family) …")
        rnd_payload, rnd_results, sim = run_optimization(
            cell,
            method="random_search",
            device=device,
            seed=seed,
            reward_mode="hybrid_qloss",
            n_random=n_random,
            simulator=sim,
        )
        rnd_path.write_text(json.dumps(rnd_payload, indent=2, default=str))
        _try_plot_profiles(
            rnd_results, cell, out_dir / "random_best_profiles.png", " (random)",
        )
        print(f"  Random done in {time.time() - t0:.1f}s → {rnd_path}")

    if resume and bo_path.is_file():
        print(f"  Resume: reusing {bo_path.name}")
        return bo_path, rnd_path

    elite = {
        fid: (entry.get("history") or [])
        for fid, entry in (rnd_payload.get("families") or {}).items()
    }

    t1 = time.time()
    print(f"  GP-BO (n_calls={n_calls}, n_initial={n_initial}/family) …")
    bo_payload, bo_results, _ = run_optimization(
        cell,
        method="gp_bo",
        device=device,
        seed=seed + 1,
        reward_mode="hybrid_qloss",
        n_calls=n_calls,
        n_initial=n_initial,
        elite_histories=elite,
        simulator=sim,
    )
    bo_path.write_text(json.dumps(bo_payload, indent=2, default=str))
    _try_plot_profiles(
        bo_results, cell, out_dir / "gp_bo_best_profiles.png", " (GP-BO)",
    )
    print(f"  GP-BO done in {time.time() - t1:.1f}s → {bo_path}")
    return bo_path, rnd_path


def _make_figures(
    cell_id: str, bo_path: Path, rnd_path: Path, out_dir: Path, device: str,
) -> None:
    print(f"  Figures for {cell_id} …")
    rows, info, cloud = build_comparison_rows(bo_path, rnd_path, device=device)
    _save_csv(rows, out_dir / "bo_vs_cc_degradation.csv", info)
    (out_dir / "bo_vs_cc_degradation_meta.json").write_text(
        json.dumps(info, indent=2, default=str),
    )
    plot_simple_one_axis(rows, info, out_dir / "fig8_bo_vs_cc_degradation.png")
    plot_degradation_comparison(rows, info, out_dir / "fig8b_bo_vs_cc_degradation_detail.png")
    plot_pareto_cloud(rows, cloud, info, out_dir / "fig8c_bo_vs_cc_pareto.png")

    for r in rows:
        print(
            f"    {r['group']:10s}  Q={float(r['qloss_total']):.5f}  "
            f"feas={r['feasible']}  t={float(r['duration_min']):.1f} min"
        )

    policies, life_info = _collect_policies(bo_path, rnd_path, device=device)
    _, curves, scale = project_fade(policies)
    measured = _load_measured_cell(cell_id)
    plot_lifetime_vs_cycles(
        policies, curves, life_info,
        scale=scale, anchor_cycle=400, soh_anchor_pct=80.0,
        out_path=out_dir / "fig9_lifetime_fade_vs_cycles.png",
    )
    plot_delta_vs_halfc(
        policies, curves, life_info,
        out_path=out_dir / "fig9d_lifetime_delta_vs_halfC.png",
    )
    plot_lifetime_vs_ah(
        policies, curves, life_info, measured,
        out_path=out_dir / "fig9b_lifetime_fade_vs_throughput.png",
    )
    plot_lifetime_vs_ref_style(
        policies, curves, life_info, measured,
        out_path=out_dir / "fig9c_lifetime_capacity_vs_cycle_index.png",
    )
    save_projection_csv(policies, curves, out_dir / "lifetime_fade_projection.csv")
    (out_dir / "lifetime_fade_projection_meta.json").write_text(
        json.dumps({**life_info, "scale": scale}, indent=2, default=str),
    )
    print(f"  Wrote fig8*/fig9* → {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=["RW10", "RW11", "RW12"])
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-calls", type=int, default=40)
    ap.add_argument("--n-initial", type=int, default=10)
    ap.add_argument("--n-random", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--figures-only",
        action="store_true",
        help="Skip optimization; reuse existing JSONs in each cell dir",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run optimizations even if result JSONs already exist",
    )
    ap.add_argument(
        "--energy-fraction",
        type=float,
        default=None,
        help="Override energy fraction for all cells",
    )
    args = ap.parse_args()

    for cell_id in args.cells:
        cell_id = cell_id.upper()
        out_dir = args.out_root / cell_id
        efrac = args.energy_fraction
        if efrac is None:
            efrac = energy_fraction_for(cell_id)

        if args.figures_only:
            bo_path = out_dir / "gp_bo_results.json"
            rnd_path = out_dir / "random_search_results.json"
            if not bo_path.is_file() or not rnd_path.is_file():
                raise FileNotFoundError(f"Missing results under {out_dir}")
        else:
            bo_path, rnd_path = _run_pair(
                cell_id,
                out_dir,
                n_calls=args.n_calls,
                n_initial=args.n_initial,
                n_random=args.n_random,
                device=args.device,
                seed=args.seed,
                energy_fraction=efrac,
                resume=not args.no_resume,
            )

        _make_figures(cell_id, bo_path, rnd_path, out_dir, args.device)

    readme = args.out_root / "README_multi_cell.txt"
    readme.write_text(
        "Per-cell grounded BO/CC/lifetime figures\n"
        "========================================\n"
        "Each RW10/RW11/RW12 folder uses that cell's best finetuned BDT\n"
        "(frac 0.60) and default energy window (RW10=0.55, RW11/12=0.40).\n"
        "\n"
        "Contents per cell/: fig8*, fig9*, gp_bo_results.json,\n"
        "random_search_results.json, CSVs.\n"
        "\n"
        "Regenerate:\n"
        "  python -m Constrained_BO.run_grounded_multi_cell --device cuda\n",
        encoding="utf-8",
    )
    print(f"\nDone. See {args.out_root}/RW*/")


if __name__ == "__main__":
    main()
