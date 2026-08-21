#!/usr/bin/env python3
"""
Export paper-ready Digital Twin figures for RW9–RW12 into one folder.

Includes finetune adapt fractions **20% / 40% / 60%** for RW10–RW12,
plus the RW9 source twin.

Output layout
-------------
    outputs/final_twin_for_all_cell/
      README.txt
      summary_mape_across_cells.png       — RW9 + each cell@60%
      summary_mape_by_fraction.png        — RW10–12 × {20,40,60}%
      summary_metrics.json / .csv
      RW9/                                — source twin
      RW10/frac0.20|0.40|0.60/            — per-fraction figures + metrics.json
      RW10/                               — copy of frac0.60 (paper default)
      … same for RW11, RW12

Usage
-----
    python scripts/export_final_twin_paper_figs.py --device cpu
    python scripts/export_final_twin_paper_figs.py --device cuda --fractions 0.20 0.40 0.60
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.visualize_twin import _plot_twin_validation  # noqa: E402
from rw_transfer.config import load_config  # noqa: E402
from rw_transfer.data.author_dataset import AuthorChunkDataset, random_split_author_dataset  # noqa: E402
from rw_transfer.data.author_loader import load_author_stitched_series  # noqa: E402
from rw_transfer.training.twin_trainer import TwinTrainer  # noqa: E402
from rw_transfer.viz.twin_validation_plots import (  # noqa: E402
    compute_val_mean_trajectories,
    pick_best_validation_chunks,
    plot_cross_cell_twin_summary,
    plot_finetune_fraction_summary,
    plot_voltage_error_panel,
)

DEFAULT_OUT = ROOT / "outputs" / "final_twin_for_all_cell"
SOURCE_CKPT = ROOT / "outputs/twin_source/20260610_111409/twin_source_RW9.pt"
SOURCE_PLOTS = ROOT / "outputs/twin_source/20260610_111409/plots"

FINETUNE_CELLS = ("RW10", "RW11", "RW12")
DEFAULT_FRACTIONS = (0.20, 0.40, 0.60)
PAPER_DEFAULT_FRAC = 0.60


def _registry_metrics(run_dir: Path, cell: str, frac: float) -> Dict[str, Any]:
    """Pull held-out registry metrics (RMSE/MAE/MAPE/R²) if present."""
    path = run_dir / "registry" / "finetune_registry.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    key = f"{cell}_frac{frac:.2f}"
    entry = (data.get("entries") or {}).get(key) or {}
    out: Dict[str, Any] = {}
    for domain in ("voltage", "temperature"):
        block = entry.get(domain) or {}
        if not block:
            continue
        for k in ("rmse", "mae", "mape_pct", "r2", "mse", "n"):
            if k in block:
                out[f"{domain}_{k}"] = block[k]
    for k in ("n_adapt_windows", "n_eval_windows", "train_time_s", "infer_ms", "n_params"):
        if k in entry:
            out[k] = entry[k]
    return out


def _export_cell(
    *,
    cell: str,
    ckpt: Path,
    out_dir: Path,
    title_suffix: str,
    n_panels: int,
    burn_in: int,
    seed: int,
    device: str,
    fraction: Optional[float] = None,
    copy_extras: Optional[Sequence[Tuple[Path, str]]] = None,
    registry_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg = load_config(None)
    twin_cfg = cfg["twin"]
    matlab_dir = cfg["data"]["matlab_dir"]
    decimation = int(cfg["data"].get("decimation", 1))
    chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
    split_cfg = twin_cfg.get("author_split", {})
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {cell}  frac={fraction}  ckpt={ckpt.name}  →  {out_dir}")

    stitched = load_author_stitched_series(matlab_dir, cell, decimation=decimation)
    dataset = AuthorChunkDataset(stitched, chunk_size=chunk_size)
    train_set, val_set, test_set = random_split_author_dataset(
        dataset, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )
    trainer = TwinTrainer.load(ckpt, device=device, seq_len=chunk_size)

    _plot_twin_validation(
        trainer,
        cell=cell,
        stitched=stitched,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        out_dir=out_dir,
        chunk_size=chunk_size,
        seed=seed,
        n_panels=n_panels,
        burn_in=burn_in,
        name_tag="",
        title_suffix=title_suffix,
        window="best",
    )

    samples = pick_best_validation_chunks(
        trainer, test_set, stitched, n=n_panels, burn_in=burn_in,
        age_min=0.25, age_max=0.75,
    )
    if not samples:
        samples = pick_best_validation_chunks(
            trainer, test_set, stitched, n=n_panels, burn_in=burn_in,
        )
    if samples:
        plot_voltage_error_panel(
            samples,
            out_dir / "voltage_residual_test_chunks.png",
            cell_id=cell,
        )
        print("       Saved voltage_residual_test_chunks.png")

    val_stats = compute_val_mean_trajectories(
        trainer, val_set, stitched, burn_in=burn_in, seed=seed,
    )
    row: Dict[str, Any] = {
        "cell": cell,
        "fraction": fraction,
        "ckpt": str(ckpt),
        "mape_v": float(val_stats["pooled_mape_v_pct"]) if val_stats else float("nan"),
        "mape_t": float(val_stats["pooled_mape_t_pct"]) if val_stats else float("nan"),
        "n_val_windows": int(val_stats["n_windows_used"]) if val_stats else 0,
        "title_suffix": title_suffix,
    }
    if registry_extra:
        row.update(registry_extra)

    for src, name in (copy_extras or []):
        if src.is_file():
            shutil.copy2(src, out_dir / name)
            print(f"       Copied {name}")

    (out_dir / "metrics.json").write_text(json.dumps(row, indent=2))
    return row


def _mirror_default_frac(frac_dir: Path, cell_dir: Path) -> None:
    """Copy paper-default fraction artifacts to the cell root for compatibility."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "digital_twin_validation.png",
        "digital_twin_validation_val_mean.png",
        "voltage_residual_test_chunks.png",
        "metrics.json",
    ):
        src = frac_dir / name
        if src.is_file():
            shutil.copy2(src, cell_dir / name)
    for png in frac_dir.glob("finetune_curves_*.png"):
        shutil.copy2(png, cell_dir / png.name)


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    # Stable column order
    preferred = [
        "cell", "fraction", "title_suffix", "mape_v", "mape_t", "n_val_windows",
        "voltage_rmse", "voltage_mae", "voltage_mape_pct", "voltage_r2",
        "temperature_rmse", "temperature_mae", "temperature_mape_pct", "temperature_r2",
        "n_adapt_windows", "n_eval_windows", "train_time_s", "infer_ms", "n_params", "ckpt",
    ]
    keys: List[str] = []
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-panels", type=int, default=3)
    ap.add_argument("--burn-in", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_FRACTIONS),
        help="Finetune adapt-data fractions for RW10–RW12 (default 0.20 0.40 0.60).",
    )
    ap.add_argument(
        "--skip-rw9",
        action="store_true",
        help="Skip regenerating RW9 source twin figures.",
    )
    args = ap.parse_args()

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    fractions = tuple(float(f) for f in args.fractions)
    summary_rows: List[Dict[str, Any]] = []

    # RW9 source twin
    if not args.skip_rw9:
        if not SOURCE_CKPT.is_file():
            raise FileNotFoundError(SOURCE_CKPT)
        extras = []
        for name in ("twin_train_curves.png", "soc_estimation.png", "soc_variant_comparison.png"):
            p = SOURCE_PLOTS / name
            if not p.is_file():
                p = SOURCE_CKPT.parent / name
            extras.append((p, name))
        summary_rows.append(
            _export_cell(
                cell="RW9",
                ckpt=SOURCE_CKPT,
                out_dir=out_root / "RW9",
                title_suffix="source twin",
                n_panels=args.n_panels,
                burn_in=args.burn_in,
                seed=args.seed,
                device=args.device,
                fraction=None,
                copy_extras=extras,
            )
        )
    elif (out_root / "RW9" / "metrics.json").is_file():
        summary_rows.append(json.loads((out_root / "RW9" / "metrics.json").read_text()))

    for cell in FINETUNE_CELLS:
        run_dir = ROOT / f"outputs/finetune_two_stage_{cell}"
        for frac in fractions:
            ckpt = run_dir / "registry" / f"finetune_{cell}_frac{frac:.2f}.pt"
            if not ckpt.is_file():
                print(f"WARNING: missing {ckpt} — skip {cell} @{frac:.0%}")
                continue
            frac_dir = out_root / cell / f"frac{frac:.2f}"
            extras = []
            for stage in ("stage1", "stage2"):
                log_png = run_dir / "plots" / f"finetune_curves_{cell}_frac{frac:.2f}_{stage}.png"
                extras.append((log_png, log_png.name))
            row = _export_cell(
                cell=cell,
                ckpt=ckpt,
                out_dir=frac_dir,
                title_suffix=f"finetune {frac:.0%} adapt data",
                n_panels=args.n_panels,
                burn_in=args.burn_in,
                seed=args.seed,
                device=args.device,
                fraction=frac,
                copy_extras=extras,
                registry_extra=_registry_metrics(run_dir, cell, frac),
            )
            summary_rows.append(row)
            if abs(frac - PAPER_DEFAULT_FRAC) < 1e-9:
                _mirror_default_frac(frac_dir, out_root / cell)

    # Summaries
    default_rows = [
        r for r in summary_rows
        if r.get("cell") == "RW9" or (
            r.get("fraction") is not None and abs(float(r["fraction"]) - PAPER_DEFAULT_FRAC) < 1e-9
        )
    ]
    if default_rows:
        plot_cross_cell_twin_summary(
            default_rows,
            out_root / "summary_mape_across_cells.png",
            footnote="RW9 = source twin; RW10–RW12 = finetune frac 0.60 (paper default).",
        )
    frac_rows = [r for r in summary_rows if r.get("fraction") is not None]
    if frac_rows:
        plot_cross_cell_twin_summary(
            frac_rows,
            out_root / "summary_mape_all_fractions.png",
            footnote="RW10–RW12 finetune adapt fractions 20% / 40% / 60%.",
        )
        plot_finetune_fraction_summary(
            frac_rows,
            out_root / "summary_mape_by_fraction.png",
        )

    (out_root / "summary_metrics.json").write_text(json.dumps(summary_rows, indent=2))
    _write_summary_csv(summary_rows, out_root / "summary_metrics.csv")

    (out_root / "README.txt").write_text(
        "Paper-ready Digital Twin figures (NASA RW9–RW12)\n"
        "================================================\n"
        "\n"
        "RW9/\n"
        "  Source twin (no finetune fractions).\n"
        "  digital_twin_validation.png, val_mean, residual, metrics.json, SOC/train plots\n"
        "\n"
        "RW10|RW11|RW12/\n"
        "  frac0.20/  frac0.40/  frac0.60/   — full figure set + metrics.json each\n"
        "  Cell root mirrors frac0.60 (paper default).\n"
        "\n"
        "Root summaries\n"
        "  summary_mape_across_cells.png   — RW9 + cells @ 60%\n"
        "  summary_mape_by_fraction.png    — V/T MAPE vs 20/40/60% (RW10–12)\n"
        "  summary_mape_all_fractions.png  — flat bar view of all frac runs\n"
        "  summary_metrics.json / .csv     — pooled val MAPE + registry RMSE/MAE/R²\n"
        "\n"
        "Display note (senior review: noisy voltage predictions)\n"
        "-------------------------------------------------------\n"
        "Voltage/temperature predicted traces use a light Savitzky–Golay\n"
        "overlay for paper readability; the faint dotted line is the raw\n"
        "twin output. Reported MAPE is always computed on raw predictions.\n"
        "\n"
        "Checkpoints\n"
        "  RW9  : outputs/twin_source/20260610_111409/twin_source_RW9.pt\n"
        "  RW10–12 : outputs/finetune_two_stage_RW*/registry/finetune_*_frac{0.20,0.40,0.60}.pt\n"
        "\n"
        "Regenerate:\n"
        "  python scripts/export_final_twin_paper_figs.py --device cpu\n"
        "  python scripts/export_final_twin_paper_figs.py --device cuda "
        "--fractions 0.20 0.40 0.60\n",
        encoding="utf-8",
    )

    print(f"\nDone → {out_root}")
    for r in summary_rows:
        frac = r.get("fraction")
        tag = "source" if frac is None else f"{float(frac):.0%}"
        print(
            f"  {r['cell']:4s} {tag:>6s}: "
            f"V-MAPE={r['mape_v']:.3f}%  T-MAPE={r['mape_t']:.3f}%"
        )


if __name__ == "__main__":
    main()
