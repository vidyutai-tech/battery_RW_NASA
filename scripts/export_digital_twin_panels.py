#!/usr/bin/env python3
"""
Export separate square digital-twin validation panels for LaTeX.

For each cell under final_seperate_graphs/{RW9,RW10,RW11,RW12}/ creates:

    digital_twin_panels/chunk{idx}_voltage.png
    digital_twin_panels/chunk{idx}_temperature.png

No overall figure title; minimal margins; manuscript-sized serif fonts.

Usage
-----
    python scripts/export_digital_twin_panels.py --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rw_transfer.config import load_config
from rw_transfer.data.author_dataset import AuthorChunkDataset, random_split_author_dataset
from rw_transfer.data.author_loader import load_author_stitched_series
from rw_transfer.training.twin_trainer import TwinTrainer
from rw_transfer.viz.twin_validation_plots import (
    pick_best_validation_chunks,
    plot_digital_twin_validation_separate,
)

BASE = ROOT / "outputs" / "final_twin_for_all_cell" / "final_seperate_graphs"
CELLS = ("RW9", "RW10", "RW11", "RW12")


def _export_cell(cell: str, device: str, n_panels: int, burn_in: int, seed: int) -> None:
    metrics_path = BASE / cell / "metrics.json"
    if not metrics_path.is_file():
        print(f"  skip {cell}: no metrics.json")
        return
    meta = json.loads(metrics_path.read_text())
    ckpt = Path(meta["ckpt"])
    if not ckpt.is_file():
        print(f"  skip {cell}: missing ckpt {ckpt}")
        return

    cfg = load_config(None)
    twin_cfg = cfg["twin"]
    matlab_dir = cfg["data"]["matlab_dir"]
    decimation = int(cfg["data"].get("decimation", 1))
    chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
    split_cfg = twin_cfg.get("author_split", {})
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))

    out_dir = BASE / cell / "digital_twin_panels"
    print(f"\n=== {cell}  ckpt={ckpt.name}  →  {out_dir}")

    stitched = load_author_stitched_series(matlab_dir, cell, decimation=decimation)
    dataset = AuthorChunkDataset(stitched, chunk_size=chunk_size)
    _, _, test_set = random_split_author_dataset(
        dataset, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )
    trainer = TwinTrainer.load(ckpt, device=device, seq_len=chunk_size)

    samples = pick_best_validation_chunks(
        trainer, test_set, stitched, n=n_panels, burn_in=burn_in,
        age_min=0.25, age_max=0.75,
    )
    if not samples:
        samples = pick_best_validation_chunks(
            trainer, test_set, stitched, n=n_panels, burn_in=burn_in,
        )

    saved = plot_digital_twin_validation_separate(samples, out_dir)
    for p in saved:
        print(f"  saved {p.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_panels", type=int, default=3)
    ap.add_argument("--burn_in", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cells", nargs="+", default=list(CELLS))
    args = ap.parse_args()

    for cell in args.cells:
        _export_cell(cell, args.device, args.n_panels, args.burn_in, args.seed)
    print("\nDone.")


if __name__ == "__main__":
    main()
