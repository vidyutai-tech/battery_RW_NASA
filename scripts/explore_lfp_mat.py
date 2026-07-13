#!/usr/bin/env python3
"""Inspect lfp_processed.mat without training (fast metadata + optional stitch stats)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lfp_transfer.constants import LFP_STEP_MODE_COMMENTS
from lfp_transfer.data.lfp_loader import load_lfp_stitched_series


def _step_comment(comments_arr: np.ndarray) -> str:
    if comments_arr.size == 0:
        return ""
    raw = comments_arr.flat[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace").strip()
    return str(raw).strip()


def main() -> None:
    p = argparse.ArgumentParser(description="Explore LFP MAT file structure")
    p.add_argument("--mat", default="lfp_processed.mat")
    p.add_argument(
        "--comment_mode",
        default="rw_only",
        choices=list(LFP_STEP_MODE_COMMENTS.keys()),
    )
    p.add_argument("--decimation", type=int, default=10)
    p.add_argument("--stitch", action="store_true", help="Run full stitch (slower)")
    p.add_argument(
        "--export-csv",
        default=None,
        metavar="PATH",
        help="Export stitched series to CSV (creates parent dirs if needed)",
    )
    args = p.parse_args()

    mat_path = Path(args.mat)
    if not mat_path.is_absolute():
        mat_path = ROOT / mat_path

    print(f"Loading {mat_path} ...")
    raw = loadmat(str(mat_path), squeeze_me=True)

    step_comments = [_step_comment(c) for c in raw["Comments_all"]]
    counts = Counter(step_comments)
    lens = [len(v) for v in raw["Voltage_all"]]

    print(f"\nSteps in file     : {len(step_comments)}")
    print(f"Step comments     : {dict(counts)}")
    print(f"Samples per step  : min={min(lens):,}  max={max(lens):,}  mean={np.mean(lens):,.0f}")

    allowed = LFP_STEP_MODE_COMMENTS[args.comment_mode]
    if allowed is not None:
        kept = [i for i, c in enumerate(step_comments) if c in allowed]
        kept_samples = sum(lens[i] for i in kept)
        print(f"\ncomment_mode={args.comment_mode!r}")
        print(f"  kept steps      : {len(kept)}")
        print(f"  kept samples    : {kept_samples:,} (before decimation)")

    v = np.concatenate(raw["Voltage_all"])
    i = np.concatenate(raw["Current_all"])
    t = np.concatenate(raw["Temperature_all"])
    print(f"\nConcatenated (all steps, full resolution)")
    print(f"  samples         : {len(v):,}")
    print(f"  V range         : {v.min():.4f} – {v.max():.4f} V")
    print(f"  I range         : {i.min():.4f} – {i.max():.4f} A")
    print(f"  T range         : {t.min():.2f} – {t.max():.2f} °C")

    if args.stitch:
        print(f"\nStitching (comment_mode={args.comment_mode}, decimation={args.decimation}) ...")
        series = load_lfp_stitched_series(
            mat_path,
            comment_mode=args.comment_mode,
            decimation=args.decimation,
        )
        print(f"  cell_id         : {series.cell_id}")
        print(f"  n_steps         : {series.n_steps}")
        print(f"  n_samples       : {series.n_samples:,}")
        print(f"  duration        : {series.duration_hours:.1f} h")
        chunk_size = 150
        n_chunks = max(0, series.n_samples // chunk_size - 1)
        print(f"  chunks (size={chunk_size}) : {n_chunks:,}")

    if args.export_csv:
        import pandas as pd

        print(f"\nExporting stitched CSV (decimation={args.decimation}) ...")
        series = load_lfp_stitched_series(
            mat_path,
            comment_mode=args.comment_mode,
            decimation=args.decimation,
        )
        df = pd.DataFrame({
            "time_s": series.non_relative_time_s,
            "voltage_v": series.voltage_v,
            "current_a": series.current_a,
            "temperature_c": series.temperature_c,
            "age": series.age,
        })
        out = Path(args.export_csv)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"  wrote {len(df):,} rows → {out}")


if __name__ == "__main__":
    main()
