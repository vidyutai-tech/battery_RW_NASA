#!/usr/bin/env python3
"""
Publication-style measured vs predicted plots for LFP fine-tuned BDT checkpoints.

Outputs (under ``<run_dir>/plots/`` or ``<run_dir>/plots/<window>/``):

  digital_twin_validation_LFP_frac0.20.png
  digital_twin_validation_val_mean_LFP_frac0.20.png
  ... (one pair per finetune fraction)

Window modes
------------
  best   — lowest MAPE (can pick idle/rest windows; misleading for cross-chem)
  active — lowest MAPE among windows with |I| > 0.5 A (recommended default for LFP)
  pulsed — largest voltage swings with current transitions (charge pulse legs)

Usage
-----
    python3 scripts/visualize_lfp_finetune.py \\
        --run_dir outputs/lfp_finetune/finetune_percent/20260707_083400

    python3 scripts/visualize_lfp_finetune.py \\
        --run_dir outputs/lfp_finetune/finetune_percent/20260707_083400 \\
        --window pulsed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

from lfp_transfer.config import load_lfp_config
from lfp_transfer.data.lfp_loader import load_lfp_stitched_series
from rw_transfer.data.author_dataset import AuthorChunkDataset, random_split_author_dataset
from rw_transfer.training.twin_trainer import TwinTrainer
from rw_transfer.viz.plots import plot_finetune_training_curves
from rw_transfer.viz.twin_validation_plots import (
    compute_val_mean_trajectories,
    pick_active_validation_chunks,
    pick_best_validation_chunks,
    pick_pulsed_validation_chunks,
    plot_digital_twin_validation,
    plot_digital_twin_validation_val_mean,
)

# LFP plateau is narrow (~3.2–3.6 V); pulse legs still swing ~0.5–1.0 V.
LFP_PULSED_KW = dict(min_voltage_range_v=0.15, min_current_transitions=2)
LFP_ACTIVE_KW = dict(min_mean_current_a=0.5, min_voltage_range_v=0.05)
LFP_VAL_MEAN_PULSED_KW = dict(
    pulsed_only=True, min_voltage_range_v=0.15, min_current_transitions=2,
)


def _resolve_run_dir(run_dir: Path) -> tuple[Path, Path]:
    run_dir = Path(run_dir)
    registry_dir = run_dir / "registry"
    if registry_dir.is_dir():
        return run_dir, registry_dir

    parent = run_dir.parent
    parent_registry = parent / "registry"
    if parent_registry.is_dir():
        print(
            f"  Note: checkpoints live in {parent_registry}\n"
            f"        (use --run_dir {parent}, not {run_dir})",
            flush=True,
        )
        return parent, parent_registry

    return run_dir, registry_dir


def _discover_fractions(registry_dir: Path, target: str = "LFP") -> list[float]:
    fracs: list[float] = []
    for ckpt in sorted(registry_dir.glob(f"finetune_{target}_frac*.pt")):
        try:
            fracs.append(float(ckpt.stem.split("frac")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(fracs)


def _pick_validation_samples(
    trainer: TwinTrainer,
    test_set,
    stitched,
    *,
    window: str,
    n_panels: int,
    burn_in: int,
):
    if window == "pulsed":
        samples = pick_pulsed_validation_chunks(
            trainer, test_set, stitched, n=n_panels, burn_in=burn_in, **LFP_PULSED_KW,
        )
    elif window == "active":
        samples = pick_active_validation_chunks(
            trainer, test_set, stitched, n=n_panels, burn_in=burn_in, **LFP_ACTIVE_KW,
        )
    else:
        samples = pick_best_validation_chunks(
            trainer, test_set, stitched, n=n_panels, burn_in=burn_in,
            age_min=0.25, age_max=0.75,
        )
        if not samples:
            samples = pick_best_validation_chunks(
                trainer, test_set, stitched, n=n_panels, burn_in=burn_in,
            )
    return samples


def run_visualize_lfp_finetune(
    config_path: str | None = None,
    run_dir: Path | None = None,
    target: str = "LFP",
    fraction: float | None = None,
    out_dir: Path | None = None,
    n_panels: int = 3,
    burn_in: int = 5,
    window: str = "best",
) -> None:
    if run_dir is None:
        raise ValueError("--run_dir is required")

    cfg = load_lfp_config(config_path)
    data_cfg = cfg["data"]
    twin_cfg = cfg["twin"]
    mat_path = data_cfg["lfp_mat"]
    comment_mode = str(data_cfg.get("comment_mode", "rw_only"))
    decimation = int(data_cfg.get("decimation", 1))
    chunk_size = int(twin_cfg.get("chunk_size", cfg["windows"]["seq_len"]))
    split_cfg = twin_cfg.get("author_split", {})
    train_frac = float(split_cfg.get("train_frac", 0.6))
    val_frac = float(split_cfg.get("val_frac", 0.2))
    seed = int(cfg.get("seed", 42))

    run_dir, registry_dir = _resolve_run_dir(Path(run_dir))
    if out_dir is None:
        base_out = run_dir / "plots"
        out_dir = base_out / window if window != "best" else base_out
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fraction is not None:
        fractions = [fraction]
    else:
        fractions = _discover_fractions(registry_dir, target)
        if not fractions:
            raise FileNotFoundError(
                f"No finetune_{target}_frac*.pt checkpoints in {registry_dir}",
            )

    print(f"\n{'='*60}")
    print("  LFP — Finetuned twin visualization")
    print(f"{'='*60}")
    print(f"  Run dir      : {run_dir}")
    print(f"  Target       : {target}")
    print(f"  Fractions    : {[f'{f:.0%}' for f in fractions]}")
    print(f"  Window       : {window}")
    print(f"  Output       : {out_dir}\n")

    print("  Loading LFP stitched series …", flush=True)
    stitched = load_lfp_stitched_series(
        mat_path,
        target,
        comment_mode=comment_mode,
        decimation=decimation,
    )
    dataset = AuthorChunkDataset(stitched, chunk_size=chunk_size)
    train_set, val_set, test_set = random_split_author_dataset(
        dataset, train_frac=train_frac, val_frac=val_frac, seed=seed,
    )
    print(f"  Chunks: train {len(train_set)} / val {len(val_set)} / test {len(test_set)}")

    window_label = {
        "best": "lowest MAPE",
        "active": "active current",
        "pulsed": "pulsed voltage",
    }.get(window, window)

    for frac in fractions:
        ckpt_path = registry_dir / f"finetune_{target}_frac{frac:.2f}.pt"
        if not ckpt_path.is_file():
            print(f"  WARNING: missing checkpoint {ckpt_path}")
            continue

        frac_tag = f"{target}_frac{frac:.2f}"
        print(f"\n  --- {frac_tag} ---")
        print(f"  Checkpoint : {ckpt_path}", flush=True)

        trainer = TwinTrainer.load(ckpt_path, seq_len=chunk_size)

        print(f"  [1] digital_twin_validation_{frac_tag}.png ({window_label}) …", flush=True)
        samples = _pick_validation_samples(
            trainer, test_set, stitched, window=window, n_panels=n_panels, burn_in=burn_in,
        )
        title_suffix = f"finetune {frac:.0%} adapt data"
        if window != "best":
            title_suffix = f"{title_suffix} — {window} windows"
        plot_digital_twin_validation(
            samples,
            out_dir / f"digital_twin_validation_{frac_tag}.png",
            cell_id=target,
            seq_len=chunk_size,
            title_suffix=title_suffix,
        )
        print(f"       Saved digital_twin_validation_{frac_tag}.png  ({len(samples)} panels)")

        print(f"  [1c] digital_twin_validation_val_mean_{frac_tag}.png ({window_label}) …", flush=True)
        val_kw: dict = {"burn_in": burn_in, "seed": seed}
        if window == "pulsed":
            val_kw.update(LFP_VAL_MEAN_PULSED_KW)
        elif window == "active":
            val_kw["min_mean_current_a"] = LFP_ACTIVE_KW["min_mean_current_a"]
            val_kw["min_voltage_range_v"] = LFP_ACTIVE_KW["min_voltage_range_v"]
        val_stats = compute_val_mean_trajectories(trainer, val_set, stitched, **val_kw)
        if val_stats:
            plot_digital_twin_validation_val_mean(
                val_stats,
                out_dir / f"digital_twin_validation_val_mean_{frac_tag}.png",
                title_suffix=title_suffix if window != "best" else "",
            )
            print(f"       Saved digital_twin_validation_val_mean_{frac_tag}.png")

        if window == "best":
            for stage in ("stage1", "stage2"):
                log_path = registry_dir / f"train_log_{frac_tag}_{stage}.jsonl"
                out_path = out_dir / f"finetune_curves_{frac_tag}_{stage}.png"
                if log_path.is_file():
                    plot_finetune_training_curves(
                        log_path,
                        out_path,
                        stage_label=f"{target} {frac:.0%} — {stage.replace('stage', 'Stage ')}",
                    )

    print(f"\n{'='*60}")
    print(f"  Done — figures in {out_dir}")
    print(f"{'='*60}\n")


def main() -> None:
    p = argparse.ArgumentParser(description="LFP finetuned twin validation plots")
    p.add_argument("--config", default="configs/lfp_finetune.yaml")
    p.add_argument(
        "--run_dir",
        required=True,
        help="LFP finetune run root, e.g. outputs/lfp_finetune/finetune_percent/<timestamp>",
    )
    p.add_argument("--target", default="LFP")
    p.add_argument("--fraction", type=float, default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--n_panels", type=int, default=3)
    p.add_argument("--burn_in", type=int, default=5)
    p.add_argument(
        "--window",
        choices=["best", "active", "pulsed"],
        default="best",
        help="Chunk selection: best MAPE, active-current only, or pulsed voltage",
    )
    args = p.parse_args()

    run_visualize_lfp_finetune(
        config_path=args.config,
        run_dir=Path(args.run_dir),
        target=args.target,
        fraction=args.fraction,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        n_panels=args.n_panels,
        burn_in=args.burn_in,
        window=args.window,
    )


if __name__ == "__main__":
    main()
