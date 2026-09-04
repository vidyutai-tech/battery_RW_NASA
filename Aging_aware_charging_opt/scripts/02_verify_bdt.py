#!/usr/bin/env python
"""STAGE 2 — verify the inference-only BDT against inherited checkpoints.

Two checks, both reported in ``results/02_bdt_verification/``:

1. **Numerical equivalence** (gate): ``aacopt.bdt.BatteryDigitalTwin.predict``
   vs the original ``TwinTrainer`` wrapper on randomized current profiles.
   Max-abs voltage and temperature must be < 1e-5 on every cell.
   ``rw_transfer`` is imported *only* here, and only for this comparison.

2. **Held-out NASA accuracy** (verification of the inherited artifact, not a
   new training claim): the author 60/20/20 chunk split (seed 42) test set,
   scored with *this* project's loader. Reported as RMSE/MAE/MAPE/R².
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Aging_aware_charging_opt" / "src"))
sys.path.insert(0, str(REPO))

from aacopt.bdt import FrozenBDT, load_twin  # noqa: E402
from aacopt.config import (  # noqa: E402
    Paths, ensure_matplotlib_cache, file_hash, provenance, stage_dir, write_json,
)

STAGE = "02_bdt_verification"
EQ_TOL = 1e-5


def _random_profiles(rng: np.random.Generator) -> list:
    """Charge-signed currents (NASA convention: negative = charge)."""
    out = []
    for n, i_cc in ((80, -1.1), (150, -2.2), (300, -4.4), (450, -0.75)):
        out.append(np.full(n, i_cc, dtype=np.float32))
    n = 200
    i = np.zeros(n, dtype=np.float32)
    i[:80] = -3.0
    i[80:120] = 0.0
    i[120:] = -1.5
    out.append(i)
    rw = -rng.uniform(0.75, 4.8, size=375).astype(np.float32)
    out.append(rw)
    return out


def _max_abs(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b)))


def equivalence_vs_twin_trainer(cell: str, ckpt: Path, *, n_extra: int = 4) -> dict:
    from rw_transfer.training.twin_trainer import TwinTrainer

    ours = load_twin(ckpt, device="cpu")
    trainer = TwinTrainer.load(ckpt, device="cpu")
    trainer.model.eval()
    for p in trainer.model.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(20260904 + sum(ord(c) for c in cell))
    profiles = _random_profiles(rng)
    starts = [
        {"age": 0.0, "v0": 3.55, "t0": 24.0},
        {"age": 0.35, "v0": 3.72, "t0": 26.5},
        {"age": 0.8, "v0": 3.40, "t0": 22.0},
    ]
    records = []
    max_v, max_t = 0.0, 0.0
    for prof in profiles:
        for st in starts:
            v_a, t_a = ours.predict(st["age"], st["v0"], st["t0"], prof)
            v_b, t_b = trainer.model.predict(st["age"], st["v0"], st["t0"], prof)
            dv, dt = _max_abs(v_a, v_b), _max_abs(t_a, t_b)
            max_v, max_t = max(max_v, dv), max(max_t, dt)
            records.append({
                "n": int(prof.size), "age": st["age"], "v0": st["v0"], "t0": st["t0"],
                "max_abs_v": dv, "max_abs_t": dt,
            })
    for _ in range(n_extra):
        n = int(rng.integers(40, 500))
        prof = rng.uniform(-5.0, 0.0, size=n).astype(np.float32)
        st = {
            "age": float(rng.uniform(0, 1)),
            "v0": float(rng.uniform(3.3, 4.0)),
            "t0": float(rng.uniform(20.0, 35.0)),
        }
        v_a, t_a = ours.predict(st["age"], st["v0"], st["t0"], prof)
        v_b, t_b = trainer.model.predict(st["age"], st["v0"], st["t0"], prof)
        dv, dt = _max_abs(v_a, v_b), _max_abs(t_a, t_b)
        max_v, max_t = max(max_v, dv), max(max_t, dt)
        records.append({
            "n": n, **st, "max_abs_v": dv, "max_abs_t": dt,
        })
    passed = (max_v < EQ_TOL) and (max_t < EQ_TOL)
    return {
        "max_abs_voltage": max_v,
        "max_abs_temperature": max_t,
        "tolerance": EQ_TOL,
        "n_comparisons": len(records),
        "passed": bool(passed),
        "worst": records,
    }


def _metrics(pred, ref) -> dict:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    err = pred - ref
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((ref - ref.mean()) ** 2))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(ref), 1e-8)) * 100.0)
    return {
        "n": int(pred.size),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape_pct": mape,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "bias": float(np.mean(err)),
    }


def heldout_nasa(cell: str, ckpt: Path, device: str) -> dict:
    """Author test split (seed 42, 60/20/20), scored with this project's twin."""
    from rw_transfer.data.author_dataset import (
        AuthorChunkDataset, author_subset_to_window_batch, random_split_author_dataset,
    )
    from rw_transfer.data.author_loader import load_author_stitched_series

    paths = Paths.load()
    stitched = load_author_stitched_series(str(paths.matlab_dir), cell, decimation=1)
    dataset = AuthorChunkDataset(stitched, chunk_size=150)
    _, _, test_set = random_split_author_dataset(dataset, train_frac=0.6, val_frac=0.2, seed=42)
    batch = author_subset_to_window_batch(test_set, max_windows=None)
    twin = FrozenBDT(ckpt, device=device)
    v_hat, t_hat = twin.predict_windows(batch.X, batch_size=64)
    return {
        "split": "author_random_60_20_20_seed42_test",
        "n_windows": int(batch.X.shape[0]),
        "seq_len": int(batch.X.shape[1] - 3) if batch.X.size else 0,
        "voltage": _metrics(v_hat, batch.Y_voltage),
        "temperature": _metrics(t_hat, batch.Y_temperature),
        "note": (
            "Inherited-artifact verification on the original author test split. "
            "Not a new training result. Fine-tunes used a prefix of the *train* "
            "split only; this test split was not used to update weights."
        ),
        "example_window": {
            "v_meas": batch.Y_voltage[0].tolist() if batch.X.shape[0] else [],
            "v_pred": v_hat[0].tolist() if v_hat.shape[0] else [],
            "t_meas": batch.Y_temperature[0].tolist() if batch.X.shape[0] else [],
            "t_pred": t_hat[0].tolist() if t_hat.shape[0] else [],
        },
    }


def _plot_cell(cell: str, eq: dict, hold: dict, out_png: Path) -> None:
    ensure_matplotlib_cache()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
    ex = hold.get("example_window") or {}
    t = np.arange(len(ex.get("v_meas") or []))
    ax = axes[0, 0]
    if t.size:
        ax.plot(t, ex["v_meas"], color="k", lw=1.2, label="NASA measured")
        ax.plot(t, ex["v_pred"], color="C0", lw=1.0, ls="--", label="aacopt BDT")
    ax.set_ylabel("Voltage [V]")
    ax.set_title(f"{cell} — held-out window 0")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    if t.size:
        ax.plot(t, ex["t_meas"], color="k", lw=1.2, label="NASA measured")
        ax.plot(t, ex["t_pred"], color="C3", lw=1.0, ls="--", label="aacopt BDT")
    ax.set_ylabel("Temperature [°C]")
    ax.set_title(f"{cell} — held-out window 0")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(["ΔV", "ΔT"], [eq["max_abs_voltage"], eq["max_abs_temperature"]], color=["C0", "C3"])
    ax.axhline(EQ_TOL, color="k", ls="--", lw=0.8, label=f"tol {EQ_TOL:g}")
    ax.set_ylabel("max |ours − TwinTrainer|")
    ax.set_title("Equivalence (gate)")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    vv, tt = hold["voltage"], hold["temperature"]
    ax.axis("off")
    ax.set_title("Held-out NASA test split")
    ax.text(
        0.05, 0.55,
        f"V  RMSE={vv['rmse']:.4f} V   MAPE={vv['mape_pct']:.3f}%   R²={vv['r2']:.4f}\n"
        f"T  RMSE={tt['rmse']:.4f} °C  MAPE={tt['mape_pct']:.3f}%   R²={tt['r2']:.4f}\n"
        f"windows = {hold['n_windows']}",
        transform=ax.transAxes, fontsize=10, family="monospace", va="center",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-heldout", action="store_true")
    args = ap.parse_args()

    paths = Paths.load()
    cells = [c.upper() for c in (args.cells or paths.cells)]
    out_dir = stage_dir(STAGE)
    per_cell = {}
    all_pass = True
    for cell in cells:
        ckpt = paths.checkpoint(cell)
        print(f"\n=== {cell}  {ckpt} ===", flush=True)
        eq = equivalence_vs_twin_trainer(cell, ckpt)
        print(
            f"  equivalence  max|ΔV|={eq['max_abs_voltage']:.3e}  "
            f"max|ΔT|={eq['max_abs_temperature']:.3e}  "
            f"{'PASS' if eq['passed'] else 'FAIL'}",
            flush=True,
        )
        hold = {"skipped": True}
        if not args.skip_heldout:
            hold = heldout_nasa(cell, ckpt, args.device)
            print(
                f"  held-out     V RMSE={hold['voltage']['rmse']:.5f} V  "
                f"T RMSE={hold['temperature']['rmse']:.4f} °C  "
                f"n={hold['n_windows']}",
                flush=True,
            )
            _plot_cell(cell, eq, hold, out_dir / f"{cell}_bdt_verification.png")
        all_pass = all_pass and bool(eq["passed"])
        per_cell[cell] = {
            "checkpoint": str(ckpt),
            "sha256": file_hash(ckpt),
            "equivalence": {k: v for k, v in eq.items() if k != "worst"} | {
                "n_worst_shown": 5,
                "worst": sorted(eq["worst"], key=lambda r: r["max_abs_v"] + r["max_abs_t"], reverse=True)[:5],
            },
            "held_out_nasa": {k: v for k, v in hold.items() if k != "example_window"},
        }

    payload = {
        "gate": "max-abs equivalence vs TwinTrainer < 1e-5",
        "tolerance": EQ_TOL,
        "passed": bool(all_pass),
        "per_cell": per_cell,
        "provenance": provenance(
            STAGE, configs=["paths"],
            inputs=[paths.checkpoint(c) for c in cells],
        ),
    }
    write_json(out_dir / "bdt_verification.json", payload)
    print(f"\nWrote {out_dir / 'bdt_verification.json'}")
    print("PASS" if all_pass else "FAIL — do not proceed to optimization")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
