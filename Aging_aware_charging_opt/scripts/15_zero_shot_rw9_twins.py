#!/usr/bin/env python
"""Zero-shot generalization: two RW9-trained twins, no fine-tuning.

Models
  1. Old_Codes/simulator_cpu.pth          (legacy v9 state_dict)
  2. outputs/twin_source/.../twin_source_RW9.pt  (current source twin)

Each is evaluated on RW9–RW12 using the frozen author 60/20/20 seed-42 test
split. Weights are never updated. Writes
``results/15_zero_shot_rw9_twins/``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "Aging_aware_charging_opt" / "src"))
sys.path.insert(0, str(REPO))

from aacopt.bdt import BatteryDigitalTwin, load_twin, resolve_device  # noqa: E402
from aacopt.config import (  # noqa: E402
    Paths, ensure_matplotlib_cache, file_hash, provenance, stage_dir, write_json,
)
from aacopt.viz.style import apply_paper_style, savefig  # noqa: E402

CELLS = ("RW9", "RW10", "RW11", "RW12")
LEGACY_PTH = REPO / "Old_Codes" / "simulator_cpu.pth"
SOURCE_PT = REPO / "outputs" / "twin_source" / "20260610_111409" / "twin_source_RW9.pt"


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


def load_legacy_pth(path: Path, device: str = "auto") -> BatteryDigitalTwin:
    """Load Old_Codes/simulator_cpu.pth into the v9 BatteryDigitalTwin.

    Extra unused buffers (scaling_factors, layer_norm, conv1d) are ignored.
    They are present in the file but not used in the v9 forward pass.
    """
    dev = resolve_device(device)
    raw = torch.load(path, map_location=dev, weights_only=False)
    if isinstance(raw, dict) and "model_state" in raw:
        sd = raw["model_state"]
        seq_len = int(raw.get("seq_len", 150))
        d_model = int(raw.get("twin_d_model", 150))
        nhead = int(raw.get("twin_nhead", 20))
        n_layers = int(raw.get("twin_num_layers", 1))
        dropout = float(raw.get("twin_dropout", 0.1))
    else:
        sd = raw
        pe = sd["positional_encoding"]
        seq_len = int(pe.shape[0])
        d_model = int(pe.shape[1])
        nhead, n_layers, dropout = 20, 1, 0.1
    model = BatteryDigitalTwin(
        seq_len=seq_len, d_model=d_model, nhead=nhead,
        num_layers=n_layers, dropout=dropout, temp_delta_scale=0.1,
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unused_ok = {
        "scaling_factors_0th", "scaling_factors_1st",
        "layer_norm.weight", "layer_norm.bias",
        "layer_norm_final.weight", "layer_norm_final.bias",
        "conv1d_volt.weight", "conv1d_volt.bias",
        "conv1d_temp.weight", "conv1d_temp.bias",
    }
    unexpected_extra = [k for k in unexpected if k not in unused_ok]
    if missing:
        raise RuntimeError(f"{path}: missing weights {missing}")
    if unexpected_extra:
        raise RuntimeError(f"{path}: unexpected weights {unexpected_extra}")
    model.to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class AnyTwin:
    def __init__(self, model: BatteryDigitalTwin):
        self.model = model
        self.device = next(model.parameters()).device
        self.seq_len = model.seq_len

    @torch.no_grad()
    def predict_windows(self, X: np.ndarray, batch_size: int = 64):
        self.model.eval()
        x = np.asarray(X, dtype=np.float32)
        if x.size == 0:
            return np.empty((0, self.seq_len)), np.empty((0, self.seq_len))
        v_parts, t_parts = [], []
        for i in range(0, x.shape[0], int(batch_size)):
            xb = torch.from_numpy(x[i : i + batch_size]).to(self.device)
            v_hat, t_hat = self.model(xb[:, 0], xb[:, 1], xb[:, 2], xb[:, 3:])
            v_parts.append(v_hat.cpu().numpy())
            t_parts.append(t_hat.cpu().numpy())
        return np.concatenate(v_parts, axis=0), np.concatenate(t_parts, axis=0)


def load_test_batch(cell: str, *, all_windows: bool = False):
    from rw_transfer.data.author_dataset import (
        AuthorChunkDataset, author_subset_to_window_batch, random_split_author_dataset,
    )
    from rw_transfer.data.author_loader import load_author_stitched_series

    paths = Paths.load()
    stitched = load_author_stitched_series(str(paths.matlab_dir), cell, decimation=1)
    dataset = AuthorChunkDataset(stitched, chunk_size=150)
    if all_windows:
        from torch.utils.data import Subset
        subset = Subset(dataset, list(range(len(dataset))))
        split = "all_windows"
    else:
        _, _, subset = random_split_author_dataset(
            dataset, train_frac=0.6, val_frac=0.2, seed=42,
        )
        split = "author_random_60_20_20_seed42_test"
    batch = author_subset_to_window_batch(subset, max_windows=None)
    return batch, split, int(stitched.n_samples)


def score_cell(twin: AnyTwin, cell: str, *, all_windows: bool) -> dict:
    batch, split, n_samples = load_test_batch(cell, all_windows=all_windows)
    v_hat, t_hat = twin.predict_windows(batch.X, batch_size=64)
    return {
        "cell": cell,
        "split": split,
        "n_windows": int(batch.X.shape[0]),
        "n_samples_stitched": n_samples,
        "seq_len": int(batch.X.shape[1] - 3) if batch.X.size else 0,
        "voltage": _metrics(v_hat, batch.Y_voltage),
        "temperature": _metrics(t_hat, batch.Y_temperature),
        "example_window": {
            "v_meas": batch.Y_voltage[0].tolist() if batch.X.shape[0] else [],
            "v_pred": v_hat[0].tolist() if v_hat.shape[0] else [],
            "t_meas": batch.Y_temperature[0].tolist() if batch.X.shape[0] else [],
            "t_pred": t_hat[0].tolist() if t_hat.shape[0] else [],
        },
    }


def plot_bars(rows: List[dict], *, metric: str, ylabel: str, title: str, out: Path) -> None:
    apply_paper_style()
    import matplotlib.pyplot as plt

    models = []
    for r in rows:
        if r["model_id"] not in models:
            models.append(r["model_id"])
    x = np.arange(len(CELLS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    colors = ["#2563eb", "#ea580c"]
    for i, mid in enumerate(models):
        vals = []
        for cell in CELLS:
            hit = next(r for r in rows if r["model_id"] == mid and r["cell"] == cell and r["split_kind"] == "test")
            vals.append(hit[metric])
        ax.bar(x + (i - 0.5) * width, vals, width, label=mid, color=colors[i % 2])
    ax.set_xticks(x)
    ax.set_xticklabels(list(CELLS))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig(fig, out)


def plot_examples(payload: dict, out: Path) -> None:
    apply_paper_style()
    import matplotlib.pyplot as plt

    models = list(payload["models"].keys())
    fig, axes = plt.subplots(len(CELLS), 2, figsize=(11.2, 10.8), sharex=True)
    for r, cell in enumerate(CELLS):
        for c, (key, ylab) in enumerate((("v", "Voltage (V)"), ("t", "Temperature (°C)"))):
            ax = axes[r, c]
            drawn = False
            for mid, color, ls in zip(models, ("#2563eb", "#ea580c"), ("--", ":")):
                ex = payload["models"][mid]["per_cell"][cell]["test"]["example_window"]
                meas = np.asarray(ex["v_meas" if key == "v" else "t_meas"])
                pred = np.asarray(ex["v_pred" if key == "v" else "t_pred"])
                t = np.arange(meas.size)
                if not drawn:
                    ax.plot(t, meas, color="k", lw=1.6, label="NASA measured")
                    drawn = True
                ax.plot(t, pred, color=color, lw=1.4, ls=ls, label=mid)
            ax.set_ylabel(ylab)
            if c == 0:
                ax.set_title(cell, loc="left")
            if r == 0 and c == 1:
                ax.legend(frameon=False, fontsize=10)
    axes[-1, 0].set_xlabel("Sample in window")
    axes[-1, 1].set_xlabel("Sample in window")
    fig.suptitle("Held-out window 0 — zero-shot (no fine-tune)", y=0.995)
    fig.tight_layout()
    savefig(fig, out)


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def build_report(payload: dict) -> str:
    models = list(payload["models"].keys())
    rows_v, rows_t = [], []
    for cell in CELLS:
        rv, rt = [cell], [cell]
        for mid in models:
            m = payload["models"][mid]["per_cell"][cell]["test"]
            rv += [f"{m['voltage']['rmse']:.4f}", f"{m['voltage']['mae']:.4f}",
                   f"{m['voltage']['mape_pct']:.3f}", f"{m['voltage']['r2']:.4f}"]
            rt += [f"{m['temperature']['rmse']:.4f}", f"{m['temperature']['mae']:.4f}",
                   f"{m['temperature']['mape_pct']:.3f}", f"{m['temperature']['r2']:.4f}"]
        rows_v.append(rv)
        rows_t.append(rt)
    hdr_v = ["Cell"]
    hdr_t = ["Cell"]
    for mid in models:
        short = "legacy" if "legacy" in mid.lower() or "old" in mid.lower() or "cpu" in mid.lower() else "source"
        # keep model id compact
        tag = mid
        hdr_v += [f"{tag} V RMSE", f"{tag} V MAE", f"{tag} V MAPE %", f"{tag} V R²"]
        hdr_t += [f"{tag} T RMSE", f"{tag} T MAE", f"{tag} T MAPE %", f"{tag} T R²"]

    # transfer means RW10-12
    transfer_lines = []
    for mid in models:
        v_rmses, t_rmses = [], []
        for cell in ("RW10", "RW11", "RW12"):
            m = payload["models"][mid]["per_cell"][cell]["test"]
            v_rmses.append(m["voltage"]["rmse"])
            t_rmses.append(m["temperature"]["rmse"])
        rw9 = payload["models"][mid]["per_cell"]["RW9"]["test"]
        transfer_lines.append(
            [mid,
             f"{rw9['voltage']['rmse']:.4f}", f"{float(np.mean(v_rmses)):.4f}",
             f"{rw9['temperature']['rmse']:.4f}", f"{float(np.mean(t_rmses)):.4f}"]
        )

    winner_v = min(models, key=lambda mid: np.mean([
        payload["models"][mid]["per_cell"][c]["test"]["voltage"]["rmse"]
        for c in ("RW10", "RW11", "RW12")
    ]))
    winner_t = min(models, key=lambda mid: np.mean([
        payload["models"][mid]["per_cell"][c]["test"]["temperature"]["rmse"]
        for c in ("RW10", "RW11", "RW12")
    ]))
    winner_both = min(models, key=lambda mid: np.mean([
        payload["models"][mid]["per_cell"][c]["test"]["voltage"]["rmse"]
        + payload["models"][mid]["per_cell"][c]["test"]["temperature"]["rmse"]
        for c in ("RW10", "RW11", "RW12")
    ]))

    return f"""# Zero-shot RW9 twin comparison (no fine-tuning)

Both models were trained on **RW9 only**. Weights are frozen. Each cell uses the
same author test split (60/20/20, seed 42, chunk length 150). RW9 is in-distribution;
RW10–RW12 are **zero-shot transfer**.

## Voltage — held-out test

{_md_table(hdr_v, rows_v)}

## Temperature — held-out test

{_md_table(hdr_t, rows_t)}

## In-distribution vs transfer (mean RMSE)

{_md_table(
    ["Model", "RW9 V RMSE", "mean RW10–12 V RMSE", "RW9 T RMSE", "mean RW10–12 T RMSE"],
    transfer_lines,
)}

## Recommendation (this metric, this split)

- Lower mean **voltage** RMSE on RW10–RW12: **{winner_v}**
- Lower mean **temperature** RMSE on RW10–RW12: **{winner_t}**
- Lower mean **V+T** RMSE on RW10–RW12: **{winner_both}**

Pick the generalized model from the transfer columns (RW10–RW12), not from RW9
test error. RW9 test error only says which model fit its training cell.

If voltage error on transfer cells is the charging-constraint quantity (4.20 V),
prefer the voltage winner.
"""


def main() -> int:
    ensure_matplotlib_cache()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--legacy", default=str(LEGACY_PTH))
    ap.add_argument("--source", default=str(SOURCE_PT))
    args = ap.parse_args()

    out = stage_dir("15_zero_shot_rw9_twins", create=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    models_spec = [
        {"model_id": "legacy_cpu_pth", "path": Path(args.legacy), "kind": "legacy_pth"},
        {"model_id": "source_RW9_pt", "path": Path(args.source), "kind": "source_pt"},
    ]

    payload: Dict[str, Any] = {"models": {}, "cells": list(CELLS)}
    flat_rows: List[dict] = []

    for spec in models_spec:
        path = spec["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Loading {spec['model_id']} from {path}", flush=True)
        if spec["kind"] == "legacy_pth":
            model = load_legacy_pth(path, device=args.device)
        else:
            model = load_twin(path, device=args.device)
        twin = AnyTwin(model)
        per_cell: Dict[str, Any] = {}
        for cell in CELLS:
            print(f"  {spec['model_id']} → {cell} test …", flush=True)
            test = score_cell(twin, cell, all_windows=False)
            print(f"  {spec['model_id']} → {cell} all windows …", flush=True)
            full = score_cell(twin, cell, all_windows=True)
            per_cell[cell] = {"test": test, "all_windows": full}
            for kind, blob in (("test", test), ("all", full)):
                flat_rows.append({
                    "model_id": spec["model_id"],
                    "checkpoint": str(path),
                    "cell": cell,
                    "split_kind": kind,
                    "n_windows": blob["n_windows"],
                    "v_rmse": blob["voltage"]["rmse"],
                    "v_mae": blob["voltage"]["mae"],
                    "v_mape_pct": blob["voltage"]["mape_pct"],
                    "v_r2": blob["voltage"]["r2"],
                    "v_bias": blob["voltage"]["bias"],
                    "t_rmse": blob["temperature"]["rmse"],
                    "t_mae": blob["temperature"]["mae"],
                    "t_mape_pct": blob["temperature"]["mape_pct"],
                    "t_r2": blob["temperature"]["r2"],
                    "t_bias": blob["temperature"]["bias"],
                })
                print(
                    f"    {kind:4} V RMSE={blob['voltage']['rmse']:.4f}  "
                    f"T RMSE={blob['temperature']['rmse']:.4f}  n={blob['n_windows']}",
                    flush=True,
                )
        payload["models"][spec["model_id"]] = {
            "path": str(path),
            "sha256": file_hash(path),
            "per_cell": per_cell,
        }

    payload["provenance"] = provenance(
        "15_zero_shot_rw9_twins",
        configs=["paths"],
        inputs=[Path(s["path"]) for s in models_spec],
    )
    write_json(out / "zero_shot_metrics.json", payload)

    fields = list(flat_rows[0].keys())
    with (out / "zero_shot_metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(flat_rows)

    (out / "REPORT.md").write_text(build_report(payload), encoding="utf-8")

    plot_bars(flat_rows, metric="v_rmse", ylabel="Voltage RMSE (V)",
              title="Zero-shot voltage RMSE (held-out test, no fine-tune)",
              out=fig_dir / "voltage_rmse.png")
    plot_bars(flat_rows, metric="t_rmse", ylabel="Temperature RMSE (°C)",
              title="Zero-shot temperature RMSE (held-out test, no fine-tune)",
              out=fig_dir / "temperature_rmse.png")
    plot_examples(payload, fig_dir / "example_windows.png")

    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
