"""
FinetuneRegistry
────────────────
Tracks every fine-tuned checkpoint with timing, size, and full metrics.

The registry JSON lives at  <out_dir>/registry/finetune_registry.json.
Each entry is keyed by  "<target>_frac<fraction>"  and records:

  • when the run completed
  • source checkpoint used
  • data split (n_adapt, n_eval windows)
  • wall-clock fine-tune time, model size, param count, inference latency
  • full metric bundle for voltage and temperature on the test set:
      MSE, RMSE, MAE, MAPE%, R²

Usage (inside twin_finetune_percent.py)
---------------------------------------
    from rw_transfer.registry import FinetuneRegistry

    reg = FinetuneRegistry(out_dir / "registry", source_ckpt)
    reg.register_fraction(
        target="RW10", fraction=0.20,
        n_adapt=6877, n_eval=11461,
        train_time_s=258.4,
        model_size_mb=7.655, n_params=2_004_462, infer_ms=1.13,
        voltage_metrics={"mse": ..., "rmse": ..., "mae": ..., "mape_pct": ..., "r2": ...},
        temperature_metrics={...},
        ckpt_path=Path("registry/finetune_RW10_frac0.20.pt"),
    )
    reg.save()
    reg.print_summary()
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ── Inference timing helper ───────────────────────────────────────────────────

def measure_infer_ms(
    model: torch.nn.Module,
    device: torch.device,
    seq_len: int = 150,
    n_warmup: int = 10,
    n_timed: int = 100,
) -> float:
    """Average forward-pass time (ms) over *n_timed* single-chunk calls.

    Uses ``forward_author(starting_state, actions)`` which matches fine-tuning.
    Shapes match ``AuthorChunkDataset.__getitem__`` after DataLoader batching:
      starting_state : (1, 3)        — [age, v0, t0] scalars
      actions        : (1, seq_len, 1) — current sequence, last dim = 1
    """
    model.eval()
    state  = torch.zeros(1, 3, device=device)
    action = torch.zeros(1, seq_len, 1, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model.forward_author(state, action)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_timed):
            model.forward_author(state, action)
    if device.type == "cuda":
        torch.cuda.synchronize()

    return round((time.perf_counter() - t0) / n_timed * 1000, 3)


def file_size_mb(path: Path) -> float:
    path = Path(path)
    return round(path.stat().st_size / 1024 / 1024, 3) if path.exists() else 0.0


# ── FinetuneRegistry ──────────────────────────────────────────────────────────

class FinetuneRegistry:
    """Persistent per-fraction registry for fine-tuned twin checkpoints."""

    _REGISTRY_FILE = "finetune_registry.json"

    def __init__(self, registry_dir: Path, source_ckpt: Path):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.source_ckpt = Path(source_ckpt)
        self._path = self.registry_dir / self._REGISTRY_FILE
        self._data: Dict[str, Any] = self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            with self._path.open(encoding="utf-8") as f:
                return json.load(f)
        return {"source_ckpt": str(self.source_ckpt), "entries": {}}

    def save(self) -> None:
        self._data["source_ckpt"] = str(self.source_ckpt)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)

    # ── registration ─────────────────────────────────────────────────────────

    def register_fraction(
        self,
        *,
        target: str,
        fraction: float,
        n_adapt: int,
        n_eval: int,
        train_time_s: float,
        model_size_mb: float,
        n_params: int,
        infer_ms: float,
        voltage_metrics: Dict[str, Any],
        temperature_metrics: Dict[str, Any],
        ckpt_path: Optional[Path] = None,
        stage1_epochs_run: int = 0,
        stage2_epochs_run: int = 0,
    ) -> None:
        key = f"{target}_frac{fraction:.2f}"
        self._data.setdefault("entries", {})[key] = {
            "completed_at": datetime.now(tz=timezone.utc).isoformat(),
            "target": target,
            "fraction": fraction,
            "n_adapt_windows": n_adapt,
            "n_eval_windows": n_eval,
            "train_time_s": round(train_time_s, 1),
            "train_time_human": _fmt_time(train_time_s),
            "model_size_mb": model_size_mb,
            "n_params": n_params,
            "infer_ms": infer_ms,
            "stage1_epochs_run": stage1_epochs_run,
            "stage2_epochs_run": stage2_epochs_run,
            "checkpoint": str(ckpt_path) if ckpt_path else None,
            "voltage": voltage_metrics,
            "temperature": temperature_metrics,
        }

    # ── summary printing ─────────────────────────────────────────────────────

    def print_summary(self) -> None:
        entries = self._data.get("entries", {})
        if not entries:
            print("  [FinetuneRegistry] No entries yet.")
            return

        W = 110
        print()
        print("  " + "─" * W)
        print(f"  Finetune Registry  —  source: {self.source_ckpt}")
        print(f"  Location: {self._path}")
        print("  " + "─" * W)
        hdr = (
            f"  {'Target':<6} {'Frac':>5} {'Adapt':>8} {'Time':>9} "
            f"{'Params':>10} {'MB':>7} {'Infer':>8}  "
            f"{'V-RMSE':>8} {'V-MAPE%':>8} {'V-R²':>7}  "
            f"{'T-RMSE':>8} {'T-MAPE%':>8} {'T-R²':>7}"
        )
        print(hdr)
        print("  " + "─" * W)

        for key in sorted(entries):
            e = entries[key]
            v = e.get("voltage", {})
            t = e.get("temperature", {})
            print(
                f"  {e['target']:<6} {e['fraction']:>4.0%} {e['n_adapt_windows']:>8,} "
                f"{e['train_time_human']:>9} "
                f"{e['n_params']:>10,} {e['model_size_mb']:>7.3f} {e['infer_ms']:>7.3f}ms  "
                f"{v.get('rmse', float('nan')):>8.5f} {v.get('mape_pct', float('nan')):>8.3f} "
                f"{v.get('r2', float('nan')):>7.4f}  "
                f"{t.get('rmse', float('nan')):>8.4f} {t.get('mape_pct', float('nan')):>8.3f} "
                f"{t.get('r2', float('nan')):>7.4f}"
            )
        print("  " + "─" * W)
        print()

    def get_entry(self, target: str, fraction: float) -> Optional[Dict[str, Any]]:
        return self._data.get("entries", {}).get(f"{target}_frac{fraction:.2f}")

    def all_rows(self) -> List[Dict[str, Any]]:
        return list(self._data.get("entries", {}).values())


# ── Source model registry (written once after RW9 training) ──────────────────

class SourceModelRegistry:
    """
    Lightweight registry for the source (RW9) twin checkpoint.

    Written to  <run_dir>/source_registry.json  by  scripts/build_source_registry.py
    — no retraining required; reads from the existing .pt file and JSONL logs.
    """

    _FILE = "source_registry.json"

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._path = self.run_dir / self._FILE
        self._data: Dict[str, Any] = {}

    def build_from_checkpoint(
        self,
        ckpt_path: Path,
        device: torch.device,
        *,
        seq_len: int = 150,
        test_voltage_metrics: Optional[Dict[str, Any]] = None,
        test_temp_metrics: Optional[Dict[str, Any]] = None,
        train_log_path: Optional[Path] = None,
    ) -> None:
        """Populate registry by reading the saved checkpoint — no retraining."""
        ckpt_path = Path(ckpt_path)

        raw = torch.load(ckpt_path, map_location=device, weights_only=False)
        from rw_transfer.models.digital_twin import BatteryDigitalTwin
        model = BatteryDigitalTwin(
            seq_len=int(raw.get("seq_len", seq_len)),
            d_model=int(raw.get("twin_d_model", 150)),
            nhead=int(raw.get("twin_nhead", 20)),
            num_layers=int(raw.get("twin_num_layers", 1)),
            dropout=float(raw.get("twin_dropout", 0.1)),
            temp_delta_scale=float(raw.get("temp_delta_scale", 0.1)),
        ).to(device)
        model.load_state_dict(raw["model_state"])

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        size_mb  = file_size_mb(ckpt_path)
        infer_ms = measure_infer_ms(model, device, seq_len=int(raw.get("seq_len", seq_len)))

        train_time_s: Optional[float] = None
        best_mape_v = best_mape_t = None
        if train_log_path and Path(train_log_path).exists():
            rows = []
            with Path(train_log_path).open(encoding="utf-8") as f:
                for line in f:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            if rows:
                best_row = min(rows, key=lambda r: r.get("val_loss", float("inf")))
                best_mape_v = best_row.get("mape_v")
                best_mape_t = best_row.get("mape_t")

        self._data = {
            "checkpoint": str(ckpt_path),
            "profiled_at": datetime.now(tz=timezone.utc).isoformat(),
            "n_params": n_params,
            "model_size_mb": size_mb,
            "infer_ms_per_chunk": infer_ms,
            "seq_len": int(raw.get("seq_len", seq_len)),
            "train_time_s": train_time_s,
            "best_val_mape_v_pct": best_mape_v,
            "best_val_mape_t_pct": best_mape_t,
            "test_voltage": test_voltage_metrics or {},
            "test_temperature": test_temp_metrics or {},
        }

    def save(self) -> None:
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, default=str)
        print(f"  Source registry saved → {self._path}")

    def print_summary(self) -> None:
        d = self._data
        if not d:
            print("  [SourceModelRegistry] Not built yet.")
            return
        W = 70
        print()
        print("  " + "─" * W)
        print("  Source Model Registry")
        print(f"  Checkpoint : {d.get('checkpoint')}")
        print("  " + "─" * W)
        print(f"  Params     : {d.get('n_params', 0):,}")
        print(f"  Size       : {d.get('model_size_mb', 0):.3f} MB")
        print(f"  Infer time : {d.get('infer_ms_per_chunk', 0):.3f} ms / chunk ({d.get('seq_len', 150)} steps)")
        v = d.get("test_voltage", {})
        t = d.get("test_temperature", {})
        if v:
            print(f"  Test V     : RMSE={v.get('rmse','?')}  MAPE={v.get('mape_pct','?')}%  R²={v.get('r2','?')}")
        if t:
            print(f"  Test T     : RMSE={t.get('rmse','?')}  MAPE={t.get('mape_pct','?')}%  R²={t.get('r2','?')}")
        if d.get("best_val_mape_v_pct") is not None:
            print(f"  Best val   : MAPE_V={d['best_val_mape_v_pct']:.3f}%  MAPE_T={d['best_val_mape_t_pct']:.3f}%")
        print("  " + "─" * W)
        print()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    """Human-readable duration: '4.3 min', '1.2 h', '45 s'."""
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"
