"""Train SOC MLPs on measured V/T (never twin predictions for labels or default features)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from rw_transfer.data.series import BatteryTimeSeries
from rw_transfer.data.soc_labels import coulomb_soc_stitched_operational
from rw_transfer.metrics import metric_bundle
from rw_transfer.models.soc_model import SOCModel, SOC_VARIANTS


def soc_sample_indices(
    series: BatteryTimeSeries,
    stride: int = 1,
    *,
    exclude_rest: bool = True,
) -> np.ndarray:
    """Sample indices used by :func:`build_soc_arrays` (stride + optional rest drop)."""
    idx = np.arange(0, len(series.time_s), max(stride, 1), dtype=np.int64)
    if exclude_rest:
        rest = np.array(
            ["rest" in str(c).lower() for c in series.comment[idx]], dtype=bool
        )
        idx = idx[~rest]
    return idx


def build_soc_arrays(
    series: BatteryTimeSeries,
    q_rated_as: float,
    q_norm: str = "per_file",
    variant: str = "vta",
    stride: int = 1,
    *,
    exclude_rest: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row-level (N, D) features and SOC labels from **measured** V, T, age, |I|.

    Labels: per-step Coulomb counting on stitched RW operational timeline.
    """
    soc_full = coulomb_soc_stitched_operational(
        series,
        q_rated_as=q_rated_as,
        q_norm=q_norm,
    )
    idx = soc_sample_indices(series, stride, exclude_rest=exclude_rest)

    feats = SOCModel.stack_features(
        voltage=series.voltage_v[idx],
        temperature=series.temperature_c[idx],
        relative_age=series.age[idx].astype(np.float32),
        current_abs=np.abs(series.current_a[idx]),
        variant=variant,
    )
    return feats.astype(np.float32), soc_full[idx].astype(np.float32)


class SOCTrainer:
    def __init__(
        self,
        variant: str = "vta",
        hidden: int = 64,
        lr: float = 1e-3,
        device: str = "auto",
    ):
        if variant not in SOC_VARIANTS:
            raise ValueError(f"Unknown soc_variant {variant!r}; expected {list(SOC_VARIANTS)}")
        self.variant = variant
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            ("cpu" if device == "auto" else device)
        )
        self.model = SOCModel(
            input_dim=SOC_VARIANTS[variant], hidden=hidden
        ).to(self.device)
        self.lr = lr
        self.criterion = nn.MSELoss()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 1000,
        batch_size: int = 256,
        log_path: Optional[Path] = None,
        log_every: int = 50,
    ) -> Dict[str, Any]:
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).unsqueeze(1),
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        opt = Adam(self.model.parameters(), lr=self.lr)
        best = float("inf")
        best_state = None
        history: List[Dict[str, Any]] = []

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                opt.step()
                train_losses.append(loss.item())

            m = self.evaluate(X_val, y_val)
            avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
            improved = m["rmse"] < best
            if improved:
                best = m["rmse"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            marker = " ✓" if improved else ""

            row = {
                "variant": self.variant,
                "epoch": epoch,
                "train_loss": avg_train,
                "val_rmse": m["rmse"],
                "val_mape_pct": m.get("mape_pct"),
                "val_r2": m.get("r2"),
                "lr": self.lr,
            }
            history.append(row)
            if log_path:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

            if log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
                print(
                    f"\n          Epoch {epoch:>5}/{epochs}  "
                    f"train={avg_train:.5f}  val_rmse={m['rmse']:.4f}  "
                    f"MAPE={m.get('mape_pct', float('nan')):.2f}%  "
                    f"R²={m.get('r2', float('nan')):.4f}{marker}",
                    flush=True,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return {"best_val_rmse": best, "epochs_run": epochs, "history": history}

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(X), 4096):
            xb = torch.tensor(X[i : i + 4096], dtype=torch.float32, device=self.device)
            out.append(self.model(xb).cpu().numpy().ravel())
        return np.concatenate(out) if out else np.array([], dtype=np.float32)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        pred = self.predict(X)
        return metric_bundle(pred, y, soc_mape=True)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "variant": self.variant,
                "input_dim": self.model.input_dim,
                "soc_input": "measured",
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> "SOCTrainer":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        variant = ckpt.get("variant", "vta")
        trainer = cls(variant=variant, device=device)
        trainer.model.load_state_dict(ckpt["state_dict"])
        return trainer
