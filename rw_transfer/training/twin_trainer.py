"""Train and fine-tune the v9 digital twin."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

from rw_transfer.data.windows import TwinWindowDataset, WindowBatch
from rw_transfer.metrics import twin_metrics
from rw_transfer.models.digital_twin import BatteryDigitalTwin
from rw_transfer.training.losses import (
    author_mape_pct,
    author_train_loss,
    author_val_loss,
    temp_aware_finetune_loss,
    twin_training_loss,
)

EarlyStopMetric = Literal["val_loss", "mape_sum", "mape_max"]


def _device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _smooth_temp_windows(Y: np.ndarray, seq_len: int) -> np.ndarray:
    """Legacy per-window Savitzky–Golay (slow; prefer series-level smoothing)."""
    wl = seq_len - 1 if seq_len % 2 == 0 else seq_len
    wl = max(wl, 5)
    return np.array(
        [savgol_filter(r, window_length=wl, polyorder=2) for r in Y],
        dtype=np.float32,
    )


def _mape_pct(pred: np.ndarray, ref: np.ndarray, eps: float) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(np.mean(np.abs(pred - ref) / (np.abs(ref) + eps)) * 100.0)


@torch.no_grad()
def predict_twin_batch(
    model: BatteryDigitalTwin,
    batch: WindowBatch,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    v_parts, t_parts = [], []
    n = len(batch.X)
    for i in range(0, n, batch_size):
        xb = batch.X[i : i + batch_size]
        age = torch.tensor(xb[:, 0], device=device, dtype=torch.float32)
        v0 = torch.tensor(xb[:, 1], device=device, dtype=torch.float32)
        t0 = torch.tensor(xb[:, 2], device=device, dtype=torch.float32)
        curr = torch.tensor(xb[:, 3:], device=device, dtype=torch.float32)
        v_hat, t_hat = model(age, v0, t0, curr)
        v_parts.append(v_hat.cpu().numpy())
        t_parts.append(t_hat.cpu().numpy())
    if not v_parts:
        seq = batch.Y_voltage.shape[1] if len(batch.Y_voltage.shape) > 1 else 0
        return np.empty((0, seq)), np.empty((0, seq))
    return np.concatenate(v_parts, axis=0), np.concatenate(t_parts, axis=0)


def evaluate_twin_windows(
    model: BatteryDigitalTwin,
    batch: WindowBatch,
    device: torch.device,
    *,
    temperature_ref: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    if len(batch.X) == 0:
        return {"voltage": {}, "temperature": {}, "voltage_rmse": float("nan"),
                "temperature_rmse": float("nan")}
    v_pred, t_pred = predict_twin_batch(model, batch, device)
    t_ref = batch.Y_temperature if temperature_ref is None else temperature_ref
    return twin_metrics(
        v_pred.ravel(), batch.Y_voltage.ravel(),
        t_pred.ravel(), t_ref.ravel(),
    )


def trainer_from_twin_config(twin_cfg: Dict[str, Any], seq_len: int, device: str = "auto") -> "TwinTrainer":
    """Build ``TwinTrainer`` from the ``twin:`` section of YAML config."""
    pipeline = str(twin_cfg.get("pipeline", "author"))
    loss = twin_cfg.get("loss", {})
    if pipeline == "author":
        return TwinTrainer(
            seq_len=seq_len,
            d_model=int(twin_cfg["d_model"]),
            nhead=int(twin_cfg["nhead"]),
            num_layers=int(twin_cfg["num_layers"]),
            dropout=float(twin_cfg["dropout"]),
            temp_delta_scale=float(twin_cfg.get("temp_delta_scale", 0.1)),
            mse_v_weight=float(twin_cfg.get("voltage_weight", 100.0)),
            mse_t_weight=float(twin_cfg.get("temp_weight", 1.0)),
            mape_v_weight=0.0,
            mape_t_weight=0.0,
            corr_t_weight=0.0,
            lr=float(twin_cfg["lr"]),
            early_stop_metric="val_loss",
            optimizer=str(twin_cfg.get("optimizer", "adam")),
            author_style=True,
            device=device,
        )
    return TwinTrainer(
        seq_len=seq_len,
        d_model=int(twin_cfg["d_model"]),
        nhead=int(twin_cfg["nhead"]),
        num_layers=int(twin_cfg["num_layers"]),
        dropout=float(twin_cfg["dropout"]),
        temp_delta_scale=float(twin_cfg["temp_delta_scale"]),
        mse_v_weight=float(loss.get("mse_voltage", twin_cfg.get("voltage_weight", 10.0))),
        mse_t_weight=float(loss.get("mse_temp", twin_cfg.get("temp_weight", 10.0))),
        mape_v_weight=float(loss.get("mape_voltage", 100.0)),
        mape_t_weight=float(loss.get("mape_temp", 100.0)),
        corr_t_weight=float(loss.get("corr_temp", twin_cfg.get("temp_corr_weight", 5.0))),
        mape_eps_v=float(twin_cfg.get("mape_eps_voltage", 0.02)),
        mape_eps_t=float(twin_cfg.get("mape_eps_temp", 0.2)),
        grad_clip=float(twin_cfg.get("grad_clip", 1.0)),
        lr=float(twin_cfg["lr"]),
        early_stop_metric=str(twin_cfg.get("early_stop_metric", "mape_sum")),
        optimizer=str(twin_cfg.get("optimizer", "adamw")),
        author_style=bool(twin_cfg.get("author_style", False)),
        device=device,
    )


class TwinTrainer:
    """Wraps BatteryDigitalTwin training and fine-tuning."""

    def __init__(
        self,
        seq_len: int = 150,
        d_model: int = 150,
        nhead: int = 20,
        num_layers: int = 1,
        dropout: float = 0.1,
        temp_delta_scale: float = 1.0,
        mse_v_weight: float = 10.0,
        mse_t_weight: float = 10.0,
        mape_v_weight: float = 100.0,
        mape_t_weight: float = 100.0,
        corr_t_weight: float = 5.0,
        mape_eps_v: float = 0.02,
        mape_eps_t: float = 0.2,
        grad_clip: float = 1.0,
        lr: float = 3e-4,
        early_stop_metric: str = "mape_sum",
        optimizer: str = "adamw",
        author_style: bool = True,
        device: str = "auto",
        # Backward compatibility with old kw names
        voltage_weight: Optional[float] = None,
        temp_weight: Optional[float] = None,
        temp_corr_weight: Optional[float] = None,
    ):
        if voltage_weight is not None:
            mse_v_weight = voltage_weight
        if temp_weight is not None:
            mse_t_weight = temp_weight
        if temp_corr_weight is not None:
            corr_t_weight = temp_corr_weight

        self.seq_len = seq_len
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.temp_delta_scale = temp_delta_scale
        self.mse_v_weight = mse_v_weight
        self.mse_t_weight = mse_t_weight
        self.mape_v_weight = mape_v_weight
        self.mape_t_weight = mape_t_weight
        self.corr_t_weight = corr_t_weight
        self.mape_eps_v = mape_eps_v
        self.mape_eps_t = mape_eps_t
        self.grad_clip = grad_clip
        self.lr = lr
        self.early_stop_metric: EarlyStopMetric = (
            early_stop_metric if early_stop_metric in ("val_loss", "mape_sum", "mape_max")
            else "mape_sum"
        )
        self.optimizer_name = optimizer.lower()
        self.device = _device(device)
        self.model = BatteryDigitalTwin(
            seq_len=seq_len,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            temp_delta_scale=temp_delta_scale,
            author_style=author_style,
        ).to(self.device)

    def _make_optimizer(self, params):
        if self.optimizer_name == "adam":
            return Adam(params, lr=self.lr)
        return AdamW(params, lr=self.lr, weight_decay=1e-4)

    def _compute_loss(
        self,
        v_hat: torch.Tensor,
        t_hat: torch.Tensor,
        yv: torch.Tensor,
        yt: torch.Tensor,
    ) -> torch.Tensor:
        return twin_training_loss(
            v_hat, t_hat, yv, yt,
            mse_v_w=self.mse_v_weight,
            mse_t_w=self.mse_t_weight,
            mape_v_w=self.mape_v_weight,
            mape_t_w=self.mape_t_weight,
            corr_t_w=self.corr_t_weight,
            mape_eps_v=self.mape_eps_v,
            mape_eps_t=self.mape_eps_t,
        )

    def _stop_score(self, avg_val: float, mape_v: float, mape_t: float) -> float:
        if self.early_stop_metric == "val_loss":
            return avg_val
        if self.early_stop_metric == "mape_max":
            return max(mape_v, mape_t)
        return mape_v + mape_t

    def fit(
        self,
        train: WindowBatch,
        val: WindowBatch,
        epochs: int = 200,
        batch_size: int = 64,
        early_stop_patience: int = 30,
        plateau_patience: int = 20,
        smooth_temp_targets: bool = False,
        log_path: Optional[Path] = None,
        num_workers: int = 0,
    ) -> Dict[str, Any]:
        if smooth_temp_targets and len(train.Y_temperature) > 0:
            seq_len = train.Y_temperature.shape[1]
            Yt_train = _smooth_temp_windows(train.Y_temperature, seq_len)
            Yt_val = _smooth_temp_windows(val.Y_temperature, seq_len)
            print(f"        Temp targets: per-window Savitzky-Golay (window≈{seq_len})", flush=True)
        else:
            Yt_train = train.Y_temperature
            Yt_val = val.Y_temperature

        train_ds = TwinWindowDataset(train.X, train.Y_voltage, Yt_train)
        val_ds = TwinWindowDataset(val.X, val.Y_voltage, Yt_val)
        pin = self.device.type == "cuda"
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
        )

        opt = self._make_optimizer(self.model.parameters())
        sch = CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=self.lr / 100)
        plateau_sch = ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=plateau_patience, min_lr=self.lr / 1000,
        )

        best_score = float("inf")
        best_state = None
        stale = 0
        history = []
        LOG_EVERY = 10

        print(f"        Device: {self.device}  |  "
              f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}", flush=True)
        print(
            f"        Loss = {self.mse_v_weight:g}·MSE_V + {self.mape_v_weight:g}·MAPE_V"
            f"  +  {self.mse_t_weight:g}·MSE_T + {self.mape_t_weight:g}·MAPE_T"
            f"  +  {self.corr_t_weight:g}·(1-PearsonR_T)   early_stop={self.early_stop_metric}",
            flush=True,
        )

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for age, v0, t0, curr, yv, yt in train_loader:
                age, v0, t0 = age.to(self.device), v0.to(self.device), t0.to(self.device)
                curr, yv, yt = curr.to(self.device), yv.to(self.device), yt.to(self.device)
                opt.zero_grad()
                v_hat, t_hat = self.model(age, v0, t0, curr)
                loss = self._compute_loss(v_hat, t_hat, yv, yt)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses, v_preds, v_tgts, t_preds, t_tgts = [], [], [], [], []
            with torch.no_grad():
                for age, v0, t0, curr, yv, yt in val_loader:
                    age, v0, t0 = age.to(self.device), v0.to(self.device), t0.to(self.device)
                    curr, yv, yt = curr.to(self.device), yv.to(self.device), yt.to(self.device)
                    v_hat, t_hat = self.model(age, v0, t0, curr)
                    val_losses.append(self._compute_loss(v_hat, t_hat, yv, yt).item())
                    v_preds.append(v_hat.cpu().numpy())
                    v_tgts.append(yv.cpu().numpy())
                    t_preds.append(t_hat.cpu().numpy())
                    t_tgts.append(yt.cpu().numpy())

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_losses))
            current_lr = float(opt.param_groups[0]["lr"])

            sch.step()
            plateau_sch.step(avg_val)

            vp = np.concatenate(v_preds).ravel()
            vt = np.concatenate(v_tgts).ravel()
            tp = np.concatenate(t_preds).ravel()
            tt = np.concatenate(t_tgts).ravel()
            mape_v = _mape_pct(vp, vt, self.mape_eps_v)
            mape_t = _mape_pct(tp, tt, self.mape_eps_t)
            val_v_rmse = float(np.sqrt(np.mean((vp - vt) ** 2)))
            val_t_rmse = float(np.sqrt(np.mean((tp - tt) ** 2)))
            stop_score = self._stop_score(avg_val, mape_v, mape_t)

            improved = stop_score < best_score
            if improved:
                best_score = stop_score
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
                marker = " ✓"
            else:
                stale += 1
                marker = f"  (no improve {stale}/{early_stop_patience})"
                if early_stop_patience and stale >= early_stop_patience:
                    print(f"        Early stop at epoch {epoch}", flush=True)
                    break

            row = {
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "val_voltage_rmse": val_v_rmse,
                "val_temp_rmse": val_t_rmse,
                "mape_v": mape_v,
                "mape_t": mape_t,
                "stop_score": stop_score,
                "lr": current_lr,
            }
            history.append(row)
            if log_path:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

            if epoch % LOG_EVERY == 0 or epoch == 1:
                print(
                    f"        Epoch {epoch:>5}/{epochs}  "
                    f"train={avg_train:.5f}  val={avg_val:.5f}  "
                    f"lr={current_lr:.2e}  "
                    f"MAPE_V={mape_v:.3f}%  MAPE_T={mape_t:.3f}%"
                    f"{marker}",
                    flush=True,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        best_row = min(history, key=lambda r: r["stop_score"]) if history else {}
        return {
            "best_val_voltage_rmse": best_row.get("val_voltage_rmse", float("nan")),
            "best_val_temp_rmse": best_row.get("val_temp_rmse", float("nan")),
            "best_val_mape_v": best_row.get("mape_v", float("nan")),
            "best_val_mape_t": best_row.get("mape_t", float("nan")),
            "best_stop_score": best_score,
            "epochs_run": len(history),
            "history": history,
        }

    def fit_two_stage_author(
        self,
        train_set: Subset,
        val_set: Subset,
        *,
        stage1_epochs: int = 150,
        stage1_voltage_weight: float = 1.0,
        stage1_temp_weight: float = 100.0,
        stage1_pearson_weight: float = 5.0,
        stage1_lr: Optional[float] = None,
        stage2_epochs: int = 500,
        stage2_voltage_weight: float = 10.0,
        stage2_temp_weight: float = 50.0,
        stage2_pearson_weight: float = 5.0,
        batch_size: int = 128,
        early_stop_patience: int = 50,
        plateau_patience: int = 3,
        plateau_factor: float = 0.1,
        log_path: Optional[Path] = None,
        num_workers: int = 0,
    ) -> Dict[str, Any]:
        """
        Two-stage temperature-aware fine-tuning.

        **Stage 1 — Output head warmup (backbone frozen):**
          Freezes the transformer backbone and all input embeddings; only the
          three output projection layers (``linear_out1/2/3``) are trained.
          Uses a temperature-biased loss so the head quickly learns to map the
          frozen representation to the target cell's temperature scale.
          A higher LR is safe here since the backbone is frozen.

        **Stage 2 — Full fine-tuning (all layers, balanced loss):**
          Unfreezes everything and trains with a balanced voltage+temperature
          loss plus a Pearson correlation term on temperature for shape matching.
          Uses the caller's ``self.lr`` (typically the fine-tune LR, e.g. 5e-7).

        Returns a dict with ``stage1`` and ``stage2`` sub-dicts and combined
        ``best_val_voltage_rmse`` / ``best_val_temp_rmse`` from stage 2.
        """
        # ── Stage 1: backbone frozen, output head only ─────────────────────
        print(f"\n        ── Stage 1: output-head warmup  "
              f"(backbone frozen, {self.model.n_trainable_params:,} → ", end="", flush=True)
        self.model.freeze_backbone()
        print(f"{self.model.n_trainable_params:,} trainable params)  "
              f"epochs={stage1_epochs}", flush=True)

        s1_lr = stage1_lr if stage1_lr is not None else self.lr * 1000
        orig_lr = self.lr
        self.lr = s1_lr

        s1_log = Path(str(log_path).replace(".jsonl", "_stage1.jsonl")) if log_path else None
        stage1_info = self.fit_author(
            train_set, val_set,
            epochs=stage1_epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience,
            plateau_patience=plateau_patience,
            plateau_factor=plateau_factor,
            log_path=s1_log,
            num_workers=num_workers,
            voltage_weight=stage1_voltage_weight,
            temp_weight=stage1_temp_weight,
            pearson_temp_weight=stage1_pearson_weight,
        )
        self.lr = orig_lr

        # ── Stage 2: full fine-tuning, balanced loss ───────────────────────
        print(f"\n        ── Stage 2: full fine-tune  "
              f"(all {self.model.n_trainable_params:,} params unfrozen)  "
              f"epochs={stage2_epochs}", flush=True)
        self.model.unfreeze_all()

        s2_log = Path(str(log_path).replace(".jsonl", "_stage2.jsonl")) if log_path else None
        stage2_info = self.fit_author(
            train_set, val_set,
            epochs=stage2_epochs,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience,
            plateau_patience=plateau_patience,
            plateau_factor=plateau_factor,
            log_path=s2_log,
            num_workers=num_workers,
            voltage_weight=stage2_voltage_weight,
            temp_weight=stage2_temp_weight,
            pearson_temp_weight=stage2_pearson_weight,
        )

        return {
            "stage1": stage1_info,
            "stage2": stage2_info,
            "best_val_voltage_rmse": stage2_info.get("best_val_voltage_rmse", float("nan")),
            "best_val_temp_rmse": stage2_info.get("best_val_temp_rmse", float("nan")),
            "best_val_mape_v": stage2_info.get("best_val_mape_v", float("nan")),
            "best_val_mape_t": stage2_info.get("best_val_mape_t", float("nan")),
            "best_val_loss": stage2_info.get("best_val_loss", float("nan")),
            "epochs_run": stage1_info.get("epochs_run", 0) + stage2_info.get("epochs_run", 0),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "seq_len": self.seq_len,
                "twin_d_model": self.d_model,
                "twin_nhead": self.nhead,
                "twin_num_layers": self.num_layers,
                "twin_dropout": self.dropout,
                "temp_delta_scale": self.temp_delta_scale,
                "corr_t_weight": self.corr_t_weight,
                "grad_clip": self.grad_clip,
                "early_stop_metric": self.early_stop_metric,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "auto", **kwargs) -> "TwinTrainer":
        dev = _device(device)
        ckpt = torch.load(path, map_location=dev, weights_only=False)
        extra = dict(kwargs)
        seq_len = int(extra.pop("seq_len", ckpt.get("seq_len", 150)))
        trainer = cls(
            seq_len=seq_len,
            d_model=int(ckpt.get("twin_d_model", 150)),
            nhead=int(ckpt.get("twin_nhead", 20)),
            num_layers=int(ckpt.get("twin_num_layers", 1)),
            dropout=float(ckpt.get("twin_dropout", 0.1)),
            temp_delta_scale=float(ckpt.get("temp_delta_scale", 0.1)),
            device=device,
            **extra,
        )
        trainer.model.load_state_dict(ckpt["model_state"])
        return trainer

    def fit_author(
        self,
        train_set: Subset,
        val_set: Subset,
        epochs: int = 10000,
        batch_size: int = 128,
        early_stop_patience: int = 50,
        plateau_patience: int = 3,
        plateau_factor: float = 0.1,
        log_path: Optional[Path] = None,
        num_workers: int = 0,
        voltage_weight: Optional[float] = None,
        temp_weight: Optional[float] = None,
        pearson_temp_weight: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Train with the original author's loop:
          * ``voltage_weight·MSE_V + temp_weight·MSE_T`` train loss
            (defaults: 100 / 1 — the author recipe; override for fine-tuning)
          * Optional ``pearson_temp_weight·(1 - PearsonR_T)`` shape term
          * plain MSE validation loss
          * ``ReduceLROnPlateau`` (factor=0.1, patience=3)
          * checkpoint on best val loss
        """
        v_w = voltage_weight if voltage_weight is not None else self.mse_v_weight
        t_w = temp_weight if temp_weight is not None else self.mse_t_weight
        use_temp_aware = (
            (voltage_weight is not None or temp_weight is not None)
            or pearson_temp_weight > 0.0
        )

        pin = self.device.type == "cuda"
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin,
        )
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin,
        )

        opt = self._make_optimizer(
            [p for p in self.model.parameters() if p.requires_grad]
        )
        plateau_sch = ReduceLROnPlateau(
            opt, mode="min", factor=plateau_factor, patience=plateau_patience,
        )

        best_val = float("inf")
        best_state = None
        stale = 0
        history = []
        LOG_EVERY = 10

        print(f"        Device: {self.device}  |  Author pipeline", flush=True)
        print(f"        Train chunks: {len(train_set)}  Val chunks: {len(val_set)}", flush=True)
        print(
            f"        Trainable params: {self.model.n_trainable_params:,}",
            flush=True,
        )
        if use_temp_aware:
            print(
                f"        Loss = {v_w:g}·MSE_V + {t_w:g}·MSE_T"
                + (f" + {pearson_temp_weight:g}·(1-PearsonR_T)" if pearson_temp_weight > 0 else "")
                + "  [temp-aware]  |  val = MSE(V,T)  |  early_stop=val_loss",
                flush=True,
            )
        else:
            print(
                f"        Loss = {v_w:g}·MSE_V + {t_w:g}·MSE_T"
                f"  |  val = MSE(V,T)  |  early_stop=val_loss",
                flush=True,
            )

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            for state, action, next_state in train_loader:
                state = state.to(self.device)
                action = action.to(self.device)
                next_state = next_state.to(self.device)
                opt.zero_grad()
                output = self.model.forward_author(state, action)
                yv, yt = next_state[:, :, 0], next_state[:, :, 1]
                v_hat, t_hat = output[:, :, 0], output[:, :, 1]
                if use_temp_aware:
                    loss = temp_aware_finetune_loss(
                        v_hat, t_hat, yv, yt,
                        voltage_weight=v_w,
                        temp_weight=t_w,
                        pearson_weight=pearson_temp_weight,
                    )
                else:
                    loss = author_train_loss(
                        v_hat, t_hat, yv, yt, voltage_weight=v_w,
                    )
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                opt.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses = []
            v_preds, v_tgts = [], []
            with torch.no_grad():
                for state, action, next_state in val_loader:
                    state = state.to(self.device)
                    action = action.to(self.device)
                    next_state = next_state.to(self.device)
                    output = self.model.forward_author(state, action)
                    yv, yt = next_state[:, :, 0], next_state[:, :, 1]
                    v_hat, t_hat = output[:, :, 0], output[:, :, 1]
                    # When training with a temperature-biased loss, use the same
                    # weights for validation so early stopping is aligned with what
                    # the model is actually optimising.  The plain author_val_loss
                    # (unweighted MSE) would see voltage degradation as bad even when
                    # Stage 1 is intentionally trading voltage for temperature gain,
                    # causing early stopping to fire at epoch 1.
                    if use_temp_aware:
                        vl = (
                            v_w * torch.nn.functional.mse_loss(v_hat, yv)
                            + t_w * torch.nn.functional.mse_loss(t_hat, yt)
                        ).item()
                    else:
                        vl = author_val_loss(v_hat, t_hat, yv, yt).item()
                    val_losses.append(vl)
                    v_preds.append(output.cpu())
                    v_tgts.append(next_state.cpu())

            avg_train = float(np.mean(train_losses))
            avg_val = float(np.mean(val_losses))
            current_lr = float(opt.param_groups[0]["lr"])
            plateau_sch.step(avg_val)

            if v_preds:
                pred = torch.cat(v_preds, dim=0)
                tgt = torch.cat(v_tgts, dim=0)
                mape_v, mape_t = author_mape_pct(pred, tgt)
                vp = pred[:, :, 0].numpy().ravel()
                vt = tgt[:, :, 0].numpy().ravel()
                tp = pred[:, :, 1].numpy().ravel()
                tt = tgt[:, :, 1].numpy().ravel()
                val_v_rmse = float(np.sqrt(np.mean((vp - vt) ** 2)))
                val_t_rmse = float(np.sqrt(np.mean((tp - tt) ** 2)))
                val_v_mse  = float(np.mean((vp - vt) ** 2))
                val_t_mse  = float(np.mean((tp - tt) ** 2))
                val_v_mae  = float(np.mean(np.abs(vp - vt)))
                val_t_mae  = float(np.mean(np.abs(tp - tt)))
                _vss = float(np.sum((vt - np.mean(vt)) ** 2))
                _tss = float(np.sum((tt - np.mean(tt)) ** 2))
                val_v_r2 = float(1 - np.sum((vp - vt) ** 2) / _vss) if _vss > 1e-12 else float("nan")
                val_t_r2 = float(1 - np.sum((tp - tt) ** 2) / _tss) if _tss > 1e-12 else float("nan")
            else:
                mape_v = mape_t = float("nan")
                val_v_rmse = val_t_rmse = float("nan")
                val_v_mse = val_t_mse = float("nan")
                val_v_mae = val_t_mae = float("nan")
                val_v_r2 = val_t_r2 = float("nan")

            improved = avg_val < best_val
            if improved:
                best_val = avg_val
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                stale = 0
                marker = " ✓"
            else:
                stale += 1
                marker = f"  (no improve {stale}/{early_stop_patience})"
                if early_stop_patience and stale >= early_stop_patience:
                    print(f"        Early stop at epoch {epoch}", flush=True)
                    break

            row = {
                "epoch": epoch,
                "train_loss": avg_train,
                "val_loss": avg_val,
                "val_voltage_mse": val_v_mse,
                "val_voltage_rmse": val_v_rmse,
                "val_voltage_mae": val_v_mae,
                "val_voltage_r2": val_v_r2,
                "val_temp_mse": val_t_mse,
                "val_temp_rmse": val_t_rmse,
                "val_temp_mae": val_t_mae,
                "val_temp_r2": val_t_r2,
                "mape_v": mape_v,
                "mape_t": mape_t,
                "stop_score": avg_val,
                "lr": current_lr,
            }
            history.append(row)
            if log_path:
                with Path(log_path).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

            if epoch % LOG_EVERY == 0 or epoch == 1:
                print(
                    f"        Epoch {epoch:>5}/{epochs}  "
                    f"train={avg_train:.5f}  val={avg_val:.5f}  "
                    f"lr={current_lr:.2e}  "
                    f"MAPE_V={mape_v:.3f}%  MAPE_T={mape_t:.3f}%"
                    f"{marker}",
                    flush=True,
                )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        best_row = min(history, key=lambda r: r["val_loss"]) if history else {}
        return {
            "best_val_voltage_rmse": best_row.get("val_voltage_rmse", float("nan")),
            "best_val_temp_rmse": best_row.get("val_temp_rmse", float("nan")),
            "best_val_mape_v": best_row.get("mape_v", float("nan")),
            "best_val_mape_t": best_row.get("mape_t", float("nan")),
            "best_val_loss": best_val,
            "epochs_run": len(history),
            "history": history,
        }
