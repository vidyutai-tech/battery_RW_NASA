"""Training losses for the digital twin (MAPE-aligned objectives)."""

from __future__ import annotations

from typing import Tuple

import torch


def pearson_corr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - mean Pearson r over batch sequences."""
    pred_c = pred - pred.mean(dim=1, keepdim=True)
    target_c = target - target.mean(dim=1, keepdim=True)
    num = (pred_c * target_c).sum(dim=1)
    den = torch.sqrt((pred_c ** 2).sum(dim=1) * (target_c ** 2).sum(dim=1) + eps)
    return (1.0 - num / den).mean()


def relative_mse(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    denom = target.abs() + eps
    return torch.mean(((pred - target) / denom) ** 2)


def mape_fraction(pred: torch.Tensor, target: torch.Tensor, eps: float) -> torch.Tensor:
    """Mean absolute percentage error as a fraction (0.01 = 1%)."""
    denom = target.abs() + eps
    return torch.mean(torch.abs(pred - target) / denom)


def author_train_loss(
    v_hat: torch.Tensor,
    t_hat: torch.Tensor,
    yv: torch.Tensor,
    yt: torch.Tensor,
    voltage_weight: float = 100.0,
) -> torch.Tensor:
    """``100 * MSE_V + MSE_T`` — original ``training.py`` loss_normalized."""
    return (
        voltage_weight * torch.nn.functional.mse_loss(v_hat, yv)
        + torch.nn.functional.mse_loss(t_hat, yt)
    )


def author_val_loss(
    v_hat: torch.Tensor,
    t_hat: torch.Tensor,
    yv: torch.Tensor,
    yt: torch.Tensor,
) -> torch.Tensor:
    """Full MSE on stacked outputs — original validation criterion."""
    pred = torch.stack([v_hat, t_hat], dim=-1)
    target = torch.stack([yv, yt], dim=-1)
    return torch.nn.functional.mse_loss(pred, target)


def author_mape_pct(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    """
  MAPE per channel — matches ``calculate_dimensional_mape`` (target + 1e-8).
    """
    eps = 1e-8
    t = target + eps
    p = pred + eps
    mape_v = torch.mean(torch.abs((t[:, :, 0] - p[:, :, 0]) / t[:, :, 0])) * 100.0
    mape_t = torch.mean(torch.abs((t[:, :, 1] - p[:, :, 1]) / t[:, :, 1])) * 100.0
    return float(mape_v.item()), float(mape_t.item())


def temp_aware_finetune_loss(
    v_hat: torch.Tensor,
    t_hat: torch.Tensor,
    yv: torch.Tensor,
    yt: torch.Tensor,
    voltage_weight: float = 10.0,
    temp_weight: float = 50.0,
    pearson_weight: float = 5.0,
) -> torch.Tensor:
    """
    Temperature-aware loss for fine-tuning.

    Replaces the author's ``100·MSE_V + 1·MSE_T`` with a balanced objective
    that gives the temperature head meaningful gradient signal:

    - ``voltage_weight`` / ``temp_weight``: MSE terms (default 10/50 — temp gets
      5× more relative weight than the author recipe's 1/100 ratio).
    - ``pearson_weight``: shape-matching term — forces the predicted temperature
      *trajectory* to be correlated with ground truth, not just the mean value.
      Critical when ``temp_delta_scale=0.1`` suppresses magnitude gradients.
    """
    loss = (
        voltage_weight * torch.nn.functional.mse_loss(v_hat, yv)
        + temp_weight * torch.nn.functional.mse_loss(t_hat, yt)
    )
    if pearson_weight > 0.0:
        loss = loss + pearson_weight * pearson_corr_loss(t_hat, yt)
    return loss


def twin_training_loss(
    v_hat: torch.Tensor,
    t_hat: torch.Tensor,
    yv: torch.Tensor,
    yt: torch.Tensor,
    *,
    mse_v_w: float,
    mse_t_w: float,
    mape_v_w: float,
    mape_t_w: float,
    corr_t_w: float,
    mape_eps_v: float,
    mape_eps_t: float,
) -> torch.Tensor:
    """Combined voltage/temperature loss with explicit MAPE terms."""
    loss = torch.tensor(0.0, device=v_hat.device)
    if mse_v_w > 0:
        loss = loss + mse_v_w * torch.nn.functional.mse_loss(v_hat, yv)
    if mse_t_w > 0:
        loss = loss + mse_t_w * torch.nn.functional.mse_loss(t_hat, yt)
    if mape_v_w > 0:
        loss = loss + mape_v_w * mape_fraction(v_hat, yv, mape_eps_v)
    if mape_t_w > 0:
        loss = loss + mape_t_w * mape_fraction(t_hat, yt, mape_eps_t)
    if corr_t_w > 0:
        loss = loss + corr_t_w * pearson_corr_loss(t_hat, yt)
    return loss
