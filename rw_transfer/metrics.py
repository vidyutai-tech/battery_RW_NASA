"""Evaluation metrics for twin and SOC models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def mse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean((pred - ref) ** 2))


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, ref)))


def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - ref)))


def mape(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-8) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = np.maximum(np.abs(ref), eps)
    return float(np.mean(np.abs(pred - ref) / denom) * 100.0)


def r2_score(pred: np.ndarray, ref: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def metric_bundle(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    soc_mape: bool = False,
) -> Dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    if pred.size == 0:
        return {}
    mp = mape(pred, ref)
    if soc_mape:
        mask = np.abs(ref) > 1e-3
        mp = mape(pred[mask], ref[mask]) if np.any(mask) else float("nan")
    return {
        "n": int(pred.size),
        "rmse": round(rmse(pred, ref), 6),
        "mae": round(mae(pred, ref), 6),
        "mape_pct": round(mp, 4),
        "r2": round(r2_score(pred, ref), 6),
    }


def twin_metrics(
    v_pred: np.ndarray,
    v_ref: np.ndarray,
    t_pred: np.ndarray,
    t_ref: np.ndarray,
) -> Dict[str, Any]:
    return {
        "voltage": metric_bundle(v_pred, v_ref),
        "temperature": metric_bundle(t_pred, t_ref),
        "voltage_rmse": rmse(v_pred, v_ref),
        "temperature_rmse": rmse(t_pred, t_ref),
    }
