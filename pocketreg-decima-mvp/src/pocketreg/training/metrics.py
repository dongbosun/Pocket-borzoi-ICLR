"""Regression metrics for teacher-output distillation."""

from __future__ import annotations

import logging

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LOGGER = logging.getLogger(__name__)


def _valid_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return y_true[mask], y_pred[mask]


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation with guards for tiny or constant arrays."""
    y_true, y_pred = _valid_pair(y_true, y_pred)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        LOGGER.warning("Pearson is undefined for <2 points or constant arrays.")
        return float("nan")
    return float(stats.pearsonr(y_true, y_pred).statistic)


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman correlation with guards for tiny or constant arrays."""
    y_true, y_pred = _valid_pair(y_true, y_pred)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        LOGGER.warning("Spearman is undefined for <2 points or constant arrays.")
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).statistic)


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, train_mean: float | None = None
) -> dict[str, float | int]:
    """Compute scalar regression metrics on raw-scale predictions."""
    y_true, y_pred = _valid_pair(y_true, y_pred)
    if y_true.size == 0:
        return {"n": 0}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    baseline = np.full_like(y_true, float(np.mean(y_true) if train_mean is None else train_mean))
    baseline_rmse = float(np.sqrt(mean_squared_error(y_true, baseline)))
    try:
        r2 = float(r2_score(y_true, y_pred))
        baseline_r2 = float(r2_score(y_true, baseline))
    except ValueError:
        r2 = float("nan")
        baseline_r2 = float("nan")
    return {
        "n": int(y_true.size),
        "pearson": safe_pearson(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
        "r2": r2,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "y_true_mean": float(np.mean(y_true)),
        "y_true_std": float(np.std(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
        "y_pred_std": float(np.std(y_pred)),
        "baseline_mean_rmse": baseline_rmse,
        "baseline_mean_r2": baseline_r2,
    }
