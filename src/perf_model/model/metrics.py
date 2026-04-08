"""Regression metrics."""

from __future__ import annotations

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def percentile_relative_error(
    y_true: np.ndarray, y_pred: np.ndarray, percentile: float
) -> float:
    denom = np.clip(np.abs(y_true), 1e-8, None)
    relative_error = np.abs((y_true - y_pred) / denom) * 100.0
    return float(np.percentile(relative_error, percentile))
