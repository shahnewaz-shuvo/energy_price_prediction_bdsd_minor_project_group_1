from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metric_dict(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    residual = pred - actual
    denom = np.where(np.abs(actual) < 1e-6, np.nan, np.abs(actual))
    return {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(math.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)) if len(actual) > 1 else np.nan,
        "bias": float(np.mean(residual)),
        "MAPE": float(np.nanmean(np.abs(residual) / denom) * 100),
        "near_zero_actual_count_abs_lt_1": int(np.sum(np.abs(actual) < 1)),
    }
