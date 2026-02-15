from __future__ import annotations

"""
Local fidelity metrics for LIME explanations.

R2 and RMSE are computed with weights applied to the perturbed samples,
ensuring that points closer to the reference instance are more important
for evaluating the quality of the surrogate model. The functions here
implement weighted versions of these metrics.
"""

import numpy as np


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the weighted coefficient of determination (R^2).

    Args:
        y_true: True values of shape (n_samples,).
        y_pred: Predicted values of shape (n_samples,).
        w: Weights of shape (n_samples,).

    Returns:
        Weighted R^2 statistic.
    """
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    w = w.astype(float)

    w_sum = np.sum(w) + 1e-12
    y_mean = np.sum(w * y_true) / w_sum

    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - y_mean) ** 2) + 1e-12
    return float(1.0 - ss_res / ss_tot)


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> float:
    """
    Compute the weighted root mean squared error.

    Args:
        y_true: True values of shape (n_samples,).
        y_pred: Predicted values of shape (n_samples,).
        w: Weights of shape (n_samples,).

    Returns:
        Weighted RMSE.
    """
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)
    w = w.astype(float)
    w_sum = np.sum(w) + 1e-12
    mse = np.sum(w * (y_true - y_pred) ** 2) / w_sum
    return float(np.sqrt(mse))
