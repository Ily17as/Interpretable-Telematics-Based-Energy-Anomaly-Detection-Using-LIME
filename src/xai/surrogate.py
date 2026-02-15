from __future__ import annotations

"""
Surrogate model utilities for LIME.

The local surrogate model is a weighted linear regression (with ridge
regularization) fitted on the perturbed samples. The closed‑form solution
for ridge regression with an intercept is implemented here, along with a
helper function to select the top‑k features based on the absolute
magnitude of their coefficients.
"""

import numpy as np
from typing import List, Tuple


def weighted_ridge_closed_form(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """
    Solve weighted ridge regression with an intercept using the closed‑form
    solution.

    The objective is to minimize:
        sum_i w_i (y_i - (b + X_i^T beta))^2 + alpha * ||beta||^2
    where b is the intercept and beta are the coefficients for the features.

    Args:
        X: Design matrix of shape (n_samples, n_features).
        y: Target vector of shape (n_samples,).
        w: Weight vector of shape (n_samples,) used for weighting.
        alpha: Ridge (L2) regularization strength (only applied to beta).

    Returns:
        (beta, intercept) where beta is an array of shape (n_features,) and
        intercept is a scalar.
    """
    X = X.astype(float)
    y = y.astype(float)
    w = w.astype(float)

    n, d = X.shape
    # Augment X with a column of ones for the intercept
    X1 = np.concatenate([np.ones((n, 1)), X], axis=1)

    # Form the diagonal weight matrix
    W = np.diag(w)
    XtW = X1.T @ W
    A = XtW @ X1

    # Ridge penalty: only apply to coefficients (not intercept)
    R = np.zeros_like(A)
    R[1:, 1:] = alpha * np.eye(d)

    b_vec = XtW @ y
    theta = np.linalg.solve(A + R, b_vec)

    intercept = float(theta[0])
    beta = theta[1:]
    return beta, intercept


def select_top_k(beta: np.ndarray, feature_names: List[str], k: int) -> List[Tuple[str, float]]:
    """
    Return the top‑k features ranked by the absolute value of their coefficients.

    Args:
        beta: Array of regression coefficients of shape (n_features,).
        feature_names: Names of features corresponding to beta.
        k: Number of top features to return.

    Returns:
        List of (feature_name, coefficient) tuples sorted by descending
        absolute value of the coefficient.
    """
    idx = np.argsort(np.abs(beta))[::-1][:k]
    return [(feature_names[i], float(beta[i])) for i in idx]
