from __future__ import annotations

"""Surrogate model utilities for LIME."""

import numpy as np
from typing import List, Tuple


def weighted_ridge_closed_form(X: np.ndarray, y: np.ndarray, w: np.ndarray, alpha: float) -> Tuple[np.ndarray, float]:
    """Solve weighted ridge regression with an intercept.

    Why this implementation:
    - the earlier version explicitly built the full diagonal matrix ``W``
    - that is unnecessary and scales poorly in memory
    - here we use the standard sqrt-weight trick and a pseudo-inverse for a
      more stable small-project implementation
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)

    n, d = X.shape
    X1 = np.concatenate([np.ones((n, 1)), X], axis=1)

    sqrt_w = np.sqrt(w + 1e-12)
    Xw = X1 * sqrt_w[:, None]
    yw = y * sqrt_w

    ridge = np.zeros((d + 1, d + 1), dtype=float)
    ridge[1:, 1:] = alpha * np.eye(d)

    theta = np.linalg.pinv(Xw.T @ Xw + ridge) @ (Xw.T @ yw)
    intercept = float(theta[0])
    beta = theta[1:]
    return beta, intercept


def select_top_k(beta: np.ndarray, feature_names: List[str], k: int) -> List[Tuple[str, float]]:
    idx = np.argsort(np.abs(beta))[::-1][:k]
    return [(feature_names[i], float(beta[i])) for i in idx]
