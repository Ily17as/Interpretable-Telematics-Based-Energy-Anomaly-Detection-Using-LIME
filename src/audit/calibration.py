from __future__ import annotations
import pandas as pd

def positive_residual_threshold(residuals: pd.Series, quantile: float = 0.95) -> float:
    positive = residuals[residuals > 0]
    if positive.empty:
        return 0.0
    return float(positive.quantile(quantile))

def apply_threshold(residuals: pd.Series, threshold: float) -> pd.Series:
    return (residuals > threshold).astype(int)
