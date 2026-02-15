from __future__ import annotations

"""
Visualization utilities for the energy anomaly detection project.

This module provides simple functions to visualize residuals and LIME
explanations. These plots are intended to be saved to disk and used in
reports or notebooks. All functions ensure the output directory exists
before writing to file.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.utils.io import ensure_dir


def plot_residual_hist(df: pd.DataFrame, residual_col: str, out_path: str | Path) -> None:
    """
    Plot a histogram of residuals.

    Args:
        df: DataFrame containing residuals.
        residual_col: Name of the residual column.
        out_path: Path to save the figure.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    plt.figure()
    plt.hist(df[residual_col].dropna().values, bins=60)
    plt.xlabel("Residual (actual - predicted)")
    plt.ylabel("Count")
    plt.title("Residual distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_actual_vs_pred(df: pd.DataFrame, y_col: str, yhat_col: str, out_path: str | Path) -> None:
    """
    Scatter plot of actual vs predicted target values.

    Args:
        df: DataFrame containing the target and predictions.
        y_col: Column name for actual target.
        yhat_col: Column name for predictions.
        out_path: Path to save the figure.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    plt.figure()
    plt.scatter(df[y_col].values, df[yhat_col].values, s=6)
    plt.xlabel("Actual energy_per_km")
    plt.ylabel("Predicted energy_per_km")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_lime_bar(top_features: List[Tuple[str, float]], out_path: str | Path, title: str = "LIME feature contributions") -> None:
    """
    Horizontal bar plot for the top feature contributions of a LIME explanation.

    Args:
        top_features: List of tuples (feature_name, contribution) sorted by importance.
        out_path: Path to save the figure.
        title: Title of the plot.
    """
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    labels = [t[0] for t in top_features][::-1]
    vals = [t[1] for t in top_features][::-1]

    plt.figure(figsize=(8, max(3, 0.25 * len(labels) + 1)))
    plt.barh(labels, vals)
    plt.xlabel("Contribution (beta * x0)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
