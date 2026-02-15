from __future__ import annotations

"""
Perturbation utilities for tabular LIME.

To generate local explanations, LIME requires synthetic samples around a point
of interest. This module implements functions to infer feature types
(numeric vs categorical), standardize numeric data, sample categorical
variables from the empirical distribution, and generate perturbed samples
around a given instance.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerturbationSpec:
    """
    Encodes which features are considered numeric and which are categorical.
    Numeric features will be perturbed with Gaussian noise (in absolute units).
    Categorical features will be sampled from the empirical distribution of
    observed values.
    """
    numeric_cols: List[str]
    categorical_cols: List[str]


def infer_feature_types(df: pd.DataFrame, feature_cols: List[str], max_cat_unique: int = 20) -> PerturbationSpec:
    """
    Infer whether each feature column should be treated as numeric or categorical.

    A feature is considered numeric if it has a numeric dtype and more than
    `max_cat_unique` unique values; otherwise it is treated as categorical.

    Args:
        df: DataFrame containing the background data.
        feature_cols: List of feature column names to inspect.
        max_cat_unique: Threshold on unique values for numeric/categorical split.

    Returns:
        PerturbationSpec listing numeric and categorical feature names.
    """
    numeric, cat = [], []
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > max_cat_unique:
            numeric.append(c)
        else:
            cat.append(c)
    return PerturbationSpec(numeric_cols=numeric, categorical_cols=cat)


def fit_standardizer(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the mean and standard deviation for numeric columns.

    When the standard deviation of a column is extremely small, it is replaced
    with 1.0 to avoid division by zero during perturbation.

    Args:
        df: Background data.
        numeric_cols: List of numeric column names.

    Returns:
        A tuple (mu, sigma) where mu and sigma are arrays of means and
        standard deviations for each numeric column.
    """
    mu = df[numeric_cols].mean(axis=0).to_numpy(dtype=float)
    sigma = df[numeric_cols].std(axis=0, ddof=0).to_numpy(dtype=float)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return mu, sigma


def sample_categorical(df: pd.DataFrame, cat_cols: List[str], n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Sample categorical feature values from the empirical distribution.

    Args:
        df: Background data.
        cat_cols: Names of categorical columns.
        n: Number of samples to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        Dictionary mapping each categorical column name to an array of sampled values.
    """
    out: Dict[str, np.ndarray] = {}
    for c in cat_cols:
        # Drop NaNs to avoid sampling missing values
        vals = df[c].dropna().unique()
        if len(vals) == 0:
            out[c] = np.array([None] * n, dtype=object)
        else:
            out[c] = rng.choice(vals, size=n, replace=True)
    return out


def perturb_tabular(
    x0: pd.Series,
    background_df: pd.DataFrame,
    feature_cols: List[str],
    n_samples: int,
    random_state: int = 42,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Generate perturbed samples around a point x0.

    Numeric features are perturbed with Gaussian noise scaled by the
    standard deviation of each feature. Categorical features are sampled
    from the empirical distribution. Missing values in the background
    distribution result in None entries.

    Args:
        x0: Series representing the reference instance (one row of features).
        background_df: DataFrame used to estimate distributions.
        feature_cols: List of feature columns used by the model.
        n_samples: Number of perturbations to generate.
        random_state: Seed for reproducible sampling.
        noise_scale: Scaling factor applied to the Gaussian noise for
                     numeric perturbations.

    Returns:
        A DataFrame of shape (n_samples, len(feature_cols)) with perturbed
        samples.
    """
    rng = np.random.default_rng(random_state)

    spec = infer_feature_types(background_df, feature_cols)
    # Compute standard deviations for numeric features
    mu, sigma = fit_standardizer(background_df, spec.numeric_cols) if spec.numeric_cols else (np.array([]), np.array([]))

    # Prepare empty DataFrame for perturbations
    Z = pd.DataFrame(index=range(n_samples), columns=feature_cols)

    # Perturb numeric columns
    if spec.numeric_cols:
        x0_num = x0[spec.numeric_cols].to_numpy(dtype=float)
        eps = rng.normal(loc=0.0, scale=noise_scale, size=(n_samples, len(spec.numeric_cols)))
        # Multiply by sigma to scale noise to observed standard deviation
        Z_num = x0_num + eps * sigma
        for j, c in enumerate(spec.numeric_cols):
            Z[c] = Z_num[:, j]

    # Sample categorical columns
    if spec.categorical_cols:
        cat_samples = sample_categorical(background_df, spec.categorical_cols, n_samples, rng)
        for c in spec.categorical_cols:
            Z[c] = cat_samples[c]

    # Ensure column types are the same as in the background_df
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(background_df[c]):
            Z[c] = Z[c].astype(float)
    return Z
