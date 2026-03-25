from __future__ import annotations

"""Utilities for loading model artifacts in a robust and explicit way."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.modeling.design import RawFeatureSpec, build_feature_spec_from_artifact, make_design_matrix
from src.utils.io import load_model


@dataclass(frozen=True)
class LoadedArtifact:
    model: Any
    design_columns: list[str]
    numeric_fill: dict[str, float]
    raw_feature_spec: RawFeatureSpec | None
    background_raw: pd.DataFrame | None
    metadata: dict[str, Any]


def _coerce_model_payload(payload: Any) -> LoadedArtifact:
    """Normalize the different artifact styles used in the repository."""
    if not isinstance(payload, dict):
        raise TypeError(
            "Expected a dictionary artifact. The committed .joblib file stores a dict, not a bare model."
        )

    model = payload.get("model", payload)

    # Full XAI artifact from the final notebook.
    if "design_columns" in payload:
        return LoadedArtifact(
            model=model,
            design_columns=list(payload["design_columns"]),
            numeric_fill=dict(payload.get("numeric_fill_values", {})),
            raw_feature_spec=build_feature_spec_from_artifact(payload),
            background_raw=payload.get("background_raw"),
            metadata={
                "artifact_type": "full_xai",
                "best_xgb_params": payload.get("best_xgb_params"),
                "test_metrics": payload.get("test_metrics"),
            },
        )

    # Baseline regression artifact from regression_model notebook.
    if "feature_columns" in payload:
        feature_columns = list(payload["feature_columns"])
        return LoadedArtifact(
            model=model,
            design_columns=feature_columns,
            numeric_fill=dict(payload.get("feature_medians", {})),
            raw_feature_spec=None,
            background_raw=payload.get("lime_background"),
            metadata={"artifact_type": "baseline_regression"},
        )

    raise ValueError("Unsupported artifact format. Missing design_columns/feature_columns metadata.")


def load_training_artifact(path: str | Path) -> LoadedArtifact:
    """Load a saved training artifact and normalize its schema."""
    payload = load_model(path)
    return _coerce_model_payload(payload)


def build_prediction_fn(artifact: LoadedArtifact) -> Callable[[pd.DataFrame], np.ndarray]:
    """Create a prediction function that matches the artifact format.

    - If raw-feature metadata is available, convert raw telematics rows to the
      design matrix before prediction.
    - Otherwise, assume the incoming DataFrame already contains aligned model features.
    """
    model = artifact.model

    def predict(df: pd.DataFrame) -> np.ndarray:
        if artifact.raw_feature_spec is not None:
            X, _, _ = make_design_matrix(
                df,
                spec=artifact.raw_feature_spec,
                design_columns=artifact.design_columns,
                numeric_fill=artifact.numeric_fill,
            )
            return np.asarray(model.predict(X), dtype=float)

        X = df.reindex(columns=artifact.design_columns, fill_value=0.0).copy()
        for col, fill_value in artifact.numeric_fill.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)
        return np.asarray(model.predict(X), dtype=float)

    return predict
