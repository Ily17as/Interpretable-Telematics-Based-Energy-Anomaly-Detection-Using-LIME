from __future__ import annotations

"""
Schema and configuration dataclasses for the VED energy anomaly detection project.

These dataclasses define the structure of the per‑trip table, configuration
parameters for LIME explanations, anomaly detection thresholds, and a container
for the names of feature columns used by the model. Keeping these definitions
centralized helps avoid hard‑coding string literals throughout the codebase
and makes it easy to update column names or configuration defaults.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TripSchema:
    """
    Contract for the per‑trip table produced by the preprocessing notebook.
    The per‑trip table must include identifiers for each trip and vehicle, the
    target column for energy consumption per kilometre, the model's
    predictions for that target and the residuals (actual minus predicted).
    """

    # Unique identifier for a trip (string). If your raw data uses a
    # different column name for the trip ID, adjust this accordingly.
    trip_id: str = "trip_id"

    # Identifier for the vehicle. The model is trained on groups of trips
    # grouped by vehicle_id to avoid data leakage.
    vehicle_id: str = "vehicle_id"

    # Name of the target column (energy consumed per kilometre).
    y: str = "energy_per_km"

    # Name of the model prediction column. After training, predictions are
    # stored here.
    yhat: str = "predicted_energy_per_km"

    # Name of the residual column (actual minus predicted). This column is
    # produced after computing the difference between the target and the
    # prediction.
    residual: str = "residual"


@dataclass(frozen=True)
class LIMEConfig:
    """
    Configuration for tabular LIME explanations.
    Adjust the number of samples, kernel width, distance metric,
    regularization strength and number of top features to inspect.
    """

    n_samples: int = 5000
    kernel_width: float | None = None  # None triggers heuristic (median distance)
    distance_metric: str = "euclidean"  # options: "euclidean", "cosine"
    ridge_alpha: float = 1.0  # L2 regularization for the surrogate model
    top_k: int = 10  # number of features to report
    random_state: int = 42  # seed for reproducibility


@dataclass(frozen=True)
class AnomalyConfig:
    """
    Configuration for anomaly detection thresholds.
    By default it uses a quantile threshold to detect high positive residuals.
    Alternatively, the MAD-z method can be used for robust z-score thresholding.
    """

    method: str = "quantile"  # "quantile" or "mad_z"
    quantile: float = 0.98  # quantile to define anomalies on the positive side
    mad_z: float = 3.5  # z-score threshold for MAD-based detection
    side: str = "positive"  # side to detect anomalies: "positive" or "both"


@dataclass(frozen=True)
class FeaturesConfig:
    """
    Container for the list of feature column names used by the model.
    This is optional and can be omitted if you infer feature columns at runtime.
    If you choose to store feature names explicitly, populate this class when
    training your model and save it to disk.
    """

    feature_cols: List[str]
