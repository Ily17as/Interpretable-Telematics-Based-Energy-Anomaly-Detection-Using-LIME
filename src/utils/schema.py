from __future__ import annotations

"""Schema and configuration dataclasses for the telematics anomaly project."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TripSchema:
    """Contract for trip-level scored tables.

    The project currently uses VED-derived tables with these identifiers:
    - ``trip_id``: string key used throughout the repo
    - ``VehId``: vehicle identifier from VED
    - ``Trip``: trip counter within a vehicle

    The original version hard-coded ``vehicle_id`` even though the committed
    tables store ``VehId``. This dataclass now reflects the actual saved data.
    """

    trip_id: str = "trip_id"
    vehicle_id: str = "VehId"
    trip_number: str = "Trip"
    y: str = "energy_per_km"
    yhat: str = "predicted_energy_per_km"
    residual: str = "residual"


@dataclass(frozen=True)
class LIMEConfig:
    n_samples: int = 5000
    kernel_width: float | None = None
    distance_metric: str = "euclidean"
    ridge_alpha: float = 1.0
    top_k: int = 10
    random_state: int = 42


@dataclass(frozen=True)
class AnomalyConfig:
    method: str = "quantile"
    quantile: float = 0.98
    mad_z: float = 3.5
    side: str = "positive"


@dataclass(frozen=True)
class FeaturesConfig:
    feature_cols: List[str]
