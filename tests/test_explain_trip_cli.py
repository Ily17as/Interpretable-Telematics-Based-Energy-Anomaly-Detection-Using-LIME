import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.modeling.design import make_design_matrix, RawFeatureSpec
from src.utils.artifact import build_prediction_fn, load_training_artifact


class ArtifactFlowTests(unittest.TestCase):
    def test_full_artifact_prediction_fn(self):
        spec = RawFeatureSpec(
            numeric_features=["distance_km", "speed_mean"],
            categorical_features=["VehicleType"],
        )
        raw = pd.DataFrame(
            {
                "distance_km": [1.0, 2.0, 3.0, 4.0],
                "speed_mean": [10.0, 20.0, 30.0, 40.0],
                "VehicleType": ["ICE", "ICE", "EV", "EV"],
            }
        )
        X, numeric_fill, design_columns = make_design_matrix(raw, spec=spec)
        y = 0.5 * X["distance_km"] + 0.1 * X["speed_mean"]
        model = LinearRegression().fit(X, y)

        artifact = {
            "model": model,
            "design_columns": design_columns,
            "numeric_fill_values": numeric_fill,
            "raw_feature_columns": spec.raw_features,
            "numeric_feature_columns": spec.numeric_features,
            "categorical_feature_columns": spec.categorical_features,
            "background_raw": raw.copy(),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.joblib"
            joblib.dump(artifact, path)
            loaded = load_training_artifact(path)
            predict_fn = build_prediction_fn(loaded)
            pred = predict_fn(raw.iloc[[0]])
            self.assertEqual(pred.shape, (1,))
            self.assertTrue(np.isfinite(pred[0]))


if __name__ == "__main__":
    unittest.main()
