import unittest

import numpy as np
import pandas as pd

from src.anomaly.thresholds import flag_anomalies


class ThresholdTests(unittest.TestCase):
    def test_quantile_positive_flags_upper_tail(self):
        s = pd.Series([0, 1, 2, 3, 100], dtype=float)
        mask, meta = flag_anomalies(s, method="quantile", side="positive", quantile=0.8)
        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(bool(mask.iloc[-1]))
        self.assertIn("threshold_pos", meta)

    def test_mad_both_flags_large_absolute_residual(self):
        s = pd.Series([0, 0, 0, 0, -10, 12], dtype=float)
        mask, meta = flag_anomalies(s, method="mad_z", side="both", mad_z=3.5)
        self.assertGreaterEqual(int(mask.sum()), 2)
        self.assertEqual(meta["threshold_robust_z"], 3.5)


if __name__ == "__main__":
    unittest.main()
