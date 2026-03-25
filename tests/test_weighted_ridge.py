import unittest

import numpy as np

from src.xai.surrogate import weighted_ridge_closed_form


class WeightedRidgeTests(unittest.TestCase):
    def test_closed_form_recovers_simple_linear_relation(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = 1.0 + 2.0 * X[:, 0]
        w = np.ones(len(X))
        beta, intercept = weighted_ridge_closed_form(X, y, w, alpha=1e-9)
        self.assertAlmostEqual(intercept, 1.0, places=5)
        self.assertAlmostEqual(beta[0], 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
