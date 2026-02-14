"""
Unit tests for LIME implementation.

Run with: python -m pytest test_lime.py -v
or: python test_lime.py
"""

import numpy as np
import unittest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from lime_explainer import LimeExplainer


class TestLimeExplainer(unittest.TestCase):
    """Test cases for LIME explainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple dataset
        self.X, self.y = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        
        # Train simple model
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)
        
        # Create test instance
        self.instance = self.X[0]
        
        # Define prediction function
        self.predict_fn = lambda x: self.model.predict_proba(x)
        
        # Feature names
        self.feature_names = [f"feature_{i}" for i in range(5)]
    
    def test_explainer_initialization(self):
        """Test LIME explainer initialization."""
        explainer = LimeExplainer(feature_names=self.feature_names, random_state=42)
        self.assertIsNotNone(explainer)
        self.assertEqual(explainer.feature_names, self.feature_names)
        self.assertEqual(explainer.random_state, 42)
    
    def test_generate_neighborhood(self):
        """Test neighborhood generation."""
        explainer = LimeExplainer(random_state=42)
        
        num_samples = 1000
        perturbed = explainer.generate_neighborhood(
            self.instance, num_samples=num_samples, std_dev=0.1
        )
        
        # Check shape
        self.assertEqual(perturbed.shape[0], num_samples)
        self.assertEqual(perturbed.shape[1], len(self.instance))
        
        # Check that samples are different from instance
        self.assertFalse(np.allclose(perturbed[0], self.instance))
    
    def test_compute_weights(self):
        """Test weight computation."""
        explainer = LimeExplainer(random_state=42)
        
        # Generate samples
        perturbed = explainer.generate_neighborhood(self.instance, num_samples=100)
        
        # Compute weights
        weights = explainer.compute_weights(self.instance, perturbed, kernel_width=0.25)
        
        # Check properties
        self.assertEqual(len(weights), len(perturbed))
        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.all(weights <= 1))
        
        # Weights should be higher for closer samples
        distances = np.linalg.norm(perturbed - self.instance, axis=1)
        closest_idx = np.argmin(distances)
        farthest_idx = np.argmax(distances)
        self.assertGreater(weights[closest_idx], weights[farthest_idx])
    
    def test_explain_instance(self):
        """Test instance explanation."""
        explainer = LimeExplainer(feature_names=self.feature_names, random_state=42)
        
        explanation = explainer.explain_instance(
            instance=self.instance,
            predict_fn=self.predict_fn,
            num_samples=500,
            num_features=3,
            kernel_width=0.25
        )
        
        # Check explanation structure
        self.assertIn('feature_weights', explanation)
        self.assertIn('feature_names', explanation)
        self.assertIn('intercept', explanation)
        self.assertIn('r2_score', explanation)
        self.assertIn('local_pred', explanation)
        
        # Check feature weights
        feature_weights = explanation['feature_weights']
        self.assertEqual(len(feature_weights), 3)
        
        # Check that weights are tuples of (index, weight)
        for idx, weight in feature_weights:
            self.assertIsInstance(idx, (int, np.integer))
            self.assertIsInstance(weight, (float, np.floating))
    
    def test_explain_instance_with_data(self):
        """Test instance explanation with data."""
        explainer = LimeExplainer(feature_names=self.feature_names, random_state=42)
        
        explanation = explainer.explain_instance_with_data(
            instance=self.instance,
            predict_fn=self.predict_fn,
            num_samples=500,
            num_features=3
        )
        
        # Check additional data
        self.assertIn('perturbed_samples', explanation)
        self.assertIn('predictions', explanation)
        self.assertIn('weights', explanation)
        
        # Check shapes
        self.assertEqual(
            explanation['perturbed_samples'].shape[0],
            explanation['predictions'].shape[0]
        )
        self.assertEqual(
            explanation['predictions'].shape[0],
            explanation['weights'].shape[0]
        )
    
    def test_reproducibility(self):
        """Test that explanations are reproducible with same random state."""
        explainer1 = LimeExplainer(feature_names=self.feature_names, random_state=42)
        explainer2 = LimeExplainer(feature_names=self.feature_names, random_state=42)
        
        exp1 = explainer1.explain_instance(
            self.instance, self.predict_fn, num_samples=500, num_features=3
        )
        exp2 = explainer2.explain_instance(
            self.instance, self.predict_fn, num_samples=500, num_features=3
        )
        
        # Check that explanations are identical
        weights1 = [w for _, w in exp1['feature_weights']]
        weights2 = [w for _, w in exp2['feature_weights']]
        
        np.testing.assert_array_almost_equal(weights1, weights2, decimal=5)
    
    def test_different_num_features(self):
        """Test with different number of features requested."""
        explainer = LimeExplainer(feature_names=self.feature_names, random_state=42)
        
        for num_features in [1, 3, 5]:
            explanation = explainer.explain_instance(
                self.instance, self.predict_fn, 
                num_samples=500, num_features=num_features
            )
            
            self.assertEqual(len(explanation['feature_weights']), num_features)
    
    def test_kernel_width_effect(self):
        """Test that kernel width affects weights."""
        explainer = LimeExplainer(random_state=42)
        perturbed = explainer.generate_neighborhood(self.instance, num_samples=100)
        
        # Smaller kernel width should give more concentrated weights
        weights_small = explainer.compute_weights(
            self.instance, perturbed, kernel_width=0.1
        )
        weights_large = explainer.compute_weights(
            self.instance, perturbed, kernel_width=1.0
        )
        
        # Variance should be larger for smaller kernel width
        self.assertGreater(np.var(weights_small), np.var(weights_large))
    
    def test_binary_classification(self):
        """Test with binary classification output."""
        explainer = LimeExplainer(feature_names=self.feature_names, random_state=42)
        
        # Prediction function returns probabilities for 2 classes
        def binary_predict_fn(samples):
            return self.model.predict_proba(samples)
        
        explanation = explainer.explain_instance(
            self.instance, binary_predict_fn, num_samples=500, num_features=3
        )
        
        self.assertIsNotNone(explanation)
        self.assertEqual(len(explanation['feature_weights']), 3)
    
    def test_r2_score_range(self):
        """Test that R² score is in valid range."""
        explainer = LimeExplainer(feature_names=self.feature_names, random_state=42)
        
        explanation = explainer.explain_instance(
            self.instance, self.predict_fn, num_samples=5000, num_features=3
        )
        
        # R² can be negative for very poor fits and has no theoretical lower bound,
        # but for this test we just ensure it's not unreasonably bad (< -10) and not > 1
        self.assertGreaterEqual(explanation['r2_score'], -10.0)
        self.assertLessEqual(explanation['r2_score'], 1.0)


class TestTelematicsDataGenerator(unittest.TestCase):
    """Test cases for telematics data generator."""
    
    def test_import(self):
        """Test that we can import the data generator."""
        from telematics_data_generator import TelematicsDataGenerator
        generator = TelematicsDataGenerator(random_state=42)
        self.assertIsNotNone(generator)
    
    def test_generate_normal_data(self):
        """Test normal data generation."""
        from telematics_data_generator import TelematicsDataGenerator
        generator = TelematicsDataGenerator(random_state=42)
        
        data = generator.generate_normal_data(n_samples=100)
        
        self.assertEqual(len(data), 100)
        self.assertEqual(data.shape[1], 9)
        
        # Check that all values are non-negative (for most features)
        self.assertTrue(np.all(data['speed'] >= 0))
        self.assertTrue(np.all(data['rpm'] >= 0))
        self.assertTrue(np.all(data['energy_consumption'] >= 0))
    
    def test_generate_anomalous_data(self):
        """Test anomalous data generation."""
        from telematics_data_generator import TelematicsDataGenerator
        generator = TelematicsDataGenerator(random_state=42)
        
        data = generator.generate_anomalous_data(n_samples=100)
        
        self.assertEqual(len(data), 100)
        self.assertEqual(data.shape[1], 9)
    
    def test_generate_dataset(self):
        """Test full dataset generation."""
        from telematics_data_generator import TelematicsDataGenerator
        generator = TelematicsDataGenerator(random_state=42)
        
        X, y = generator.generate_dataset(n_normal=500, n_anomalous=100)
        
        self.assertEqual(len(X), 600)
        self.assertEqual(len(y), 600)
        self.assertEqual(np.sum(y == 0), 500)
        self.assertEqual(np.sum(y == 1), 100)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
