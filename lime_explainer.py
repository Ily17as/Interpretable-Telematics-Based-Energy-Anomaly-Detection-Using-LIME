"""
LIME (Local Interpretable Model-Agnostic Explanations) Implementation from Scratch
This module implements LIME for explaining predictions of any black-box model.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_distances


class LimeExplainer:
    """
    LIME Explainer for tabular data.
    
    LIME explains predictions by fitting interpretable models (linear models) 
    locally around a prediction.
    """
    
    def __init__(self, feature_names=None, random_state=None):
        """
        Initialize LIME explainer.
        
        Parameters:
        -----------
        feature_names : list of str, optional
            Names of the features
        random_state : int, optional
            Random seed for reproducibility
        """
        self.feature_names = feature_names
        self.random_state = random_state
    
    def generate_neighborhood(self, instance, num_samples=5000, std_dev=0.1):
        """
        Generate perturbed samples around the instance.
        
        Parameters:
        -----------
        instance : numpy array
            The instance to explain (1D array)
        num_samples : int
            Number of samples to generate
        std_dev : float
            Standard deviation for perturbations (relative to feature std)
            
        Returns:
        --------
        perturbed_samples : numpy array
            Array of perturbed samples
        """
        num_features = len(instance)
        
        # Use local random state if provided
        rng = np.random.RandomState(self.random_state) if self.random_state is not None else np.random
        
        # Generate random perturbations using normal distribution
        perturbations = rng.normal(0, std_dev, size=(num_samples, num_features))
        
        # Compute per-feature standard deviation for appropriate scaling
        # Use small epsilon to avoid division by zero
        feature_std = np.abs(instance) + 1e-6
        
        # Create perturbed samples by adding scaled perturbations to the instance
        perturbed_samples = instance + perturbations * feature_std
        
        return perturbed_samples
    
    def compute_weights(self, instance, perturbed_samples, kernel_width=0.25):
        """
        Compute weights for perturbed samples based on distance to instance.
        
        Uses exponential kernel to assign higher weights to samples closer to the instance.
        
        Parameters:
        -----------
        instance : numpy array
            The original instance
        perturbed_samples : numpy array
            Perturbed samples
        kernel_width : float
            Width of the exponential kernel
            
        Returns:
        --------
        weights : numpy array
            Weights for each perturbed sample
        """
        # Compute distances between instance and perturbed samples
        distances = pairwise_distances(
            perturbed_samples,
            instance.reshape(1, -1),
            metric='euclidean'
        ).ravel()
        
        # Apply exponential kernel: w(z) = exp(-(d^2) / (kernel_width^2))
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        
        return weights
    
    def explain_instance(self, instance, predict_fn, num_samples=5000, 
                        num_features=10, kernel_width=0.25, std_dev=0.1):
        """
        Explain a prediction using LIME.
        
        Parameters:
        -----------
        instance : numpy array
            The instance to explain
        predict_fn : callable
            Function that takes samples and returns predictions
        num_samples : int
            Number of perturbed samples to generate
        num_features : int
            Number of top features to include in explanation
        kernel_width : float
            Width of the exponential kernel for weighting
        std_dev : float
            Standard deviation for perturbations
            
        Returns:
        --------
        explanation : dict
            Dictionary containing:
            - 'feature_weights': list of (feature_index, weight) tuples
            - 'feature_names': list of feature names (if provided)
            - 'intercept': intercept of the local linear model
            - 'r2_score': R² score of the local linear model
            - 'local_pred': prediction of the local model for the instance
        """
        # Generate perturbed samples
        perturbed_samples = self.generate_neighborhood(
            instance, num_samples, std_dev
        )
        
        # Get predictions for perturbed samples
        predictions = predict_fn(perturbed_samples)
        
        # For binary classification, ensure predictions are probabilities
        if len(predictions.shape) > 1 and predictions.shape[1] == 2:
            # Take probability of positive class
            predictions = predictions[:, 1]
        elif len(predictions.shape) > 1:
            # For multi-class, take the class with highest probability
            predictions = np.max(predictions, axis=1)
        
        # Compute weights based on distance
        weights = self.compute_weights(instance, perturbed_samples, kernel_width)
        
        # Fit weighted linear model
        linear_model = Ridge(alpha=1.0, fit_intercept=True)
        linear_model.fit(perturbed_samples, predictions, sample_weight=weights)
        
        # Get feature importances (coefficients of linear model)
        coefficients = linear_model.coef_
        
        # Sort features by absolute coefficient value
        feature_importance = [(i, coeff) for i, coeff in enumerate(coefficients)]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top num_features
        top_features = feature_importance[:num_features]
        
        # Compute R² score on the perturbed samples
        local_predictions = linear_model.predict(perturbed_samples)
        ss_res = np.sum(weights * (predictions - local_predictions) ** 2)
        ss_tot = np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Get local prediction for the instance
        local_pred = linear_model.predict(instance.reshape(1, -1))[0]
        
        explanation = {
            'feature_weights': top_features,
            'feature_names': self.feature_names,
            'intercept': linear_model.intercept_,
            'r2_score': r2_score,
            'local_pred': local_pred
        }
        
        return explanation
    
    def explain_instance_with_data(self, instance, predict_fn, num_samples=5000,
                                   num_features=10, kernel_width=0.25, std_dev=0.1):
        """
        Explain a prediction and return the generated data for further analysis.
        
        Returns both the explanation and the perturbed samples/predictions.
        """
        # Generate perturbed samples
        perturbed_samples = self.generate_neighborhood(
            instance, num_samples, std_dev
        )
        
        # Get predictions for perturbed samples
        predictions = predict_fn(perturbed_samples)
        
        # For binary classification
        if len(predictions.shape) > 1 and predictions.shape[1] == 2:
            predictions = predictions[:, 1]
        elif len(predictions.shape) > 1:
            predictions = np.max(predictions, axis=1)
        
        # Compute weights
        weights = self.compute_weights(instance, perturbed_samples, kernel_width)
        
        # Fit weighted linear model (same as in explain_instance)
        linear_model = Ridge(alpha=1.0, fit_intercept=True)
        linear_model.fit(perturbed_samples, predictions, sample_weight=weights)
        
        # Get feature importances (coefficients of linear model)
        coefficients = linear_model.coef_
        
        # Sort features by absolute coefficient value
        feature_importance = [(i, coeff) for i, coeff in enumerate(coefficients)]
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take top num_features
        top_features = feature_importance[:num_features]
        
        # Compute R² score on the perturbed samples
        local_predictions = linear_model.predict(perturbed_samples)
        ss_res = np.sum(weights * (predictions - local_predictions) ** 2)
        ss_tot = np.sum(weights * (predictions - np.average(predictions, weights=weights)) ** 2)
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Get local prediction for the instance
        local_pred = linear_model.predict(instance.reshape(1, -1))[0]
        
        explanation = {
            'feature_weights': top_features,
            'feature_names': self.feature_names,
            'intercept': linear_model.intercept_,
            'r2_score': r2_score,
            'local_pred': local_pred,
            'perturbed_samples': perturbed_samples,
            'predictions': predictions,
            'weights': weights
        }
        
        return explanation
