"""
Visualization utilities for LIME explanations
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_lime_explanation(explanation, title="LIME Explanation", figsize=(10, 6)):
    """
    Plot LIME explanation showing feature contributions.
    
    Parameters:
    -----------
    explanation : dict
        Explanation dictionary from LIME
    title : str
        Title for the plot
    figsize : tuple
        Figure size
    """
    feature_weights = explanation['feature_weights']
    feature_names = explanation['feature_names']
    
    # Extract features and weights
    features = []
    weights = []
    for idx, weight in feature_weights:
        if feature_names is not None:
            features.append(feature_names[idx])
        else:
            features.append(f"Feature {idx}")
        weights.append(weight)
    
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color bars based on positive/negative contribution
    colors = ['green' if w > 0 else 'red' for w in weights]
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, weights, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Weight (Contribution to Prediction)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add R² score as text
    r2_text = f"Local Model R² Score: {explanation['r2_score']:.4f}"
    ax.text(0.02, 0.98, r2_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_feature_importance_comparison(explanations, labels, figsize=(12, 8)):
    """
    Compare LIME explanations for multiple instances.
    
    Parameters:
    -----------
    explanations : list of dict
        List of LIME explanations
    labels : list of str
        Labels for each explanation
    figsize : tuple
        Figure size
    """
    n_explanations = len(explanations)
    fig, axes = plt.subplots(1, n_explanations, figsize=figsize, sharey=True)
    
    if n_explanations == 1:
        axes = [axes]
    
    for i, (exp, label) in enumerate(zip(explanations, labels)):
        feature_weights = exp['feature_weights']
        feature_names = exp['feature_names']
        
        features = []
        weights = []
        for idx, weight in feature_weights:
            if feature_names is not None:
                features.append(feature_names[idx])
            else:
                features.append(f"Feature {idx}")
            weights.append(weight)
        
        colors = ['green' if w > 0 else 'red' for w in weights]
        y_pos = np.arange(len(features))
        
        axes[i].barh(y_pos, weights, color=colors, alpha=0.7)
        axes[i].set_yticks(y_pos)
        if i == 0:
            axes[i].set_yticklabels(features)
        axes[i].set_xlabel('Weight', fontsize=10)
        axes[i].set_title(label, fontsize=12, fontweight='bold')
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    return fig


def plot_local_approximation(explanation, instance, predict_fn, 
                            feature_idx=0, figsize=(10, 6)):
    """
    Visualize how well the local linear model approximates the black-box model.
    
    Parameters:
    -----------
    explanation : dict
        Explanation with perturbed samples data
    instance : numpy array
        The explained instance
    predict_fn : callable
        Black-box model prediction function
    feature_idx : int
        Index of feature to vary for visualization
    figsize : tuple
        Figure size
    """
    if 'perturbed_samples' not in explanation:
        raise ValueError("Explanation must contain perturbed_samples. "
                        "Use explain_instance_with_data() instead.")
    
    perturbed_samples = explanation['perturbed_samples']
    predictions = explanation['predictions']
    weights = explanation['weights']
    
    # Get the feature values and sort them
    feature_values = perturbed_samples[:, feature_idx]
    sort_idx = np.argsort(feature_values)
    
    feature_values_sorted = feature_values[sort_idx]
    predictions_sorted = predictions[sort_idx]
    weights_sorted = weights[sort_idx]
    
    # Compute local linear model predictions
    # Using the fitted model from explanation
    feature_weight = None
    for idx, weight in explanation['feature_weights']:
        if idx == feature_idx:
            feature_weight = weight
            break
    
    if feature_weight is None:
        raise ValueError(f"Feature {feature_idx} not in top features")
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot with weights determining point size
    scatter = ax.scatter(feature_values, predictions, 
                        c=weights, s=weights * 50, 
                        alpha=0.5, cmap='viridis')
    
    # Mark the instance
    instance_pred = predict_fn(instance.reshape(1, -1))
    if len(instance_pred.shape) > 1 and instance_pred.shape[1] == 2:
        instance_pred = instance_pred[0, 1]
    else:
        instance_pred = instance_pred[0]
    
    ax.scatter(instance[feature_idx], instance_pred, 
              color='red', s=200, marker='*', 
              label='Explained Instance', zorder=5)
    
    feature_name = (explanation['feature_names'][feature_idx] 
                   if explanation['feature_names'] is not None 
                   else f"Feature {feature_idx}")
    
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel('Prediction', fontsize=12)
    ax.set_title('Local Model Approximation', fontsize=14, fontweight='bold')
    ax.legend()
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sample Weight', fontsize=10)
    
    plt.tight_layout()
    return fig


def plot_dataset_distribution(X, y, features_to_plot=None, figsize=(15, 10)):
    """
    Plot distribution of features for normal vs anomalous samples.
    
    Parameters:
    -----------
    X : pandas DataFrame
        Feature matrix
    y : numpy array
        Labels
    features_to_plot : list of str, optional
        Features to plot (default: all)
    figsize : tuple
        Figure size
    """
    if features_to_plot is None:
        features_to_plot = X.columns.tolist()
    
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    for i, feature in enumerate(features_to_plot):
        ax = axes[i]
        
        # Plot distributions
        normal_data = X[y == 0][feature]
        anomalous_data = X[y == 1][feature]
        
        ax.hist(normal_data, bins=30, alpha=0.6, label='Normal', color='blue')
        ax.hist(anomalous_data, bins=30, alpha=0.6, label='Anomalous', color='red')
        
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Feature Distributions: Normal vs Anomalous', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig
