"""
Main demonstration script for LIME explanations on XGBoost model
for vehicle telematics energy anomaly detection.

This script demonstrates:
1. Generating synthetic vehicle telematics data
2. Training an XGBoost model for anomaly detection
3. Using LIME (implemented from scratch) to explain predictions
4. Visualizing the explanations
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from telematics_data_generator import TelematicsDataGenerator
from lime_explainer import LimeExplainer
from visualization import (plot_lime_explanation, plot_feature_importance_comparison,
                           plot_local_approximation, plot_dataset_distribution)


def main():
    """Main execution function."""
    
    print("=" * 80)
    print("LIME for XGBoost: Vehicle Telematics Energy Anomaly Detection")
    print("=" * 80)
    print()
    
    # 1. Generate synthetic telematics data
    print("Step 1: Generating synthetic vehicle telematics data...")
    data_generator = TelematicsDataGenerator(random_state=42)
    X, y = data_generator.generate_dataset(n_normal=1000, n_anomalous=200)
    
    print(f"Generated {len(X)} samples with {X.shape[1]} features")
    print(f"Normal samples: {np.sum(y == 0)}")
    print(f"Anomalous samples: {np.sum(y == 1)}")
    print(f"\nFeatures: {list(X.columns)}")
    print()
    
    # 2. Split data
    print("Step 2: Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print()
    
    # 3. Standardize features
    print("Step 3: Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features standardized")
    print()
    
    # 4. Train XGBoost model
    print("Step 4: Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    print("XGBoost model trained")
    print()
    
    # 5. Evaluate model
    print("Step 5: Evaluating XGBoost model...")
    y_pred = xgb_model.predict(X_test_scaled)
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Normal', 'Anomalous']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba[:, 1]):.4f}")
    print()
    
    # 6. Initialize LIME explainer
    print("Step 6: Initializing LIME explainer (implemented from scratch)...")
    feature_names = list(X.columns)
    lime_explainer = LimeExplainer(feature_names=feature_names, random_state=42)
    print("LIME explainer initialized")
    print()
    
    # 7. Explain predictions for sample instances
    print("Step 7: Generating LIME explanations for sample instances...")
    
    # Select instances to explain
    # One normal instance
    normal_idx = np.where(y_test == 0)[0][0]
    normal_instance = X_test_scaled[normal_idx]
    
    # One anomalous instance
    anomalous_idx = np.where(y_test == 1)[0][0]
    anomalous_instance = X_test_scaled[anomalous_idx]
    
    # Define prediction function
    def predict_fn(samples):
        return xgb_model.predict_proba(samples)
    
    # Explain normal instance
    print("\nExplaining NORMAL instance...")
    normal_explanation = lime_explainer.explain_instance_with_data(
        normal_instance,
        predict_fn,
        num_samples=5000,
        num_features=9,
        kernel_width=0.25
    )
    
    normal_pred = xgb_model.predict_proba(normal_instance.reshape(1, -1))[0]
    print(f"Model prediction: Normal={normal_pred[0]:.4f}, Anomalous={normal_pred[1]:.4f}")
    print(f"Local model R² score: {normal_explanation['r2_score']:.4f}")
    print("\nTop contributing features:")
    for idx, weight in normal_explanation['feature_weights'][:5]:
        print(f"  {feature_names[idx]}: {weight:.4f}")
    
    # Explain anomalous instance
    print("\nExplaining ANOMALOUS instance...")
    anomalous_explanation = lime_explainer.explain_instance_with_data(
        anomalous_instance,
        predict_fn,
        num_samples=5000,
        num_features=9,
        kernel_width=0.25
    )
    
    anomalous_pred = xgb_model.predict_proba(anomalous_instance.reshape(1, -1))[0]
    print(f"Model prediction: Normal={anomalous_pred[0]:.4f}, Anomalous={anomalous_pred[1]:.4f}")
    print(f"Local model R² score: {anomalous_explanation['r2_score']:.4f}")
    print("\nTop contributing features:")
    for idx, weight in anomalous_explanation['feature_weights'][:5]:
        print(f"  {feature_names[idx]}: {weight:.4f}")
    print()
    
    # 8. Create visualizations
    print("Step 8: Creating visualizations...")
    
    # Plot 1: Dataset distribution
    fig1 = plot_dataset_distribution(X, y, figsize=(15, 10))
    fig1.savefig('dataset_distribution.png', dpi=150, bbox_inches='tight')
    print("Saved: dataset_distribution.png")
    
    # Plot 2: LIME explanation for normal instance
    fig2 = plot_lime_explanation(
        normal_explanation,
        title="LIME Explanation - Normal Instance",
        figsize=(10, 6)
    )
    fig2.savefig('lime_explanation_normal.png', dpi=150, bbox_inches='tight')
    print("Saved: lime_explanation_normal.png")
    
    # Plot 3: LIME explanation for anomalous instance
    fig3 = plot_lime_explanation(
        anomalous_explanation,
        title="LIME Explanation - Anomalous Instance",
        figsize=(10, 6)
    )
    fig3.savefig('lime_explanation_anomalous.png', dpi=150, bbox_inches='tight')
    print("Saved: lime_explanation_anomalous.png")
    
    # Plot 4: Comparison of explanations
    fig4 = plot_feature_importance_comparison(
        [normal_explanation, anomalous_explanation],
        ['Normal Instance', 'Anomalous Instance'],
        figsize=(14, 6)
    )
    fig4.savefig('lime_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: lime_comparison.png")
    
    # Plot 5: Local approximation visualization
    fig5 = plot_local_approximation(
        anomalous_explanation,
        anomalous_instance,
        predict_fn,
        feature_idx=3,  # engine_load
        figsize=(10, 6)
    )
    fig5.savefig('local_approximation.png', dpi=150, bbox_inches='tight')
    print("Saved: local_approximation.png")
    
    print()
    print("=" * 80)
    print("Demonstration completed successfully!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - dataset_distribution.png: Distribution of features")
    print("  - lime_explanation_normal.png: LIME explanation for normal instance")
    print("  - lime_explanation_anomalous.png: LIME explanation for anomalous instance")
    print("  - lime_comparison.png: Side-by-side comparison of explanations")
    print("  - local_approximation.png: Visualization of local model approximation")
    print()
    print("Key Insights:")
    print("  - LIME provides interpretable explanations for black-box XGBoost predictions")
    print("  - Each feature's contribution to the prediction is quantified")
    print("  - Normal and anomalous instances show different feature importance patterns")
    print("  - The local linear model provides a faithful approximation (check R² scores)")
    

if __name__ == "__main__":
    main()
