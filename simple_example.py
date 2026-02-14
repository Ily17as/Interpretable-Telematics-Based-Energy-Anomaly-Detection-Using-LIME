"""
Simple example demonstrating how to use the LIME explainer on custom data.

This script shows the basic usage pattern for the LIME implementation.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lime_explainer import LimeExplainer


def simple_example():
    """
    Simple example using synthetic data and Random Forest.
    """
    print("LIME Simple Example")
    print("=" * 60)
    
    # 1. Create synthetic dataset
    print("\n1. Creating synthetic dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(10)]
    
    # 2. Split and scale
    print("2. Splitting and scaling data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Train a Random Forest model
    print("3. Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    accuracy = rf_model.score(X_test_scaled, y_test)
    print(f"   Model accuracy: {accuracy:.4f}")
    
    # 4. Initialize LIME explainer
    print("\n4. Initializing LIME explainer...")
    explainer = LimeExplainer(feature_names=feature_names, random_state=42)
    
    # 5. Select an instance to explain
    instance_idx = 0
    instance = X_test_scaled[instance_idx]
    
    print(f"\n5. Explaining instance {instance_idx}...")
    print(f"   True label: {y_test[instance_idx]}")
    
    # Get model prediction
    prediction = rf_model.predict_proba(instance.reshape(1, -1))[0]
    print(f"   Model prediction: Class 0={prediction[0]:.4f}, Class 1={prediction[1]:.4f}")
    
    # 6. Generate LIME explanation
    print("\n6. Generating LIME explanation...")
    
    def predict_fn(samples):
        return rf_model.predict_proba(samples)
    
    explanation = explainer.explain_instance(
        instance=instance,
        predict_fn=predict_fn,
        num_samples=5000,
        num_features=5,
        kernel_width=0.25
    )
    
    # 7. Display results
    print("\n7. LIME Explanation:")
    print(f"   Local model R² score: {explanation['r2_score']:.4f}")
    print(f"   Local model intercept: {explanation['intercept']:.4f}")
    print(f"   Local prediction: {explanation['local_pred']:.4f}")
    print("\n   Top contributing features:")
    
    for idx, weight in explanation['feature_weights']:
        feature_name = feature_names[idx]
        direction = "increases" if weight > 0 else "decreases"
        print(f"   - {feature_name}: {weight:+.4f} ({direction} prediction)")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    

if __name__ == "__main__":
    simple_example()
