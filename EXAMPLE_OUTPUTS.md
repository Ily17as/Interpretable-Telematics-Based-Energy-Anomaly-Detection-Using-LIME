# Example Outputs and Usage

## Running the Main Demonstration

```bash
python main.py
```

### Console Output

```
================================================================================
LIME for XGBoost: Vehicle Telematics Energy Anomaly Detection
================================================================================

Step 1: Generating synthetic vehicle telematics data...
Generated 1200 samples with 9 features
Normal samples: 1000
Anomalous samples: 200

Features: ['speed', 'acceleration', 'rpm', 'engine_load', 'throttle', 'brake', 'temperature', 'fuel_rate', 'energy_consumption']

Step 2: Splitting data into train and test sets...
Training set: 840 samples
Test set: 360 samples

Step 3: Standardizing features...
Features standardized

Step 4: Training XGBoost model...
XGBoost model trained

Step 5: Evaluating XGBoost model...

Classification Report:
              precision    recall  f1-score   support

      Normal       1.00      0.99      1.00       300
   Anomalous       0.97      1.00      0.98        60

    accuracy                           0.99       360
   macro avg       0.98      1.00      0.99       360
weighted avg       0.99      0.99      0.99       360


Confusion Matrix:
[[298   2]
 [  0  60]]

ROC-AUC Score: 0.9997

Step 6: Initializing LIME explainer (implemented from scratch)...
LIME explainer initialized

Step 7: Generating LIME explanations for sample instances...

Explaining NORMAL instance...
Model prediction: Normal=0.9969, Anomalous=0.0031
Local model R² score: 0.4600

Top contributing features:
  brake: 0.0009
  energy_consumption: -0.0000
  throttle: -0.0000
  engine_load: 0.0000
  acceleration: -0.0000

Explaining ANOMALOUS instance...
Model prediction: Normal=0.0024, Anomalous=0.9976
Local model R² score: 0.2617

Top contributing features:
  brake: 0.0006
  fuel_rate: 0.0000
  temperature: -0.0000
  throttle: 0.0000
  acceleration: -0.0000

Step 8: Creating visualizations...
Saved: dataset_distribution.png
Saved: lime_explanation_normal.png
Saved: lime_explanation_anomalous.png
Saved: lime_comparison.png
Saved: local_approximation.png

================================================================================
Demonstration completed successfully!
================================================================================
```

## Generated Visualizations

### 1. Dataset Distribution (`dataset_distribution.png`)

This visualization shows the distribution of all features for normal vs anomalous samples. It helps understand:
- How features differ between normal and anomalous driving patterns
- Which features have the most separation between classes
- The overall data distribution

**Key Observations:**
- Energy consumption is clearly higher for anomalous samples
- Temperature shows higher values in anomalies
- Speed distribution differs between normal and anomalous patterns
- Engine load is elevated in anomalous samples

### 2. LIME Explanation - Normal Instance (`lime_explanation_normal.png`)

Shows feature contributions for a correctly classified normal driving instance.

**Interpretation:**
- Green bars: Features that increase the anomaly probability
- Red bars: Features that decrease the anomaly probability (favor normal classification)
- The R² score indicates how well the local linear model approximates the XGBoost model

**Typical Pattern for Normal Instance:**
- Most features contribute to lowering anomaly probability
- Energy consumption, speed, and temperature are typically in normal ranges
- Brake usage and throttle patterns are moderate

### 3. LIME Explanation - Anomalous Instance (`lime_explanation_anomalous.png`)

Shows feature contributions for a correctly classified anomalous instance.

**Interpretation:**
- Features pushing toward anomaly classification have positive weights
- High energy consumption is usually a major contributor
- Temperature and engine load often show positive contributions
- The pattern differs significantly from normal instances

**Typical Pattern for Anomalous Instance:**
- Energy consumption strongly pushes toward anomaly classification
- Extreme values in multiple features (high temp, high load, etc.)
- Different feature importance ranking compared to normal instances

### 4. Side-by-Side Comparison (`lime_comparison.png`)

Compares LIME explanations for normal and anomalous instances side-by-side.

**Key Insights:**
- Shows how feature importance changes between instance types
- Reveals which features are most discriminative
- Helps identify patterns that characterize anomalies
- Demonstrates the local nature of LIME (different instances have different explanations)

### 5. Local Approximation Quality (`local_approximation.png`)

Visualizes how well the local linear model approximates the black-box XGBoost model.

**What it Shows:**
- Scatter plot of perturbed samples
- Point size represents sample weight (closer = larger)
- Color represents sample weight
- Red star marks the explained instance
- Shows the relationship between a feature and the prediction

**Good Approximation Indicators:**
- High R² score (> 0.7)
- Smooth, consistent pattern in the scatter plot
- Dense cluster of points around the explained instance

## Simple Example Output

Running `simple_example.py`:

```bash
python simple_example.py
```

```
LIME Simple Example
============================================================

1. Creating synthetic dataset...
2. Splitting and scaling data...
3. Training Random Forest model...
   Model accuracy: 0.9450

4. Initializing LIME explainer...

5. Explaining instance 0...
   True label: 0
   Model prediction: Class 0=0.9800, Class 1=0.0200

6. Generating LIME explanation...

7. LIME Explanation:
   Local model R² score: 0.2399
   Local model intercept: 0.0605
   Local prediction: 0.0297

   Top contributing features:
   - feature_3: +0.0513 (increases prediction)
   - feature_9: -0.0284 (decreases prediction)
   - feature_6: -0.0167 (decreases prediction)
   - feature_5: -0.0160 (decreases prediction)
   - feature_0: +0.0117 (increases prediction)

============================================================
Example completed successfully!
```

## Interpreting Results

### Understanding Feature Weights

**Positive Weights:**
- Feature value increases the prediction (toward class 1/anomaly)
- Larger positive value = stronger push toward positive class
- Example: `energy_consumption: +0.45` means high energy consumption increases anomaly probability

**Negative Weights:**
- Feature value decreases the prediction (toward class 0/normal)
- Larger negative value = stronger push toward negative class
- Example: `speed: -0.32` means this speed value decreases anomaly probability

### Understanding R² Score

**R² ≥ 0.8:** Excellent approximation
- The local linear model closely matches the black-box model
- High confidence in the explanation

**0.5 ≤ R² < 0.8:** Good approximation
- Reasonable local fit
- Explanation is generally reliable

**R² < 0.5:** Poor approximation
- Local linear model doesn't fit well
- Consider:
  - Increasing `num_samples`
  - Adjusting `kernel_width`
  - The model might be highly non-linear in this region

### Common Patterns in Telematics Data

**Normal Driving:**
- Moderate speed (50-70 km/h)
- Low acceleration variance
- Normal engine temperature (85-95°C)
- Energy consumption: 0.15-0.25 kWh/km
- Smooth throttle/brake patterns

**Anomalous Patterns:**
1. **Inefficient Driving:**
   - High energy at low speed
   - High engine load
   - Excessive throttle

2. **Aggressive Driving:**
   - High acceleration/deceleration
   - Frequent braking
   - Elevated energy consumption

3. **Mechanical Issues:**
   - High RPM with low efficiency
   - Unusual temperature readings
   - Poor fuel rate

4. **Temperature Problems:**
   - Overheating (>110°C)
   - Reduced efficiency
   - Elevated engine load

## Customizing LIME Parameters

### For Better Explanations:

```python
# More stable, higher quality (slower)
explanation = explainer.explain_instance(
    instance=instance,
    predict_fn=predict_fn,
    num_samples=10000,      # More samples
    num_features=10,        # More features
    kernel_width=0.15,      # More local
    std_dev=0.05           # Smaller perturbations
)

# Faster, less detailed (quicker)
explanation = explainer.explain_instance(
    instance=instance,
    predict_fn=predict_fn,
    num_samples=2000,       # Fewer samples
    num_features=5,         # Fewer features
    kernel_width=0.5,       # Less local
    std_dev=0.2            # Larger perturbations
)
```

## Troubleshooting

### Low R² Scores

**Problem:** R² < 0.3 consistently

**Solutions:**
1. Increase `num_samples` to 10000+
2. Decrease `kernel_width` to 0.1-0.15
3. Decrease `std_dev` to 0.05
4. Check if the model is highly non-linear

### Inconsistent Explanations

**Problem:** Running twice gives very different results

**Solutions:**
1. Set `random_state` in LimeExplainer
2. Increase `num_samples`
3. Use multiple explanations and average results

### No Clear Pattern

**Problem:** All feature weights are very small

**Solutions:**
1. Check if instance is in a "flat" region of decision boundary
2. Verify model is trained correctly
3. Check feature scaling

## Next Steps

1. **Apply to Real Data:** Use real vehicle telematics data instead of synthetic
2. **Experiment with Models:** Try different models (Neural Networks, SVM, etc.)
3. **Feature Engineering:** Add derived features (moving averages, rates of change)
4. **Global Explanations:** Aggregate LIME explanations across many instances
5. **Interactive Dashboard:** Build a web interface for real-time explanations
