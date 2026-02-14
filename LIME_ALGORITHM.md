# LIME Algorithm Implementation Details

## Overview

This document provides a detailed explanation of the LIME (Local Interpretable Model-Agnostic Explanations) algorithm as implemented from scratch in this project.

## Algorithm Steps

### 1. Local Neighborhood Generation

**Purpose**: Create perturbed samples around the instance we want to explain.

**Implementation**:
```python
def generate_neighborhood(self, instance, num_samples=5000, std_dev=0.1):
    perturbations = np.random.normal(0, std_dev, size=(num_samples, num_features))
    perturbed_samples = instance + perturbations * np.std(instance)
    return perturbed_samples
```

**Key Concepts**:
- We generate `num_samples` variations of the original instance
- Perturbations follow a normal distribution with mean=0 and configurable standard deviation
- The perturbations are scaled by the standard deviation of the instance to maintain reasonable variations
- This creates a "local" neighborhood around the instance

### 2. Black-Box Model Predictions

**Purpose**: Get predictions from the complex model for all perturbed samples.

**Implementation**:
- We use the `predict_fn` provided by the user (XGBoost in our case)
- For binary classification, we extract the probability of the positive class
- These predictions represent the "ground truth" that our local model will approximate

### 3. Sample Weighting with Exponential Kernel

**Purpose**: Assign higher weights to samples closer to the instance being explained.

**Implementation**:
```python
def compute_weights(self, instance, perturbed_samples, kernel_width=0.25):
    distances = pairwise_distances(perturbed_samples, instance.reshape(1, -1), metric='euclidean').ravel()
    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
    return weights
```

**Key Concepts**:
- Euclidean distance measures how far each perturbed sample is from the original instance
- Exponential kernel: w(z) = exp(-(d²) / (kernel_width²))
- Samples closer to the instance get weights close to 1
- Samples farther away get weights close to 0
- `kernel_width` controls how quickly weights decay with distance

### 4. Local Linear Model Fitting

**Purpose**: Fit an interpretable linear model that approximates the black-box model locally.

**Implementation**:
```python
linear_model = Ridge(alpha=1.0, fit_intercept=True)
linear_model.fit(perturbed_samples, predictions, sample_weight=weights)
```

**Key Concepts**:
- We use Ridge regression (L2 regularization) to prevent overfitting
- The model is weighted by sample importance (closer samples matter more)
- The resulting linear model is simple and interpretable
- Coefficients represent feature importance/contribution

### 5. Explanation Generation

**Purpose**: Extract and present the most important features and their contributions.

**Implementation**:
```python
coefficients = linear_model.coef_
feature_importance = [(i, coeff) for i, coeff in enumerate(coefficients)]
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
top_features = feature_importance[:num_features]
```

**Key Concepts**:
- Positive coefficients indicate the feature increases the prediction
- Negative coefficients indicate the feature decreases the prediction
- Larger absolute values indicate stronger influence
- We return the top-k most influential features

## Mathematical Foundation

### Objective Function

LIME solves the following optimization problem:

```
ξ(x) = argmin L(f, g, π_x) + Ω(g)
        g∈G
```

Where:
- `x` is the instance being explained
- `f` is the black-box model
- `g` is the interpretable model (linear in our case)
- `G` is the class of interpretable models
- `L` is the loss function measuring how well g approximates f locally
- `π_x` is the proximity measure (exponential kernel)
- `Ω(g)` is the complexity measure (penalizes complex models)

### Loss Function

In our implementation:
```
L(f, g, π_x) = Σ π_x(z) * (f(z) - g(z))²
```

This is a weighted mean squared error where:
- `z` ranges over perturbed samples
- `π_x(z) = exp(-(d(x,z)²) / kernel_width²)` is the weight
- `f(z)` is the black-box prediction
- `g(z)` is the linear model prediction

## Hyperparameters

### num_samples (default: 5000)
- Number of perturbed samples to generate
- More samples → better approximation but slower
- Typical range: 1000-10000

### kernel_width (default: 0.25)
- Controls locality of the explanation
- Smaller values → more local (trust nearby samples more)
- Larger values → less local (consider wider neighborhood)
- Typical range: 0.1-1.0

### std_dev (default: 0.1)
- Standard deviation for perturbations
- Smaller values → perturbations closer to original
- Larger values → explore wider variations
- Typical range: 0.05-0.3

### num_features (default: 10)
- Number of top features to include in explanation
- Should be small enough to be interpretable
- Typical range: 5-15

## Model Fidelity Metrics

### R² Score
The R² score measures how well the local linear model approximates the black-box model:

```
R² = 1 - (SS_res / SS_tot)
```

Where:
- `SS_res = Σ w_i * (y_i - ŷ_i)²` (weighted residual sum of squares)
- `SS_tot = Σ w_i * (y_i - ȳ)²` (weighted total sum of squares)

**Interpretation**:
- R² ≈ 1: Excellent local approximation
- R² ≈ 0.8-0.9: Good approximation
- R² < 0.5: Poor approximation (consider adjusting hyperparameters)

## Advantages of LIME

1. **Model-Agnostic**: Works with any black-box model
2. **Locally Faithful**: Accurate explanations for individual predictions
3. **Interpretable**: Uses simple linear models
4. **Flexible**: Can explain any type of prediction (classification, regression)

## Limitations

1. **Instability**: Explanations can vary between runs
2. **Local Only**: Doesn't explain global model behavior
3. **Sampling Dependent**: Quality depends on neighborhood sampling
4. **Computational Cost**: Requires many model predictions

## Implementation Differences from Original LIME

Our implementation is simplified for educational purposes:

1. **No discretization**: We work with continuous features directly
2. **Simpler perturbation**: Gaussian noise instead of more complex sampling
3. **No feature selection**: We rank all features and return top-k
4. **Single kernel**: Only exponential kernel (original supports multiple)

## Best Practices

1. **Validate R² scores**: Check that local model is a good approximation
2. **Experiment with kernel_width**: Adjust based on your data
3. **Use enough samples**: At least 1000, preferably 5000+
4. **Interpret carefully**: Remember this is a local explanation
5. **Check multiple instances**: Don't rely on a single explanation

## References

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should i trust you?" Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 1135-1144).

2. Original LIME Paper: https://arxiv.org/abs/1602.04938

3. LIME GitHub Repository: https://github.com/marcotcr/lime
