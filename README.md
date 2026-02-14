# Interpretable Telematics-Based Energy Anomaly Detection Using LIME

This project implements **LIME (Local Interpretable Model-Agnostic Explanations) from scratch** to explain predictions of an **XGBoost** model for vehicle telematics energy anomaly detection and predictive maintenance.

## Overview

This implementation demonstrates how LIME can make black-box machine learning models interpretable by explaining individual predictions through local linear approximations. The project focuses on vehicle telematics data to detect energy consumption anomalies.

## Features

- ✅ **LIME Implementation from Scratch**: Complete implementation of the LIME algorithm without using external LIME libraries
- ✅ **XGBoost Model**: Gradient Boosting classifier for anomaly detection
- ✅ **Synthetic Telematics Data**: Realistic vehicle telematics data generator with normal and anomalous patterns
- ✅ **Comprehensive Visualizations**: Multiple visualization types to understand model predictions and explanations
- ✅ **Educational Code**: Well-documented code suitable for learning XAI (Explainable AI) concepts

## Project Structure

```
.
├── lime_explainer.py              # LIME implementation from scratch
├── telematics_data_generator.py   # Synthetic telematics data generator
├── visualization.py                # Visualization utilities for LIME explanations
├── main.py                         # Main demonstration script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Ily17as/Interpretable-Telematics-Based-Energy-Anomaly-Detection-Using-LIME.git
cd Interpretable-Telematics-Based-Energy-Anomaly-Detection-Using-LIME
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main demonstration script:

```bash
python main.py
```

This will:
1. Generate synthetic vehicle telematics data
2. Train an XGBoost model for anomaly detection
3. Generate LIME explanations for sample predictions
4. Create visualization plots

### Expected Output

The script will generate several PNG files:
- `dataset_distribution.png`: Shows feature distributions for normal vs anomalous samples
- `lime_explanation_normal.png`: LIME explanation for a normal instance
- `lime_explanation_anomalous.png`: LIME explanation for an anomalous instance
- `lime_comparison.png`: Side-by-side comparison of explanations
- `local_approximation.png`: Visualization of local model approximation

## How LIME Works

LIME explains predictions by:

1. **Generating a local neighborhood**: Creating perturbed samples around the instance to explain
2. **Getting predictions**: Using the black-box model to predict on perturbed samples
3. **Weighting samples**: Assigning higher weights to samples closer to the original instance
4. **Fitting a local model**: Training an interpretable linear model on the weighted samples
5. **Extracting explanations**: Using the linear model's coefficients as feature importances

## Telematics Features

The synthetic dataset includes realistic vehicle telematics features:

- **speed**: Vehicle speed (km/h)
- **acceleration**: Acceleration rate (m/s²)
- **rpm**: Engine revolutions per minute
- **engine_load**: Engine load percentage
- **throttle**: Throttle position percentage
- **brake**: Brake pressure (bar)
- **temperature**: Engine temperature (°C)
- **fuel_rate**: Fuel consumption rate (L/h)
- **energy_consumption**: Energy consumption (kWh/km) - target for anomaly detection

## Anomaly Types

The data generator creates four types of energy consumption anomalies:

1. **High energy at low speed**: Inefficient driving patterns
2. **Excessive acceleration/braking**: Aggressive driving behavior
3. **High RPM with low efficiency**: Poor gear selection or mechanical issues
4. **Temperature anomalies**: Overheating affecting efficiency

## Key Components

### LimeExplainer Class

The core LIME implementation includes:
- `generate_neighborhood()`: Creates perturbed samples
- `compute_weights()`: Calculates sample weights using exponential kernel
- `explain_instance()`: Generates explanation for a single prediction

### TelematicsDataGenerator Class

Generates synthetic vehicle telematics data:
- `generate_normal_data()`: Creates normal driving patterns
- `generate_anomalous_data()`: Creates various anomaly types
- `generate_dataset()`: Combines normal and anomalous data

### Visualization Functions

Multiple visualization utilities:
- `plot_lime_explanation()`: Bar chart of feature contributions
- `plot_feature_importance_comparison()`: Compare multiple explanations
- `plot_local_approximation()`: Show local model fit quality
- `plot_dataset_distribution()`: Feature distributions

## Educational Value

This implementation is ideal for:
- **XAI Course Projects**: Understanding explainable AI concepts
- **Machine Learning Education**: Learning how to interpret black-box models
- **Research**: Baseline for LIME-based research projects
- **Industry Applications**: Template for real-world telematics analysis

## References

- Original LIME Paper: ["Why Should I Trust You?" Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938)
- XGBoost: [XGBoost Documentation](https://xgboost.readthedocs.io/)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- LIME algorithm by Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin
- XGBoost library by DMLC team
- Inspired by real-world vehicle telematics and predictive maintenance applications
