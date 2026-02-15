# Interpretable Telematics-Based Energy Anomaly Detection using LIME

## Overview

This project implements an interpretable machine learning pipeline for
detecting anomalous vehicle trips based on telematics-derived energy
consumption.

The system consists of:

1.  **Energy Consumption Regression Model (XGBoost)**
2.  **Residual-based Anomaly Detection**
3.  **Local Explanations using LIME**
4.  **Visualization and CLI tools for analysis**

The primary objective is not only to detect abnormal trips, but also to
provide **human-interpretable explanations** describing why a trip is
considered anomalous.

------------------------------------------------------------------------

## Problem Statement

Given vehicle telematics data, we:

-   Train a regression model to predict expected energy consumption.
-   Compute residuals:

```{=html}
<!-- -->
```
    residual = actual_energy - predicted_energy

-   Detect anomalous trips using residual thresholds.
-   Generate local explanations using LIME to identify which features
    contribute most to the anomaly.

------------------------------------------------------------------------

## Repository Structure

    regression_model/
        ved-energy-regression.ipynb
        models/
        outputs/

    src/
        anomaly/
        cli/
        utils/
        viz/
        xai/

### regression_model/

Contains model training artifacts: - XGBoost energy regressor - Saved
model artifacts (`.joblib`) - Residual tables (`.parquet`)

### src/

Core production-style pipeline.

-   `anomaly/` --- threshold logic and anomaly generation
-   `cli/` --- command line interfaces
-   `utils/` --- I/O and schema definitions
-   `viz/` --- visualization utilities
-   `xai/` --- LIME implementation components:
    -   perturbation
    -   kernel weighting
    -   surrogate modeling
    -   fidelity evaluation

------------------------------------------------------------------------

## Installation

Create a virtual environment and install dependencies:

``` bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Training the Energy Model

Open:

    regression_model/ved-energy-regression.ipynb

This notebook:

-   Loads and preprocesses telematics data
-   Performs feature engineering
-   Splits data by vehicle ID (no leakage)
-   Trains XGBoost regressor
-   Saves:
    -   trained model
    -   residuals table

------------------------------------------------------------------------

## Detecting Anomalies

Example CLI usage:

``` bash
python -m src.cli.detect_anomalies --input residuals.parquet
```

An anomaly is defined via configurable residual thresholding.

------------------------------------------------------------------------

## Explaining a Trip

``` bash
python -m src.cli.explain_trip --trip_id <ID>
```

This will:

1.  Generate perturbed samples around the trip
2.  Query the black-box XGBoost model
3.  Fit a local linear surrogate
4.  Output feature importance explanation

------------------------------------------------------------------------

## XAI Methodology

We implement a tabular variant of **LIME**:

-   Local sampling around a target trip
-   Kernel-based proximity weighting
-   Linear surrogate fit
-   Feature attribution extraction

Fidelity metrics are computed to assess explanation reliability.

------------------------------------------------------------------------

## Research Contribution

This project demonstrates:

-   Residual-based anomaly detection for telematics
-   Practical LIME implementation from scratch
-   End-to-end interpretable ML workflow
-   CLI integration for reproducibility

------------------------------------------------------------------------

## Reproducibility

To reproduce results:

1.  Install dependencies
2.  Train the regression model
3.  Generate residuals
4.  Run anomaly detection
5.  Generate explanations

------------------------------------------------------------------------

## Requirements

See `requirements.txt`

------------------------------------------------------------------------

## License

MIT License
