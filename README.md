# Interpretable Telematics-Based Energy Anomaly Detection (LIME + XGBoost)

Explainable anomaly detection for **vehicle energy consumption** using:

- **XGBoost regression** to predict *expected (normal) energy consumption per km*
- **Residual-based anomaly detection**: `residual = actual_energy - predicted_energy`
- **LIME (from scratch)** for local, human-interpretable explanations of suspicious trips

> Course project (proposal date: 2026-02-12)

---

## Problem

Modern vehicles generate large volumes of **OBD + GPS telematics**, but drivers usually can’t interpret early signs of vehicle degradation.

This project detects **abnormal energy consumption patterns** and explains *why* a trip looks suspicious, improving transparency and user trust.

---

## Dataset

**Vehicle Energy Dataset (VED)**

- Real-world OBD and GPS telematics
- 383 vehicles
- ~600k km of driving data
- Multivariate time-series (e.g., speed, acceleration, energy consumption)

---

## Method Overview

### 1) ML model (black-box)

**Model:** Gradient Boosting (**XGBoost**) for regression.

**Target:** expected energy consumption per km.

**Features (aggregated per trip):**
- mean/variance of speed
- acceleration statistics
- stop-and-go ratio
- idle time
- time-of-day / seasonality

### 2) Anomaly definition

We flag trips with unusually high residual:

```text
residual = actual_energy - predicted_energy
```

High residual → unexplained excess energy use → potential vehicle issue.

### 3) Explainability (LIME)

We implement **LIME** for tabular trip features to explain *local behavior* of the XGBoost model around each anomalous trip.

**Core objective (LIME):**

\[
\hat{g} = \arg\min_{g \in G} \sum_{i=1}^{N} \pi_x(z_i)\,(f(z_i) - g(z_i))^2 + \Omega(g)
\]

Where:
- `f` is the black-box model (XGBoost)
- `g` is a simple local surrogate (linear model)
- `z_i` are perturbed samples around the selected trip `x`
- `π_x(z_i)` is locality weighting
- `Ω(g)` is a complexity penalty (encourages simplicity)

---

## Deliverables

- Reproducible Python code (this repo)
- Trained anomaly detection model (XGBoost + residual thresholding)
- 3–5 LIME explanation case studies
- Visualizations:
  - predicted vs actual energy
  - LIME local feature importance per anomaly
- Validation with regression metrics: **MAE / RMSE**
- Technical write-up / blog-style report

---

## Repository Structure (suggested)

```text
.
├── data/                      # (optional) dataset pointers / processed samples (avoid uploading raw VED if restricted)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_train_xgboost.ipynb
│   └── 03_lime_explanations.ipynb
├── src/
│   ├── features.py            # trip aggregation and feature engineering
│   ├── model.py               # XGBoost training + evaluation
│   ├── anomaly.py             # residual computation + thresholding
│   ├── lime_from_scratch.py   # LIME implementation
│   └── viz.py                 # plots (pred vs actual, feature importances)
├── reports/                   # figures / short write-up (optional)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run pipeline (recommended order)

1. **Feature engineering** (aggregate trip-level features)
2. **Train XGBoost** (predict expected energy)
3. **Compute residuals** + select anomalies
4. **Run LIME** on selected anomalies
5. **Generate plots** and save explanation examples

---

## Evaluation

### Regression
- MAE
- RMSE

### Explanation artifacts
- For each anomalous trip:
  - top positive contributors (features increasing predicted energy)
  - top negative contributors (features decreasing predicted energy)
  - confidence/quality diagnostics (optional)

---

## Timeline (from proposal)

- 2026-02-12 — Proposal submission
- 2026-03-12 — Base implementation (ML model + anomaly detection)
- 2026-04-15 — Visualizations and testing (LIME)
- 2026-04-21 — Final presentation

---

## Notes

- If VED licensing restricts redistribution, store only **data preparation scripts** and provide instructions for obtaining the dataset.
- Keep all experiments **reproducible**: fixed random seeds, documented environment, and consistent preprocessing.

---

## Authors

- Ilyas Galiev  
- Daria Alexandrova  
- Kamilya Shakirova  
