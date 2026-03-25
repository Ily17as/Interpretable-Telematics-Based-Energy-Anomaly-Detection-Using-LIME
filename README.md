# Interpretable Telematics-Based Energy Anomaly Detection with LIME

This repository studies **trip-level vehicle energy anomaly detection** and explains suspicious predictions with **LIME-style local surrogate models**.

The core workflow is:

1. build a trip-level table from the Vehicle Energy Dataset (VED);
2. train an **XGBoost regressor** to predict expected `energy_per_km`;
3. define anomalies as trips with unusually large positive residuals;
4. explain anomalous trips with local linear surrogates around each trip.

---

## Dataset

The project was built and tested with the Kaggle-hosted version of VED:

**Kaggle dataset:** <https://www.kaggle.com/datasets/galievilyas/ved-dataset/data>

This is the dataset version you should attach when running the notebooks in Kaggle.

---

## Recommended execution environment

### Why Kaggle is recommended

Although parts of the project can be used locally, the **recommended runtime is Kaggle Notebook environment**.

Why:

- the dataset is already hosted on Kaggle;
- the main notebooks were developed and executed in a Kaggle-style workflow;
- the exported artifacts in the repository match that workflow;
- it is the simplest way to reproduce the baseline end-to-end without path and storage issues.

### Practical recommendation

> **Run the main notebooks in Kaggle first.**  
> Use the local repository as the structured code, report, and packaging view of the project.

---

## Repository structure

```text
Interpretable-Telematics-Based-Energy-Anomaly-Detection-Using-LIME-main/
├── LIME_implementation/
│   ├── final-telematics-lime.ipynb
│   └── output.zip
├── regression_model/
│   ├── ved-energy-regression.ipynb
│   ├── models/
│   │   ├── xgb_energy_artifact.joblib
│   │   └── xgb_energy_regressor.joblib
│   └── outputs/
│       └── residuals.parquet
├── src/
│   ├── anomaly/
│   ├── cli/
│   ├── modeling/
│   ├── utils/
│   ├── viz/
│   └── xai/
├── tests/
│   ├── test_explain_trip_cli.py
│   ├── test_thresholds.py
│   └── test_weighted_ridge.py
├── README.md
├── requirements.txt
└── XAI_Course_Project_Proposal.pdf
```

---

## What is already implemented

The repository already contains:

- a baseline regression workflow for trip-level energy prediction;
- a full end-to-end notebook for anomaly detection and local explanation;
- saved model artifacts;
- saved scored trip tables;
- saved tuning tables for XGBoost and LIME;
- saved explanation summaries, JSON files, and bar plots;
- a modular `src/` package for anomaly logic, artifact loading, plotting, and LIME utilities;
- smoke tests for thresholding, surrogate fitting, and CLI behavior.

---

## Where the main logic currently lives

The project is still primarily **notebook-driven**.

Measured from the current repository:

- there are **2 main notebooks**;
- together they contain roughly **1847 lines of code**;
- the modular `src/` package contains roughly **1327 lines of Python code** (excluding tests);
- therefore, about **58%** of the notebook + `src/` implementation lives in notebooks.

So the current situation is:

- **notebooks = main research workflow**;
- **`src/` = support layer for reproducibility, CLI usage, and cleaner packaging**.

This is important for anyone grading or reusing the repository: the best way to understand the full project is still to start with the notebooks.

---

## Main notebooks

### 1. `regression_model/ved-energy-regression.ipynb`

Purpose:

- builds the baseline trip-level energy regressor;
- trains and evaluates the XGBoost model;
- exports prediction artifacts.

### 2. `LIME_implementation/final-telematics-lime.ipynb`

Purpose:

- runs the end-to-end anomaly + explanation workflow;
- scores trips with the trained model;
- identifies anomalous trips;
- tunes LIME settings;
- generates local explanation outputs.

If you only want the full baseline workflow, this second notebook is the most important one.

---

## Modular `src/` package

The `src/` package contains a cleaner implementation of reusable parts of the pipeline.

### `src/anomaly/`
Residual thresholding and anomaly-table generation.

### `src/cli/`
Command-line entry points such as anomaly detection and trip explanation.

### `src/modeling/`
Helpers for design-matrix construction and feature alignment.

### `src/utils/`
Artifact loading, I/O helpers, and schema utilities.

### `src/viz/`
Plotting utilities for reporting and inspection.

### `src/xai/`
LIME-related utilities: perturbation, kernel weighting, surrogate fitting, and fidelity metrics.

---

## Current shipped artifacts

The repository and shipped output package already include artifacts that demonstrate a working baseline:

- `outputs_final/cache/trip_table_scored.csv.gz`;
- `outputs_final/cache/xgb_tuning_results.csv`;
- `outputs_final/cache/lime_tuning_results.csv`;
- `outputs_final/models/xgb_energy_artifact.joblib`;
- `outputs_final/lime_explanations/lime_explanation_summary.csv`;
- explanation JSON files and bar plots for multiple anomalous trips.

These are enough to show that the project is already beyond the proposal stage.

---

## Current baseline status

From the current shipped artifacts:

- total trips: **3249**;
- split sizes: **1957 train / 594 val / 698 test**;
- current test metrics from shipped scored table: **MAE 0.0670 / RMSE 0.1332 / R² 0.5377**;
- flagged test anomalies: **41** out of **698** test trips (**5.87%**);
- generated explanation cases: **4**;
- mean local explanation fidelity in shipped summary: **local R² ≈ 0.7970**.

These numbers describe the **current repository outputs as shipped**, not an idealized future version.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Recommended workflow

### Option A — recommended: run in Kaggle

1. Open a Kaggle notebook.
2. Attach the dataset: `galievilyas/ved-dataset`.
3. Upload or clone the repository.
4. Run:
   - `regression_model/ved-energy-regression.ipynb`
   - then `LIME_implementation/final-telematics-lime.ipynb`
5. Export outputs if needed.

### Option B — local inspection / partial reuse

Use the repository locally to:

- inspect the notebooks;
- reuse the modular `src/` code;
- read the reports and artifacts;
- run smoke tests.

This is convenient for development, but it is **not the primary recommended path** for first reproduction.

---

## Why this repository is structured this way

This project evolved from exploratory research notebooks into a more structured repository. Because of that, the repository currently has **two complementary forms**:

1. **notebook form** — where the main experimental logic lives;
2. **package form** — where reusable anomaly/XAI utilities are collected.

This hybrid structure is useful for a course project because it preserves the original experimental workflow while also improving readability and reproducibility.

---

## Limitations

- the workflow is still primarily notebook-centric;
- residual anomalies are not direct fault labels;
- LIME explanations are local and can vary with perturbation settings;
- trip aggregation loses sequence information;
- some exogenous factors may still be missing.

---

## References

1. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). **“Why Should I Trust You?”: Explaining the Predictions of Any Classifier.** KDD 2016.  
2. Oh, G. S., LeBlanc, D. J., & Peng, H. (2019). **Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy Consumption Research.** arXiv:1905.02081.  
3. Chen, T., & Guestrin, C. (2016). **XGBoost: A Scalable Tree Boosting System.** KDD 2016.

---

## License

MIT
