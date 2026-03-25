# Interpretable Telematics-Based Energy Anomaly Detection with LIME

## What this repository does

This project studies **energy anomaly detection for vehicle trips** and explains suspicious predictions with **LIME**.

The core pipeline is:

1. build a trip-level table from the Vehicle Energy Dataset (VED)
2. train an **XGBoost regressor** to predict expected energy consumption per km
3. define anomalies as trips with unusually large positive residuals
4. explain anomalous trips with **local linear surrogates** around each trip

## Why this version is stronger

The previous repository mixed together two different artifact formats and the explanation CLI could not be reproduced from the files that were committed. This refactored version makes the workflow explicit:

- `regression_model/` stores baseline artifacts
- `LIME_implementation/final_telematics_lime_project.ipynb` is the full research notebook
- `src/` contains a cleaner, modular pipeline for anomaly detection and explanation
- `docs/` contains a polished proposal and a baseline report template
- `tests/` contains small smoke tests for the main mathematical components

## Research question

**Can we detect trips with unexpectedly high energy consumption and provide locally faithful, human-readable explanations of which telematics factors made those trips look anomalous?**

This formulation is stronger than “use LIME on XGBoost” because it defines:

- the prediction target
- the anomaly criterion
- the explainability objective
- the evaluation target for the final report

## Repository structure

```text
LIME_implementation/
  final_telematics_lime_project.ipynb   # full end-to-end notebook
  README_run_xai_project.md

regression_model/
  ved-energy-regression.ipynb           # baseline model notebook
  models/
  outputs/

src/
  anomaly/                             # threshold logic
  cli/                                 # command line entry points
  modeling/                            # raw-feature to design-matrix helpers
  utils/                               # I/O and artifact loading
  viz/                                 # plotting
  xai/                                 # LIME utilities

docs/
  IMPROVED_PROPOSAL.md
  BASELINE_REPORT_TEMPLATE.md
  TA_CHECKLIST.md

tests/
  test_thresholds.py
  test_weighted_ridge.py
```

## Important reproducibility note

There are **two artifact styles** in this repository:

1. **baseline regression artifact**
   - saved in `regression_model/models/xgb_energy_regressor.joblib`
   - contains a dictionary with the trained model and the design columns
2. **full XAI artifact**
   - produced by `LIME_implementation/final_telematics_lime_project.ipynb`
   - contains raw-feature metadata, preprocessing info, background samples, and model metrics

For trip-level explanations, the recommended path is the **full XAI artifact**, because LIME needs the raw trip features and a background distribution.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Baseline anomaly detection

```bash
python -m src.cli.detect_anomalies   --residuals_path regression_model/outputs/residuals.parquet   --out_anomalies_path outputs/anomalies.parquet   --out_meta_path outputs/anomaly_config.json
```

## Trip explanation (recommended workflow)

First run the full notebook and produce:

- `outputs_final/cache/trip_table_scored.csv.gz`
- `outputs_final/models/xgb_energy_artifact.joblib`

Then explain a trip:

```bash
python -m src.cli.explain_trip   --scored_table_path outputs_final/cache/trip_table_scored.csv.gz   --artifact_path outputs_final/models/xgb_energy_artifact.joblib   --trip_id 8_706   --out_dir outputs/explanations
```

## Why the final report should be convincing

A good final submission should clearly separate four questions:

1. **Prediction quality** — how well does XGBoost estimate expected energy consumption?
2. **Anomaly logic** — why is a residual-based anomaly definition reasonable here?
3. **Explanation quality** — how locally faithful is the surrogate? (local R², local RMSE)
4. **Practical usefulness** — do the explanations point to understandable trip factors such as speed variance, idling, distance, acceleration, and vehicle properties?

## Known limitations

- residual-based anomaly detection can confuse true faults with unusual but benign driving conditions
- LIME explanations are local and can vary with perturbation settings
- explanations are only as good as the feature engineering used upstream
- the repository does not ship the full raw VED dataset, so users must recreate the scored trip table from the notebook

## References to cite in the report

1. Ribeiro, Singh, Guestrin. *“Why Should I Trust You?”: Explaining the Predictions of Any Classifier.* KDD 2016.
2. Oh, LeBlanc, Peng. *Vehicle Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy Consumption Research.* arXiv:1905.02081.
3. Chen, Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD 2016.

## License

MIT
