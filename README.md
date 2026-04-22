# Explaining Vehicle Energy Anomalies in Telematics

A tutorial-style case study on detecting unexpectedly energy-inefficient vehicle trips with XGBoost and explaining those alerts with manual SHAP-style attributions and LIME.

## Project Summary

The project models expected trip-level `energy_per_km` from telematics features, flags trips with unusually high positive residuals, and explains those alerts with two post-hoc methods:

- **Manual SHAP-style attribution**: a permutation-based Shapley approximation implemented in the explanation notebook, without using the `shap` library.
- **LIME**: a local surrogate baseline implemented in `src/xai/` and used for comparison.

The workflow also includes a trustworthiness audit covering:

- leakage-aware modeling
- validation-only threshold calibration
- local explanation stability
- subgroup robustness

## Main Results

On the shipped baseline test split:

- **MAE:** 0.0670
- **RMSE:** 0.1332
- **R2:** 0.5377
- **Test trips:** 698
- **Flagged anomalies:** 41
- **Anomaly rate:** 5.87%

The audit shows that leakage can substantially inflate apparent performance, LIME fidelity varies by trip, and the `Transmission = NO DATA` subgroup has much higher error and anomaly rate than the CVT subgroup.

## Repository Structure

```text
docs/
  blogpost_draft.md      Final narrative draft
  results_tables.md      Final metrics and interpretation tables
  reproducibility.md     Notebook order and output locations

notebooks/
  05_blogpost_main_flow.ipynb
  06_shap_vs_lime.ipynb
  07_trustworthiness_audit.ipynb
  baseline_implementation/
    final-telematics-lime.ipynb
    output.zip           Compact baseline artifact used by final notebooks

src/
  anomaly/               Residual thresholding and anomaly table generation
  audit/                 Calibration, leakage, stability, subgroup checks
  cli/                   Command-line entry points
  modeling/              Feature design helpers
  utils/                 I/O, artifact, schema helpers
  viz/                   Plotting utilities
  xai/                   LIME perturbation, kernel, surrogate, fidelity code

outputs_blogpost/
  figures/               Main figures
  tables/                Summary CSVs
  audit/                 Audit figures and tables
  xai/                   Manual SHAP-style, LIME, and comparison outputs

tests/
  Unit tests for thresholds, LIME surrogate logic, CLI helpers, and audits
```

## Recommended Notebook Order

1. `regression_model/ved-energy-regression.ipynb`
2. `notebooks/05_blogpost_main_flow.ipynb`
3. `notebooks/06_shap_vs_lime.ipynb`
4. `notebooks/07_trustworthiness_audit.ipynb`

## Final Deliverables

- [docs/blogpost_draft.md](docs/blogpost_draft.md)
- [docs/results_tables.md](docs/results_tables.md)
- [docs/reproducibility.md](docs/reproducibility.md)
- final figures and tables under `outputs_blogpost/`
- reusable LIME/anomaly/audit utilities under `src/`

## Interpretation Note

This is a decision-support workflow, not a direct mechanical diagnostic tool. A flagged trip means "unexpected under the model", not "definitely faulty".
