# VED Energy Baseline: `src/` usage

This folder contains the **post-training pipeline** for the VED coursework project:
- detect anomalous trips using **residuals** (`actual - predicted`)
- generate **LIME** explanations for selected trips
- (optional) save basic plots for reporting

> **Assumption:** you already ran the notebook and produced:
> - `regression_model/outputs/residuals.parquet` (trip-level table with features + actual/predicted/residual)
> - `regression_model/models/xgb_energy_regressor.joblib` (trained XGBoost model)

---

## Folder structure

```
src/
  anomaly/        # residual thresholding + anomaly table generation
  xai/            # LIME (perturbations, kernel weights, weighted ridge surrogate, fidelity)
  viz/            # helper plots (residual histogram, actual vs predicted, LIME bar)
  cli/            # runnable scripts (detect anomalies, explain a trip)
  utils/          # IO helpers + column/config schema
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## 1) Detect anomalies (from residuals)

Creates an anomalies table based on a threshold (quantile or MAD-z).

### Quantile threshold (recommended baseline)

```bash
python -m src.cli.detect_anomalies \
  --residuals_path outputs/residuals.parquet \
  --method quantile --quantile 0.98 \
  --out_anomalies_path outputs/anomalies.parquet \
  --out_meta_path outputs/anomaly_config.json
```

Output:
- `outputs/anomalies.parquet` — anomalous trips (ranked)
- `outputs/anomaly_config.json` — threshold metadata

### MAD-z (robust thresholding)

```bash
python -m src.cli.detect_anomalies \
  --residuals_path outputs/residuals.parquet \
  --method mad_z --mad_z 3.5 \
  --out_anomalies_path outputs/anomalies.parquet \
  --out_meta_path outputs/anomaly_config.json
```

---

## 2) Explain a specific trip with LIME

Generates a local explanation around one trip using tabular LIME:
- perturb samples around the trip
- query the black-box regressor
- fit a weighted ridge surrogate
- save top feature contributions

```bash
python -m src.cli.explain_trip \
  --residuals_path outputs/residuals.parquet \
  --model_path regression_model/models/xgb_energy_regressor.joblib \
  --trip_id <TRIP_ID> \
  --n_samples 5000 \
  --top_k 10
```

Outputs:
- `outputs/explanations/trip_<TRIP_ID>_lime.json`
- `outputs/explanations/trip_<TRIP_ID>_lime_bar.png`

Key fields in the JSON:
- `top_features`: top local contributions (beta * x0)
- `local_r2`, `local_rmse`: surrogate fidelity near the instance
- `kernel_width`: kernel width used for proximity weighting

---

## 3) (Optional) Basic plots for reporting

You can call plotting utilities from a notebook:

```python
import pandas as pd
from src.viz.plots import plot_residual_hist, plot_actual_vs_pred

df = pd.read_parquet("outputs/residuals.parquet")
plot_residual_hist(df, residual_col="residual", out_path="outputs/residual_hist.png")
plot_actual_vs_pred(df, y_col="energy_per_km", yhat_col="predicted_energy_per_km", out_path="outputs/actual_vs_pred.png")
```

---

## Notes / common pitfalls

- **Feature mismatch errors**: ensure `residuals.parquet` includes the same feature columns that were used to train the model.
- **Categoricals**: the pipeline uses `pandas.get_dummies` (one-hot). If categories differ between splits, always reindex to a shared column set.
- **LIME stability**: increase `--n_samples` (e.g., 10000) and/or increase ridge regularization if explanations are noisy.
