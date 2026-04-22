# Explaining Vehicle Energy Anomalies in Telematics: A Tutorial Case Study with XGBoost, Manual SHAP, and LIME

> **One-line summary.** We build a tutorial-style workflow for detecting unexpectedly energy-inefficient trips and explaining those alerts with modern post-hoc XAI, while auditing when the explanations deserve trust.

## 1. Why anomaly scores are not enough

Vehicle telematics produces rich signals, but an anomaly score by itself is rarely actionable. A trip may look inefficient because of stop-go traffic, prolonged idling, unusual route conditions, sensor issues, or genuinely abnormal behavior. A useful system therefore needs both:
1. a way to flag unexpected trips, and
2. a way to explain *why* the model found them unusual.

Our goal is not to diagnose faults directly. Instead, we identify trips with unusually high energy consumption relative to a learned baseline and explain the model’s reasoning in a human-readable way.

## 2. Problem setup

We use the **Vehicle Energy Dataset (VED)** and aggregate each trip into a single tabular row. The target is trip-level:

`energy_per_km`

The black-box model is an **XGBoost regressor**. The anomaly score is the positive residual:

\[
\text{residual} = \text{actual\_energy\_per\_km} - \text{predicted\_energy\_per\_km}
\]

A trip is treated as anomalous when this positive residual exceeds the selected threshold.

### Main predictive result
On the shipped baseline test split:
- **MAE:** 0.0670
- **RMSE:** 0.1332
- **R²:** 0.5377
- **Test trips:** 698
- **Flagged anomalies:** 41
- **Anomaly rate:** 5.87%

Final figures:
- `outputs_blogpost/figures/pipeline_diagram.png`
- `outputs_blogpost/figures/predicted_vs_actual_test.png`
- `outputs_blogpost/figures/residual_distribution_test.png`

## 3. Why XGBoost?

XGBoost is a strong baseline for heterogeneous tabular telematics data:
- nonlinear feature interactions,
- mixed-scale numeric inputs,
- robust performance without sequence-model overhead,
- natural fit for aggregated trip-level features.

The model predicts expected energy use from features such as:
- speed statistics,
- acceleration statistics,
- duration and distance,
- idle time,
- vehicle descriptors such as transmission and drive wheels.

## 4. Why XAI is needed here

The anomaly score tells us **what** happened: the trip consumed more energy than expected.  
XAI tells us **why** the model made that judgment.

For this project, we compare two explanation approaches:
- **Manual SHAP**, implemented from first principles using Shapley-style attribution
- **LIME**, a local surrogate model used as a baseline explanation method

This pairing is useful pedagogically:
- SHAP provides additive feature attributions grounded in cooperative game theory,
- LIME highlights the strengths and weaknesses of local surrogate explanations.

## 5. Manual SHAP as the primary method

Instead of using the `shap` library, we implement a manual Shapley-style explainer. For each feature, we estimate its average marginal contribution to the prediction across random feature orderings. This is not the optimized tree-specific SHAP algorithm, but it captures the core Shapley logic while making the method transparent.

For a feature \(j\), the Shapley value is the average marginal contribution of that feature across coalitions:

\[
\phi_j
=
\sum_{S \subseteq F \setminus \{j\}}
\frac{|S|!(M-|S|-1)!}{M!}
\left(v(S \cup \{j\}) - v(S)\right)
\]

In practice, exact computation is exponential, so we use a **permutation-based approximation** for larger feature sets and exact computation only for very small subsets.

### Global explanation
The global summary plot shows which features matter across many trips.

Final figure: `outputs_blogpost/xai/shap/shap_summary.png`

In our run, the manual SHAP summaries repeatedly emphasized:
- `Transmission_*`
- `accel_var`
- `Generalized_Weight`
- `distance_km`
- `duration_min`

These features therefore anchor the global interpretation of the energy model.

## 6. LIME as a comparison method

LIME explains one prediction by perturbing samples around the original point, querying the black box on those perturbations, and fitting a weighted local surrogate model. This is intuitive and educational, but the explanation quality depends on the perturbation design and the local surrogate fidelity.

We keep LIME because:
1. it is a classic XAI baseline,
2. it provides interpretable local coefficients,
3. it exposes an important trust issue: a readable explanation is not automatically faithful.

## 7. SHAP vs LIME on the same anomaly

To compare both methods directly, we generated side-by-side explanations for selected anomalous trips.

Final figure: `outputs_blogpost/xai/comparisons/shap_vs_lime_560_141.png`

This comparison is especially useful because the two methods behave differently:

- **Manual SHAP** tends to highlight feature contributions derived from the global black-box structure.
- **LIME** reflects a local linear approximation and may emphasize different factors if the local neighborhood is unstable.

In our strongest failure example, trip `560_141`, the LIME local fidelity is very low:
- **local R² ≈ 0.101**
- **local RMSE ≈ 0.0041**

This means the explanation is readable, but the surrogate fit is weak and should be treated cautiously.

## 8. Case studies

We focus on four anomalous trips.

### Case 1 — `536_340` (strong anomaly, LIME fits well)
- actual energy per km: **1.5590**
- predicted energy per km: **0.4924**
- residual: **1.0666**
- transmission: **NO DATA**
- LIME local R²: **0.8557**

**Manual SHAP top contributors**
- `Transmission_CVT: 0.0896`
- `accel_var: 0.0221`
- `Transmission_NO DATA: 0.0148`

**LIME top contributors**
- `speed_var: -4.0517`
- `speed_mean: -0.1761`
- `Generalized_Weight: 0.0194`

Interpretation: this is a high-confidence anomalous trip where the two methods both indicate that the trip differs strongly from the model’s normal operating profile, but they emphasize different aspects of that deviation.

### Case 2 — `11_3013` (moderate anomaly, stable local fit)
- actual energy per km: **1.1895**
- predicted energy per km: **0.8825**
- residual: **0.3070**
- transmission: **CVT**
- LIME local R²: **0.8172**

Interpretation: this is a smaller anomaly than the top outliers, but the local explanation remains reasonably faithful.

### Case 3 — `443_1365` (moderate anomaly, stable local fit)
- actual energy per km: **0.7369**
- predicted energy per km: **0.4854**
- residual: **0.2515**
- transmission: **CVT**
- LIME local R²: **0.8306**

Interpretation: another example where the alert is weaker than the largest outliers, yet both explanation methods still identify a coherent local explanation.

### Case 4 — `560_141` (failure / cautionary case)
- actual energy per km: **2.2628**
- predicted energy per km: **0.6627**
- residual: **1.6001**
- transmission: **CVT**
- LIME local R²: **0.1011**

Interpretation: this is the strongest anomaly in the test split, but the LIME surrogate fits poorly. This makes it an ideal cautionary case: the largest anomaly is not automatically the easiest one to explain faithfully with a local linear surrogate.

## 9. How not to misuse XAI in this setting

### 9.1 Leakage matters
The leakage audit is one of the strongest findings in this project.

| variant | n_model_features | val_mae | val_rmse | val_r2 | test_mae | test_rmse | test_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 20 | 0.0589 | 0.0885 | 0.7320 | 0.0667 | 0.1276 | 0.5763 |
| leaky | 24 | 0.0271 | 0.0419 | 0.9399 | 0.0415 | 0.0859 | 0.8079 |

The leaky model looks dramatically better, but that improvement is not methodologically trustworthy. This is exactly why high predictive quality alone is not enough.

Final figure: `outputs_blogpost/audit/figures/leakage_ablation.png`

### 9.2 Threshold calibration matters
Using validation residuals instead of test residuals gives a cleaner protocol.

| threshold_name | threshold | test_anomaly_rate | n_test_anomalies |
|---|---:|---:|---:|
| val_positive_q95 | 0.0864 | 0.0960 | 67 |
| val_positive_q98 | 0.1326 | 0.0688 | 48 |

This shows that anomaly prevalence depends substantially on the calibration choice.

Final figure: `outputs_blogpost/audit/figures/validation_threshold_calibration.png`

### 9.3 LIME stability matters
Average local explanation stability across the selected trips:

| trip_id | mean_jaccard_top5 | mean_sign_consistency | mean_local_r2 | mean_local_rmse |
|---|---:|---:|---:|---:|
| 536_340 | 1.0000 | 0.7250 | 0.5528 | 0.0125 |
| 11_3013 | 0.7500 | 0.6708 | 0.4702 | 0.0192 |
| 443_1365 | 0.7381 | 0.6905 | 0.5804 | 0.0147 |
| 560_141 | 0.7262 | 0.8125 | 0.5265 | 0.0033 |

The top-5 overlap is decent but not perfect, and the fidelity is clearly case-dependent.

Final figure: `outputs_blogpost/audit/figures/lime_stability.png`

### 9.4 Subgroup robustness matters
The largest subgroup gap appears for `Transmission`.

| group_value | n | mae | rmse | r2 | anomaly_rate | mean_residual |
|---|---:|---:|---:|---:|---:|---:|
| CVT | 636 | 0.0473 | 0.0882 | 0.6143 | 0.0330 | -0.0183 |
| NO DATA | 62 | 0.2689 | 0.3466 | -0.4561 | 0.7419 | 0.2324 |

The `NO DATA` subgroup has much higher error and a dramatically higher anomaly rate. That does not indicate classical demographic unfairness, but it does reveal an important **operational robustness issue**.

Final figure: `outputs_blogpost/audit/figures/subgroup_transmission_anomaly_rate.png`

## 10. Reproducibility

### Recommended notebook order
1. `regression_model/ved-energy-regression.ipynb`
2. `notebooks/05_blogpost_main_flow.ipynb`
3. `notebooks/06_shap_vs_lime.ipynb`
4. `notebooks/07_trustworthiness_audit.ipynb`

### Key output locations
- predictive figures: `outputs_blogpost/figures/`
- explanation outputs: `outputs_blogpost/xai/`
- trustworthiness figures and tables: `outputs_blogpost/audit/`
- summary tables: `outputs_blogpost/tables/`

### Runtime
The project is easiest to reproduce in Kaggle-style notebook environments, but the shipped artifacts also allow local inspection of the final results.

## 11. Takeaways

- Residual anomaly detection is useful for telematics, but the anomaly score alone is not enough.
- Manual SHAP gives a transparent Shapley-style explanation framework for a tree-based black box.
- LIME remains useful as a comparison baseline, but its fidelity is case-dependent.
- Trustworthy XAI requires more than plots: it requires leakage auditing, calibration discipline, stability analysis, and subgroup robustness checks.

The main lesson of the project is simple:

> **A working XAI pipeline is not automatically a trustworthy one.**
