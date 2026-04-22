# Results Tables

## Table 1 - Main Predictive Performance

| split | MAE | RMSE | R2 | n_trips | n_anomalies | anomaly_rate |
|---|---:|---:|---:|---:|---:|---:|
| test | 0.0670 | 0.1332 | 0.5377 | 698 | 41 | 0.0587 |

## Table 2 - Split Summary

| split | n_trips | n_vehicles | anomaly_rate | mean_target | mean_residual |
|---|---:|---:|---:|---:|---:|
| train | 1957 | 14 | 0.0000 | 0.3204 | 0.0043 |
| val | 594 | 5 | 0.0000 | 0.3523 | -0.0027 |
| test | 698 | 5 | 0.0587 | 0.4016 | 0.0040 |

## Table 3 - Top Anomalous Trips In The Test Split

| trip_id | VehId | energy_per_km | predicted_energy_per_km | residual | VehicleType | Transmission | Drive Wheels |
|---|---:|---:|---:|---:|---|---|---|
| 560_141 | 560 | 2.2628 | 0.6627 | 1.6001 | PHEV | CVT | FWD |
| 536_340 | 536 | 1.5590 | 0.4924 | 1.0666 | PHEV | NO DATA | FWD |
| 536_370 | 536 | 1.4280 | 0.6133 | 0.8147 | PHEV | NO DATA | FWD |
| 536_544 | 536 | 1.1299 | 0.3671 | 0.7628 | PHEV | NO DATA | FWD |
| 536_643 | 536 | 0.9522 | 0.3504 | 0.6018 | PHEV | NO DATA | FWD |

## Table 4 - XAI Case Summary

| trip_id | VehId | energy_per_km | predicted_energy_per_km | residual | VehicleType | Transmission | local_r2 | local_rmse |
|---|---:|---:|---:|---:|---|---|---:|---:|
| 560_141 | 560 | 2.2628 | 0.6627 | 1.6001 | PHEV | CVT | 0.1011 | 0.0041 |
| 536_340 | 536 | 1.5590 | 0.4924 | 1.0666 | PHEV | NO DATA | 0.8557 | 0.0183 |
| 11_3013 | 11 | 1.1895 | 0.8825 | 0.3070 | PHEV | CVT | 0.8172 | 0.0142 |
| 443_1365 | 443 | 0.7369 | 0.4854 | 0.2515 | PHEV | CVT | 0.8306 | 0.0130 |

## Table 5 - Leakage Ablation

| variant | n_model_features | val_mae | val_rmse | val_r2 | test_mae | test_rmse | test_r2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 20 | 0.0589 | 0.0885 | 0.7320 | 0.0667 | 0.1276 | 0.5763 |
| leaky | 24 | 0.0271 | 0.0419 | 0.9399 | 0.0415 | 0.0859 | 0.8079 |

## Table 6 - Validation-Only Threshold Calibration

| threshold_name | threshold | test_anomaly_rate | n_test_anomalies |
|---|---:|---:|---:|
| val_positive_q95 | 0.0864 | 0.0960 | 67 |
| val_positive_q98 | 0.1326 | 0.0688 | 48 |

## Table 7 - LIME Stability Summary

| trip_id | mean_jaccard_top5 | mean_sign_consistency | mean_local_r2 | mean_local_rmse |
|---|---:|---:|---:|---:|
| 536_340 | 1.0000 | 0.7250 | 0.5528 | 0.0125 |
| 11_3013 | 0.7500 | 0.6708 | 0.4702 | 0.0192 |
| 443_1365 | 0.7381 | 0.6905 | 0.5804 | 0.0147 |
| 560_141 | 0.7262 | 0.8125 | 0.5265 | 0.0033 |

## Table 8 - Subgroup Robustness By Transmission

| group_value | n | mae | rmse | r2 | anomaly_rate | mean_residual |
|---|---:|---:|---:|---:|---:|---:|
| CVT | 636 | 0.0473 | 0.0882 | 0.6143 | 0.0330 | -0.0183 |
| NO DATA | 62 | 0.2689 | 0.3466 | -0.4561 | 0.7419 | 0.2324 |

## Table 9 - Top Local Features From Manual SHAP

| trip_id | top features |
|---|---|
| 560_141 | accel_var:-0.0125; Transmission_CVT:-0.0107; Generalized_Weight:0.0070; Transmission_NO DATA:-0.0035 |
| 536_340 | Transmission_CVT:0.0896; accel_var:0.0221; Transmission_NO DATA:0.0148; distance_km:-0.0027 |
| 11_3013 | Transmission_CVT:-0.0137; Transmission_NO DATA:-0.0033; Generalized_Weight:-0.0027; distance_km:-0.0021 |
| 443_1365 | accel_var:-0.0151; Transmission_CVT:-0.0126; Transmission_NO DATA:-0.0036; Generalized_Weight:-0.0036 |

## Table 10 - Top Local Features From LIME

| trip_id | top features |
|---|---|
| 560_141 | Generalized_Weight:0.0112; speed_mean:-0.0011; distance_km:0.0005; duration_min:0.0001 |
| 536_340 | speed_var:-4.0517; speed_mean:-0.1761; Generalized_Weight:0.0194; duration_min:-0.0052 |
| 11_3013 | speed_var:-3.1555; speed_mean:-0.0543; Generalized_Weight:0.0159; duration_min:-0.0118 |
| 443_1365 | speed_var:-1.7238; speed_mean:-0.0541; Generalized_Weight:0.0161; idle_time_min:0.0109 |
