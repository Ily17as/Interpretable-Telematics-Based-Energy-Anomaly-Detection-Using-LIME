import pandas as pd
from src.audit.subgroup import subgroup_regression_metrics

def test_subgroup_metrics_has_expected_columns():
    df = pd.DataFrame({
        "grp": ["a", "a", "b", "b"],
        "y": [1.0, 2.0, 3.0, 4.0],
        "pred": [1.1, 1.9, 2.5, 4.5],
        "anom": [0, 1, 0, 1],
    })
    out = subgroup_regression_metrics(df, "grp", "y", "pred", "anom")
    expected = {"group_col", "group_value", "n", "mae", "rmse", "r2", "mean_residual", "anomaly_rate"}
    assert expected.issubset(set(out.columns))
