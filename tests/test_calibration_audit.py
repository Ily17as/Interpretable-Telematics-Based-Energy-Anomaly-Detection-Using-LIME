import pandas as pd
from src.audit.calibration import positive_residual_threshold, apply_threshold

def test_positive_threshold_non_negative():
    s = pd.Series([-1.0, 0.0, 0.5, 1.0, 2.0])
    thr = positive_residual_threshold(s, quantile=0.5)
    assert thr >= 0.0

def test_apply_threshold_binary():
    s = pd.Series([0.1, 0.6, 1.2])
    out = apply_threshold(s, 0.5)
    assert set(out.unique()) <= {0, 1}
