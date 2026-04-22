from __future__ import annotations
import pandas as pd

DEFAULT_SUSPECT_FEATURES = [
    "energy_kWh",
    "fuel_energy_kWh",
    "battery_energy_kWh",
    "ac_energy_kWh",
    "heater_energy_kWh",
]

def drop_leakage_features(df: pd.DataFrame, suspect_features: list[str] | None = None) -> pd.DataFrame:
    suspect = suspect_features or DEFAULT_SUSPECT_FEATURES
    cols = [c for c in df.columns if c not in suspect]
    return df.loc[:, cols].copy()
