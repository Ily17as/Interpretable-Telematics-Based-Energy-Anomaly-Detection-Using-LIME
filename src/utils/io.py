from __future__ import annotations

"""
IO utilities for the energy anomaly detection project.

This module defines helper functions to read and write tabular data
consistently, save and load models, and store JSON configuration or
metadata files. Ensuring these operations are centralized simplifies
error handling and makes it easier to adjust file formats later.
"""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Ensure that a directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path: str | Path) -> pd.DataFrame:
    """
    Read a tabular dataset from CSV or Parquet.

    Args:
        path: Path to the file.

    Returns:
        A pandas DataFrame containing the table contents.
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in [".csv", ".gz"]:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    """
    Write a DataFrame to CSV or Parquet.

    Args:
        df: DataFrame to write.
        path: Destination file path. Format is inferred from extension.
    """
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if path.suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported table format: {path.suffix}")


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """
    Save a Python dictionary as a JSON file with UTF‑8 encoding.

    Args:
        obj: Object to serialize.
        path: Destination path. Will be created if it doesn't exist.
    """
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file as a Python dictionary."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(model: Any, path: str | Path) -> None:
    """Serialize and save a model using joblib."""
    path = Path(path)
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    """Load a model saved with joblib."""
    return joblib.load(path)
