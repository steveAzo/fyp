"""
Dataset loading and preprocessing for all three experimental datasets.

Each loader returns a DatasetInfo with a clean reference/stream split:
  - X_reference / y_reference : used to train the model and as the drift reference window
  - X_stream    / y_stream    : the production stream; drift is injected here later

All numerical features are StandardScaler-fitted on the reference set and
applied to the stream (no leakage).  Categorical columns are ordinal-encoded
consistently across both splits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DatasetInfo:
    name: str
    X_reference: pd.DataFrame
    y_reference: pd.Series
    X_stream: pd.DataFrame
    y_stream: pd.Series
    feature_names: list[str]
    numerical_features: list[str]
    categorical_features: list[str]

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(name={self.name!r}, "
            f"reference={self.X_reference.shape}, "
            f"stream={self.X_stream.shape})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_categoricals(
    df_ref: pd.DataFrame,
    df_stream: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit LabelEncoder on the union of both splits to avoid unseen-label errors."""
    df_ref = df_ref.copy()
    df_stream = df_stream.copy()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_ref[col], df_stream[col]]).astype(str)
        le.fit(combined)
        df_ref[col] = le.transform(df_ref[col].astype(str))
        df_stream[col] = le.transform(df_stream[col].astype(str))
    return df_ref, df_stream


def _scale_numericals(
    df_ref: pd.DataFrame,
    df_stream: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on reference only; transform both."""
    scaler = StandardScaler()
    df_ref = df_ref.copy()
    df_stream = df_stream.copy()
    df_ref[num_cols] = scaler.fit_transform(df_ref[num_cols])
    df_stream[num_cols] = scaler.transform(df_stream[num_cols])
    return df_ref, df_stream


def _split(
    X: pd.DataFrame,
    y: pd.Series,
    reference_ratio: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y,
        train_size=reference_ratio,
        random_state=random_state,
        stratify=y,
    )


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_credit_default(
    path: str = "data/credit_raw.csv",
    reference_ratio: float = 0.6,
    random_state: int = 42,
) -> DatasetInfo:
    """
    UCI Default of Credit Card Clients  (30 000 × 24 features after ID drop).
    Target: dpnm  (1 = default next month).
    """
    df = pd.read_csv(path)
    df = df.drop(columns=["ID"])

    X = df.drop(columns=["dpnm"])
    y = df["dpnm"].rename("target")

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    X_ref, X_str, y_ref, y_str = _split(X, y, reference_ratio, random_state)

    if cat_cols:
        X_ref, X_str = _encode_categoricals(X_ref, X_str, cat_cols)

    X_ref, X_str = _scale_numericals(X_ref, X_str, num_cols)

    return DatasetInfo(
        name="credit_default",
        X_reference=X_ref.reset_index(drop=True),
        y_reference=y_ref.reset_index(drop=True),
        X_stream=X_str.reset_index(drop=True),
        y_stream=y_str.reset_index(drop=True),
        feature_names=list(X.columns),
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )


def load_telco_churn(
    path: str = "data/telco_raw.csv",
    reference_ratio: float = 0.6,
    random_state: int = 42,
) -> DatasetInfo:
    """
    IBM Telco Customer Churn  (~7 000 × 19 usable features).
    Target: Churn Value  (1 = churned).

    Dropped at load time: geo columns, identifier columns, leakage columns
    (Churn Label, Churn Score, Churn Reason, CLTV).
    """
    _DROP = [
        "CustomerID", "Count",
        "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude",
        "Churn Label", "Churn Score", "Churn Reason", "CLTV",
    ]
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in _DROP if c in df.columns])

    # Total Charges arrives as object with blank cells — coerce and drop the ~11 bad rows
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df = df.dropna(subset=["Total Charges"]).reset_index(drop=True)

    X = df.drop(columns=["Churn Value"])
    y = df["Churn Value"].rename("target")

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    X_ref, X_str, y_ref, y_str = _split(X, y, reference_ratio, random_state)

    if cat_cols:
        X_ref, X_str = _encode_categoricals(X_ref, X_str, cat_cols)

    X_ref, X_str = _scale_numericals(X_ref, X_str, num_cols)

    return DatasetInfo(
        name="telco_churn",
        X_reference=X_ref.reset_index(drop=True),
        y_reference=y_ref.reset_index(drop=True),
        X_stream=X_str.reset_index(drop=True),
        y_stream=y_str.reset_index(drop=True),
        feature_names=list(X.columns),
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )


def load_predictive_maintenance(
    path: str = "data/predictive_maintenance.csv",
    reference_ratio: float = 0.6,
    random_state: int = 42,
) -> DatasetInfo:
    """
    AI4I 2020 Predictive Maintenance  (10 000 × 6 features after cleaning).
    Target: Machine failure  (1 = failure).

    Sub-failure columns (TWF, HDF, PWF, OSF, RNF) are dropped — they are
    post-hoc failure codes, not predictive features.
    Type is ordinal-encoded L=0, M=1, H=2 (reflects load level ordering).
    """
    _DROP = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in _DROP if c in df.columns])

    type_map = {"L": 0, "M": 1, "H": 2}
    df["Type"] = df["Type"].map(type_map)

    X = df.drop(columns=["Machine failure"])
    y = df["Machine failure"].rename("target")

    num_cols = list(X.columns)  # all numeric after ordinal encoding
    cat_cols: list[str] = []

    X_ref, X_str, y_ref, y_str = _split(X, y, reference_ratio, random_state)
    X_ref, X_str = _scale_numericals(X_ref, X_str, num_cols)

    return DatasetInfo(
        name="predictive_maintenance",
        X_reference=X_ref.reset_index(drop=True),
        y_reference=y_ref.reset_index(drop=True),
        X_stream=X_str.reset_index(drop=True),
        y_stream=y_str.reset_index(drop=True),
        feature_names=list(X.columns),
        numerical_features=num_cols,
        categorical_features=cat_cols,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_LOADERS = {
    "credit":      load_credit_default,
    "telco":       load_telco_churn,
    "maintenance": load_predictive_maintenance,
}


def load_dataset(name: str, **kwargs) -> DatasetInfo:
    """
    Load a dataset by short name.

    Args:
        name: One of "credit", "telco", "maintenance".
        **kwargs: Forwarded to the specific loader
                  (path, reference_ratio, random_state).

    Returns:
        DatasetInfo with pre-split reference and stream sets.
    """
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {list(_LOADERS)}")
    return _LOADERS[name](**kwargs)
