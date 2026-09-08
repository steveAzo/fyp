"""
Dataset loading and preprocessing for the multivariate follow-up study.

Ported directly from 02_implementation/data/loader.py (the validated Paper 1
pipeline) rather than re-derived, so reference/stream splits and encodings
stay comparable across the two papers:
  - X_reference / y_reference : used to train the model and as the drift
    reference window
  - X_stream    / y_stream    : the "production" stream; drift is injected
    here later

Reference/stream split is a STRATIFIED RANDOM split (not positional/row-order),
because these datasets have no real temporal order and several of the raw
CSVs are sorted by fields correlated with the target (e.g. telco_raw.csv is
sorted such that the last ~40% of rows contain zero churners) -- a positional
split would confound "drift" with a pre-existing distributional split.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


@dataclass
class DatasetSplit:
    X_reference: pd.DataFrame
    y_reference: pd.Series
    X_stream: pd.DataFrame
    y_stream: pd.Series


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _resolve_path(raw_path: str) -> Path:
    p = Path(__file__).resolve().parent.parent / raw_path
    if p.exists():
        return p
    fallback = Path(__file__).resolve().parents[2] / "02_implementation" / "data" / Path(raw_path).name
    if fallback.exists():
        return fallback
    return p


def _split(X: pd.DataFrame, y: pd.Series, reference_ratio: float, random_state: int):
    return train_test_split(
        X, y,
        train_size=reference_ratio,
        random_state=random_state,
        stratify=y,
    )


def _encode_categoricals(df_ref: pd.DataFrame, df_stream: pd.DataFrame, cat_cols: list[str]):
    df_ref = df_ref.copy()
    df_stream = df_stream.copy()
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_ref[col], df_stream[col]]).astype(str)
        le.fit(combined)
        df_ref[col] = le.transform(df_ref[col].astype(str))
        df_stream[col] = le.transform(df_stream[col].astype(str))
    return df_ref, df_stream


def _scale_numericals(df_ref: pd.DataFrame, df_stream: pd.DataFrame, num_cols: list[str]):
    scaler = StandardScaler()
    df_ref = df_ref.copy()
    df_stream = df_stream.copy()
    df_ref[num_cols] = scaler.fit_transform(df_ref[num_cols])
    df_stream[num_cols] = scaler.transform(df_stream[num_cols])
    return df_ref, df_stream


def _load_credit(cfg: dict, random_state: int) -> DatasetSplit:
    ds_cfg = cfg["datasets"]["credit"]
    df = pd.read_csv(_resolve_path(ds_cfg["path"]))
    df = df.drop(columns=[c for c in ds_cfg.get("drop_cols", []) if c in df.columns])

    X = df.drop(columns=[ds_cfg["target"]])
    y = df[ds_cfg["target"]].rename("target")

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    ref_ratio = cfg["split"]["reference_ratio"]
    X_ref, X_str, y_ref, y_str = _split(X, y, ref_ratio, random_state)

    if cat_cols:
        X_ref, X_str = _encode_categoricals(X_ref, X_str, cat_cols)
    X_ref, X_str = _scale_numericals(X_ref, X_str, num_cols)

    return DatasetSplit(
        X_ref.reset_index(drop=True), y_ref.reset_index(drop=True),
        X_str.reset_index(drop=True), y_str.reset_index(drop=True),
    )


def _load_telco(cfg: dict, random_state: int) -> DatasetSplit:
    ds_cfg = cfg["datasets"]["telco"]
    df = pd.read_csv(_resolve_path(ds_cfg["path"]))
    df = df.drop(columns=[c for c in ds_cfg.get("drop_cols", []) if c in df.columns])

    # "Total Charges" arrives as object dtype with blank cells for new
    # customers (tenure=0) -- coerce to numeric and drop the few bad rows.
    # Without this it gets caught by select_dtypes(exclude=number) below and
    # one-hot/label-encoded as a near-unique categorical, which both destroys
    # it as a continuous feature and breaks any downstream reference to the
    # literal column name "Total Charges" (used as a multivariate drift feature).
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
        df = df.dropna(subset=["Total Charges"]).reset_index(drop=True)

    X = df.drop(columns=[ds_cfg["target"]])
    y = df[ds_cfg["target"]].rename("target")

    num_cols = list(X.select_dtypes(include=[np.number]).columns)
    cat_cols = list(X.select_dtypes(exclude=[np.number]).columns)

    ref_ratio = cfg["split"]["reference_ratio"]
    X_ref, X_str, y_ref, y_str = _split(X, y, ref_ratio, random_state)

    if cat_cols:
        X_ref, X_str = _encode_categoricals(X_ref, X_str, cat_cols)
    X_ref, X_str = _scale_numericals(X_ref, X_str, num_cols)

    return DatasetSplit(
        X_ref.reset_index(drop=True), y_ref.reset_index(drop=True),
        X_str.reset_index(drop=True), y_str.reset_index(drop=True),
    )


def _load_maintenance(cfg: dict, random_state: int) -> DatasetSplit:
    ds_cfg = cfg["datasets"]["maintenance"]
    df = pd.read_csv(_resolve_path(ds_cfg["path"]))
    df = df.drop(columns=[c for c in ds_cfg.get("drop_cols", []) if c in df.columns])

    if "Type" in df.columns:
        df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})

    X = df.drop(columns=[ds_cfg["target"]])
    y = df[ds_cfg["target"]].rename("target")

    num_cols = list(X.columns)  # all numeric after ordinal-mapping Type

    ref_ratio = cfg["split"]["reference_ratio"]
    X_ref, X_str, y_ref, y_str = _split(X, y, ref_ratio, random_state)
    X_ref, X_str = _scale_numericals(X_ref, X_str, num_cols)

    return DatasetSplit(
        X_ref.reset_index(drop=True), y_ref.reset_index(drop=True),
        X_str.reset_index(drop=True), y_str.reset_index(drop=True),
    )


_LOADERS = {
    "credit": _load_credit,
    "telco": _load_telco,
    "maintenance": _load_maintenance,
}


def load_dataset(dataset_name: str, random_state: int = 42) -> DatasetSplit:
    cfg = load_config()
    if dataset_name not in _LOADERS:
        raise ValueError(f"Unknown dataset {dataset_name!r}. Choose from: {list(_LOADERS)}")
    return _LOADERS[dataset_name](cfg, random_state)
