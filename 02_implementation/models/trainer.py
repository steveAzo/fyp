"""
Model training for the drift detection experiments.

We deliberately keep models simple (Random Forest, Logistic Regression,
LightGBM) to isolate the effect of resource constraints on *drift
detection*, not on model complexity.  A complicated model would confound
the results.

Supported model_type values: "random_forest", "logistic_regression", "lightgbm"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def _safe_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Replace special JSON characters in column names for LightGBM compatibility.

    LightGBM rejects characters like [ ] , { } in feature names.
    Returns the renamed DataFrame and the old→new mapping.
    """
    mapping = {col: re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns}
    return df.rename(columns=mapping), mapping


@dataclass
class TrainingResult:
    model: object          # RandomForestClassifier | LogisticRegression | LGBMClassifier
    model_type: str
    accuracy: float
    roc_auc: float
    _col_mapping: dict = field(default_factory=dict, repr=False)

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply column renaming for LightGBM; no-op for other models."""
        if self._col_mapping:
            return X.rename(columns=self._col_mapping)
        return X

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare(X))

    def accuracy_on(self, X: pd.DataFrame, y: pd.Series) -> float:
        return float(accuracy_score(y, self.model.predict(self._prepare(X))))


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> TrainingResult:
    """
    Train a classifier on X_train / y_train.

    Args:
        model_type: "random_forest" | "logistic_regression" | "lightgbm"

    If X_val / y_val are provided, accuracy and ROC-AUC are evaluated on
    that held-out set.  Otherwise both metrics are 0.0.
    """
    col_mapping: dict = {}

    if model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "logistic_regression":
        model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
        )
    elif model_type == "lightgbm":
        model = LGBMClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )
        # Sanitize column names — LightGBM rejects [ ] and other JSON chars
        X_train, col_mapping = _safe_columns(X_train)
        if X_val is not None:
            X_val = X_val.rename(columns=col_mapping)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}. "
                         f"Choose from: random_forest, logistic_regression, lightgbm")

    model.fit(X_train, y_train)

    accuracy, roc_auc = 0.0, 0.0
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        accuracy = float(accuracy_score(y_val, y_pred))
        if len(np.unique(y_val)) == 2:
            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = float(roc_auc_score(y_val, y_prob))

    return TrainingResult(
        model=model,
        model_type=model_type,
        accuracy=accuracy,
        roc_auc=roc_auc,
        _col_mapping=col_mapping,
    )
