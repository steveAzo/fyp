from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

@dataclass
class ModelResults:
    model: Any
    accuracy: float

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

def train_model(
    X_ref: pd.DataFrame,
    y_ref: pd.Series,
    X_stream: pd.DataFrame,
    y_stream: pd.Series,
    model_type: str = "random_forest",
    random_state: int = 42,
) -> ModelResults:
    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif model_type == "logistic_regression":
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif model_type == "lightgbm":
        clf = LGBMClassifier(n_estimators=100, random_state=random_state, verbose=-1, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_type {model_type!r}")

    clf.fit(X_ref, y_ref)
    preds = clf.predict(X_stream)
    acc = float(accuracy_score(y_stream, preds))

    return ModelResults(model=clf, accuracy=acc)
