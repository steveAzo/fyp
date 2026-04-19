"""
Model training for the drift detection experiments.

We deliberately keep models simple (Random Forest) to isolate the effect
of resource constraints on *drift detection*, not on model complexity.
A complicated model would confound the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


@dataclass
class TrainingResult:
    model: RandomForestClassifier
    accuracy: float
    roc_auc: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def accuracy_on(self, X: pd.DataFrame, y: pd.Series) -> float:
        return float(accuracy_score(y, self.model.predict(X)))


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    n_estimators: int = 100,
    random_state: int = 42,
) -> TrainingResult:
    """
    Train a Random Forest classifier on X_train / y_train.

    If X_val / y_val are provided, accuracy and ROC-AUC are evaluated on
    that held-out set.  Otherwise both metrics are 0.0.

    Returns:
        TrainingResult with the fitted model and evaluation metrics.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    accuracy, roc_auc = 0.0, 0.0
    if X_val is not None and y_val is not None:
        y_pred = model.predict(X_val)
        accuracy = float(accuracy_score(y_val, y_pred))
        if len(np.unique(y_val)) == 2:
            y_prob = model.predict_proba(X_val)[:, 1]
            roc_auc = float(roc_auc_score(y_val, y_prob))

    return TrainingResult(model=model, accuracy=accuracy, roc_auc=roc_auc)
