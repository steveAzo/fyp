from __future__ import annotations

import numpy as np
import pandas as pd
from .base import DriftSimulator, DriftSpec

class SuddenDriftSimulator(DriftSimulator):
    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        magnitude: float = 2.0,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        X = X.copy()
        magnitude_, std_val, ceiling = self._shift_and_ceiling(X[feature], magnitude)
        X.loc[drift_start_idx:, feature] = self.saturate(
            X.loc[drift_start_idx:, feature], magnitude_, std_val, ceiling
        )

        spec = DriftSpec(
            drift_type="sudden",
            feature=feature,
            drift_start_idx=drift_start_idx,
            parameters={"magnitude": magnitude, "ceiling": ceiling},
        )
        return X, spec
