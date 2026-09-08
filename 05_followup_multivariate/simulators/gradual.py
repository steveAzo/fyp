from __future__ import annotations

import numpy as np
import pandas as pd
from .base import DriftSimulator, DriftSpec

class GradualDriftSimulator(DriftSimulator):
    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        magnitude: float = 2.0,
        transition_length: int = 500,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        X = X.copy()
        n = len(X)
        magnitude_, std_val, ceiling = self._shift_and_ceiling(X[feature], magnitude)

        end_idx = min(drift_start_idx + transition_length, n)
        n_ramp = end_idx - drift_start_idx

        magnitude_ramp = np.linspace(0.0, magnitude_, n_ramp)
        X.loc[drift_start_idx : end_idx - 1, feature] = self.saturate(
            X.loc[drift_start_idx : end_idx - 1, feature].to_numpy(), magnitude_ramp, std_val, ceiling
        )

        if end_idx < n:
            X.loc[end_idx:, feature] = self.saturate(
                X.loc[end_idx:, feature], magnitude_, std_val, ceiling
            )

        spec = DriftSpec(
            drift_type="gradual",
            feature=feature,
            drift_start_idx=drift_start_idx,
            parameters={"magnitude": magnitude, "transition_length": transition_length},
        )
        return X, spec
