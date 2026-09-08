from __future__ import annotations

import pandas as pd
from .base import DriftSimulator, DriftSpec

class RecurringDriftSimulator(DriftSimulator):
    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        magnitude: float = 2.0,
        cycle_length: int = 200,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        X = X.copy()
        n = len(X)
        magnitude_, std_val, ceiling = self._shift_and_ceiling(X[feature], magnitude)

        for idx in range(drift_start_idx, n):
            cycle_pos = (idx - drift_start_idx) % (2 * cycle_length)
            if cycle_pos < cycle_length:
                X.loc[idx, feature] = float(self.saturate(X.loc[idx, feature], magnitude_, std_val, ceiling))

        spec = DriftSpec(
            drift_type="recurring",
            feature=feature,
            drift_start_idx=drift_start_idx,
            parameters={"magnitude": magnitude, "cycle_length": cycle_length},
        )
        return X, spec
