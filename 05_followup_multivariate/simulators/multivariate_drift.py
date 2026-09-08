from __future__ import annotations

import numpy as np
import pandas as pd
from .base import DriftSimulator, DriftSpec

class MultivariateDriftSimulator(DriftSimulator):
    def apply_multivariate(
        self,
        X: pd.DataFrame,
        features: list[str],
        drift_start_idx: int,
        magnitude: float = 2.0,
        drift_type: str = "sudden",
        transition_length: int = 500,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        X = X.copy()
        n = len(X)

        stats = {feat: self._shift_and_ceiling(X[feat], magnitude) for feat in features}
        stds = {feat: s[1] for feat, s in stats.items()}
        ceilings = {feat: s[2] for feat, s in stats.items()}

        if drift_type == "sudden":
            for feat in features:
                col_idx = X.columns.get_loc(feat)
                X.iloc[drift_start_idx:, col_idx] = self.saturate(
                    X.iloc[drift_start_idx:, col_idx], magnitude, stds[feat], ceilings[feat]
                )

        elif drift_type == "gradual":
            end_idx = min(drift_start_idx + transition_length, n)
            n_ramp = end_idx - drift_start_idx
            for feat in features:
                col_idx = X.columns.get_loc(feat)
                magnitude_ramp = np.linspace(0.0, magnitude, n_ramp)
                X.iloc[drift_start_idx:end_idx, col_idx] = self.saturate(
                    X.iloc[drift_start_idx:end_idx, col_idx].to_numpy(), magnitude_ramp, stds[feat], ceilings[feat]
                )
                if end_idx < n:
                    X.iloc[end_idx:, col_idx] = self.saturate(
                        X.iloc[end_idx:, col_idx], magnitude, stds[feat], ceilings[feat]
                    )

        elif drift_type == "recurring":
            cycle_length = 200
            for feat in features:
                col_idx = X.columns.get_loc(feat)
                # Create mask for active recurring cycles
                positions = np.arange(drift_start_idx, n)
                cycle_pos = (positions - drift_start_idx) % (2 * cycle_length)
                active_mask = cycle_pos < cycle_length
                active_indices = positions[active_mask]
                X.iloc[active_indices, col_idx] = self.saturate(
                    X.iloc[active_indices, col_idx], magnitude, stds[feat], ceilings[feat]
                )

        spec = DriftSpec(
            drift_type=f"multivariate_{drift_type}",
            feature="+".join(features),
            drift_start_idx=drift_start_idx,
            parameters={
                "features": features,
                "magnitude": magnitude,
                "drift_type": drift_type,
            },
        )
        return X, spec

    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        **kwargs,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        features = [f.strip() for f in feature.split("+")]
        return self.apply_multivariate(X, features, drift_start_idx, **kwargs)
