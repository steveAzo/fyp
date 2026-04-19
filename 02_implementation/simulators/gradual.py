from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DriftSimulator, DriftSpec


class GradualDriftSimulator(DriftSimulator):
    """
    Gradual drift: the shift ramps linearly from 0 to max_shift over
    transition_length samples, then stays constant.

    Parameters
    ----------
    magnitude         : float  — max shift in units of the feature's std.
    transition_length : int    — number of samples over which drift ramps up.
    """

    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        magnitude: float = 2.0,
        transition_length: int = 500,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        X = X.copy()
        std = float(X[feature].std())
        max_shift = magnitude * std

        end_idx = min(drift_start_idx + transition_length, len(X))
        n_ramp = end_idx - drift_start_idx

        # Linear ramp during transition window
        ramp = np.linspace(0.0, max_shift, n_ramp)
        X.loc[drift_start_idx : end_idx - 1, feature] += ramp

        # Constant full shift after transition
        if end_idx < len(X):
            X.loc[end_idx:, feature] += max_shift

        return X, DriftSpec(
            drift_type="gradual",
            feature=feature,
            drift_start_idx=drift_start_idx,
            parameters={
                "magnitude": magnitude,
                "transition_length": transition_length,
                "max_shift": max_shift,
            },
        )
