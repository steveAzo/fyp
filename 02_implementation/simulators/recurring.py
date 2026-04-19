from __future__ import annotations

import numpy as np
import pandas as pd

from .base import DriftSimulator, DriftSpec


class RecurringDriftSimulator(DriftSimulator):
    """
    Recurring drift: the feature alternates between shifted and unshifted
    states in cycles of `cycle_length` samples.  This models seasonal or
    periodic distribution changes.

    Parameters
    ----------
    magnitude    : float — shift size in units of the feature's std.
    cycle_length : int   — length of one on/off cycle in samples.
    """

    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        magnitude: float = 2.0,
        cycle_length: int = 200,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        X = X.copy()
        std = float(X[feature].std())
        shift = magnitude * std

        stream_len = len(X) - drift_start_idx
        if stream_len <= 0:
            return X, DriftSpec(
                drift_type="recurring",
                feature=feature,
                drift_start_idx=drift_start_idx,
                parameters={"magnitude": magnitude, "cycle_length": cycle_length, "shift": 0.0},
            )

        position_in_cycle = np.arange(stream_len) % (cycle_length * 2)
        is_on = position_in_cycle < cycle_length
        shift_values = np.where(is_on, shift, 0.0)

        X.loc[drift_start_idx:, feature] += shift_values

        return X, DriftSpec(
            drift_type="recurring",
            feature=feature,
            drift_start_idx=drift_start_idx,
            parameters={
                "magnitude": magnitude,
                "cycle_length": cycle_length,
                "shift": shift,
            },
        )
