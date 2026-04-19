from __future__ import annotations

import pandas as pd

from .base import DriftSimulator, DriftSpec


class SuddenDriftSimulator(DriftSimulator):
    """
    Sudden (abrupt) drift: adds a constant shift to the feature starting
    at drift_start_idx.  The shift magnitude is expressed as a multiple of
    the feature's standard deviation in the stream, making it comparable
    across datasets.

    Parameters
    ----------
    magnitude : float
        Shift size in units of the feature's std.  Default 2.0 corresponds
        to a 2-sigma shift — detectable but not trivially obvious.
    """

    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        magnitude: float = 2.0,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        X = X.copy()
        std = float(X[feature].std())
        shift = magnitude * std

        X.loc[drift_start_idx:, feature] += shift

        return X, DriftSpec(
            drift_type="sudden",
            feature=feature,
            drift_start_idx=drift_start_idx,
            parameters={"magnitude": magnitude, "shift_applied": shift},
        )
