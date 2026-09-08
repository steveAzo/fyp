from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

@dataclass
class DriftSpec:
    drift_type: str
    feature: str
    drift_start_idx: int
    parameters: dict[str, Any]

class DriftSimulator(ABC):
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @abstractmethod
    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        magnitude: float = 2.0,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        pass

    @staticmethod
    def _shift_and_ceiling(series: pd.Series, magnitude: float, cap_percentile: float = 0.995) -> tuple[float, float, float]:
        """Returns (magnitude, std_val, ceiling) to be applied via
        `saturate(value, magnitude, std_val, ceiling)` below (NOT a plain
        additive shift, and NOT a hard `min(value + shift, ceiling)` clip).

        A plain "add magnitude * std to every value" pushes the bulk of a
        feature past its observed range whenever the feature isn't sharply
        peaked -- e.g. telco's Monthly Charges is close to uniformly spread
        across its range, so even a shift capped to the gap between the mean
        and the 99.5th percentile still pushed ~55% of all samples beyond
        anything the model was ever trained on. A hard `min(value+shift,
        ceiling)` clip fixes that but creates its own artifact: for
        maintenance's Tool wear / Air temperature, over 45% of samples
        collapsed onto the exact same identical ceiling value -- an
        artificial spike no real sensor saturation produces (real saturation
        still has some spread), which measurably inflated the accuracy-
        degradation numbers by concentrating far more mass right at a single
        decision-boundary-crossing point than a realistic shift would.
        """
        std_val = float(series.std())
        ceiling = float(series.quantile(cap_percentile))
        return magnitude, std_val, ceiling

    @staticmethod
    def saturate(values, magnitude: float, std_val: float, ceiling: float) -> np.ndarray:
        """Move `values` toward `ceiling` with smooth exponential saturation,
        instead of a hard clip or a plain additive shift.

        For a point with headroom = ceiling - value > 0 and a nominal
        (unsaturated) shift of nominal = magnitude * std_val:

            new_value = ceiling - headroom * exp(-nominal / headroom)

        Verified numerically: when headroom >> nominal (points far from the
        ceiling, e.g. credit's long-tailed features), this reduces to
        approximately `value + nominal` -- the same plain additive shift as
        before, since the Taylor expansion of exp(-eps) for small
        eps=nominal/headroom gives new_value = value + nominal - O(eps^2).
        When headroom is small or comparable to nominal (telco/maintenance),
        the shift compresses smoothly toward the ceiling instead of a hard
        clip, so distinct original values stay distinct (no pile-up) while
        the result can approach but never reach or exceed `ceiling`.

        Points that are already at or beyond `ceiling` before any shift is
        applied (up to (1 - cap_percentile) of the reference data, by
        definition of the percentile) have headroom <= 0, where this formula
        is undefined/wrong (verified: it can overshoot past the ceiling) --
        those are left unchanged, since they're already at the extreme this
        cap is meant to represent.
        """
        values = np.asarray(values, dtype=float)
        headroom = ceiling - values
        nominal = magnitude * std_val
        safe_headroom = np.where(headroom > 0, headroom, 1.0)  # placeholder to avoid div-by-zero/negative
        saturated = ceiling - safe_headroom * np.exp(-nominal / safe_headroom)
        return np.where(headroom > 0, saturated, values)
