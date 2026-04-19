from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from .base import DetectionResult, DriftDetector


class WassersteinDetector(DriftDetector):
    """
    First Wasserstein distance (Earth Mover's Distance).

    Because features are StandardScaler-normalised before reaching the
    detector, a threshold of 0.1 corresponds to a ~0.1σ shift in the
    underlying distribution — a practically meaningful but conservative
    sensitivity level.

    score   : Wasserstein distance  (higher → more drift)
    is_drift: distance > threshold
    """

    def __init__(self, threshold: float = 0.10) -> None:
        super().__init__(threshold)

    def detect(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        feature: str = "",
    ) -> DetectionResult:
        ref = self._clean(self._to_array(reference))
        tst = self._clean(self._to_array(test))

        t0 = self._now_ms()
        distance = float(wasserstein_distance(ref, tst))
        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=distance > self.threshold,
            score=distance,
            detector_name="Wasserstein",
            feature=feature,
            computation_time_ms=elapsed,
        )
