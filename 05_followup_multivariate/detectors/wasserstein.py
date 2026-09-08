from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance
from .base import DetectionResult, DriftDetector

class WassersteinDetector(DriftDetector):
    def __init__(self, threshold: float = 0.25) -> None:
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

        if ref.ndim == 2:
            distances = [float(wasserstein_distance(ref[:, col], tst[:, col])) for col in range(ref.shape[1])]
            distance = float(np.max(distances))
        else:
            distance = float(wasserstein_distance(ref, tst))

        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=bool(distance > self.threshold),
            score=distance,
            detector_name="Wasserstein",
            feature=feature,
            computation_time_ms=elapsed,
        )
