from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from .base import DetectionResult, DriftDetector


class KSTestDetector(DriftDetector):
    """
    Two-sample Kolmogorov-Smirnov test.

    score   : p-value  (lower → stronger evidence of drift)
    is_drift: p_value < threshold
    threshold default: 0.05  (standard α level)
    """

    def __init__(self, threshold: float = 0.05) -> None:
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
        _, p_value = ks_2samp(ref, tst)
        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=bool(p_value < self.threshold),
            score=float(p_value),
            detector_name="KS-Test",
            feature=feature,
            computation_time_ms=elapsed,
        )
