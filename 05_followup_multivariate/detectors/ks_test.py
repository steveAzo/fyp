from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from .base import DetectionResult, DriftDetector

class KSTestDetector(DriftDetector):
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

        if ref.ndim == 2:
            p_values = []
            for col in range(ref.shape[1]):
                _, p_val = ks_2samp(ref[:, col], tst[:, col])
                p_values.append(p_val)
            min_p = float(np.min(p_values))
            # Bonferroni correction over features
            p_value = min(1.0, min_p * ref.shape[1])
        else:
            _, p_value = ks_2samp(ref, tst)

        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=bool(p_value < self.threshold),
            score=float(p_value),
            detector_name="KS-Test",
            feature=feature,
            computation_time_ms=elapsed,
        )
