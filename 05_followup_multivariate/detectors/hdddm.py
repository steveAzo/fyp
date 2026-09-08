from __future__ import annotations

import numpy as np
from .base import DetectionResult, DriftDetector

class HDDDMDetector(DriftDetector):
    def __init__(self, threshold: float = 0.15, n_bins: int = 10) -> None:
        super().__init__(threshold)
        self.n_bins = n_bins

    def _histogram_1d(self, ref: np.ndarray, tst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        min_val, max_val = ref.min(), ref.max()
        if min_val == max_val:
            min_val -= 0.5
            max_val += 0.5
        bins = np.linspace(min_val, max_val, self.n_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(ref, bins=bins)
        tst_counts, _ = np.histogram(tst, bins=bins)

        p = ref_counts / max(len(ref), 1)
        q = tst_counts / max(len(tst), 1)

        return p, q

    def _hellinger_1d(self, ref: np.ndarray, tst: np.ndarray) -> float:
        p, q = self._histogram_1d(ref, tst)
        bc = np.sum(np.sqrt(p * q))
        return float(np.sqrt(max(0.0, 1.0 - bc)))

    def detect(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        feature: str = "",
    ) -> DetectionResult:
        ref = self._clean(self._to_array(reference))
        tst = self._clean(self._to_array(test))

        t0 = self._now_ms()

        if ref.ndim == 1:
            hd = self._hellinger_1d(ref, tst)
        else:
            # Ditzler & Polikar (2011) Eq. (1): the multivariate Hellinger
            # distance is the AVERAGE of the per-feature Hellinger distances,
            # not a single distance over concatenated per-feature histograms.
            d = ref.shape[1]
            per_feature_hd = [self._hellinger_1d(ref[:, col], tst[:, col]) for col in range(d)]
            hd = float(np.mean(per_feature_hd))

        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=bool(hd > self.threshold),
            score=hd,
            detector_name="HDDDM",
            feature=feature,
            computation_time_ms=elapsed,
        )
