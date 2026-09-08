from __future__ import annotations

import numpy as np
from .base import DetectionResult, DriftDetector

class PSIDetector(DriftDetector):
    def __init__(self, threshold: float = 0.25, n_bins: int = 10) -> None:
        super().__init__(threshold)
        self.n_bins = n_bins

    def _psi_1d(self, ref: np.ndarray, tst: np.ndarray) -> float:
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        breakpoints = np.percentile(ref, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(ref, bins=breakpoints)[0]
        tst_counts = np.histogram(tst, bins=breakpoints)[0]

        ref_props = ref_counts / max(len(ref), 1)
        tst_props = tst_counts / max(len(tst), 1)

        ref_props = np.where(ref_props == 0, 1e-4, ref_props)
        tst_props = np.where(tst_props == 0, 1e-4, tst_props)

        return float(np.sum((tst_props - ref_props) * np.log(tst_props / ref_props)))

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
            psi_vals = [self._psi_1d(ref[:, col], tst[:, col]) for col in range(ref.shape[1])]
            psi = float(np.max(psi_vals))
        else:
            psi = self._psi_1d(ref, tst)

        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=bool(psi > self.threshold),
            score=psi,
            detector_name="PSI",
            feature=feature,
            computation_time_ms=elapsed,
        )
