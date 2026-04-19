from __future__ import annotations

import numpy as np

from .base import DetectionResult, DriftDetector


class PSIDetector(DriftDetector):
    """
    Population Stability Index (PSI).

    Bins are derived from percentiles of the reference distribution so that
    each bin holds an equal proportion of reference observations.

    Conventional thresholds:
        PSI < 0.10  : no significant drift
        0.10 – 0.25 : moderate drift  (this class uses 0.10 as default)
        PSI > 0.25  : significant drift

    score   : PSI value   (higher → more drift)
    is_drift: psi > threshold
    """

    def __init__(self, threshold: float = 0.10, n_bins: int = 10) -> None:
        super().__init__(threshold)
        self.n_bins = n_bins

    def detect(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        feature: str = "",
    ) -> DetectionResult:
        ref = self._clean(self._to_array(reference))
        tst = self._clean(self._to_array(test))

        t0 = self._now_ms()

        # Build equal-frequency bins from reference
        percentiles = np.linspace(0, 100, self.n_bins + 1)
        breakpoints = np.percentile(ref, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts = np.histogram(ref, bins=breakpoints)[0]
        tst_counts = np.histogram(tst, bins=breakpoints)[0]

        ref_props = ref_counts / max(len(ref), 1)
        tst_props = tst_counts / max(len(tst), 1)

        # Replace zeros to avoid log(0); floor at 1e-4
        ref_props = np.where(ref_props == 0, 1e-4, ref_props)
        tst_props = np.where(tst_props == 0, 1e-4, tst_props)

        psi = float(np.sum((tst_props - ref_props) * np.log(tst_props / ref_props)))

        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=psi > self.threshold,
            score=psi,
            detector_name="PSI",
            feature=feature,
            computation_time_ms=elapsed,
        )
