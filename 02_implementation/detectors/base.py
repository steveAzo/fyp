"""
Abstract base class for all drift detectors.

Every detector must implement detect(reference, test, feature) and return a
DetectionResult.  The base class provides helpers for array coercion,
NaN removal, and computation-time measurement.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectionResult:
    is_drift: bool
    score: float
    detector_name: str
    feature: str
    computation_time_ms: float


class DriftDetector(ABC):
    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    @abstractmethod
    def detect(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        feature: str = "",
    ) -> DetectionResult:
        """
        Compare test distribution against reference.

        Args:
            reference: 1-D array representing the reference (training) distribution.
            test:      1-D array representing the current monitoring window.
            feature:   Feature name — stored in result for traceability.

        Returns:
            DetectionResult with is_drift, score, and timing.
        """

    # ------------------------------------------------------------------
    # Shared helpers — call these in subclass detect() implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _to_array(x) -> np.ndarray:
        if hasattr(x, "values"):
            x = x.values
        return np.asarray(x, dtype=float)

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        return x[~np.isnan(x)]

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"
