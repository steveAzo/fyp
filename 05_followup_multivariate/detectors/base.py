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
        pass

    @staticmethod
    def _to_array(x) -> np.ndarray:
        if hasattr(x, "values"):
            x = x.values
        return np.asarray(x, dtype=float)

    @staticmethod
    def _clean(x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            return x[~np.isnan(x)]
        elif x.ndim == 2:
            return x[~np.isnan(x).any(axis=1)]
        return x

    @staticmethod
    def _now_ms() -> float:
        return time.perf_counter() * 1000.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(threshold={self.threshold})"
