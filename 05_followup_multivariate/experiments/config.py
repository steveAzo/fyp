from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentConfig:
    dataset: str
    detector: str
    drift_type: str
    drift_feature: str
    model: str = "random_forest"
    monitoring_frequency: int = 100
    window_size: int = 100
    reference_size: Optional[int] = None
    drift_start_ratio: float = 0.5
    drift_magnitude: float = 2.0
    repetition: int = 0
    random_state: int = 42

    def key(self) -> tuple:
        ref_s = self.reference_size if self.reference_size is not None else -1
        return (
            self.dataset,
            self.detector,
            self.drift_type,
            self.drift_feature,
            self.model,
            self.monitoring_frequency,
            self.window_size,
            ref_s,
            self.repetition,
        )
