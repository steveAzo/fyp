from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentConfig:
    """
    All parameters that define a single experimental run.

    Varying these systematically (via grid.py) is what produces the data
    needed to answer the research questions about resource constraints.
    """

    # What we are studying
    dataset: str                  # "credit" | "telco" | "maintenance"
    detector: str                 # "ks"     | "psi"   | "wasserstein"
    drift_type: str               # "sudden" | "gradual" | "recurring"
    drift_feature: str            # column name to inject drift into

    # Resource constraint dimensions (independent variables)
    monitoring_frequency: int     # check every N stream samples
    window_size: int              # test window size in samples
    reference_size: Optional[int] # max reference samples kept; None = unlimited

    # Drift injection parameters
    drift_start_ratio: float = 0.5   # drift injected after this fraction of the stream
    drift_magnitude: float = 2.0     # shift size in units of feature std

    # Reproducibility
    repetition: int = 0
    random_state: int = 42

    def key(self) -> tuple:
        """Unique identifier for deduplication when resuming runs."""
        return (
            self.dataset, self.detector, self.drift_type, self.drift_feature,
            self.monitoring_frequency, self.window_size,
            self.reference_size if self.reference_size is not None else -1,
            self.repetition,
        )
