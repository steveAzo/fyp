"""
Abstract base class for drift simulators.

All simulators receive a copy of the stream DataFrame, modify a single
feature starting from drift_start_idx, and return both the modified
DataFrame and a DriftSpec describing exactly what was done (for
reproducibility and metric computation downstream).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DriftSpec:
    drift_type: str
    feature: str
    drift_start_idx: int
    parameters: dict = field(default_factory=dict)


class DriftSimulator(ABC):
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    @abstractmethod
    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        **kwargs,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        """
        Inject drift into X[feature] starting at drift_start_idx.

        Returns:
            X_drifted : modified copy of X
            spec      : DriftSpec documenting what was applied
        """
