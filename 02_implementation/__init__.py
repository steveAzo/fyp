"""
FYP: Data Drift Detection in Resource-Constrained ML Systems

A systematic empirical evaluation of drift detection methods under constraints.

Module structure:
- data: Dataset loading and preprocessing
- models: Model training and inference
- detectors: Drift detection implementations (KS, PSI, Wasserstein)
- simulators: Drift pattern simulation
- experiments: Experiment orchestration and execution
- evaluation: Metrics calculation and analysis
- utils: Logging and utilities

See PROJECT_STRUCTURE.md for complete architecture overview.
"""

__version__ = "0.1.0"
__author__ = "Stephen Kwaku Azongo"

from . import data
from . import detectors
from . import evaluation
from . import experiments
from . import models
from . import simulators
from . import utils

__all__ = [
    'data',
    'detectors',
    'evaluation',
    'experiments',
    'models',
    'simulators',
    'utils',
]
