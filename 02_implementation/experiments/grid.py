"""
Generate lists of ExperimentConfig covering the full experimental design.

Full grid  (8 100 configs):
    3 datasets × 3 detectors × 3 drift types × 5 monitoring frequencies
    × 4 window sizes × 3 reference sizes × 5 repetitions

Quick grid (~108 configs):
    One constraint level per axis — for smoke-testing the pipeline before
    committing to the full run.
"""
from __future__ import annotations

from itertools import product
from typing import Optional

from .config import ExperimentConfig

# ---------------------------------------------------------------------------
# Primary drift feature per dataset
# (the feature most likely to exhibit natural distribution shift)
# ---------------------------------------------------------------------------
PRIMARY_FEATURES: dict[str, str] = {
    "credit":      "AGE",
    "telco":       "Monthly Charges",
    "maintenance": "Tool wear [min]",
}


def generate_full_grid(
    datasets: list[str] | None = None,
    detectors: list[str] | None = None,
    drift_types: list[str] | None = None,
    monitoring_frequencies: list[int] | None = None,
    window_sizes: list[int] | None = None,
    reference_sizes: list[Optional[int]] | None = None,
    model: str = "random_forest",
    n_repetitions: int = 5,
    drift_magnitude: float = 2.0,
    random_state: int = 42,
) -> list[ExperimentConfig]:
    """
    Return the full cross-product of experimental configurations.

    All parameters have sensible defaults matching the research design.
    Override any axis to run a partial grid.
    """
    datasets            = datasets             or ["credit", "telco", "maintenance"]
    detectors           = detectors            or ["ks", "psi", "wasserstein"]
    drift_types         = drift_types          or ["sudden", "gradual", "recurring"]
    monitoring_frequencies = monitoring_frequencies or [10, 50, 100, 500, 1000]
    window_sizes        = window_sizes         or [50, 100, 500, 1000]
    reference_sizes     = reference_sizes      or [100, 500, None]

    configs: list[ExperimentConfig] = []

    for dataset, detector, drift_type, freq, win, ref_size, rep in product(
        datasets, detectors, drift_types,
        monitoring_frequencies, window_sizes, reference_sizes,
        range(n_repetitions),
    ):
        feature = PRIMARY_FEATURES[dataset]
        configs.append(
            ExperimentConfig(
                dataset=dataset,
                detector=detector,
                drift_type=drift_type,
                drift_feature=feature,
                model=model,
                monitoring_frequency=freq,
                window_size=win,
                reference_size=ref_size,
                drift_magnitude=drift_magnitude,
                repetition=rep,
                random_state=random_state,
            )
        )

    return configs


def generate_quick_grid(
    datasets: list[str] | None = None,
    n_repetitions: int = 2,
) -> list[ExperimentConfig]:
    """
    A small representative grid for pipeline validation.

    Uses one value per constraint axis — fast enough to finish in seconds.
    """
    return generate_full_grid(
        datasets=datasets or ["credit", "telco", "maintenance"],
        detectors=["ks", "psi", "wasserstein"],
        drift_types=["sudden", "gradual", "recurring"],
        monitoring_frequencies=[100],
        window_sizes=[100],
        reference_sizes=[500, None],
        n_repetitions=n_repetitions,
        random_state=0,
    )


def generate_model_comparison_grid(
    datasets: list[str] | None = None,
    n_repetitions: int = 5,
) -> list[ExperimentConfig]:
    """
    Cross all three models at a single representative constraint configuration.

    Holds constraints fixed (monitoring_frequency=100, window_size=500,
    reference_size=None) so that any difference in results is attributable
    to model choice, not constraint variation.

    3 models × 3 datasets × 3 detectors × 3 drift types × 5 reps = 405 configs.
    """
    models = ["random_forest", "logistic_regression", "lightgbm"]
    configs: list[ExperimentConfig] = []

    for model, dataset, detector, drift_type, rep in product(
        models,
        datasets or ["credit", "telco", "maintenance"],
        ["ks", "psi", "wasserstein"],
        ["sudden", "gradual", "recurring"],
        range(n_repetitions),
    ):
        feature = PRIMARY_FEATURES[dataset]
        configs.append(
            ExperimentConfig(
                dataset=dataset,
                detector=detector,
                drift_type=drift_type,
                drift_feature=feature,
                model=model,
                monitoring_frequency=100,
                window_size=500,
                reference_size=None,
                repetition=rep,
                random_state=42,
            )
        )

    return configs
