from __future__ import annotations

from itertools import product
from typing import Optional
from .config import ExperimentConfig

MULTIVARIATE_FEATURES: dict[str, list[str]] = {
    "credit":      ["AGE", "BILL_AMT1"],
    "telco":       ["Monthly Charges", "Total Charges"],
    "maintenance": ["Tool wear [min]", "Air temperature [K]"],
}

def generate_multivariate_grid(
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
    datasets               = datasets               or ["credit", "telco", "maintenance"]
    detectors              = detectors              or ["ks", "psi", "wasserstein", "mmd", "c2st", "hdddm"]
    drift_types            = drift_types            or ["sudden", "gradual", "recurring"]
    monitoring_frequencies = monitoring_frequencies or [10, 50, 100, 500, 1000]
    window_sizes           = window_sizes           or [50, 100, 500, 1000]
    reference_sizes        = reference_sizes        or [100, 500, None]

    configs: list[ExperimentConfig] = []

    for dataset, detector, drift_type, freq, win, ref_size, rep in product(
        datasets, detectors, drift_types,
        monitoring_frequencies, window_sizes, reference_sizes,
        range(n_repetitions),
    ):
        features = MULTIVARIATE_FEATURES[dataset]
        drift_feature_str = "+".join(features)
        configs.append(
            ExperimentConfig(
                dataset=dataset,
                detector=detector,
                drift_type=drift_type,
                drift_feature=drift_feature_str,
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

def generate_correlation_grid(
    datasets: list[str] | None = None,
    detectors: list[str] | None = None,
    monitoring_frequencies: list[int] | None = None,
    window_sizes: list[int] | None = None,
    reference_sizes: list[Optional[int]] | None = None,
    model: str = "random_forest",
    n_repetitions: int = 5,
    correlation_strength: float = 0.8,
    random_state: int = 42,
) -> list[ExperimentConfig]:
    """Supplementary grid for the correlation-only (marginal-preserving) drift
    condition. Deliberately smaller than the full F x W x R grid used for the
    marginal-shift condition (see follow-up.md 5.5: don't multiply the full
    factorial for a secondary condition) -- this answers "does the detector
    ranking change once marginals stop leaking the signal" without doubling
    the total compute budget on top of an already MMD-heavy full grid.
    """
    datasets               = datasets               or ["credit", "telco", "maintenance"]
    detectors              = detectors              or ["ks", "psi", "wasserstein", "mmd", "c2st", "hdddm"]
    drift_types             = ["correlation_sudden", "correlation_gradual", "correlation_recurring"]
    monitoring_frequencies = monitoring_frequencies or [10, 100, 1000]
    window_sizes           = window_sizes           or [50, 500]
    reference_sizes        = reference_sizes        or [500, None]

    configs: list[ExperimentConfig] = []

    for dataset, detector, drift_type, freq, win, ref_size, rep in product(
        datasets, detectors, drift_types,
        monitoring_frequencies, window_sizes, reference_sizes,
        range(n_repetitions),
    ):
        features = MULTIVARIATE_FEATURES[dataset]
        drift_feature_str = "+".join(features)
        configs.append(
            ExperimentConfig(
                dataset=dataset,
                detector=detector,
                drift_type=drift_type,
                drift_feature=drift_feature_str,
                model=model,
                monitoring_frequency=freq,
                window_size=win,
                reference_size=ref_size,
                drift_magnitude=correlation_strength,
                repetition=rep,
                random_state=random_state,
            )
        )

    return configs

def generate_quick_multivariate_grid(
    datasets: list[str] | None = None,
    n_repetitions: int = 2,
) -> list[ExperimentConfig]:
    return generate_multivariate_grid(
        datasets=datasets or ["credit", "telco", "maintenance"],
        detectors=["ks", "psi", "wasserstein", "mmd", "c2st", "hdddm"],
        drift_types=["sudden", "gradual", "recurring"],
        monitoring_frequencies=[100],
        window_sizes=[100],
        reference_sizes=[500, None],
        n_repetitions=n_repetitions,
        random_state=0,
    )
