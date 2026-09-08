from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass
class ExperimentMetrics:
    dataset: str
    detector: str
    drift_type: str
    drift_feature: str
    model: str
    monitoring_frequency: int
    window_size: int
    reference_size: int
    repetition: int
    drift_start_idx: int
    baseline_accuracy: float
    post_drift_accuracy: float
    accuracy_degradation: float
    matched_baseline_accuracy: float
    accuracy_degradation_matched: float
    first_detection_idx: Optional[int]
    detection_delay_samples: Optional[int]
    detection_delay_windows: Optional[int]
    missed_detection: bool
    false_positives: int
    n_checks_performed: int
    total_computation_ms: float

def compute_metrics(
    detection_flags: list[tuple[int, bool]],
    drift_start_idx: int,
    baseline_accuracy: float,
    post_drift_accuracy: float,
    matched_baseline_accuracy: float,
    total_computation_ms: float,
    dataset: str,
    detector: str,
    drift_type: str,
    drift_feature: str,
    model: str,
    monitoring_frequency: int,
    window_size: int,
    reference_size: int,
    repetition: int,
) -> ExperimentMetrics:
    first_det_idx = None
    for idx, is_drift in detection_flags:
        if is_drift and idx >= drift_start_idx:
            first_det_idx = idx
            break

    missed = first_det_idx is None
    delay_samples = (first_det_idx - drift_start_idx) if first_det_idx is not None else None
    delay_windows = (delay_samples // monitoring_frequency) if delay_samples is not None else None
    acc_deg = baseline_accuracy - post_drift_accuracy
    acc_deg_matched = matched_baseline_accuracy - post_drift_accuracy

    false_positives = sum(1 for idx, is_drift in detection_flags if is_drift and idx < drift_start_idx)
    n_checks_performed = len(detection_flags)

    return ExperimentMetrics(
        dataset=dataset,
        detector=detector,
        drift_type=drift_type,
        drift_feature=drift_feature,
        model=model,
        monitoring_frequency=monitoring_frequency,
        window_size=window_size,
        reference_size=reference_size,
        repetition=repetition,
        drift_start_idx=drift_start_idx,
        baseline_accuracy=baseline_accuracy,
        post_drift_accuracy=post_drift_accuracy,
        accuracy_degradation=acc_deg,
        matched_baseline_accuracy=matched_baseline_accuracy,
        accuracy_degradation_matched=acc_deg_matched,
        first_detection_idx=first_det_idx,
        detection_delay_samples=delay_samples,
        detection_delay_windows=delay_windows,
        missed_detection=missed,
        false_positives=false_positives,
        n_checks_performed=n_checks_performed,
        total_computation_ms=total_computation_ms,
    )
