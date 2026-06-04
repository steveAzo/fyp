"""
Evaluation metrics for a single drift detection experiment.

The core measurement is detection_delay — how many stream samples elapsed
between when drift was injected and when the detector first flagged it.
All other metrics contextualise this number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperimentMetrics:
    # --- Experiment identity ---
    dataset: str
    detector: str
    drift_type: str
    drift_feature: str
    model: str
    monitoring_frequency: int
    window_size: int
    reference_size: int          # -1 means unlimited
    repetition: int

    # --- Ground truth ---
    drift_introduced_at: int     # stream index where drift was injected

    # --- Detection outcome ---
    drift_detected_at: Optional[int]   # first detection at or after injection; None if missed
    detection_delay: Optional[int]     # drift_detected_at - drift_introduced_at; None if missed
    missed_detection: bool             # True if drift was never detected in the stream

    # --- False alarm rate ---
    false_positives: int               # detections that fired *before* drift_introduced_at
    n_checks_performed: int            # total monitoring checks executed

    # --- Model impact ---
    baseline_accuracy: float           # accuracy before drift
    post_drift_accuracy: float         # accuracy after drift injection point
    accuracy_degradation: float        # baseline - post_drift

    # --- Resource usage ---
    total_computation_ms: float        # cumulative detector compute time


def compute_metrics(
    *,
    detection_flags: list[tuple[int, bool]],
    drift_start_idx: int,
    baseline_accuracy: float,
    post_drift_accuracy: float,
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
    """
    Derive all metrics from the raw per-check detection flags.

    Args:
        detection_flags : list of (stream_index, is_drift) for each check.
        drift_start_idx : stream index at which drift was injected.
        ...rest          : scalar metadata forwarded to ExperimentMetrics.
    """
    false_positives = sum(
        1 for idx, flagged in detection_flags if flagged and idx < drift_start_idx
    )

    drift_detected_at: Optional[int] = None
    for idx, flagged in detection_flags:
        if flagged and idx >= drift_start_idx:
            drift_detected_at = idx
            break

    detection_delay = (
        drift_detected_at - drift_start_idx if drift_detected_at is not None else None
    )

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
        drift_introduced_at=drift_start_idx,
        drift_detected_at=drift_detected_at,
        detection_delay=detection_delay,
        missed_detection=drift_detected_at is None,
        false_positives=false_positives,
        n_checks_performed=len(detection_flags),
        baseline_accuracy=baseline_accuracy,
        post_drift_accuracy=post_drift_accuracy,
        accuracy_degradation=baseline_accuracy - post_drift_accuracy,
        total_computation_ms=total_computation_ms,
    )
