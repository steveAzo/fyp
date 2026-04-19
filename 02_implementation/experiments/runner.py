"""
Core experiment execution logic.

run_experiment(cfg)        → Optional[ExperimentMetrics]
run_experiment_batch(...)  → pd.DataFrame  (incremental CSV save, resume-safe)
"""
from __future__ import annotations

import dataclasses
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data.loader import load_dataset
from detectors import KSTestDetector, PSIDetector, WassersteinDetector, DriftDetector
from evaluation.metrics import compute_metrics, ExperimentMetrics
from experiments.config import ExperimentConfig
from models.trainer import train_model
from simulators import GradualDriftSimulator, RecurringDriftSimulator, SuddenDriftSimulator
from utils.logger import get_logger

logger = get_logger("runner")

# ---------------------------------------------------------------------------
# Internal registries
# ---------------------------------------------------------------------------

def _make_detector(name: str) -> DriftDetector:
    if name == "ks":
        return KSTestDetector(threshold=0.05)
    if name == "psi":
        # 0.25 = industry-standard "significant change" threshold.
        # The pre-drift p99 on normalised data is ~0.23, so 0.25 keeps
        # false alarms at <1% while still catching a 2-sigma shift.
        return PSIDetector(threshold=0.25, n_bins=10)
    if name == "wasserstein":
        # Pre-drift mean on normalised data is ~0.12; p99 is ~0.24.
        # 0.25 puts us safely above the noise floor.
        return WassersteinDetector(threshold=0.25)
    raise ValueError(f"Unknown detector {name!r}")


def _make_simulator(drift_type: str, random_state: int):
    if drift_type == "sudden":
        return SuddenDriftSimulator(random_state=random_state)
    if drift_type == "gradual":
        return GradualDriftSimulator(random_state=random_state)
    if drift_type == "recurring":
        return RecurringDriftSimulator(random_state=random_state)
    raise ValueError(f"Unknown drift type {drift_type!r}")


def _reference_window(X_reference: pd.DataFrame, feature: str, reference_size: Optional[int]) -> np.ndarray:
    """Return (possibly truncated) reference distribution for a feature."""
    data = X_reference[feature].values
    if reference_size is not None and len(data) > reference_size:
        data = data[-reference_size:]   # keep most-recent to simulate memory limit
    return data


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(cfg: ExperimentConfig) -> Optional[ExperimentMetrics]:
    """
    Execute one experiment and return its metrics.

    Returns None if the experiment fails for any reason (logged at ERROR
    level) so that a batch run can continue past isolated failures.
    """
    try:
        # 1. Load data
        dataset = load_dataset(cfg.dataset, random_state=cfg.random_state)

        if cfg.drift_feature not in dataset.X_stream.columns:
            raise ValueError(
                f"Feature {cfg.drift_feature!r} not found in {cfg.dataset} dataset. "
                f"Available: {list(dataset.X_stream.columns)}"
            )

        # 2. Train model on reference set; evaluate baseline on stream
        training = train_model(
            dataset.X_reference, dataset.y_reference,
            dataset.X_stream, dataset.y_stream,
            random_state=cfg.random_state,
        )
        baseline_accuracy = training.accuracy

        # 3. Inject drift into the stream
        sim_seed = cfg.random_state + cfg.repetition
        simulator = _make_simulator(cfg.drift_type, sim_seed)

        drift_start_idx = max(
            int(len(dataset.X_stream) * cfg.drift_start_ratio),
            cfg.window_size,   # guarantee at least one pre-drift check
        )

        X_drifted, drift_spec = simulator.apply(
            dataset.X_stream,
            feature=cfg.drift_feature,
            drift_start_idx=drift_start_idx,
            magnitude=cfg.drift_magnitude,
        )

        # 4. Measure post-drift model accuracy
        post_X = X_drifted.iloc[drift_start_idx:]
        post_y = dataset.y_stream.iloc[drift_start_idx:]
        post_drift_accuracy = (
            float(accuracy_score(post_y, training.predict(post_X)))
            if len(post_X) > 0
            else baseline_accuracy
        )

        # 5. Build reference window for the detector
        ref_array = _reference_window(dataset.X_reference, cfg.drift_feature, cfg.reference_size)

        # 6. Simulate monitoring: slide a window over the stream, check at frequency
        detector = _make_detector(cfg.detector)
        stream_arr = X_drifted[cfg.drift_feature].values
        n_samples = len(stream_arr)

        detection_flags: list[tuple[int, bool]] = []
        total_ms = 0.0

        for check_idx in range(cfg.window_size, n_samples, cfg.monitoring_frequency):
            test_window = stream_arr[check_idx - cfg.window_size : check_idx]
            result = detector.detect(ref_array, test_window, feature=cfg.drift_feature)
            detection_flags.append((check_idx, result.is_drift))
            total_ms += result.computation_time_ms

        if not detection_flags:
            logger.warning(
                "No checks performed for %s — window_size=%d may exceed stream length %d",
                cfg.key(), cfg.window_size, n_samples,
            )
            return None

        # 7. Compute metrics
        metrics = compute_metrics(
            detection_flags=detection_flags,
            drift_start_idx=drift_start_idx,
            baseline_accuracy=baseline_accuracy,
            post_drift_accuracy=post_drift_accuracy,
            total_computation_ms=total_ms,
            dataset=cfg.dataset,
            detector=cfg.detector,
            drift_type=cfg.drift_type,
            drift_feature=cfg.drift_feature,
            monitoring_frequency=cfg.monitoring_frequency,
            window_size=cfg.window_size,
            reference_size=cfg.reference_size if cfg.reference_size is not None else -1,
            repetition=cfg.repetition,
        )

        return metrics

    except Exception:
        logger.error("Experiment failed: %s\n%s", cfg.key(), traceback.format_exc())
        return None


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_experiment_batch(
    configs: list[ExperimentConfig],
    output_path: str = "results/results.csv",
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run a list of ExperimentConfig objects, saving results to CSV after each
    successful run.

    Args:
        configs     : list of ExperimentConfig to execute.
        output_path : CSV file path for incremental results.
        resume      : if True and the CSV already exists, skip configs whose
                      key already appears in the file (crash-safe).

    Returns:
        DataFrame of all results (including those from a prior partial run).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build set of already-completed keys
    completed: set[tuple] = set()
    if not resume and out.exists():
        out.unlink()   # delete stale file so we start fresh
        logger.info("--no-resume: deleted existing %s", out)

    if resume and out.exists():
        existing = pd.read_csv(out)
        logger.info("Resuming: %d experiments already in %s", len(existing), out)
        for _, row in existing.iterrows():
            completed.add((
                row["dataset"], row["detector"], row["drift_type"], row["drift_feature"],
                int(row["monitoring_frequency"]), int(row["window_size"]),
                int(row["reference_size"]), int(row["repetition"]),
            ))

    write_header = not out.exists()
    n_skipped = n_success = n_failed = 0

    for cfg in tqdm(configs, desc="Experiments", unit="run"):
        if cfg.key() in completed:
            n_skipped += 1
            continue

        metrics = run_experiment(cfg)

        if metrics is None:
            n_failed += 1
            continue

        row = dataclasses.asdict(metrics)
        pd.DataFrame([row]).to_csv(out, mode="a", header=write_header, index=False)
        write_header = False
        n_success += 1

    logger.info(
        "Batch complete — success: %d  skipped: %d  failed: %d",
        n_success, n_skipped, n_failed,
    )

    return pd.read_csv(out) if out.exists() else pd.DataFrame()
