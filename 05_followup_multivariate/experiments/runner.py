from __future__ import annotations

import dataclasses
import os
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from data.loader import load_dataset
from detectors import (
    C2STDetector,
    DriftDetector,
    HDDDMDetector,
    KSTestDetector,
    MMDDetector,
    PSIDetector,
    WassersteinDetector,
)
from evaluation.metrics import ExperimentMetrics, compute_metrics
from experiments.config import ExperimentConfig
from models.trainer import train_model
from simulators import (
    CorrelationDriftSimulator,
    GradualDriftSimulator,
    MultivariateDriftSimulator,
    RecurringDriftSimulator,
    SuddenDriftSimulator,
)
from utils.logger import get_logger

logger = get_logger("runner")

def _make_detector(name: str) -> DriftDetector:
    if name == "ks":
        return KSTestDetector(threshold=0.05)
    if name == "psi":
        return PSIDetector(threshold=0.25, n_bins=10)
    if name == "wasserstein":
        return WassersteinDetector(threshold=0.25)
    if name == "mmd":
        return MMDDetector(threshold=0.01, n_permutations=199, max_samples=300)
    if name == "c2st":
        return C2STDetector(threshold=0.01)
    if name == "hdddm":
        return HDDDMDetector(threshold=0.15)
    raise ValueError(f"Unknown detector {name!r}")

def _base_drift_type(drift_type: str) -> str:
    """Strip the 'correlation_' marker to get the underlying temporal profile."""
    return drift_type[len("correlation_"):] if drift_type.startswith("correlation_") else drift_type

def _make_simulator(drift_type: str, feature: str, random_state: int):
    if drift_type.startswith("correlation_"):
        return CorrelationDriftSimulator(random_state=random_state)
    if "+" in feature:
        return MultivariateDriftSimulator(random_state=random_state)
    if drift_type == "sudden":
        return SuddenDriftSimulator(random_state=random_state)
    if drift_type == "gradual":
        return GradualDriftSimulator(random_state=random_state)
    if drift_type == "recurring":
        return RecurringDriftSimulator(random_state=random_state)
    raise ValueError(f"Unknown drift type {drift_type!r}")

def _reference_window(X_reference: pd.DataFrame, feature: str, reference_size: Optional[int]) -> np.ndarray:
    if "+" in feature:
        features = [f.strip() for f in feature.split("+")]
        data = X_reference[features].values
    else:
        data = X_reference[feature].values

    if reference_size is not None and len(data) > reference_size:
        data = data[-reference_size:]
    return data

def run_experiment(cfg: ExperimentConfig) -> Optional[ExperimentMetrics]:
    try:
        # `repetition` must actually change something, or repeated runs of the
        # same config are byte-identical duplicates that provide zero
        # statistical replication (this was previously the case: data split
        # and model training were both pinned to the constant cfg.random_state
        # regardless of repetition, so every "repeat" reproduced the exact
        # same reference/stream split and the exact same trained model).
        # Varying the seed by repetition here, the same way it already varies
        # for the drift simulator below, makes each repetition a genuinely
        # independent trial (different stratified split -> different trained
        # model -> genuinely different baseline/post-drift accuracy).
        rep_seed = cfg.random_state + cfg.repetition

        dataset = load_dataset(cfg.dataset, random_state=rep_seed)

        is_multivariate = "+" in cfg.drift_feature
        features = [f.strip() for f in cfg.drift_feature.split("+")] if is_multivariate else [cfg.drift_feature]

        for feat in features:
            if feat not in dataset.X_stream.columns:
                raise ValueError(
                    f"Feature {feat!r} not found in {cfg.dataset} dataset. "
                    f"Available: {list(dataset.X_stream.columns)}"
                )

        training = train_model(
            dataset.X_reference, dataset.y_reference,
            dataset.X_stream, dataset.y_stream,
            model_type=cfg.model,
            random_state=rep_seed,
        )
        baseline_accuracy = training.accuracy

        simulator = _make_simulator(cfg.drift_type, cfg.drift_feature, rep_seed)

        drift_start_idx = max(
            int(len(dataset.X_stream) * cfg.drift_start_ratio),
            cfg.window_size,
        )

        if is_multivariate:
            X_drifted, drift_spec = simulator.apply_multivariate(
                dataset.X_stream,
                features=features,
                drift_start_idx=drift_start_idx,
                magnitude=cfg.drift_magnitude,
                drift_type=_base_drift_type(cfg.drift_type),
            )
        else:
            X_drifted, drift_spec = simulator.apply(
                dataset.X_stream,
                feature=cfg.drift_feature,
                drift_start_idx=drift_start_idx,
                magnitude=cfg.drift_magnitude,
            )

        post_X = X_drifted.iloc[drift_start_idx:]
        post_y = dataset.y_stream.iloc[drift_start_idx:]
        post_drift_accuracy = (
            float(accuracy_score(post_y, training.predict(post_X)))
            if len(post_X) > 0
            else baseline_accuracy
        )

        # `baseline_accuracy` is measured on the WHOLE stream, but
        # `post_drift_accuracy` only on its second half -- two different
        # populations, so their gap is partly natural sampling variation
        # between stream halves, not just the drift's effect (confirmed
        # empirically: the undrifted second half of telco's stream is
        # already ~0.15pp easier than the whole stream with zero drift
        # involved). This counterfactual measures accuracy on the SAME
        # index range as post_drift_accuracy but WITHOUT drift applied,
        # isolating the drift's true causal effect from that population gap.
        undrifted_post_X = dataset.X_stream.iloc[drift_start_idx:]
        matched_baseline_accuracy = (
            float(accuracy_score(post_y, training.predict(undrifted_post_X)))
            if len(undrifted_post_X) > 0
            else baseline_accuracy
        )

        ref_array = _reference_window(dataset.X_reference, cfg.drift_feature, cfg.reference_size)

        detector = _make_detector(cfg.detector)
        stream_arr = X_drifted[features].values if is_multivariate else X_drifted[cfg.drift_feature].values
        n_samples = len(stream_arr)

        detection_flags: list[tuple[int, bool]] = []
        total_ms = 0.0

        for check_idx in range(cfg.window_size, n_samples, cfg.monitoring_frequency):
            test_window = stream_arr[check_idx - cfg.window_size : check_idx]
            result = detector.detect(ref_array, test_window, feature=cfg.drift_feature)
            detection_flags.append((check_idx, result.is_drift))
            total_ms += result.computation_time_ms

        if not detection_flags:
            logger.warning("No checks performed for %s", cfg.key())
            return None

        metrics = compute_metrics(
            detection_flags=detection_flags,
            drift_start_idx=drift_start_idx,
            baseline_accuracy=baseline_accuracy,
            post_drift_accuracy=post_drift_accuracy,
            matched_baseline_accuracy=matched_baseline_accuracy,
            total_computation_ms=total_ms,
            dataset=cfg.dataset,
            detector=cfg.detector,
            drift_type=cfg.drift_type,
            drift_feature=cfg.drift_feature,
            model=cfg.model,
            monitoring_frequency=cfg.monitoring_frequency,
            window_size=cfg.window_size,
            reference_size=cfg.reference_size if cfg.reference_size is not None else -1,
            repetition=cfg.repetition,
        )

        return metrics

    except Exception:
        logger.error("Experiment failed: %s\n%s", cfg.key(), traceback.format_exc())
        return None

def run_experiment_batch(
    configs: list[ExperimentConfig],
    output_path: str = "results/results.csv",
    resume: bool = True,
) -> pd.DataFrame:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # A second concurrent process writing to the same CSV interleaves/corrupts
    # rows (this has actually happened -- two accidental `--mode full`
    # instances collided and produced a headerless, truncated file). Refuse
    # to start rather than risk silently corrupting hours of prior work.
    lock_path = out.with_name(out.name + ".lock")
    if lock_path.exists():
        stale = True
        try:
            old_pid = int(lock_path.read_text().strip())
            os.kill(old_pid, 0)  # raises OSError if that PID isn't alive
            stale = False
        except (ValueError, OSError):
            pass  # unreadable, or the PID is dead -- treat the lock as stale

        if not stale:
            raise RuntimeError(
                f"Lock file {lock_path} exists and PID {old_pid} is still running, "
                f"meaning another run_experiment_batch process is already writing to "
                f"{out}. Running two instances against the same file corrupts it."
            )
        logger.warning(
            "Found stale lock file %s (owning process is no longer running -- likely "
            "killed by closing its terminal, which skips cleanup). Removing it and continuing.",
            lock_path,
        )
    lock_path.write_text(str(os.getpid()))

    try:
        completed: set[tuple] = set()
        if not resume and out.exists():
            out.unlink()

        if resume and out.exists():
            existing = pd.read_csv(out)
            for _, row in existing.iterrows():
                completed.add((
                    row["dataset"], row["detector"], row["drift_type"], row["drift_feature"],
                    row["model"],
                    int(row["monitoring_frequency"]), int(row["window_size"]),
                    int(row["reference_size"]), int(row["repetition"]),
                ))

        write_header = not out.exists() or out.stat().st_size == 0
        n_skipped = n_success = n_failed = 0

        for cfg in tqdm(configs, desc="Experiments", unit="run"):
            if cfg.key() in completed:
                n_skipped += 1
                continue

            try:
                metrics = run_experiment(cfg)

                if metrics is None:
                    n_failed += 1
                    continue

                row = dataclasses.asdict(metrics)
                pd.DataFrame([row]).to_csv(out, mode="a", header=write_header, index=False)
                write_header = False
                n_success += 1
            except Exception:
                # run_experiment() already catches its own internal errors and
                # returns None; this outer guard is for anything else (e.g. a
                # transient CSV write failure) so one bad row can't kill an
                # unattended multi-hour batch.
                n_failed += 1
                logger.error("Unexpected failure writing %s\n%s", cfg.key(), traceback.format_exc())

        logger.info("Batch complete — success: %d  skipped: %d  failed: %d", n_success, n_skipped, n_failed)
        return pd.read_csv(out) if out.exists() else pd.DataFrame()
    finally:
        lock_path.unlink(missing_ok=True)
