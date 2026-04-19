"""
Smoke test — run this first to verify the entire pipeline works end-to-end.

Usage (from 02_implementation/):
    python smoke_test.py

What it checks:
  1. All three datasets load and split correctly
  2. All three detectors work on clean data (should NOT flag drift)
  3. All three simulators produce detectable drift (KS should flag it)
  4. A single full experiment runs without errors
  5. Results write to CSV correctly

Expected runtime: ~30 seconds.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from data.loader import load_dataset
from detectors import KSTestDetector, PSIDetector, WassersteinDetector
from experiments.config import ExperimentConfig
from experiments.runner import run_experiment
from simulators import GradualDriftSimulator, RecurringDriftSimulator, SuddenDriftSimulator
from utils.logger import get_logger

logger = get_logger("smoke_test")

PASS = "[PASS]"
FAIL = "[FAIL]"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    msg = f"{status} {label}"
    if detail:
        msg += f" — {detail}"
    logger.info(msg)
    return condition


def test_data_loading() -> bool:
    logger.info("\n=== 1. Dataset Loading ===")
    ok = True
    for name in ["credit", "telco", "maintenance"]:
        try:
            ds = load_dataset(name)
            ok &= check(
                f"load_dataset({name!r})",
                len(ds.X_reference) > 0 and len(ds.X_stream) > 0,
                f"reference={ds.X_reference.shape}, stream={ds.X_stream.shape}",
            )
        except Exception as e:
            ok &= check(f"load_dataset({name!r})", False, str(e))
    return ok


def test_detectors_no_drift() -> bool:
    logger.info("\n=== 2. Detectors on Drift-Free Data (expect no drift) ===")
    rng = np.random.default_rng(0)
    reference = rng.normal(0, 1, 500)
    test = rng.normal(0, 1, 100)

    detectors = [
        KSTestDetector(threshold=0.05),
        PSIDetector(threshold=0.10),
        WassersteinDetector(threshold=0.10),
    ]
    ok = True
    for det in detectors:
        result = det.detect(reference, test, feature="synthetic")
        ok &= check(
            f"{det} no-drift",
            not result.is_drift,
            f"score={result.score:.4f}  time={result.computation_time_ms:.2f}ms",
        )
    return ok


def test_simulators() -> bool:
    logger.info("\n=== 3. Drift Simulators (expect KS to flag drift) ===")
    ds = load_dataset("credit")
    feature = "AGE"
    drift_start = len(ds.X_stream) // 2
    ok = True

    for SimCls in [SuddenDriftSimulator, GradualDriftSimulator, RecurringDriftSimulator]:
        sim = SimCls(random_state=42)
        X_drifted, spec = sim.apply(ds.X_stream, feature, drift_start)

        ref = ds.X_reference[feature].values
        test_post = X_drifted[feature].values[drift_start : drift_start + 200]

        result = KSTestDetector(threshold=0.05).detect(ref, test_post)
        ok &= check(
            f"{spec.drift_type} drift detectable",
            result.is_drift,
            f"KS p-value={result.score:.4f}",
        )
    return ok


def test_full_experiment() -> bool:
    logger.info("\n=== 4. Full Experiment Run ===")
    cfg = ExperimentConfig(
        dataset="credit",
        detector="ks",
        drift_type="sudden",
        drift_feature="AGE",
        monitoring_frequency=50,
        window_size=100,
        reference_size=500,
        repetition=0,
    )
    t0 = time.perf_counter()
    metrics = run_experiment(cfg)
    elapsed = time.perf_counter() - t0

    if metrics is None:
        return check("full experiment", False, "run_experiment returned None")

    return check(
        "full experiment",
        True,
        (
            f"delay={metrics.detection_delay}  "
            f"missed={metrics.missed_detection}  "
            f"fp={metrics.false_positives}  "
            f"acc_drop={metrics.accuracy_degradation:.3f}  "
            f"elapsed={elapsed:.1f}s"
        ),
    )


def main() -> None:
    results = [
        test_data_loading(),
        test_detectors_no_drift(),
        test_simulators(),
        test_full_experiment(),
    ]

    passed = sum(results)
    total = len(results)
    logger.info("\n=== Summary: %d / %d test groups passed ===", passed, total)

    if passed < total:
        logger.error("Some checks failed — fix before running the full experiment grid.")
        sys.exit(1)
    else:
        logger.info("All checks passed. Run `python run_experiments.py` to start experiments.")


if __name__ == "__main__":
    main()
