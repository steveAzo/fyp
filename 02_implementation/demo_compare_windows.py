"""Defense demo: isolate the effect of detection-window size.

Run from the 02_implementation directory:
    python demo_compare_windows.py

The two runs use the same dataset, detector, drift type, monitoring
frequency, reference size, model, and random seed. Only window_size changes.
"""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from experiments.config import ExperimentConfig
from experiments.runner import run_experiment


def main() -> None:
    common = dict(
        dataset="credit",
        detector="ks",
        drift_type="sudden",
        drift_feature="AGE",
        model="random_forest",
        monitoring_frequency=100,
        reference_size=500,
        drift_start_ratio=0.5,
        drift_magnitude=2.0,
        repetition=0,
        random_state=42,
    )

    configurations = [
        ExperimentConfig(window_size=50, **common),
        ExperimentConfig(window_size=1000, **common),
    ]

    results = []
    for config in configurations:
        metrics = run_experiment(config)
        if metrics is None:
            raise RuntimeError(f"Experiment failed for configuration: {config.key()}")
        results.append(asdict(metrics))

    table = pd.DataFrame(results)
    display_columns = [
        "detector",
        "window_size",
        "monitoring_frequency",
        "reference_size",
        "drift_introduced_at",
        "drift_detected_at",
        "detection_delay",
        "missed_detection",
        "false_positives",
        "baseline_accuracy",
        "post_drift_accuracy",
        "accuracy_degradation",
    ]

    print("\nControlled comparison: only window_size changes")
    print(table[display_columns].to_string(index=False))

    first, second = table.iloc[0], table.iloc[1]
    if pd.notna(first["detection_delay"]) and pd.notna(second["detection_delay"]):
        ratio = second["detection_delay"] / max(first["detection_delay"], 1)
        print(
            f"\nInterpretation: increasing the window from "
            f"{int(first['window_size'])} to {int(second['window_size'])} "
            f"changed detection delay from {int(first['detection_delay'])} "
            f"to {int(second['detection_delay'])} samples "
            f"({ratio:.2f}x in this single run)."
        )
    else:
        print(
            "\nInterpretation: at least one configuration missed the drift; "
            "compare missed_detection and false_positives above."
        )

    print(
        "\nThis is a controlled demonstration of the mechanism. "
        "The reported 37-fold range comes from the full factorial experiment."
    )


if __name__ == "__main__":
    main()
