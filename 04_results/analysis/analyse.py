"""
Main analysis entry point.

Usage (from 04_results/analysis/):
    python analyse.py

Generates:
    04_results/plots/   — 7 publication-ready figures (PNG, 300 DPI)
    04_results/experiments/ — 4 summary CSV tables for the thesis
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from load import PLOTS_DIR, TABLES_DIR, detected_only, load_results
from plots import (
    plot_accuracy_degradation,
    plot_compute_time,
    plot_delay_by_drift_type,
    plot_delay_heatmap,
    plot_false_positives,
    plot_missed_by_constraint,
    plot_tradeoff_curves,
)

# Ensure tables dir exists
TABLES_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def export_constraint_impact(df: pd.DataFrame) -> None:
    """
    Table 1 — Effect of each constraint dimension on detection performance.
    One row per (constraint_name, constraint_value, detector).
    """
    rows = []
    for dim, label in [
        ("monitoring_frequency", "Monitoring Frequency"),
        ("window_size",          "Window Size"),
        ("reference_size",       "Reference Size"),
    ]:
        agg = (
            df.groupby([dim, "detector_label"])
            .agg(
                missed_rate=("missed_detection", "mean"),
                mean_delay=("detection_delay", "mean"),
                mean_fp=("false_positives", "mean"),
            )
            .reset_index()
        )
        agg.insert(0, "constraint", label)
        agg.rename(columns={dim: "value"}, inplace=True)
        rows.append(agg)

    out = pd.concat(rows, ignore_index=True).round(4)
    path = TABLES_DIR / "table1_constraint_impact.csv"
    out.to_csv(path, index=False)
    print(f"  Saved {path.name}  ({len(out)} rows)")


def export_detector_comparison(df: pd.DataFrame) -> None:
    """
    Table 2 — Per-detector summary across all configurations.
    """
    det = detected_only(df)
    agg = (
        df.groupby("detector_label")
        .agg(
            missed_rate=("missed_detection", "mean"),
            mean_fp=("false_positives", "mean"),
            median_fp=("false_positives", "median"),
            mean_compute_ms=("total_computation_ms", "mean"),
        )
        .join(
            det.groupby("detector_label")["detection_delay"]
            .agg(mean_delay="mean", median_delay="median")
        )
        .reset_index()
        .round(4)
    )
    path = TABLES_DIR / "table2_detector_comparison.csv"
    agg.to_csv(path, index=False)
    print(f"  Saved {path.name}  ({len(agg)} rows)")


def export_dataset_comparison(df: pd.DataFrame) -> None:
    """
    Table 3 — Per-dataset accuracy degradation and baseline accuracy.
    """
    _labels = {
        "credit":      "Credit Default",
        "telco":       "Telco Churn",
        "maintenance": "Predictive Maint.",
    }
    agg = (
        df.groupby(["dataset", "drift_label"])
        .agg(
            baseline_accuracy=("baseline_accuracy", "mean"),
            post_drift_accuracy=("post_drift_accuracy", "mean"),
            accuracy_degradation=("accuracy_degradation", "mean"),
        )
        .reset_index()
        .round(4)
    )
    agg["dataset"] = agg["dataset"].map(_labels)
    path = TABLES_DIR / "table3_dataset_comparison.csv"
    agg.to_csv(path, index=False)
    print(f"  Saved {path.name}  ({len(agg)} rows)")


def export_delay_heatmap_table(df: pd.DataFrame) -> None:
    """
    Table 4 — Detection delay pivot (monitoring_frequency × window_size),
    averaged over all detectors.  Ready to paste into the thesis as a table.
    """
    det = detected_only(df)
    freq_order = sorted(det["monitoring_frequency"].unique())
    win_order  = sorted(det["window_size"].unique())

    pivot = (
        det.groupby(["monitoring_frequency", "window_size"])["detection_delay"]
        .mean()
        .round(1)
        .unstack("window_size")
        .reindex(index=freq_order, columns=win_order)
    )
    pivot.index.name = "Monitoring Frequency \\ Window Size"
    path = TABLES_DIR / "table4_delay_pivot.csv"
    pivot.to_csv(path)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    det = detected_only(df)
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total experiments      : {len(df):,}")
    print(f"Missed detections      : {df['missed_detection'].sum()} "
          f"({df['missed_detection'].mean()*100:.2f}%)")
    print()

    print("-- Missed rate by monitoring_frequency --")
    print(df.groupby("monitoring_frequency")["missed_pct"].mean().round(2).to_string())
    print()

    print("-- Missed rate by window_size --")
    print(df.groupby("window_size")["missed_pct"].mean().round(2).to_string())
    print()

    print("-- Mean detection delay (detected only) by detector --")
    print(det.groupby("detector_label")["detection_delay"].mean().round(1).to_string())
    print()

    print("-- Mean accuracy degradation by dataset --")
    print(df.groupby("dataset")["accuracy_degradation"].mean().round(4).to_string())
    print()

    print("-- Mean compute time (ms) by detector --")
    print(df.groupby("detector_label")["total_computation_ms"].mean().round(1).to_string())
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results...")
    df = load_results()
    print(f"  {len(df):,} experiments loaded from results.csv")

    print_summary(df)

    print("Generating plots...")
    plot_fns = [
        ("Fig 1 — Detection delay heatmap",        plot_delay_heatmap),
        ("Fig 2 — Missed detection by constraint",  plot_missed_by_constraint),
        ("Fig 3 — Trade-off curves",                plot_tradeoff_curves),
        ("Fig 4 — Accuracy degradation",            plot_accuracy_degradation),
        ("Fig 5 — Compute time",                    plot_compute_time),
        ("Fig 6 — Delay by drift type",             plot_delay_by_drift_type),
        ("Fig 7 — False positives",                 plot_false_positives),
    ]
    for label, fn in plot_fns:
        path = fn(df, PLOTS_DIR)
        print(f"  {label:45s} -> {path.name}")

    print("\nExporting summary tables...")
    export_constraint_impact(df)
    export_detector_comparison(df)
    export_dataset_comparison(df)
    export_delay_heatmap_table(df)

    print(f"\nAll outputs saved to:")
    print(f"  Plots  : {PLOTS_DIR}")
    print(f"  Tables : {TABLES_DIR}")



if __name__ == "__main__":
    main()
