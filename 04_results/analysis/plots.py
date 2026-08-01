"""
Plot generation functions.

Every function:
  - accepts a DataFrame (from load.load_results())
  - saves the figure to output_dir
  - returns the Path of the saved file
  - never calls plt.show() — safe for headless / batch execution
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from load import detected_only

# ---------------------------------------------------------------------------
# Global style — applied once when this module is imported
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

_DETECTOR_ORDER = ["KS-Test", "PSI", "Wasserstein"]
_PALETTE = {"KS-Test": "#4878CF", "PSI": "#6ACC65", "Wasserstein": "#D65F5F"}
_DRIFT_ORDER = ["Sudden", "Gradual", "Recurring"]
_DATASET_ORDER = ["credit", "telco", "maintenance"]
_DATASET_LABELS = {
    "credit":      "Credit Default",
    "telco":       "Telco Churn",
    "maintenance": "Predictive Maint.",
}


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Fig 1 — Detection delay heatmap (monitoring_frequency × window_size)
# ---------------------------------------------------------------------------

def plot_delay_heatmap(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    3-panel heatmap: one panel per detector.
    Rows = monitoring_frequency, Cols = window_size, Cell = mean detection delay.
    Only detected experiments are included (missed = excluded from mean).
    """
    det = detected_only(df)
    detectors = _DETECTOR_ORDER
    freq_order = sorted(df["monitoring_frequency"].unique())
    win_order  = sorted(df["window_size"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "Mean Detection Delay (samples)\nby Monitoring Frequency × Window Size",
        fontsize=14, y=1.02,
    )

    vmin = det["detection_delay"].min()
    vmax = det["detection_delay"].max()

    for ax, det_label in zip(axes, detectors):
        subset = det[det["detector_label"] == det_label]
        pivot = (
            subset.groupby(["monitoring_frequency", "window_size"])["detection_delay"]
            .mean()
            .round(0)
            .unstack("window_size")
            .reindex(index=freq_order, columns=win_order)
        )

        sns.heatmap(
            pivot,
            ax=ax,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            cbar=(ax is axes[-1]),
        )
        ax.set_title(det_label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Window Size (samples)")
        ax.set_ylabel("Monitoring Frequency (every N samples)" if ax is axes[0] else "")

    plt.tight_layout()
    return _save(fig, output_dir, "fig1_delay_heatmap.png")


# ---------------------------------------------------------------------------
# Fig 2 — Missed detection rate by each constraint dimension
# ---------------------------------------------------------------------------

def plot_missed_by_constraint(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    3-panel bar chart showing how each resource constraint independently
    affects the missed detection rate.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Missed Detection Rate by Constraint Dimension", fontsize=14, y=1.02)

    dims = [
        ("monitoring_frequency", "Monitoring Frequency\n(every N samples)", axes[0]),
        ("window_size",          "Window Size (samples)",                   axes[1]),
        ("ref_label",            "Reference Size (samples)",                axes[2]),
    ]

    for col, xlabel, ax in dims:
        agg = (
            df.groupby([col, "detector_label"], observed=True)["missed_pct"]
            .mean()
            .reset_index()
        )
        sns.barplot(
            data=agg,
            x=col,
            y="missed_pct",
            hue="detector_label",
            hue_order=_DETECTOR_ORDER,
            palette=_PALETTE,
            ax=ax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Missed Detection Rate (%)" if ax is axes[0] else "")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
        ax.get_legend().remove()
        ax.set_title(xlabel.split("\n")[0])

    # Single shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Detector",
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
    )
    plt.tight_layout()
    return _save(fig, output_dir, "fig2_missed_by_constraint.png")


# ---------------------------------------------------------------------------
# Fig 3 — Detection delay trade-off curves
# ---------------------------------------------------------------------------

def plot_tradeoff_curves(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Line plots: detection delay vs monitoring_frequency, one line per
    window_size, one panel per detector.  Shows the resource trade-off surface.
    """
    det = detected_only(df)
    freq_order = sorted(det["monitoring_frequency"].unique())
    win_sizes  = sorted(det["window_size"].unique())
    cmap = plt.cm.get_cmap("viridis", len(win_sizes))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle(
        "Detection Delay vs. Monitoring Frequency\n(by Window Size)",
        fontsize=14, y=1.02,
    )

    for ax, det_label in zip(axes, _DETECTOR_ORDER):
        subset = det[det["detector_label"] == det_label]
        for i, ws in enumerate(win_sizes):
            data = (
                subset[subset["window_size"] == ws]
                .groupby("monitoring_frequency")["detection_delay"]
                .mean()
                .reindex(freq_order)
            )
            ax.plot(
                freq_order, data.values,
                marker="o", label=f"w={ws}",
                color=cmap(i), linewidth=2,
            )
        ax.set_title(det_label, fontsize=13, fontweight="bold")
        ax.set_xlabel("Monitoring Frequency (every N samples)")
        ax.set_ylabel("Mean Detection Delay (samples)" if ax is axes[0] else "")
        ax.legend(title="Window", fontsize=9)

    plt.tight_layout()
    return _save(fig, output_dir, "fig3_tradeoff_curves.png")


# ---------------------------------------------------------------------------
# Fig 4 — Accuracy degradation by dataset and drift type
# ---------------------------------------------------------------------------

def plot_accuracy_degradation(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Grouped bar chart: x = dataset, hue = drift type.
    Answers: "how much does undetected drift hurt the model?"
    """
    agg = (
        df.groupby(["dataset", "drift_label"])["accuracy_degradation"]
        .mean()
        .reset_index()
    )
    agg["dataset_label"] = agg["dataset"].map(_DATASET_LABELS)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=agg,
        x="dataset_label",
        y="accuracy_degradation",
        hue="drift_label",
        hue_order=_DRIFT_ORDER,
        palette="muted",
        ax=ax,
    )

    # Annotate bars with values
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)

    ax.set_title("Model Accuracy Degradation After Drift Injection", fontsize=13)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy Degradation (baseline − post-drift)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(title="Drift Type")
    plt.tight_layout()
    return _save(fig, output_dir, "fig4_accuracy_degradation.png")


# ---------------------------------------------------------------------------
# Fig 5 — Computational cost by detector
# ---------------------------------------------------------------------------

def plot_compute_time(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Bar chart comparing mean total computation time per run, grouped by
    detector and window size (larger windows = more compute).
    """
    agg = (
        df.groupby(["detector_label", "window_size"])["total_computation_ms"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=agg,
        x="detector_label",
        y="total_computation_ms",
        hue="window_size",
        order=_DETECTOR_ORDER,
        palette="Blues_d",
        ax=ax,
    )
    ax.set_title("Mean Computation Time per Monitoring Run\nby Detector and Window Size", fontsize=13)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Mean Total Computation Time (ms)")
    ax.legend(title="Window Size", loc="upper right")
    plt.tight_layout()
    return _save(fig, output_dir, "fig5_compute_time.png")


# ---------------------------------------------------------------------------
# Fig 6 — Detection delay by drift type
# ---------------------------------------------------------------------------

def plot_delay_by_drift_type(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Box plot: how much does drift type affect detection delay?
    Gradual drift is expected to take longer to detect than sudden.
    """
    det = detected_only(df)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=det,
        x="drift_label",
        y="detection_delay",
        hue="detector_label",
        hue_order=_DETECTOR_ORDER,
        order=_DRIFT_ORDER,
        palette=_PALETTE,
        flierprops={"marker": ".", "alpha": 0.4},
        ax=ax,
    )
    ax.set_title("Detection Delay Distribution by Drift Type and Detector", fontsize=13)
    ax.set_xlabel("Drift Type")
    ax.set_ylabel("Detection Delay (samples)")
    ax.legend(title="Detector", loc="upper left", bbox_to_anchor=(0.01, 0.99))
    plt.tight_layout()
    return _save(fig, output_dir, "fig6_delay_by_drift_type.png")


# ---------------------------------------------------------------------------
# Fig 8 — Model robustness to drift (accuracy degradation by model, per dataset)
# ---------------------------------------------------------------------------

_MODEL_ORDER = ["random_forest", "lightgbm", "logistic_regression"]
_MODEL_LABELS = {
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "logistic_regression": "Logistic Regression",
}


def plot_model_robustness(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Grouped bar chart: x = dataset, hue = model.
    Answers RQ4: which model degrades least under drift?
    Expects the model_comparison.csv schema (dataset, model, accuracy_degradation).
    """
    agg = (
        df.groupby(["dataset", "model"])["accuracy_degradation"]
        .mean()
        .reset_index()
    )
    agg["dataset_label"] = agg["dataset"].map(_DATASET_LABELS)
    agg["model_label"] = agg["model"].map(_MODEL_LABELS)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=agg,
        x="dataset_label",
        y="accuracy_degradation",
        hue="model_label",
        hue_order=[_MODEL_LABELS[m] for m in _MODEL_ORDER],
        order=[_DATASET_LABELS[d] for d in _DATASET_ORDER],
        palette="muted",
        ax=ax,
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=9, padding=2)

    ax.set_title("Model Accuracy Degradation Under Drift, by Model", fontsize=13)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy Degradation (baseline − post-drift)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(title="Model")
    plt.tight_layout()
    return _save(fig, output_dir, "fig8_model_robustness.png")


# ---------------------------------------------------------------------------
# Fig 7 — False positive rate by detector
# ---------------------------------------------------------------------------

def plot_false_positives(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Box plot of false positive counts per run, grouped by detector.
    Shows how often each detector fires before drift is actually introduced.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: distribution (log scale to handle outliers)
    sns.boxplot(
        data=df,
        x="detector_label",
        y="false_positives",
        hue="detector_label",
        order=_DETECTOR_ORDER,
        hue_order=_DETECTOR_ORDER,
        palette=_PALETTE,
        legend=False,
        flierprops={"marker": ".", "alpha": 0.3},
        ax=axes[0],
    )
    axes[0].set_title("False Positive Distribution")
    axes[0].set_xlabel("Detector")
    axes[0].set_ylabel("False Positives per Run")
    axes[0].set_yscale("symlog")

    # Right: mean FP by monitoring_frequency (frequency affects # of pre-drift checks)
    agg = (
        df.groupby(["monitoring_frequency", "detector_label"])["false_positives"]
        .mean()
        .reset_index()
    )
    sns.lineplot(
        data=agg,
        x="monitoring_frequency",
        y="false_positives",
        hue="detector_label",
        hue_order=_DETECTOR_ORDER,
        palette=_PALETTE,
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("Mean False Positives vs. Monitoring Frequency")
    axes[1].set_xlabel("Monitoring Frequency (every N samples)")
    axes[1].set_ylabel("Mean False Positives per Run")
    axes[1].legend(title="Detector")

    fig.suptitle("False Positive Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    return _save(fig, output_dir, "fig7_false_positives.png")
