"""
Predictive Impact and Class-Imbalance Analysis.

Evaluates counterfactual-matched degradation for:
  1. Standard Accuracy
  2. Balanced Accuracy (macro-averaged recall across imbalanced classes)
  3. ROC-AUC (probability discrimination)

Across 3 datasets x 2 conditions (marginal vs dependency) x 3 drift types x 5 repetitions.
Computes 95% confidence intervals and generates multi-metric comparison figures.

Outputs:
  - results/predictive_impact/predictive_impact_runs.csv
  - results/predictive_impact/predictive_impact_summary.csv
  - results/predictive_impact/predictive_impact_summary.txt
  - paper/figures/fig6_predictive_impact_multimetric.png
  - results/plots/fig6_predictive_impact_multimetric.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from data.loader import load_dataset
from experiments.grid import MULTIVARIATE_FEATURES
from models.trainer import train_model
from simulators import CorrelationDriftSimulator, MultivariateDriftSimulator
from utils.logger import get_logger

logger = get_logger("predictive_impact")

OUT_DIR = Path("results/predictive_impact")
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("results/plots").mkdir(parents=True, exist_ok=True)
Path("paper/figures").mkdir(parents=True, exist_ok=True)

DATASETS = ["credit", "telco", "maintenance"]
DRIFT_TYPES_MARG = ["sudden", "gradual", "recurring"]
DRIFT_TYPES_DEP = ["sudden", "gradual", "recurring"]
N_REPETITIONS = 5
BASE_SEED = 42


def evaluate_all_runs() -> pd.DataFrame:
    """Evaluate matched counterfactual degradation across all configurations."""
    rows = []
    total = len(DATASETS) * (len(DRIFT_TYPES_MARG) + len(DRIFT_TYPES_DEP)) * N_REPETITIONS
    pbar_step = 0
    t0 = time.time()

    logger.info("Evaluating predictive impact across %d trials (accuracy, balanced accuracy, ROC-AUC)...", total)

    for dataset_name in DATASETS:
        features = MULTIVARIATE_FEATURES[dataset_name]

        for rep in range(N_REPETITIONS):
            rep_seed = BASE_SEED + rep
            ds = load_dataset(dataset_name, random_state=rep_seed)

            # Train model once per (dataset, rep)
            tm = train_model(
                ds.X_reference, ds.y_reference,
                ds.X_stream, ds.y_stream,
                model_type="random_forest",
                random_state=rep_seed,
            )

            drift_start_idx = int(len(ds.X_stream) * 0.50)
            undrifted_post_X = ds.X_stream.iloc[drift_start_idx:]
            post_y = ds.y_stream.iloc[drift_start_idx:]

            # Matched counterfactual baseline performance
            base_pred = tm.predict(undrifted_post_X)
            base_prob = tm.model.predict_proba(undrifted_post_X)[:, 1]

            match_acc = float(accuracy_score(post_y, base_pred))
            match_bacc = float(balanced_accuracy_score(post_y, base_pred))
            match_auc = float(roc_auc_score(post_y, base_prob))

            # 1. Marginal shift runs
            sim_marg = MultivariateDriftSimulator(random_state=rep_seed)
            for dtype in DRIFT_TYPES_MARG:
                X_drift, _ = sim_marg.apply_multivariate(
                    ds.X_stream,
                    features=features,
                    drift_start_idx=drift_start_idx,
                    magnitude=2.0,
                    drift_type=dtype,
                )
                post_X = X_drift.iloc[drift_start_idx:]
                drift_pred = tm.predict(post_X)
                drift_prob = tm.model.predict_proba(post_X)[:, 1]

                post_acc = float(accuracy_score(post_y, drift_pred))
                post_bacc = float(balanced_accuracy_score(post_y, drift_pred))
                post_auc = float(roc_auc_score(post_y, drift_prob))

                rows.append({
                    "dataset": dataset_name,
                    "condition": "marginal",
                    "drift_type": dtype,
                    "repetition": rep,
                    "matched_acc": match_acc,
                    "post_acc": post_acc,
                    "deg_acc": match_acc - post_acc,
                    "matched_bacc": match_bacc,
                    "post_bacc": post_bacc,
                    "deg_bacc": match_bacc - post_bacc,
                    "matched_auc": match_auc,
                    "post_auc": post_auc,
                    "deg_auc": match_auc - post_auc,
                })
                pbar_step += 1

            # 2. Dependency shift runs
            sim_dep = CorrelationDriftSimulator(random_state=rep_seed)
            for dtype in DRIFT_TYPES_DEP:
                X_drift_c, _ = sim_dep.apply_multivariate(
                    ds.X_stream,
                    features=features,
                    drift_start_idx=drift_start_idx,
                    magnitude=0.8,
                    drift_type=dtype,
                )
                post_X_c = X_drift_c.iloc[drift_start_idx:]
                drift_pred_c = tm.predict(post_X_c)
                drift_prob_c = tm.model.predict_proba(post_X_c)[:, 1]

                post_acc_c = float(accuracy_score(post_y, drift_pred_c))
                post_bacc_c = float(balanced_accuracy_score(post_y, drift_pred_c))
                post_auc_c = float(roc_auc_score(post_y, drift_prob_c))

                rows.append({
                    "dataset": dataset_name,
                    "condition": "dependency",
                    "drift_type": f"correlation_{dtype}",
                    "repetition": rep,
                    "matched_acc": match_acc,
                    "post_acc": post_acc_c,
                    "deg_acc": match_acc - post_acc_c,
                    "matched_bacc": match_bacc,
                    "post_bacc": post_bacc_c,
                    "deg_bacc": match_bacc - post_bacc_c,
                    "matched_auc": match_auc,
                    "post_auc": post_auc_c,
                    "deg_auc": match_auc - post_auc_c,
                })
                pbar_step += 1

    df_runs = pd.DataFrame(rows)
    logger.info("Completed %d evaluations in %.1fs", len(df_runs), time.time() - t0)
    return df_runs


def compute_summary_with_ci(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Compute mean degradation, standard error, and 95% confidence intervals by dataset and condition."""
    summary_rows = []

    for (dataset, cond), grp in df_runs.groupby(["dataset", "condition"]):
        n = len(grp)

        def ci_stats(vals: pd.Series):
            m = vals.mean()
            s = vals.std(ddof=1)
            se = s / np.sqrt(n)
            # t-distribution critical value for df = n - 1
            t_crit = stats.t.ppf(0.975, df=n - 1) if n > 1 else 1.96
            margin = t_crit * se
            return m, se, margin, m - margin, m + margin

        acc_m, acc_se, acc_err, acc_lo, acc_hi = ci_stats(grp["deg_acc"])
        bacc_m, bacc_se, bacc_err, bacc_lo, bacc_hi = ci_stats(grp["deg_bacc"])
        auc_m, auc_se, auc_err, auc_lo, auc_hi = ci_stats(grp["deg_auc"])

        summary_rows.append({
            "dataset": dataset,
            "condition": cond,
            "N_trials": n,
            # Accuracy degradation (% points)
            "deg_acc_pp": np.round(acc_m * 100, 2),
            "deg_acc_err_pp": np.round(acc_err * 100, 2),
            "deg_acc_ci95": f"[{acc_lo*100:.2f}, {acc_hi*100:.2f}]",
            # Balanced accuracy degradation (% points)
            "deg_bacc_pp": np.round(bacc_m * 100, 2),
            "deg_bacc_err_pp": np.round(bacc_err * 100, 2),
            "deg_bacc_ci95": f"[{bacc_lo*100:.2f}, {bacc_hi*100:.2f}]",
            # ROC-AUC degradation (% points)
            "deg_auc_pp": np.round(auc_m * 100, 2),
            "deg_auc_err_pp": np.round(auc_err * 100, 2),
            "deg_auc_ci95": f"[{auc_lo*100:.2f}, {auc_hi*100:.2f}]",
            # Baseline means
            "base_acc_mean": np.round(grp["matched_acc"].mean() * 100, 2),
            "base_bacc_mean": np.round(grp["matched_bacc"].mean() * 100, 2),
            "base_auc_mean": np.round(grp["matched_auc"].mean() * 100, 2),
        })

    return pd.DataFrame(summary_rows)


def plot_multimetric_impact(df_summary: pd.DataFrame) -> None:
    """Generate 3-panel publication figure showing accuracy, balanced accuracy, and ROC-AUC degradation with 95% CIs."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#898781",
        "ytick.color": "#898781",
        "text.color": "#0b0b0b",
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300, sharey=False)

    datasets = ["credit", "telco", "maintenance"]
    labels = ["Credit Card\n(22.1% pos)", "Telco Churn\n(26.5% pos)", "AI4I Maint.\n(3.4% pos)"]
    x = np.arange(len(datasets))
    width = 0.32

    metrics = [
        ("deg_acc_pp", "deg_acc_err_pp", "(a) Plain Accuracy Degradation", "Accuracy Loss (pp)"),
        ("deg_bacc_pp", "deg_bacc_err_pp", "(b) Balanced Accuracy Degradation", "Balanced Accuracy Loss (pp)"),
        ("deg_auc_pp", "deg_auc_err_pp", "(c) ROC-AUC Degradation", "ROC-AUC Loss (pp)"),
    ]

    blue = "#2a78d6"    # Marginal shift
    orange = "#eb6834"  # Dependency shift

    for ax, (val_col, err_col, title, ylabel) in zip(axes, metrics):
        marg_vals = [
            df_summary[(df_summary["dataset"] == d) & (df_summary["condition"] == "marginal")][val_col].values[0]
            for d in datasets
        ]
        marg_errs = [
            df_summary[(df_summary["dataset"] == d) & (df_summary["condition"] == "marginal")][err_col].values[0]
            for d in datasets
        ]

        dep_vals = [
            df_summary[(df_summary["dataset"] == d) & (df_summary["condition"] == "dependency")][val_col].values[0]
            for d in datasets
        ]
        dep_errs = [
            df_summary[(df_summary["dataset"] == d) & (df_summary["condition"] == "dependency")][err_col].values[0]
            for d in datasets
        ]

        rects1 = ax.bar(
            x - width/2, marg_vals, width, yerr=marg_errs, capsize=4,
            label="Marginal Shift", color=blue, edgecolor="#184f95", linewidth=0.8, alpha=0.9
        )
        rects2 = ax.bar(
            x + width/2, dep_vals, width, yerr=dep_errs, capsize=4,
            label="Dependency Shift", color=orange, edgecolor="#b34417", linewidth=0.8, alpha=0.9
        )

        ax.axhline(0, color="#898781", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel, fontsize=10, fontweight="medium")
        ax.grid(True, linestyle=":", color="#e1e0d9", alpha=0.7, axis="y")

        if ax == axes[0]:
            ax.legend(loc="upper left", frameon=True, framealpha=0.9, facecolor="#ffffff", edgecolor="#c3c2b7", fontsize=8.5)

    plt.tight_layout()

    p1 = "paper/figures/fig6_predictive_impact_multimetric.png"
    p2 = "results/plots/fig6_predictive_impact_multimetric.png"
    fig.savefig(p1, bbox_inches="tight")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved multi-metric predictive impact figure to %s and %s", p1, p2)


def run_predictive_impact_analysis():
    df_runs = evaluate_all_runs()
    df_runs.to_csv(OUT_DIR / "predictive_impact_runs.csv", index=False)

    df_summary = compute_summary_with_ci(df_runs)
    df_summary.to_csv(OUT_DIR / "predictive_impact_summary.csv", index=False)

    plot_multimetric_impact(df_summary)

    lines = [
        "=" * 85,
        "PREDICTIVE IMPACT AND CLASS-IMBALANCE ANALYSIS (WITH 95% CONFIDENCE INTERVALS)",
        "=" * 85,
        "",
        df_summary.to_string(index=False),
        "",
        "Notes:",
        "  - Values reported in percentage points (pp). Degradation = Counterfactual Matched Baseline - Post Drift.",
        "  - Positive values indicate downstream performance loss; near-zero values indicate benign drift.",
        "  - 95% Confidence Intervals computed using Student-t distribution over the 15 experimental trials per condition.",
        "  - Class imbalance: Credit (22.1% pos), Telco (26.5% pos), Maintenance (3.4% pos).",
        "=" * 85,
    ]
    summary_text = "\n".join(lines)
    print(summary_text)
    (OUT_DIR / "predictive_impact_summary.txt").write_text(summary_text, encoding="utf-8")
    logger.info("Predictive impact analysis completed. Artifacts saved to %s", OUT_DIR)


if __name__ == "__main__":
    run_predictive_impact_analysis()
