"""
Kaplan-Meier Right-Censored Survival Analysis and Visualizations.

Addresses the editorial requirement to replace detected-only median delay comparisons
with a right-censored time-to-detection survival analysis, directly resolving the
survivorship artifact.

Outputs:
  - results/significance/survival/km_summary_dependency.csv
  - results/significance/survival/km_summary_marginal.csv
  - results/significance/survival/logrank_pairwise.csv
  - results/significance/survival/survival_analysis_summary.txt
  - paper/figures/fig7_kaplan_meier_survival.png
  - results/plots/fig7_kaplan_meier_survival.png
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.duration.survfunc import SurvfuncRight, survdiff

from utils.logger import get_logger

logger = get_logger("survival_analysis")

OUT_DIR = Path("results/significance/survival")
OUT_DIR.mkdir(parents=True, exist_ok=True)
Path("results/plots").mkdir(parents=True, exist_ok=True)
Path("paper/figures").mkdir(parents=True, exist_ok=True)

STREAM_MAX_POST_DRIFT = {
    "credit": 6000.0,
    "telco": 1407.0,
    "maintenance": 2000.0,
}

DETECTOR_COLORS = {
    "mmd": "#2a78d6",          # Primary blue (Joint-dependence)
    "c2st": "#eb6834",         # Orange (Joint-dependence)
    "hdddm": "#184f95",        # Dark blue (Marginal-information)
    "ks": "#52514e",           # Dark gray (Marginal-information)
    "psi": "#898781",          # Medium gray (Marginal-information)
    "wasserstein": "#c3c2b7",  # Light gray (Marginal-information)
}

DETECTOR_LABELS = {
    "mmd": "MMD (Joint-dependence)",
    "c2st": "C2ST (Joint-dependence)",
    "hdddm": "HDDDM (Marginal-info)",
    "ks": "KS (Marginal-info)",
    "psi": "PSI (Marginal-info)",
    "wasserstein": "Wasserstein (Marginal-info)",
}


def prepare_survival_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add event indicator, censoring time, and observed time to dataframe."""
    df = df.copy()
    df["detected"] = (~df["missed_detection"]).astype(int)
    df["censor_max"] = df["dataset"].map(STREAM_MAX_POST_DRIFT)
    df["time"] = np.where(
        df["detected"] == 1,
        np.maximum(df["detection_delay_samples"], 1.0),
        df["censor_max"],
    )
    return df


def compute_km_table(df: pd.DataFrame, tau: float = 1400.0) -> pd.DataFrame:
    """Compute per-detector Kaplan-Meier summary statistics with RMST."""
    rows = []
    detectors = ["mmd", "ks", "c2st", "hdddm", "psi", "wasserstein"]

    for det in detectors:
        sub = df[df["detector"] == det]
        n = len(sub)
        events = sub["detected"].sum()
        censored = n - events
        miss_rate = 100.0 * censored / n

        sf = SurvfuncRight(sub["time"], sub["detected"])

        # 25th percentile of detection time (surv_prob <= 0.75)
        p75 = np.where(sf.surv_prob <= 0.75)[0]
        t25 = float(sf.surv_times[p75[0]]) if len(p75) > 0 else np.nan

        # Median detection time (surv_prob <= 0.50)
        p50 = np.where(sf.surv_prob <= 0.50)[0]
        t50 = float(sf.surv_times[p50[0]]) if len(p50) > 0 else np.nan

        # 75th percentile of detection time (surv_prob <= 0.25)
        p25 = np.where(sf.surv_prob <= 0.25)[0]
        t75 = float(sf.surv_times[p25[0]]) if len(p25) > 0 else np.nan

        # RMST up to tau (area under S(t) from 0 to tau)
        idx_tau = sf.surv_times <= tau
        t_sub = sf.surv_times[idx_tau]
        p_sub = sf.surv_prob[idx_tau]

        times = np.concatenate(([0.0], t_sub, [tau]))
        probs = np.concatenate(([1.0], p_sub))
        if len(probs) < len(times):
            probs = np.append(probs, probs[-1])

        # Trapezoidal integration
        rmst = float(np.trapezoid(probs, times))

        rows.append({
            "detector": det.upper(),
            "class": "Joint-dependence" if det in ["mmd", "c2st"] else "Marginal-info",
            "N": n,
            "events_detected": events,
            "censored_missed": censored,
            "miss_rate_pct": np.round(miss_rate, 2),
            "p25_samples": t25,
            "median_samples": t50,
            "p75_samples": t75,
            f"RMST_tau{int(tau)}": np.round(rmst, 1),
        })

    return pd.DataFrame(rows)


def run_pairwise_logrank(df: pd.DataFrame, ref_detector: str = "mmd") -> pd.DataFrame:
    """Pairwise log-rank tests comparing reference detector against all others."""
    others = [d for d in ["ks", "c2st", "hdddm", "psi", "wasserstein"] if d != ref_detector]
    rows = []

    for other in others:
        pair = df[df["detector"].isin([ref_detector, other])]
        chisq, pval = survdiff(pair["time"], pair["detected"], pair["detector"])
        rows.append({
            "comparison": f"{ref_detector.upper()} vs {other.upper()}",
            "ref_detector": ref_detector.upper(),
            "comparator": other.upper(),
            "chisq": np.round(chisq, 3),
            "p_raw": pval,
            "p_bonferroni": min(pval * len(others), 1.0),
            "p_str": "<0.001" if pval < 0.001 else f"{pval:.4f}",
            "significant_005": pval * len(others) < 0.05,
        })

    return pd.DataFrame(rows)


def plot_kaplan_meier_curves(dep: pd.DataFrame, marg: pd.DataFrame) -> None:
    """Generate publication-ready 2-panel figure for survival and cumulative detection."""
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

    detectors = ["mmd", "c2st", "hdddm", "ks", "psi", "wasserstein"]
    max_x = 1500

    # Panel A: Cumulative Detection Probability 1 - S(t) under Dependency Shift
    for det in detectors:
        sub = dep[dep["detector"] == det]
        sf = SurvfuncRight(sub["time"], sub["detected"])
        
        # Step curve: t and 1 - S(t)
        t = np.concatenate(([0.0], sf.surv_times))
        cum_det = np.concatenate(([0.0], 1.0 - sf.surv_prob))
        
        # Filter to max_x
        mask = t <= max_x
        t_plot = t[mask]
        cd_plot = cum_det[mask]
        
        # Add end point at max_x if needed
        if t_plot[-1] < max_x:
            t_plot = np.append(t_plot, max_x)
            cd_plot = np.append(cd_plot, cd_plot[-1])

        lw = 2.4 if det == "mmd" else (1.8 if det == "c2st" else 1.3)
        ls = "-" if det in ["mmd", "c2st"] else ("--" if det == "hdddm" else ":")
        alpha = 1.0 if det in ["mmd", "c2st"] else 0.85

        ax1.step(
            t_plot, cd_plot * 100,
            where="post",
            label=DETECTOR_LABELS[det],
            color=DETECTOR_COLORS[det],
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
        )

    ax1.axhline(50, color="#898781", linestyle="--", linewidth=0.8, alpha=0.6, label="50% Median Line")
    ax1.set_xlim(0, max_x)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Post-Drift Observation Delay (Samples)", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Cumulative Detection Probability (%)", fontsize=11, fontweight="medium")
    ax1.set_title("(a) Cumulative Detection Function (Dependency Shift)\n1 − S(t) over Time", fontsize=12, fontweight="bold", pad=10)
    ax1.grid(True, linestyle=":", color="#e1e0d9", alpha=0.8)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="#ffffff", edgecolor="#c3c2b7", fontsize=8.5)

    # Panel B: Survival Probability S(t) (Remaining Undetected) under Dependency Shift
    for det in detectors:
        sub = dep[dep["detector"] == det]
        sf = SurvfuncRight(sub["time"], sub["detected"])
        
        t = np.concatenate(([0.0], sf.surv_times))
        s_prob = np.concatenate(([1.0], sf.surv_prob))
        
        mask = t <= max_x
        t_plot = t[mask]
        s_plot = s_prob[mask]
        
        if t_plot[-1] < max_x:
            t_plot = np.append(t_plot, max_x)
            s_plot = np.append(s_plot, s_plot[-1])

        lw = 2.4 if det == "mmd" else (1.8 if det == "c2st" else 1.3)
        ls = "-" if det in ["mmd", "c2st"] else ("--" if det == "hdddm" else ":")
        alpha = 1.0 if det in ["mmd", "c2st"] else 0.85

        ax2.step(
            t_plot, s_plot * 100,
            where="post",
            label=DETECTOR_LABELS[det],
            color=DETECTOR_COLORS[det],
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
        )

    ax2.axhline(50, color="#898781", linestyle="--", linewidth=0.8, alpha=0.6, label="50% Threshold")
    ax2.set_xlim(0, max_x)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Post-Drift Observation Delay (Samples)", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Probability Remaining Undetected S(t) (%)", fontsize=11, fontweight="medium")
    ax2.set_title("(b) Kaplan-Meier Survival Function S(t)\nResolving Survivorship Bias", fontsize=12, fontweight="bold", pad=10)
    ax2.grid(True, linestyle=":", color="#e1e0d9", alpha=0.8)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, facecolor="#ffffff", edgecolor="#c3c2b7", fontsize=8.5)

    plt.tight_layout()

    # Save to both locations
    p1 = "paper/figures/fig7_kaplan_meier_survival.png"
    p2 = "results/plots/fig7_kaplan_meier_survival.png"
    fig.savefig(p1, bbox_inches="tight")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Kaplan-Meier plot to %s and %s", p1, p2)


def run_analysis():
    marg_raw = pd.read_csv("results/results.csv")
    dep_raw = pd.read_csv("results/results_correlation.csv")

    marg = prepare_survival_data(marg_raw)
    dep = prepare_survival_data(dep_raw)

    summary_lines = []

    def log(s):
        summary_lines.append(s)
        print(s)

    log("=" * 80)
    log("KAPLAN-MEIER RIGHT-CENSORED SURVIVAL ANALYSIS")
    log("=" * 80)

    # 1. Dependency grid summary
    log("\n--- DEPENDENCY-SHIFT CONDITION (n = 3,240, 540 per detector) ---")
    km_dep = compute_km_table(dep, tau=1400.0)
    km_dep.to_csv(OUT_DIR / "km_summary_dependency.csv", index=False)
    log(km_dep.to_string(index=False))

    # 2. Overall log-rank test
    chisq_all, pval_all = survdiff(dep["time"], dep["detected"], dep["detector"])
    log(f"\nOverall Log-Rank Test across all 6 detectors: Chi-Square = {chisq_all:.3f}, p = {pval_all:.4e}")

    # 3. Pairwise log-rank vs MMD
    log("\n--- PAIRWISE LOG-RANK TESTS (MMD vs Competitors, Dependency Shift) ---")
    pairwise = run_pairwise_logrank(dep, ref_detector="mmd")
    pairwise.to_csv(OUT_DIR / "logrank_pairwise.csv", index=False)
    log(pairwise.to_string(index=False))

    # 4. Marginal grid summary (for comparison)
    log("\n--- MARGINAL-SHIFT CONDITION (n = 16,200, 2,700 per detector) ---")
    km_marg = compute_km_table(marg, tau=1400.0)
    km_marg.to_csv(OUT_DIR / "km_summary_marginal.csv", index=False)
    log(km_marg.to_string(index=False))

    # 5. Generate figure
    plot_kaplan_meier_curves(dep, marg)

    # Save summary text
    (OUT_DIR / "survival_analysis_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Kaplan-Meier survival analysis complete. Artifacts saved to %s", OUT_DIR)


if __name__ == "__main__":
    run_analysis()
