"""
Mixed-model and Clustered Survival Inferential Analysis.

Addresses the editorial requirement that the primary inferential analysis
respect the crossed repeated-measures blocking structure of the experiment
design, rather than treating the 16,200 / 3,240 runs as independent observations.

Models fit:
  1. Model M1: Marginal-shift binary missed detection (GEE logistic, n=16,200, 54 clusters)
  2. Model M2: Dependency-shift binary missed detection (GEE logistic, n=3,240, 54 clusters)
  3. Model M3a: Combined condition x detector-class interaction on balanced common grid (GEE logistic, n=6,480, 54 clusters)
  4. Model M3b: Combined condition x detector interaction on balanced common grid (GEE logistic, n=6,480, 54 clusters)
  5. Model M4: Detection delay log-linear GEE on detected runs (n=16,054, 54 clusters)
  6. Model M5: Clustered Cox Proportional Hazards survival model with right-censoring for missed detection (PHReg, 54 clusters)

Outputs written to results/significance/mixed_model/:
  - m1_marginal_gee.csv
  - m2_dependency_gee.csv
  - m3a_class_interaction_gee.csv
  - m3b_detector_interaction_gee.csv
  - m4_marginal_delay_gee.csv
  - m5_phreg_survival.csv
  - mixed_model_summary.txt
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import statsmodels.formula.api as smf
from statsmodels.genmod.families import Binomial, Gaussian
from statsmodels.genmod.families.links import Logit
from statsmodels.duration.hazard_regression import PHReg

from utils.logger import get_logger

logger = get_logger("mixed_model")
OUT_DIR = Path("results/significance/mixed_model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MI_DETECTORS = {"ks", "psi", "wasserstein", "hdddm"}
JD_DETECTORS = {"mmd", "c2st"}

STREAM_MAX_POST_DRIFT = {
    "credit": 6000.0,
    "telco": 1407.0,
    "maintenance": 2000.0,
}


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load marginal and dependency results, format columns, and construct common grid."""
    marg = pd.read_csv("results/results.csv")
    dep = pd.read_csv("results/results_correlation.csv")

    for df, cond in [(marg, "marginal"), (dep, "dependency")]:
        df["missed"] = df["missed_detection"].astype(int)
        df["detected"] = (~df["missed_detection"]).astype(int)
        df["shift_cond"] = cond
        df["detector_class"] = df["detector"].apply(
            lambda d: "joint_dep" if d in JD_DETECTORS else "marginal_info"
        )
        df["ref_size_cat"] = df["reference_size"].apply(
            lambda r: "full" if r == -1 else str(r)
        )
        df["win_cat"] = df["window_size"].astype(str)
        df["freq_cat"] = df["monitoring_frequency"].astype(str)
        df["drift_type_clean"] = df["drift_type"].str.replace("correlation_", "", regex=False)
        df["cluster_id"] = (
            df["dataset"] + ":" + df["detector"] + ":" + df["drift_type_clean"]
        )
        # Survival time: detection delay in samples if detected; stream length - drift_start_idx if missed (censored)
        df["censor_max"] = df["dataset"].map(STREAM_MAX_POST_DRIFT)
        df["survival_time"] = np.where(
            df["detected"] == 1,
            np.maximum(df["detection_delay_samples"], 1.0),
            df["censor_max"]
        )

    # Balanced common grid (factors tested across BOTH conditions)
    common_win = [50, 500]
    common_freq = [10, 100, 1000]
    common_ref = ["500", "full"]

    marg_sub = marg[
        marg["window_size"].isin(common_win)
        & marg["monitoring_frequency"].isin(common_freq)
        & marg["ref_size_cat"].isin(common_ref)
    ].copy()

    dep_sub = dep[
        dep["window_size"].isin(common_win)
        & dep["monitoring_frequency"].isin(common_freq)
        & dep["ref_size_cat"].isin(common_ref)
    ].copy()

    shared_cols = [
        "dataset", "detector", "detector_class", "drift_type_clean",
        "monitoring_frequency", "window_size", "reference_size",
        "repetition", "missed", "detected", "shift_cond",
        "ref_size_cat", "win_cat", "freq_cat", "cluster_id",
        "detection_delay_samples", "survival_time",
    ]
    comb_sub = pd.concat([marg_sub[shared_cols], dep_sub[shared_cols]], ignore_index=True)

    logger.info("Marginal: %d rows (%d clusters)", len(marg), marg["cluster_id"].nunique())
    logger.info("Dependency: %d rows (%d clusters)", len(dep), dep["cluster_id"].nunique())
    logger.info("Balanced common grid: %d rows (%d marginal + %d dependency)",
                len(comb_sub), len(marg_sub), len(dep_sub))
    return marg, dep, comb_sub


def tidy_gee(result, exponentiate: bool = True) -> pd.DataFrame:
    """Format GEE summary into a clean dataframe with OR/multiplier, CI, and p-value."""
    params = result.params
    conf = result.conf_int()
    pvals = result.pvalues

    tidy = pd.DataFrame({
        "term": params.index,
        "coef": params.values.round(4),
        "std_err": result.bse.values.round(4),
        "z_score": (params.values / result.bse.values).round(3),
        "p_value": pvals.values,
    })
    if exponentiate:
        tidy["OR_or_mult"] = np.exp(params.values).round(4)
        tidy["CI_lower"] = np.exp(conf.iloc[:, 0].values).round(4)
        tidy["CI_upper"] = np.exp(conf.iloc[:, 1].values).round(4)
    else:
        tidy["CI_lower"] = conf.iloc[:, 0].values.round(4)
        tidy["CI_upper"] = conf.iloc[:, 1].values.round(4)

    tidy["p_str"] = pvals.apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.4f}"
    ).values
    tidy["significant"] = (tidy["p_value"] < 0.05).astype(str)
    return tidy


def run_all_models():
    marg, dep, comb = load_datasets()
    summary_lines = []

    def hdr(s):
        summary_lines.append("\n" + "=" * 78)
        summary_lines.append(s)
        summary_lines.append("=" * 78)

    def note(s):
        summary_lines.append(s)

    # -----------------------------------------------------------------------
    # M1: Marginal-shift binary missed detection
    # -----------------------------------------------------------------------
    hdr("MODEL M1: Marginal-Shift Missed Detection (GEE Logistic, n=16,200)")
    note("Outcome: missed_detection (1=missed, 0=detected)")
    note("Clusters: dataset x detector x drift_type (54 clusters, 300 obs/cluster)")
    note("Reference: detector=KS, window=50, freq=10, ref=full, dataset=credit, drift=sudden\n")

    f_m1 = (
        "missed ~ C(detector, Treatment('ks'))"
        " + C(win_cat, Treatment('50'))"
        " + C(freq_cat, Treatment('10'))"
        " + C(ref_size_cat, Treatment('full'))"
        " + C(dataset, Treatment('credit'))"
        " + C(drift_type_clean, Treatment('sudden'))"
    )
    mod_m1 = smf.gee(f_m1, "cluster_id", data=marg, family=Binomial(link=Logit()))
    res_m1 = mod_m1.fit(maxiter=200)
    tidy_m1 = tidy_gee(res_m1, exponentiate=True)
    tidy_m1.to_csv(OUT_DIR / "m1_marginal_gee.csv", index=False)

    for _, r in tidy_m1.iterrows():
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        note(f"  {r['term']:<60} OR={r['OR_or_mult']:>8.3f} [{r['CI_lower']:>7.3f}, {r['CI_upper']:>7.3f}] p={r['p_str']:<7} {sig}")

    # -----------------------------------------------------------------------
    # M2: Dependency-shift binary missed detection
    # -----------------------------------------------------------------------
    hdr("MODEL M2: Dependency-Shift Missed Detection (GEE Logistic, n=3,240)")
    note("Outcome: missed_detection (1=missed, 0=detected)")
    note("Clusters: dataset x detector x drift_type (54 clusters, 60 obs/cluster)")
    note("Reference: detector=KS, window=50, freq=10, ref=full, dataset=credit, drift=sudden\n")

    f_m2 = (
        "missed ~ C(detector, Treatment('ks'))"
        " + C(win_cat, Treatment('50'))"
        " + C(freq_cat, Treatment('10'))"
        " + C(ref_size_cat, Treatment('full'))"
        " + C(dataset, Treatment('credit'))"
        " + C(drift_type_clean, Treatment('sudden'))"
    )
    mod_m2 = smf.gee(f_m2, "cluster_id", data=dep, family=Binomial(link=Logit()))
    res_m2 = mod_m2.fit(maxiter=200)
    tidy_m2 = tidy_gee(res_m2, exponentiate=True)
    tidy_m2.to_csv(OUT_DIR / "m2_dependency_gee.csv", index=False)

    for _, r in tidy_m2.iterrows():
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        note(f"  {r['term']:<60} OR={r['OR_or_mult']:>8.3f} [{r['CI_lower']:>7.3f}, {r['CI_upper']:>7.3f}] p={r['p_str']:<7} {sig}")

    # -----------------------------------------------------------------------
    # M3a: Condition x Detector-Class Interaction (Balanced Common Grid)
    # -----------------------------------------------------------------------
    hdr("MODEL M3a: Condition x Detector-Class Interaction (Balanced Grid, n=6,480)")
    note("Outcome: missed_detection (1=missed, 0=detected)")
    note("Balanced factorial: 3 datasets x 6 detectors x 3 drift types x 2 win x 3 freq x 2 ref x 5 reps")
    note("Clusters: dataset x detector x drift_type (54 clusters, 120 obs/cluster)")
    note("Reference: shift=marginal, class=joint_dep, win=50, freq=10, ref=full, dataset=credit, drift=sudden\n")

    f_m3a = (
        "missed ~ C(shift_cond, Treatment('marginal')) * C(detector_class, Treatment('joint_dep'))"
        " + C(win_cat, Treatment('50'))"
        " + C(freq_cat, Treatment('10'))"
        " + C(ref_size_cat, Treatment('full'))"
        " + C(dataset, Treatment('credit'))"
        " + C(drift_type_clean, Treatment('sudden'))"
    )
    mod_m3a = smf.gee(f_m3a, "cluster_id", data=comb, family=Binomial(link=Logit()))
    res_m3a = mod_m3a.fit(maxiter=200)
    tidy_m3a = tidy_gee(res_m3a, exponentiate=True)
    tidy_m3a.to_csv(OUT_DIR / "m3a_class_interaction_gee.csv", index=False)

    for _, r in tidy_m3a.iterrows():
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        note(f"  {r['term']:<60} OR={r['OR_or_mult']:>8.3f} [{r['CI_lower']:>7.3f}, {r['CI_upper']:>7.3f}] p={r['p_str']:<7} {sig}")

    # -----------------------------------------------------------------------
    # M3b: Condition x Individual Detector Interaction (Balanced Common Grid)
    # -----------------------------------------------------------------------
    hdr("MODEL M3b: Condition x Detector Interaction (Balanced Grid, n=6,480)")
    note("Outcome: missed_detection (1=missed, 0=detected)")
    note("Clusters: dataset x detector x drift_type (54 clusters, 120 obs/cluster)")
    note("Reference: shift=marginal, detector=KS, win=50, freq=10, ref=full, dataset=credit, drift=sudden\n")

    f_m3b = (
        "missed ~ C(shift_cond, Treatment('marginal')) * C(detector, Treatment('ks'))"
        " + C(win_cat, Treatment('50'))"
        " + C(freq_cat, Treatment('10'))"
        " + C(ref_size_cat, Treatment('full'))"
        " + C(dataset, Treatment('credit'))"
        " + C(drift_type_clean, Treatment('sudden'))"
    )
    mod_m3b = smf.gee(f_m3b, "cluster_id", data=comb, family=Binomial(link=Logit()))
    res_m3b = mod_m3b.fit(maxiter=200)
    tidy_m3b = tidy_gee(res_m3b, exponentiate=True)
    tidy_m3b.to_csv(OUT_DIR / "m3b_detector_interaction_gee.csv", index=False)

    for _, r in tidy_m3b.iterrows():
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        note(f"  {r['term']:<60} OR={r['OR_or_mult']:>8.3f} [{r['CI_lower']:>7.3f}, {r['CI_upper']:>7.3f}] p={r['p_str']:<7} {sig}")

    # -----------------------------------------------------------------------
    # M4: Marginal Detection Delay (Log-linear GEE, Detected Cases Only)
    # -----------------------------------------------------------------------
    hdr("MODEL M4: Marginal Detection Delay (Log-Linear GEE, n=16,054 Detected Cases)")
    note("Outcome: log(detection_delay_samples)")
    note("Clusters: dataset x detector x drift_type (54 clusters)")
    note("Exponentiated coefficients exp(b) represent MULTIPLICATIVE effect on delay")
    note("Reference: detector=KS, window=50, freq=10, ref=full, dataset=credit, drift=sudden\n")

    marg_det = marg[(marg["detected"] == 1) & (marg["detection_delay_samples"] > 0)].copy()
    marg_det["log_delay"] = np.log(marg_det["detection_delay_samples"])

    f_m4 = (
        "log_delay ~ C(detector, Treatment('ks'))"
        " + C(win_cat, Treatment('50'))"
        " + C(freq_cat, Treatment('10'))"
        " + C(ref_size_cat, Treatment('full'))"
        " + C(dataset, Treatment('credit'))"
        " + C(drift_type_clean, Treatment('sudden'))"
    )
    mod_m4 = smf.gee(f_m4, "cluster_id", data=marg_det, family=Gaussian())
    res_m4 = mod_m4.fit(maxiter=200)
    tidy_m4 = tidy_gee(res_m4, exponentiate=True)
    tidy_m4.rename(columns={"OR_or_mult": "delay_multiplier"}, inplace=True)
    tidy_m4.to_csv(OUT_DIR / "m4_marginal_delay_gee.csv", index=False)

    for _, r in tidy_m4.iterrows():
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        note(f"  {r['term']:<60} Mult={r['delay_multiplier']:>7.3f} [{r['CI_lower']:>7.3f}, {r['CI_upper']:>7.3f}] p={r['p_str']:<7} {sig}")

    # -----------------------------------------------------------------------
    # M5: Clustered Cox Proportional Hazards Model (Right-Censored Delay)
    # -----------------------------------------------------------------------
    hdr("MODEL M5: Clustered Cox Proportional Hazards (Right-Censored Detection Time)")
    note("Evaluates detection rate (Hazard Ratio) accounting for missed runs right-censored at stream end")
    note("Event: detection (1=detected, 0=right-censored/missed)")
    note("Time: detection_delay_samples (detected) or stream post-drift observation length (censored)")
    note("Clusters: dataset x detector x drift_type (54 clusters)")
    note("HR > 1 indicates faster detection / higher detection rate\n")

    # Fit on dependency condition where censoring is substantial (50.4% missed)
    f_m5_dep = (
        "survival_time ~ C(detector, Treatment('ks'))"
        " + C(win_cat, Treatment('50'))"
        " + C(freq_cat, Treatment('10'))"
        " + C(ref_size_cat, Treatment('full'))"
        " + C(dataset, Treatment('credit'))"
        " + C(drift_type_clean, Treatment('sudden'))"
    )
    ph_dep = PHReg.from_formula(
        f_m5_dep,
        status=dep["detected"],
        data=dep,
        cluster=dep["cluster_id"],
    )
    res_ph_dep = ph_dep.fit()

    ph_df = pd.DataFrame({
        "term": res_ph_dep.model.exog_names,
        "log_HR": np.round(res_ph_dep.params, 4),
        "SE": np.round(res_ph_dep.bse, 4),
        "HR": np.round(np.exp(res_ph_dep.params), 4),
        "CI_lower_HR": np.round(np.exp(res_ph_dep.conf_int()[:, 0]), 4),
        "CI_upper_HR": np.round(np.exp(res_ph_dep.conf_int()[:, 1]), 4),
        "z_score": np.round(res_ph_dep.params / res_ph_dep.bse, 3),
        "p_value": res_ph_dep.pvalues,
    })
    ph_df["p_str"] = ph_df["p_value"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.4f}")
    ph_df["significant"] = (ph_df["p_value"] < 0.05).astype(str)
    ph_df.to_csv(OUT_DIR / "m5_phreg_survival.csv", index=False)

    for _, r in ph_df.iterrows():
        sig = "***" if r["p_value"] < 0.001 else ("**" if r["p_value"] < 0.01 else ("*" if r["p_value"] < 0.05 else ""))
        note(f"  {r['term']:<60} HR={r['HR']:>7.3f} [{r['CI_lower_HR']:>7.3f}, {r['CI_upper_HR']:>7.3f}] p={r['p_str']:<7} {sig}")

    summary_text = "\n".join(summary_lines)
    print(summary_text)
    (OUT_DIR / "mixed_model_summary.txt").write_text(summary_text, encoding="utf-8")
    logger.info("All inferential models fit successfully. Output saved to %s", OUT_DIR)


if __name__ == "__main__":
    run_all_models()
