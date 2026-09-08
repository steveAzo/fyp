from __future__ import annotations

import pandas as pd

from evaluation.significance import (
    compare_detector_class,
    compare_detectors,
    compare_drift_types,
    compare_missed_detection_rate,
    fdr_correct,
    summary_table,
)

OUT = "results/significance"


def save(df: pd.DataFrame, name: str) -> None:
    path = f"{OUT}/{name}.csv"
    df.to_csv(path, index=False)
    print(f"saved {path}  ({len(df)} rows)")


def main() -> None:
    marginal = pd.read_csv("results/results.csv")
    correlation = pd.read_csv("results/results_correlation.csv")

    headline_tests: list[dict] = []  # collected for the final FDR pass

    print("=" * 70)
    print("MARGINAL GRID (results.csv, n=%d)" % len(marginal))
    print("=" * 70)

    # A1: constraint effects (replicates Paper 1's RQ, now across 6 detectors)
    constraints = summary_table(marginal)
    save(constraints, "marginal_constraint_effects")
    print(constraints.to_string(index=False))
    for _, row in constraints.iterrows():
        headline_tests.append({"test": f"marginal_constraint_{row['constraint']}", "p_value": row["kruskal_p"]})

    # A2: detector comparison (6-way)
    kw_det, pw_det = compare_detectors(marginal)
    print(f"\nDetector Kruskal-Wallis: H={kw_det['h_statistic']}, p={kw_det['p_value']:.4g}, eps2={kw_det['epsilon_squared']}")
    save(pw_det, "marginal_detector_pairwise")
    headline_tests.append({"test": "marginal_detector_kruskalwallis", "p_value": kw_det["p_value"]})

    # A3: detector CLASS comparison -- the central new question
    class_cmp = compare_detector_class(marginal)
    print(f"\nDetector CLASS (univariate vs multivariate) on detection_delay_samples:")
    print(f"  median univariate={class_cmp['median_univariate']}, median multivariate={class_cmp['median_multivariate']}")
    print(f"  U={class_cmp['u_statistic']}, p={class_cmp['p_value_raw']:.4g}, r={class_cmp['effect_size_r']}")
    save(pd.DataFrame([class_cmp]), "marginal_detector_class")
    headline_tests.append({"test": "marginal_detector_class_delay", "p_value": class_cmp["p_value_raw"]})

    # A4: missed_detection rate by detector class (binary outcome -> chi-square, not Mann-Whitney)
    marginal["detector_class"] = marginal["detector"].isin(["ks", "psi", "wasserstein"]).map({True: "univariate", False: "multivariate"})
    miss_class = compare_missed_detection_rate(marginal, "detector_class")
    save(miss_class, "marginal_missed_detection_by_class")
    print(f"\nMissed-detection rate by class (chi2): p={miss_class['p_value'].iloc[0]:.4g}")
    headline_tests.append({"test": "marginal_missed_by_class", "p_value": miss_class["p_value"].iloc[0]})

    # A5: drift type comparison (replicates Paper 1's RQ)
    kw_dt, pw_dt = compare_drift_types(marginal)
    print(f"\nDrift-type Kruskal-Wallis: H={kw_dt['h_statistic']}, p={kw_dt['p_value']:.4g}")
    save(pw_dt, "marginal_drifttype_pairwise")
    headline_tests.append({"test": "marginal_drifttype_kruskalwallis", "p_value": kw_dt["p_value"]})

    print("\n" + "=" * 70)
    print("CORRELATION GRID (results_correlation.csv, n=%d)" % len(correlation))
    print("=" * 70)

    # B1: detector comparison (6-way)
    kw_det_c, pw_det_c = compare_detectors(correlation)
    print(f"\nDetector Kruskal-Wallis: H={kw_det_c['h_statistic']}, p={kw_det_c['p_value']:.4g}, eps2={kw_det_c['epsilon_squared']}")
    save(pw_det_c, "correlation_detector_pairwise")
    headline_tests.append({"test": "correlation_detector_kruskalwallis", "p_value": kw_det_c["p_value"]})

    # B2: detector CLASS comparison -- the decisive test for this paper's core claim
    class_cmp_c = compare_detector_class(correlation)
    print(f"\nDetector CLASS (univariate vs multivariate) on detection_delay_samples:")
    print(f"  median univariate={class_cmp_c['median_univariate']}, median multivariate={class_cmp_c['median_multivariate']}")
    print(f"  U={class_cmp_c['u_statistic']}, p={class_cmp_c['p_value_raw']:.4g}, r={class_cmp_c['effect_size_r']}")
    save(pd.DataFrame([class_cmp_c]), "correlation_detector_class")
    headline_tests.append({"test": "correlation_detector_class_delay", "p_value": class_cmp_c["p_value_raw"]})

    # B3: missed_detection rate by class -- the headline miss-rate-table result, formalized
    correlation["detector_class"] = correlation["detector"].isin(["ks", "psi", "wasserstein"]).map({True: "univariate", False: "multivariate"})
    miss_class_c = compare_missed_detection_rate(correlation, "detector_class")
    save(miss_class_c, "correlation_missed_detection_by_class")
    print(f"\nMissed-detection rate by class (chi2): p={miss_class_c['p_value'].iloc[0]:.4g}")
    headline_tests.append({"test": "correlation_missed_by_class", "p_value": miss_class_c["p_value"].iloc[0]})

    # B4: constraint effects within the correlation grid
    constraints_c = summary_table(correlation)
    save(constraints_c, "correlation_constraint_effects")
    print(constraints_c.to_string(index=False))
    for _, row in constraints_c.iterrows():
        headline_tests.append({"test": f"correlation_constraint_{row['constraint']}", "p_value": row["kruskal_p"]})

    print("\n" + "=" * 70)
    print("STUDY-WIDE FDR CORRECTION ACROSS ALL HEADLINE TESTS (n=%d)" % len(headline_tests))
    print("=" * 70)
    headline_df = pd.DataFrame(headline_tests)
    headline_df["p_value_fdr"] = fdr_correct(headline_df["p_value"].tolist())
    headline_df["significant_fdr"] = headline_df["p_value_fdr"] < 0.05
    save(headline_df, "headline_tests_fdr_corrected")
    print(headline_df.to_string(index=False))


if __name__ == "__main__":
    main()
