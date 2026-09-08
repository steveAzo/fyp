"""
Statistical significance testing for the multivariate follow-up.

Ported from 02_implementation/evaluation/significance.py so both papers use
identical, comparable nonparametric methodology (Kruskal-Wallis for k-group
differences, Mann-Whitney U with Bonferroni-corrected pairwise comparisons
and rank-biserial effect sizes, Spearman for monotonic constraint effects).
This is deliberate methodological reuse for comparability -- each test here
is a standard, independently-verifiable nonparametric procedure, not an
unverified Paper-1-specific implementation detail.

New in this module (not in Paper 1, since Paper 1 had no detector-class axis):
    compare_detector_class(df)   Univariate {ks,psi,wasserstein} vs
                                 multivariate {mmd,c2st,hdddm} group comparison
                                 -- this follow-up's central research question.
    fdr_correct(pvalues)         Benjamini-Hochberg correction across the
                                 whole family of headline tests in one run,
                                 mirroring Paper 1's Phase 2 correction.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

UNIVARIATE_DETECTORS = ["ks", "psi", "wasserstein"]
MULTIVARIATE_DETECTORS = ["mmd", "c2st", "hdddm"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Effect size for Mann-Whitney U: r = 1 - 2U / (n1 * n2).

    Interpretation: |r|<0.10 negligible, <0.30 small, <0.50 medium, >=0.50 large.
    """
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))


def _bonferroni(p_value: float, n_comparisons: int) -> float:
    return min(p_value * n_comparisons, 1.0)


def _pairwise_mannwhitney(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "detection_delay_samples",
    drop_nulls: bool = True,
) -> pd.DataFrame:
    groups = sorted(df[group_col].unique())
    n_pairs = len(groups) * (len(groups) - 1) // 2
    rows = []

    for a, b in itertools.combinations(groups, 2):
        vals_a = df.loc[df[group_col] == a, value_col]
        vals_b = df.loc[df[group_col] == b, value_col]

        if drop_nulls:
            vals_a = vals_a.dropna()
            vals_b = vals_b.dropna()

        if len(vals_a) < 3 or len(vals_b) < 3:
            continue

        u_stat, p_raw = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        p_adj = _bonferroni(p_raw, n_pairs)
        r = _rank_biserial(u_stat, len(vals_a), len(vals_b))

        rows.append({
            "group_a": str(a), "group_b": str(b),
            "n_a": len(vals_a), "n_b": len(vals_b),
            "u_statistic": round(u_stat, 2),
            "p_value_raw": p_raw,
            "p_value_bonferroni": p_adj,
            "effect_size_r": round(r, 3),
            "significant": p_adj < 0.05,
        })

    return pd.DataFrame(rows)


def _kruskal_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "detection_delay_samples",
    drop_nulls: bool = True,
) -> dict:
    groups = sorted(df[group_col].unique())
    samples = []
    for g in groups:
        vals = df.loc[df[group_col] == g, value_col]
        if drop_nulls:
            vals = vals.dropna()
        samples.append(vals.values)

    samples = [s for s in samples if len(s) >= 3]
    if len(samples) < 2:
        return {"groups": groups, "h_statistic": None, "p_value": None, "significant": None}

    h, p = kruskal(*samples)
    # epsilon-squared effect size for Kruskal-Wallis: H / ((n^2-1)/(n+1))
    n_total = sum(len(s) for s in samples)
    eps_sq = float(h) / ((n_total**2 - 1) / (n_total + 1)) if n_total > 1 else None
    return {
        "groups": groups,
        "h_statistic": round(float(h), 3),
        "p_value": float(p),
        "epsilon_squared": round(eps_sq, 4) if eps_sq is not None else None,
        "significant": p < 0.05,
    }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compare_detectors(df: pd.DataFrame, value_col: str = "detection_delay_samples") -> tuple[dict, pd.DataFrame]:
    """Kruskal-Wallis + pairwise Mann-Whitney across all 6 detectors."""
    kw = _kruskal_test(df, group_col="detector", value_col=value_col)
    pw = _pairwise_mannwhitney(df, group_col="detector", value_col=value_col)
    return kw, pw


def compare_detector_class(df: pd.DataFrame, value_col: str = "detection_delay_samples") -> dict:
    """
    The central new comparison this follow-up adds: does detector CLASS
    (univariate KS/PSI/Wasserstein vs multivariate MMD/C2ST/HDDDM) produce a
    significant difference, and how large is that effect vs. Paper 1's
    already-established constraint effects?
    """
    df = df.copy()
    df["detector_class"] = np.where(df["detector"].isin(UNIVARIATE_DETECTORS), "univariate", "multivariate")

    vals_uni = df.loc[df.detector_class == "univariate", value_col].dropna()
    vals_multi = df.loc[df.detector_class == "multivariate", value_col].dropna()

    u_stat, p_raw = mannwhitneyu(vals_uni, vals_multi, alternative="two-sided")
    r = _rank_biserial(u_stat, len(vals_uni), len(vals_multi))

    return {
        "value_col": value_col,
        "n_univariate": len(vals_uni),
        "n_multivariate": len(vals_multi),
        "median_univariate": float(vals_uni.median()),
        "median_multivariate": float(vals_multi.median()),
        "u_statistic": round(float(u_stat), 2),
        "p_value_raw": float(p_raw),
        "effect_size_r": round(r, 3),
    }


def compare_missed_detection_rate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    missed_detection is binary, so comparing rates across groups needs a
    proportions test (chi-square), not Mann-Whitney (which assumes an
    ordinal/continuous DV). Runs a chi-square test of independence between
    group_col and missed_detection.
    """
    from scipy.stats import chi2_contingency

    table = pd.crosstab(df[group_col], df["missed_detection"])
    chi2, p, dof, expected = chi2_contingency(table)
    return pd.DataFrame([{
        "group_col": group_col,
        "chi2_statistic": round(float(chi2), 3),
        "dof": dof,
        "p_value": float(p),
        "significant": p < 0.05,
    }])


def test_constraint_effect(df: pd.DataFrame, constraint_col: str, value_col: str = "detection_delay_samples") -> dict:
    clean = df[[value_col, constraint_col]].dropna()
    rho, sp = spearmanr(clean[constraint_col], clean[value_col])
    kw = _kruskal_test(df, group_col=constraint_col, value_col=value_col)

    return {
        "constraint": constraint_col,
        "spearman_rho": round(float(rho), 3),
        "spearman_p": float(sp),
        "kruskal_h": kw["h_statistic"],
        "kruskal_p": kw["p_value"],
        "significant": (sp < 0.05) or (kw["p_value"] is not None and kw["p_value"] < 0.05),
    }


def compare_drift_types(df: pd.DataFrame, value_col: str = "detection_delay_samples") -> tuple[dict, pd.DataFrame]:
    kw = _kruskal_test(df, group_col="drift_type", value_col=value_col)
    pw = _pairwise_mannwhitney(df, group_col="drift_type", value_col=value_col)
    return kw, pw


def summary_table(df: pd.DataFrame, value_col: str = "detection_delay_samples") -> pd.DataFrame:
    constraints = ["monitoring_frequency", "window_size", "reference_size"]
    rows = [test_constraint_effect(df, c, value_col=value_col) for c in constraints]
    return pd.DataFrame(rows)


def fdr_correct(pvalues: list[float]) -> list[float]:
    """
    Benjamini-Hochberg FDR correction across a family of p-values, implemented
    directly (no statsmodels dependency). Mirrors Paper 1's Phase 2
    correction (applied once across ALL headline tests in the study, not per
    sub-family Bonferroni), per the reviewer finding that within-family
    Bonferroni alone can be overly conservative when many independent test
    families exist across a study.

    Standard BH procedure: sort ascending, q(i) = min_{j>=i}[p(j) * m / j],
    enforced monotone via a reverse cumulative minimum, clipped to [0, 1].
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    result = np.empty(m)
    result[order] = adjusted
    return result.tolist()
