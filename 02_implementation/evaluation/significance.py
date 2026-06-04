"""
Statistical significance testing for drift detection experiment results.

All tests operate on the results DataFrame produced by run_experiment_batch.

Public API
----------
compare_detectors(df)              Kruskal-Wallis + pairwise Mann-Whitney on
                                   detection_delay, grouped by detector.

test_constraint_effect(df, col)    Spearman correlation + Kruskal-Wallis for
                                   one constraint column vs detection_delay.

compare_drift_types(df)            Same pairwise test, grouped by drift_type.

compare_models(df)                 Same pairwise test, grouped by model.
                                   Requires model-comparison results.

summary_table(df)                  One-shot table of all constraint effects,
                                   suitable for a single thesis table.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Effect size for Mann-Whitney U: r = 1 - 2U / (n1 * n2).

    Interpretation:
        |r| < 0.10  negligible
        |r| < 0.30  small
        |r| < 0.50  medium
        |r| >= 0.50 large
    """
    return float(1.0 - (2.0 * u_stat) / (n1 * n2))


def _bonferroni(p_value: float, n_comparisons: int) -> float:
    """Apply Bonferroni correction: adjusted p = min(p * n, 1.0)."""
    return min(p_value * n_comparisons, 1.0)


def _pairwise_mannwhitney(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "detection_delay",
    drop_nulls: bool = True,
) -> pd.DataFrame:
    """
    Run pairwise Mann-Whitney U tests between all levels of group_col.

    Returns a DataFrame with columns:
        group_a, group_b, n_a, n_b,
        u_statistic, p_value_raw, p_value_bonferroni, effect_size_r, significant
    """
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
            "group_a":             str(a),
            "group_b":             str(b),
            "n_a":                 len(vals_a),
            "n_b":                 len(vals_b),
            "u_statistic":         round(u_stat, 2),
            "p_value_raw":         round(p_raw, 4),
            "p_value_bonferroni":  round(p_adj, 4),
            "effect_size_r":       round(r, 3),
            "significant":         p_adj < 0.05,
        })

    return pd.DataFrame(rows)


def _kruskal_test(
    df: pd.DataFrame,
    group_col: str,
    value_col: str = "detection_delay",
    drop_nulls: bool = True,
) -> dict:
    """
    Kruskal-Wallis H-test across all levels of group_col.

    Returns a dict with keys: groups, h_statistic, p_value, significant.
    """
    groups = sorted(df[group_col].unique())
    samples = []
    for g in groups:
        vals = df.loc[df[group_col] == g, value_col]
        if drop_nulls:
            vals = vals.dropna()
        samples.append(vals.values)

    # Need at least 2 groups with data
    samples = [s for s in samples if len(s) >= 3]
    if len(samples) < 2:
        return {"groups": groups, "h_statistic": None, "p_value": None, "significant": None}

    h, p = kruskal(*samples)
    return {
        "groups":      groups,
        "h_statistic": round(float(h), 3),
        "p_value":     round(float(p), 4),
        "significant": p < 0.05,
    }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def compare_detectors(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Test whether detection_delay differs significantly across detectors.

    Returns
    -------
    kw_result : dict with Kruskal-Wallis H and p-value
    pairwise  : DataFrame of pairwise Mann-Whitney results (Bonferroni corrected)

    Example thesis sentence:
        "A Kruskal-Wallis test revealed a significant difference in detection
        delay across detectors (H=XX.X, p<0.001). Post-hoc pairwise
        Mann-Whitney tests (Bonferroni corrected) showed that KS detected
        drift significantly faster than PSI (p=0.003, r=0.41)."
    """
    kw = _kruskal_test(df, group_col="detector")
    pw = _pairwise_mannwhitney(df, group_col="detector")
    return kw, pw


def test_constraint_effect(
    df: pd.DataFrame,
    constraint_col: str,
) -> dict:
    """
    Test whether a constraint dimension significantly affects detection_delay.

    Uses Spearman correlation (captures monotonic relationships) and
    Kruskal-Wallis (group-level significance).

    Parameters
    ----------
    constraint_col : one of "monitoring_frequency", "window_size", "reference_size"

    Returns a dict with:
        spearman_rho, spearman_p, kruskal_h, kruskal_p, significant
    """
    clean = df[["detection_delay", constraint_col]].dropna()

    rho, sp = spearmanr(clean[constraint_col], clean["detection_delay"])
    kw = _kruskal_test(df, group_col=constraint_col)

    return {
        "constraint":   constraint_col,
        "spearman_rho": round(float(rho), 3),
        "spearman_p":   round(float(sp), 4),
        "kruskal_h":    kw["h_statistic"],
        "kruskal_p":    kw["p_value"],
        "significant":  (sp < 0.05) or (kw["p_value"] is not None and kw["p_value"] < 0.05),
    }


def compare_drift_types(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Test whether detection_delay differs significantly across drift types
    (sudden, gradual, recurring).

    Returns kw_result dict and pairwise DataFrame (same structure as
    compare_detectors).
    """
    kw = _kruskal_test(df, group_col="drift_type")
    pw = _pairwise_mannwhitney(df, group_col="drift_type")
    return kw, pw


def compare_models(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Test whether accuracy_degradation differs significantly across models.

    Intended for the model-comparison results (model column present).
    Uses accuracy_degradation rather than detection_delay because the
    detectors are model-agnostic; only model impact differs.

    Returns kw_result dict and pairwise DataFrame.
    """
    kw = _kruskal_test(df, group_col="model", value_col="accuracy_degradation")
    pw = _pairwise_mannwhitney(df, group_col="model", value_col="accuracy_degradation")
    return kw, pw


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman + Kruskal-Wallis for all three constraint dimensions
    in one call.  Returns a single DataFrame suitable for a thesis table.

    Columns: constraint, spearman_rho, spearman_p, kruskal_h, kruskal_p, significant
    """
    constraints = ["monitoring_frequency", "window_size", "reference_size"]
    rows = [test_constraint_effect(df, c) for c in constraints]
    return pd.DataFrame(rows)
