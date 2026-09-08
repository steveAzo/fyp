"""
Integrity checks for the correlation-shift canary. Different checks than the
marginal grid: the key properties here are marginal preservation (checked at
the simulator level separately) and that multivariate detectors actually
outperform univariate ones on this condition (the whole point of the design).
"""
import pandas as pd

df = pd.read_csv('_canary_correlation.csv')
failures = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)

print(f"Total rows: {len(df)}\n")

check("expected row count", len(df) == 3*6*3*3, f"got {len(df)}, expected {3*6*3*3}")

grp_cols = ['dataset','detector','drift_type']
rep_variation = df.groupby(grp_cols)['baseline_accuracy'].nunique()
check(
    "repetitions produce distinct baseline_accuracy",
    (rep_variation > 1).mean() > 0.8,
    f"{(rep_variation > 1).sum()}/{len(rep_variation)} groups show variation",
)

key_cols = ['dataset','detector','drift_type','repetition']
check("no duplicate config rows", df.duplicated(subset=key_cols).sum() == 0)

key_outputs = ['baseline_accuracy','post_drift_accuracy','missed_detection','false_positives','n_checks_performed']
check("no NaN in key output columns", not df[key_outputs].isna().any().any())

# The core design claim: MMD should miss far less often than KS/PSI/Wasserstein,
# since marginals are provably unchanged and only MMD/C2ST can see joint structure.
print()
miss_by_detector = df.groupby('detector')['missed_detection'].mean().sort_values()
print(miss_by_detector.round(3))
univariate_miss = df[df.detector.isin(['ks','psi','wasserstein'])]['missed_detection'].mean()
mmd_miss = df[df.detector=='mmd']['missed_detection'].mean()
check(
    "MMD misses meaningfully less than the univariate group (design's core claim)",
    mmd_miss < univariate_miss - 0.1,
    f"MMD miss={mmd_miss:.3f}, univariate group miss={univariate_miss:.3f}",
)

print()
if failures:
    print(f"=== {len(failures)} CHECK(S) FAILED: {failures} ===")
else:
    print("=== ALL CHECKS PASSED ===")
