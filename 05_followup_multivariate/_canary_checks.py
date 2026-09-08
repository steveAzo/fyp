"""
Integrity checks for the canary validation run. Run after _canary_validation.csv
exists. Every check must show PASS before recommending the full 16,200-run grid.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('_canary_validation.csv')
failures = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    if not condition:
        failures.append(name)

print(f"Total rows: {len(df)}\n")

# 1. No crashes / correct row count
check("expected row count", len(df) == 3*6*3*3, f"got {len(df)}, expected {3*6*3*3}")

# 2. Repetitions produce genuinely different results (the bug we just fixed)
grp_cols = ['dataset','detector','drift_type']
rep_variation = df.groupby(grp_cols)['baseline_accuracy'].nunique()
check(
    "repetitions produce distinct baseline_accuracy (not duplicated)",
    (rep_variation > 1).mean() > 0.8,
    f"{(rep_variation > 1).sum()}/{len(rep_variation)} groups show variation across reps",
)

# 3. No duplicate rows
key_cols = ['dataset','detector','drift_type','repetition']
check("no duplicate config rows", df.duplicated(subset=key_cols).sum() == 0)

# 4. Direction of accuracy degradation is sane per dataset (mean should not be a wild outlier)
print()
for ds in df['dataset'].unique():
    sub = df[df.dataset == ds]
    mean_deg = sub['accuracy_degradation_matched'].mean()
    print(f"  {ds}: mean accuracy_degradation_matched = {mean_deg:.4f}")
    check(f"{ds} degradation within plausible bounds (|mean| < 0.05)", abs(mean_deg) < 0.05, f"mean={mean_deg:.4f}")

# 5. No missing/NaN in key output columns
key_outputs = ['baseline_accuracy','post_drift_accuracy','missed_detection','false_positives','n_checks_performed']
check("no NaN in key output columns", not df[key_outputs].isna().any().any())

# 6. Miss rates: only flag if a detector NEVER detects anything (100% miss = broken),
# not if it detects everything (0% miss is expected for a strong 2-sigma shift at
# moderate constraints -- confirmed consistent with the full 16,200-row run earlier).
print()
for det in df['detector'].unique():
    miss_rate = df[df.detector==det]['missed_detection'].mean()
    print(f"  {det}: miss_rate={miss_rate:.3f}")
    check(f"{det} is not universally missing (100% miss would mean it's broken)", miss_rate < 1.0, f"miss_rate={miss_rate}")

print()
if failures:
    print(f"=== {len(failures)} CHECK(S) FAILED: {failures} ===")
else:
    print("=== ALL CHECKS PASSED ===")
