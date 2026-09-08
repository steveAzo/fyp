"""
Feature-pair sensitivity analysis.

Purpose: confirm that the main-grid findings are not artifacts of the specific
feature pairs chosen. For each dataset, one alternative pair is tested using
the same dependency-shift grid. The qualitative result -- whether detector
identity is the dominant factor and MMD achieves the lowest miss rate -- must
hold across pairs for the information-content taxonomy claim to be robust.

Alternative pairs and rationale
--------------------------------
Credit (main: AGE + BILL_AMT1, r=+0.06):
  Alt: BILL_AMT1 + BILL_AMT2  (r=+0.951)
  Rationale: tests the high-baseline-correlation regime; auto-calibration will
  push this pair toward decorrelation (r > 0.3 threshold), which is the
  opposite direction from the main pair but structurally the same shift type.

Telco (main: Monthly Charges + Total Charges, r=+0.660):
  Alt: Tenure Months + Monthly Charges  (r=+0.247)
  Rationale: the only other available continuous pair (Telco has 3 numeric
  features total after coercing Total Charges). Medium positive baseline;
  auto-calibration pushes positive (r < 0.3), testing a different baseline
  correlation level.

Maintenance (main: Tool wear [min] + Air temperature [K], r=+0.014):
  Alt: Rotational speed [rpm] + Torque [Nm]  (r=-0.875)
  Rationale: strong negative baseline correlation; auto-calibration will push
  this toward positive (baseline r=-0.875 < 0.3 threshold), producing a shift
  from strong-negative to near-zero or positive. This is the largest
  within-dataset correlation contrast possible.

Grid: same as the split sensitivity -- a representative slice of the main
dependency-shift grid:
  - Detectors      : all six
  - Drift types    : sudden, gradual, recurring
  - Monitoring     : intervals 10, 100, 1000
  - Window sizes   : 50, 100, 500
  - Ref sizes      : 500, None
  - Repetitions    : 3
  Total: 3 x 6 x 3 x 3 x 3 x 2 x 3 = 2,916 configs
  Estimated runtime: ~25-35 min on study hardware.

Run from the 05_followup_multivariate/ directory:
  python run_feature_pair_sensitivity.py               # run all three alt pairs
  python run_feature_pair_sensitivity.py --dataset credit
  python run_feature_pair_sensitivity.py --no-resume
  python run_feature_pair_sensitivity.py --summarize-only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
from tqdm import tqdm

from data.loader import load_dataset
from detectors import (
    C2STDetector,
    HDDDMDetector,
    KSTestDetector,
    MMDDetector,
    PSIDetector,
    WassersteinDetector,
)
from simulators import CorrelationDriftSimulator
from utils.logger import get_logger

logger = get_logger("feature_pair_sensitivity")
OUT_DIR = Path("results/supplementary/feature_pair_sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Alternative feature pairs
# ---------------------------------------------------------------------------

ALT_PAIRS: dict[str, dict] = {
    "credit": {
        "features": ["BILL_AMT1", "BILL_AMT2"],
        "baseline_r": +0.951,
        "direction_note": "decorrelating (auto-calibrated: r=0.951 > 0.3 threshold)",
        "negative": None,  # let auto-calibration handle it (will choose decorrelate)
    },
    "telco": {
        "features": ["Tenure Months", "Monthly Charges"],
        "baseline_r": +0.247,
        "direction_note": "toward positive (auto-calibrated: r=0.247 < 0.3 threshold)",
        "negative": None,
    },
    "maintenance": {
        "features": ["Rotational speed [rpm]", "Torque [Nm]"],
        "baseline_r": -0.875,
        "direction_note": "toward positive (auto-calibrated: r=-0.875 < 0.3 threshold)",
        "negative": None,
    },
}

# Main pairs for reference
MAIN_PAIRS: dict[str, list[str]] = {
    "credit":      ["AGE", "BILL_AMT1"],
    "telco":       ["Monthly Charges", "Total Charges"],
    "maintenance": ["Tool wear [min]", "Air temperature [K]"],
}

DATASETS = ["credit", "telco", "maintenance"]
DETECTORS = ["ks", "psi", "wasserstein", "mmd", "c2st", "hdddm"]
DRIFT_TYPES = ["sudden", "gradual", "recurring"]
MONITORING_INTERVALS = [10, 100, 1000]
WINDOW_SIZES = [50, 100, 500]
REFERENCE_SIZES = [500, None]
N_REPS = 3
DRIFT_START_RATIO = 0.50


# ---------------------------------------------------------------------------
# Detector factory
# ---------------------------------------------------------------------------

def _make_detector(name: str):
    if name == "ks":
        return KSTestDetector(threshold=0.05)
    if name == "psi":
        return PSIDetector(threshold=0.25, n_bins=10)
    if name == "wasserstein":
        return WassersteinDetector(threshold=0.25)
    if name == "mmd":
        return MMDDetector(threshold=0.01, n_permutations=199, max_samples=300)
    if name == "c2st":
        return C2STDetector(threshold=0.01)
    if name == "hdddm":
        return HDDDMDetector(threshold=0.15)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _rs_label(rs) -> str:
    return str(rs) if rs is not None else "full"


def _load_done(path: Path) -> set[tuple]:
    key_cols = ["dataset", "detector", "drift_type", "monitoring_interval",
                "window_size", "reference_size", "repetition"]
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        if all(c in df.columns for c in key_cols):
            return set(df[key_cols].itertuples(index=False, name=None))
    except Exception:
        pass
    return set()


def _append_row(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Monitoring function
# ---------------------------------------------------------------------------

def _monitor(detector, ref_arr: np.ndarray, stream_arr: np.ndarray,
             interval: int, window: int) -> dict:
    first_alarm = None
    n_checks = 0
    for idx in range(window, len(stream_arr), interval):
        w = stream_arr[idx - window: idx]
        res = detector.detect(ref_arr, w)
        n_checks += 1
        if res.is_drift and first_alarm is None:
            first_alarm = idx
    return {"first_alarm_idx": first_alarm, "n_checks": n_checks}


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, no_resume: bool = False) -> Path:
    pair_cfg = ALT_PAIRS[dataset]
    features = pair_cfg["features"]
    pair_label = "+".join(features)
    out = OUT_DIR / f"{dataset}_alt_pair.csv"

    if no_resume and out.exists():
        out.unlink()
        logger.info("Removed %s (--no-resume)", out)

    done = _load_done(out)
    logger.info("%s alt pair [%s] | already done: %d", dataset, pair_label, len(done))

    combos = [
        (det, dt, iv, win, rs, rep)
        for det in DETECTORS
        for dt in DRIFT_TYPES
        for iv in MONITORING_INTERVALS
        for win in WINDOW_SIZES
        for rs in REFERENCE_SIZES
        for rep in range(N_REPS)
    ]

    pending = [
        c for c in combos
        if (dataset, c[0], c[1], c[2], c[3], _rs_label(c[4]), c[5]) not in done
    ]
    logger.info("%s | remaining: %d / %d", dataset, len(pending), len(combos))

    for det_name, drift_type, iv, win, rs, rep in tqdm(
            pending, desc=f"{dataset}_alt", total=len(pending)):
        seed = 42 + rep
        rsl = _rs_label(rs)
        try:
            split = load_dataset(dataset, random_state=seed)

            # Verify both features exist in the loaded split
            for f in features:
                if f not in split.X_reference.columns:
                    raise ValueError(
                        f"Feature {f!r} not found in {dataset}. "
                        f"Available: {list(split.X_reference.columns)}"
                    )

            # Build reference array
            r_arr = split.X_reference[features].values
            if rs is not None and len(r_arr) > rs:
                r_arr = r_arr[-rs:]

            # Apply correlation shift to stream
            sim = CorrelationDriftSimulator(random_state=seed)
            drift_start = int(len(split.X_stream) * DRIFT_START_RATIO)

            # Use explicit negative=None so the simulator's auto-calibration
            # reads the actual baseline correlation of THIS pair in THIS split.
            # For BILL_AMT1+BILL_AMT2 (r~0.95) it will correctly decorrelate;
            # for Rotational speed+Torque (r~-0.875) it will push toward positive.
            X_drifted, spec = sim.apply_correlation_shift(
                split.X_stream.copy(),
                features,
                drift_start,
                correlation_strength=0.8,
                drift_type=drift_type,
                negative=pair_cfg["negative"],
            )
            s_arr = X_drifted[features].values

            detector = _make_detector(det_name)
            res = _monitor(detector, r_arr, s_arr, iv, win)

            detected = res["first_alarm_idx"] is not None
            row = {
                "dataset": dataset,
                "pair_label": pair_label,
                "baseline_r": pair_cfg["baseline_r"],
                "main_pair": "+".join(MAIN_PAIRS[dataset]),
                "detector": det_name,
                "drift_type": drift_type,
                "monitoring_interval": iv,
                "window_size": win,
                "reference_size": rsl,
                "repetition": rep,
                "drift_start": drift_start,
                "n_checks": res["n_checks"],
                "detected": int(detected),
                "miss": int(not detected),
                "first_alarm_idx": res["first_alarm_idx"] if detected else "",
                "detection_delay": (res["first_alarm_idx"] - drift_start) if detected else "",
            }
            _append_row(out, row)

        except Exception as e:
            logger.warning("Error (%s %s %s iv=%d win=%d rs=%s rep=%d): %s",
                           dataset, det_name, drift_type, iv, win, rsl, rep, e)

    logger.info("%s alt pair done. Saved to %s", dataset, out)
    return out


# ---------------------------------------------------------------------------
# Summary: compare alt-pair results against main-pair results
# ---------------------------------------------------------------------------

def summarize():
    logger.info("=== Feature-pair sensitivity summary ===")

    # Load main-grid correlation results for comparison
    main_path = Path("results/results_correlation.csv")
    if not main_path.exists():
        logger.warning("Main results not found at %s", main_path)
        return

    main_df = pd.read_csv(main_path)
    main_df["miss"] = main_df["missed_detection"].astype(int)

    rows = []

    for dataset in DATASETS:
        # Main pair stats
        main_sub = main_df[main_df.dataset == dataset]
        main_miss = main_sub.groupby("detector")["miss"].mean().round(3).to_dict()

        # Alt pair stats
        alt_path = OUT_DIR / f"{dataset}_alt_pair.csv"
        if not alt_path.exists():
            logger.warning("Alt pair results missing for %s", dataset)
            continue

        alt_df = pd.read_csv(alt_path)
        alt_miss = alt_df.groupby("detector")["miss"].mean().round(3).to_dict()

        # Epsilon-squared: detector identity on detection delay
        for label, df_use, miss_col in [
            (f"{dataset} (main pair)", main_sub, "miss"),
            (f"{dataset} (alt pair: {ALT_PAIRS[dataset]['features']})", alt_df, "miss"),
        ]:
            det_df = df_use[df_use["miss"] == 0].copy()
            delay_col = "detection_delay_samples" if "detection_delay_samples" in det_df.columns else "detection_delay"
            det_df[delay_col] = pd.to_numeric(det_df[delay_col], errors="coerce")
            det_df = det_df.dropna(subset=[delay_col])

            groups = [g[delay_col].values for _, g in det_df.groupby("detector") if len(g) > 1]
            if len(groups) >= 2:
                H, p = kruskal(*groups)
                n = sum(len(g) for g in groups)
                eps2 = round(H / (n - 1), 4)
            else:
                eps2, p = None, None

            mi = ["ks", "psi", "wasserstein", "hdddm"]
            jd = ["mmd", "c2st"]
            rows.append({
                "label": label,
                "eps2_detector": eps2,
                "miss_MMD": df_use[df_use.detector == "mmd"][miss_col].mean().round(3),
                "miss_C2ST": df_use[df_use.detector == "c2st"][miss_col].mean().round(3),
                "miss_HDDDM": df_use[df_use.detector == "hdddm"][miss_col].mean().round(3),
                "miss_marginal_info_avg": df_use[df_use.detector.isin(mi)][miss_col].mean().round(3),
                "miss_joint_dep_avg": df_use[df_use.detector.isin(jd)][miss_col].mean().round(3),
            })

    if rows:
        summary_df = pd.DataFrame(rows)
        out = OUT_DIR / "feature_pair_summary.csv"
        summary_df.to_csv(out, index=False)
        print("\n=== FEATURE PAIR SENSITIVITY SUMMARY ===")
        print(summary_df.to_string(index=False))
        logger.info("Summary saved to %s", out)
    else:
        logger.warning("No data to summarize.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=DATASETS, default=None,
                   help="Run only this dataset's alt pair. Default: run all three.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--summarize-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.summarize_only:
        # Print the plan so the user knows exactly what will run
        print("=== FEATURE PAIR SENSITIVITY: RUN PLAN ===")
        for ds in DATASETS:
            cfg = ALT_PAIRS[ds]
            main_p = " + ".join(MAIN_PAIRS[ds])
            alt_p  = " + ".join(cfg["features"])
            print(f"  {ds:12s}  main: [{main_p}]  ->  alt: [{alt_p}]  (baseline r={cfg['baseline_r']:+.3f}, {cfg['direction_note']})")
        print(f"  Grid per dataset: {len(DETECTORS)} detectors x {len(DRIFT_TYPES)} drift types x "
              f"{len(MONITORING_INTERVALS)} intervals x {len(WINDOW_SIZES)} windows x "
              f"{len(REFERENCE_SIZES)} ref sizes x {N_REPS} reps = "
              f"{len(DETECTORS)*len(DRIFT_TYPES)*len(MONITORING_INTERVALS)*len(WINDOW_SIZES)*len(REFERENCE_SIZES)*N_REPS} configs/dataset")
        print()

        datasets_to_run = [args.dataset] if args.dataset else DATASETS
        for ds in datasets_to_run:
            run_dataset(ds, no_resume=args.no_resume)

    summarize()


if __name__ == "__main__":
    main()
