"""
Supplementary analyses required by editorial review.

RESUMABLE: Every analysis writes results row-by-row to its output CSV.
If interrupted and restarted, already-completed configurations are skipped
automatically. Use --no-resume to force a full restart.

Produces (all in results/supplementary/):
  false_alarm_rates.csv           Type I error rates on no-drift streams
  false_alarm_rates_summary.csv   Mean FAR per detector (written at end)
  mmd_budget_sensitivity.csv      MMD miss rate at 4 sample caps
  c2st_sample_sensitivity.csv     C2ST miss rate at 5 window sizes x 2 classifiers
  dependency_strength.csv         Pre/post Pearson r and Spearman rho per dataset
  dependency_strength_summary.csv
  saturation_quantiles.csv        Pre/post quantiles for each drifted feature

Run from the 05_followup_multivariate/ directory:
  python run_supplementary.py                          # run all
  python run_supplementary.py --only false_alarm       # one at a time
  python run_supplementary.py --only mmd_budget
  python run_supplementary.py --only c2st_sensitivity
  python run_supplementary.py --only dep_strength
  python run_supplementary.py --only saturation
  python run_supplementary.py --no-resume              # force fresh start
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
from simulators.base import DriftSimulator
from experiments.grid import MULTIVARIATE_FEATURES
from utils.logger import get_logger

logger = get_logger("supplementary")
OUT_DIR = Path("results/supplementary")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["credit", "telco", "maintenance"]
DETECTORS_ALL = ["ks", "psi", "wasserstein", "mmd", "c2st", "hdddm"]
DRIFT_START_RATIO = 0.50


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_done(path: Path, key_cols: list[str]) -> set[tuple]:
    """Read already-completed configs from an existing CSV."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        if all(c in df.columns for c in key_cols):
            return set(df[key_cols].itertuples(index=False, name=None))
    except Exception:
        pass
    return set()


def _append_row(path: Path, row: dict, write_header: bool) -> None:
    """Append one row to a CSV, writing the header only on the first write."""
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_detector(name: str, **kwargs):
    if name == "ks":
        return KSTestDetector(threshold=0.05)
    if name == "psi":
        return PSIDetector(threshold=0.25, n_bins=10)
    if name == "wasserstein":
        return WassersteinDetector(threshold=0.25)
    if name == "mmd":
        return MMDDetector(threshold=0.01, n_permutations=199,
                           max_samples=kwargs.get("max_samples", 300))
    if name == "c2st":
        return C2STDetector(threshold=0.01)
    if name == "hdddm":
        return HDDDMDetector(threshold=0.15)
    raise ValueError(f"Unknown detector: {name!r}")


def _load(dataset: str, random_state: int = 42):
    return load_dataset(dataset, random_state=random_state)


def _ref_arr(split, features: list[str], reference_size=None) -> np.ndarray:
    data = split.X_reference[features].values
    if reference_size is not None and len(data) > reference_size:
        data = data[-reference_size:]
    return data


def _monitor(detector, ref_arr, stream_arr, interval: int, window: int) -> dict:
    first_alarm = None
    alarm_count = 0
    n_checks = 0
    n_stream = len(stream_arr)
    for idx in range(window, n_stream, interval):
        w = stream_arr[idx - window: idx]
        res = detector.detect(ref_arr, w)
        n_checks += 1
        if res.is_drift:
            alarm_count += 1
            if first_alarm is None:
                first_alarm = idx
    return {"first_alarm_idx": first_alarm, "alarm_count": alarm_count, "n_checks": n_checks}


# ---------------------------------------------------------------------------
# 1. FALSE-ALARM RATES (no-drift control)
# ---------------------------------------------------------------------------

def run_false_alarm_rates(no_resume: bool = False):
    """
    Run each detector on streams with NO drift injected.

    Grid design:
    - Fast detectors (KS, PSI, Wasserstein, HDDDM, C2ST): 3 intervals x 3 windows
      x 2 ref sizes x 3 datasets x 5 reps = 1350 runs (~5 min)
    - MMD (slow, 199 permutations/check): interval=100 only x 3 windows x 2 ref
      x 3 datasets x 5 reps = 90 runs (~30 min)
    Total: 1440 runs, estimated ~35 min.
    """
    logger.info("=== Running false-alarm rate analysis (1440 configs) ===")

    out = OUT_DIR / "false_alarm_rates.csv"
    key_cols = ["dataset", "detector", "monitoring_interval", "window_size",
                "reference_size", "repetition"]

    if no_resume and out.exists():
        out.unlink()
        logger.info("Removed existing %s (--no-resume)", out)

    done = _load_done(out, key_cols)
    logger.info("Already completed: %d configs", len(done))

    fast_dets = ["ks", "psi", "wasserstein", "hdddm", "c2st"]
    intervals_fast = [10, 100, 1000]
    intervals_mmd  = [100]
    windows = [50, 100, 500]
    ref_sizes = [500, None]
    n_reps = 5

    combos = (
        [(ds, det, iv, win, rs, rep)
         for ds in DATASETS for det in fast_dets
         for iv in intervals_fast for win in windows
         for rs in ref_sizes for rep in range(n_reps)]
        +
        [(ds, "mmd", iv, win, rs, rep)
         for ds in DATASETS
         for iv in intervals_mmd for win in windows
         for rs in ref_sizes for rep in range(n_reps)]
    )

    pending = [
        c for c in combos
        if (c[0], c[1], c[2], c[3],
            str(c[4]) if c[4] is not None else "full", c[5]) not in done
    ]
    logger.info("Remaining: %d / %d configs", len(pending), len(combos))

    for ds, det_name, iv, win, rs, rep in tqdm(pending, desc="false_alarm",
                                                total=len(pending)):
        seed = 42 + rep
        rs_label = rs if rs is not None else "full"
        try:
            split = _load(ds, random_state=seed)
            features = MULTIVARIATE_FEATURES[ds]
            r_arr = _ref_arr(split, features, reference_size=rs)
            s_arr = split.X_stream[features].values

            detector = _make_detector(det_name)
            res = _monitor(detector, r_arr, s_arr, iv, win)
            n = res["n_checks"]
            far = res["alarm_count"] / n if n > 0 else float("nan")

            row = {
                "dataset": ds, "detector": det_name,
                "monitoring_interval": iv, "window_size": win,
                "reference_size": rs_label, "repetition": rep,
                "n_checks": n, "alarm_count": res["alarm_count"],
                "false_alarm_rate": round(far, 6),
            }
            _append_row(out, row, write_header=not out.exists())

        except Exception as e:
            logger.warning("false_alarm error (%s %s iv=%d win=%d rs=%s rep=%d): %s",
                           ds, det_name, iv, win, rs_label, rep, e)

    # Summary written only when all configs are done
    if out.exists():
        df = pd.read_csv(out)
        summary = (df.groupby("detector")["false_alarm_rate"]
                   .agg(["mean", "std", "min", "max"]).round(4))
        summary.to_csv(OUT_DIR / "false_alarm_rates_summary.csv")
        logger.info("Done. Summary:\n%s", summary)
    return out


# ---------------------------------------------------------------------------
# 2. MMD EQUAL-BUDGET SENSITIVITY
# ---------------------------------------------------------------------------

def run_mmd_budget_sensitivity(no_resume: bool = False):
    """
    MMD miss rate at max_samples in {100, 200, 300, 500} on a subset of
    the dependency-shift grid (240 configs, ~30 min).
    """
    logger.info("=== Running MMD equal-budget sensitivity (240 configs) ===")

    out = OUT_DIR / "mmd_budget_sensitivity.csv"
    key_cols = ["dataset", "max_samples", "drift_type", "window_size",
                "monitoring_interval", "repetition"]

    if no_resume and out.exists():
        out.unlink()

    done = _load_done(out, key_cols)
    logger.info("Already completed: %d configs", len(done))

    budgets = [100, 200, 300, 500]
    drift_types = ["sudden", "gradual"]
    windows = [100, 500]
    interval = 100
    n_reps = 5

    combos = [
        (ds, b, dt, win, rep)
        for ds in DATASETS for b in budgets
        for dt in drift_types for win in windows for rep in range(n_reps)
    ]
    pending = [c for c in combos
               if (c[0], c[1], c[2], c[3], interval, c[4]) not in done]
    logger.info("Remaining: %d / %d configs", len(pending), len(combos))

    for ds, budget, drift_type, win, rep in tqdm(pending, desc="mmd_budget",
                                                   total=len(pending)):
        seed = 42 + rep
        try:
            split = _load(ds, random_state=seed)
            features = MULTIVARIATE_FEATURES[ds]
            r_arr = _ref_arr(split, features)

            sim = CorrelationDriftSimulator(random_state=seed)
            drift_start = int(len(split.X_stream) * DRIFT_START_RATIO)
            X_drifted, _ = sim.apply_correlation_shift(
                split.X_stream.copy(), features, drift_start,
                correlation_strength=0.8, drift_type=drift_type,
            )
            s_arr = X_drifted[features].values

            detector = _make_detector("mmd", max_samples=budget)
            t0 = time.time()
            res = _monitor(detector, r_arr, s_arr, interval, win)
            elapsed = time.time() - t0

            detected = res["first_alarm_idx"] is not None
            row = {
                "dataset": ds, "max_samples": budget, "drift_type": drift_type,
                "window_size": win, "monitoring_interval": interval,
                "repetition": rep, "detected": int(detected),
                "miss": int(not detected),
                "detection_delay": (res["first_alarm_idx"] - drift_start) if detected else "",
                "elapsed_s": round(elapsed, 3),
            }
            _append_row(out, row, write_header=not out.exists())

        except Exception as e:
            logger.warning("mmd_budget error (%s b=%d %s win=%d rep=%d): %s",
                           ds, budget, drift_type, win, rep, e)

    if out.exists():
        df = pd.read_csv(out)
        summary = (df.groupby(["max_samples", "window_size"])
                   .agg(miss_rate=("miss", "mean"), n=("miss", "count")).round(3))
        logger.info("Done. Miss rate by budget:\n%s", summary)
    return out


# ---------------------------------------------------------------------------
# 3. C2ST SAMPLE-SIZE / CLASSIFIER SENSITIVITY
# ---------------------------------------------------------------------------

def run_c2st_sensitivity(no_resume: bool = False):
    """
    C2ST miss rate at 5 window sizes and 2 classifiers on the correlation-shift
    grid (300 configs, ~20 min).
    """
    logger.info("=== Running C2ST sensitivity (300 configs) ===")

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split as sktts
    from scipy.stats import binomtest

    out = OUT_DIR / "c2st_sample_sensitivity.csv"
    key_cols = ["dataset", "classifier", "window_size", "drift_type",
                "monitoring_interval", "repetition"]

    if no_resume and out.exists():
        out.unlink()

    done = _load_done(out, key_cols)
    logger.info("Already completed: %d configs", len(done))

    windows = [50, 100, 200, 500, 1000]
    drift_types = ["sudden", "gradual"]
    interval = 100
    n_reps = 5

    clf_factories = {
        "logistic":   lambda rs: LogisticRegression(solver="lbfgs", max_iter=200,
                                                     random_state=rs),
        "rf_shallow": lambda rs: RandomForestClassifier(n_estimators=50, max_depth=3,
                                                         random_state=rs),
    }

    combos = [
        (ds, clf, win, dt, rep)
        for ds in DATASETS for clf in clf_factories
        for win in windows for dt in drift_types for rep in range(n_reps)
    ]
    pending = [c for c in combos
               if (c[0], c[1], c[2], c[3], interval, c[4]) not in done]
    logger.info("Remaining: %d / %d configs", len(pending), len(combos))

    for ds, clf_name, win, drift_type, rep in tqdm(pending, desc="c2st",
                                                     total=len(pending)):
        seed = 42 + rep
        try:
            split = _load(ds, random_state=seed)
            features = MULTIVARIATE_FEATURES[ds]
            r_arr = _ref_arr(split, features)

            sim = CorrelationDriftSimulator(random_state=seed)
            drift_start = int(len(split.X_stream) * DRIFT_START_RATIO)
            X_drifted, _ = sim.apply_correlation_shift(
                split.X_stream.copy(), features, drift_start,
                correlation_strength=0.8, drift_type=drift_type,
            )
            s_arr = X_drifted[features].values

            clf_factory = clf_factories[clf_name]
            n_stream = len(s_arr)
            first_alarm = None
            alarm_count = 0
            n_checks = 0

            for idx in range(win, n_stream, interval):
                window = s_arr[idx - win: idx]
                rng = np.random.default_rng(seed + idx)
                min_n = min(len(r_arr), len(window))
                if min_n < 6:
                    continue
                ref_sub = (r_arr[rng.choice(len(r_arr), size=min_n, replace=False)]
                           if len(r_arr) > min_n else r_arr.copy())
                tst_sub = (window[rng.choice(len(window), size=min_n, replace=False)]
                           if len(window) > min_n else window.copy())
                X_c = np.vstack([ref_sub, tst_sub])
                y_c = np.hstack([np.zeros(len(ref_sub)), np.ones(len(tst_sub))])
                try:
                    X_tr, X_ev, y_tr, y_ev = sktts(
                        X_c, y_c, test_size=0.5, random_state=seed, stratify=y_c)
                except ValueError:
                    continue
                clf = clf_factory(seed)
                clf.fit(X_tr, y_tr)
                n_correct = int((clf.predict(X_ev) == y_ev).sum())
                p_val = float(binomtest(n_correct, len(y_ev), p=0.5,
                                        alternative="greater").pvalue)
                n_checks += 1
                if p_val < 0.01:
                    alarm_count += 1
                    if first_alarm is None:
                        first_alarm = idx

            detected = first_alarm is not None
            row = {
                "dataset": ds, "classifier": clf_name, "window_size": win,
                "drift_type": drift_type, "monitoring_interval": interval,
                "repetition": rep, "n_checks": n_checks,
                "alarm_count": alarm_count, "detected": int(detected),
                "miss": int(not detected),
                "detection_delay": (first_alarm - drift_start) if detected else "",
            }
            _append_row(out, row, write_header=not out.exists())

        except Exception as e:
            logger.warning("c2st error (%s %s win=%d rep=%d): %s",
                           ds, clf_name, win, rep, e)

    if out.exists():
        df = pd.read_csv(out)
        summary = (df.groupby(["classifier", "window_size"])
                   .agg(miss_rate=("miss", "mean"), n=("miss", "count")).round(3))
        logger.info("Done. Miss rate:\n%s", summary)
    return out


# ---------------------------------------------------------------------------
# 4. DEPENDENCY STRENGTH
# ---------------------------------------------------------------------------

def run_dependency_strength(no_resume: bool = False):
    """Pre/post Pearson r and Spearman rho per dataset (45 configs, ~60 s)."""
    logger.info("=== Computing dependency strength before/after ===")
    from scipy.stats import spearmanr, pearsonr

    out = OUT_DIR / "dependency_strength.csv"
    key_cols = ["dataset", "drift_type", "repetition"]

    if no_resume and out.exists():
        out.unlink()

    done = _load_done(out, key_cols)
    logger.info("Already completed: %d configs", len(done))

    drift_types = ["sudden", "gradual", "recurring"]
    n_reps = 5

    combos = [(ds, dt, rep)
              for ds in DATASETS for dt in drift_types for rep in range(n_reps)]
    pending = [c for c in combos if c not in done]
    logger.info("Remaining: %d / %d configs", len(pending), len(combos))

    for ds, drift_type, rep in tqdm(pending, desc="dep_strength", total=len(pending)):
        seed = 42 + rep
        try:
            split = _load(ds, random_state=seed)
            features = MULTIVARIATE_FEATURES[ds]

            a_ref = split.X_reference[features[0]].values
            b_ref = split.X_reference[features[1]].values
            pre_r   = float(pearsonr(a_ref, b_ref)[0])
            pre_rho = float(spearmanr(a_ref, b_ref)[0])

            sim = CorrelationDriftSimulator(random_state=seed)
            drift_start = int(len(split.X_stream) * DRIFT_START_RATIO)
            X_drifted, spec = sim.apply_correlation_shift(
                split.X_stream.copy(), features, drift_start,
                correlation_strength=0.8, drift_type=drift_type,
            )

            if drift_type == "gradual":
                post_start = drift_start + int(
                    (len(split.X_stream) - drift_start) * 0.75)
            else:
                post_start = drift_start

            pa = X_drifted[features[0]].values[post_start:]
            pb = X_drifted[features[1]].values[post_start:]
            if len(pa) < 5:
                continue

            post_r   = float(pearsonr(pa, pb)[0])
            post_rho = float(spearmanr(pa, pb)[0])

            row = {
                "dataset": ds, "feature_pair": "+".join(features),
                "drift_type": drift_type, "repetition": rep,
                "pre_pearson_r":    round(pre_r, 4),
                "pre_spearman_rho": round(pre_rho, 4),
                "post_pearson_r":   round(post_r, 4),
                "post_spearman_rho": round(post_rho, 4),
                "delta_pearson_r":    round(post_r - pre_r, 4),
                "delta_spearman_rho": round(post_rho - pre_rho, 4),
                "shift_direction": ("decorrelating"
                                    if spec.parameters.get("negative", False)
                                    else "positive_correlation"),
            }
            _append_row(out, row, write_header=not out.exists())

        except Exception as e:
            logger.warning("dep_strength error (%s %s rep=%d): %s",
                           ds, drift_type, rep, e)

    if out.exists():
        df = pd.read_csv(out)
        summary = (
            df.groupby(["dataset", "drift_type"])
            .agg(pre_pearson=("pre_pearson_r", "mean"),
                 pre_spearman=("pre_spearman_rho", "mean"),
                 post_pearson=("post_pearson_r", "mean"),
                 post_spearman=("post_spearman_rho", "mean"),
                 delta_pearson=("delta_pearson_r", "mean"),
                 delta_spearman=("delta_spearman_rho", "mean"))
            .round(3)
        )
        summary.to_csv(OUT_DIR / "dependency_strength_summary.csv")
        logger.info("Done.\n%s", summary)
    return out


# ---------------------------------------------------------------------------
# 5. SATURATION QUANTILES
# ---------------------------------------------------------------------------

def run_saturation_quantiles(no_resume: bool = False):
    """Pre/post quantiles for each drifted feature (6 rows, instant)."""
    logger.info("=== Computing saturation quantiles ===")

    out = OUT_DIR / "saturation_quantiles.csv"
    if no_resume and out.exists():
        out.unlink()

    if out.exists():
        logger.info("Already exists, skipping (use --no-resume to redo)")
        return out

    rows = []
    for ds in DATASETS:
        split = _load(ds)
        features = MULTIVARIATE_FEATURES[ds]
        for feat in features:
            ref_series = split.X_reference[feat]
            stream_vals = split.X_stream[feat].values.copy()
            _, std_val, ceiling = DriftSimulator._shift_and_ceiling(ref_series, magnitude=2.0)
            shifted_vals = DriftSimulator.saturate(stream_vals, 2.0, std_val, ceiling)

            headroom = ceiling - stream_vals
            tol = 0.01 * abs(ceiling) if abs(ceiling) > 0 else 0.01

            def pct(arr, q):
                return round(float(np.percentile(arr, q)), 4)

            rows.append({
                "dataset": ds, "feature": feat,
                "ceiling_p99_5_ref": round(ceiling, 4),
                "std_ref": round(std_val, 4),
                "nominal_shift_2sigma": round(2.0 * std_val, 4),
                "pre_min":    pct(stream_vals, 0),
                "pre_mean":   round(float(np.mean(stream_vals)), 4),
                "pre_median": pct(stream_vals, 50),
                "pre_p95":    pct(stream_vals, 95),
                "pre_max":    pct(stream_vals, 100),
                "post_min":    pct(shifted_vals, 0),
                "post_mean":   round(float(np.mean(shifted_vals)), 4),
                "post_median": pct(shifted_vals, 50),
                "post_p95":    pct(shifted_vals, 95),
                "post_max":    pct(shifted_vals, 100),
                "frac_unchanged_above_ceiling": round(float(np.mean(headroom <= 0)), 4),
                "frac_near_ceiling_post": round(
                    float(np.mean(np.abs(shifted_vals - ceiling) <= tol)), 4),
                "frac_beyond_obs_max": round(
                    float(np.mean(shifted_vals > float(split.X_stream[feat].max()))), 6),
                "stream_obs_max": round(float(split.X_stream[feat].max()), 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    logger.info("Saved %s (%d rows)\n%s", out, len(df), df.to_string(index=False))
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Supplementary editorial analyses")
    p.add_argument(
        "--only",
        choices=["false_alarm", "mmd_budget", "c2st_sensitivity",
                 "dep_strength", "saturation"],
        default=None,
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing partial results and start fresh",
    )
    return p.parse_args()


def main():
    args = parse_args()
    only = args.only
    nr = args.no_resume

    if only is None or only == "saturation":
        run_saturation_quantiles(no_resume=nr)
    if only is None or only == "dep_strength":
        run_dependency_strength(no_resume=nr)
    if only is None or only == "false_alarm":
        run_false_alarm_rates(no_resume=nr)
    if only is None or only == "mmd_budget":
        run_mmd_budget_sensitivity(no_resume=nr)
    if only is None or only == "c2st_sensitivity":
        run_c2st_sensitivity(no_resume=nr)

    logger.info("Complete. Results in: %s", OUT_DIR)


if __name__ == "__main__":
    main()
