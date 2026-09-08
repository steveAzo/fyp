"""
Split sensitivity analysis: 50/50 and 70/30 reference/stream ratios.

Purpose: confirm that the main-grid finding (detector construction dominates
configuration under dependency-only shift; configuration dominates under
marginal-visible shift) is not an artifact of the 60/40 reference/stream
split used in the main experiments.

Design: replicate a representative slice of the dependency-shift grid at
all three split ratios for comparison:
  - Datasets      : credit, telco, maintenance (all three)
  - Detectors     : all six (ks, psi, wasserstein, mmd, c2st, hdddm)
  - Drift types   : sudden, gradual, recurring
  - Monitoring    : intervals 10, 100, 1000  (3 levels)
  - Window sizes  : 50, 100, 500             (3 levels -- skip 1000 for speed)
  - Ref sizes     : 500, None                (2 levels)
  - Repetitions   : 3 (enough to compute ε² ordering)
  Total per split ratio: 3 × 6 × 3 × 3 × 3 × 2 × 3 = 2916 configs
  Estimated runtime: ~20 min per ratio on the study hardware.

The analysis tests whether epsilon-squared rankings (window > interval >
detector) change when the split ratio changes. It does NOT rerun the full
16,200 + 3,240 main grid.

Run from the 05_followup_multivariate/ directory:
  python run_split_sensitivity.py               # run both ratios
  python run_split_sensitivity.py --ratio 0.5   # run 50/50 only
  python run_split_sensitivity.py --ratio 0.7   # run 70/30 only
  python run_split_sensitivity.py --no-resume   # force fresh start
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import data.loader as _loader_module
from data.loader import (
    DatasetSplit,
    _encode_categoricals,
    _scale_numericals,
    _resolve_path,
    load_config,
)
from detectors import (
    C2STDetector,
    HDDDMDetector,
    KSTestDetector,
    MMDDetector,
    PSIDetector,
    WassersteinDetector,
)
from simulators import CorrelationDriftSimulator
from experiments.grid import MULTIVARIATE_FEATURES
from utils.logger import get_logger

logger = get_logger("split_sensitivity")
OUT_DIR = Path("results/supplementary/split_sensitivity")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["credit", "telco", "maintenance"]
DETECTORS = ["ks", "psi", "wasserstein", "mmd", "c2st", "hdddm"]
DRIFT_TYPES = ["sudden", "gradual", "recurring"]
MONITORING_INTERVALS = [10, 100, 1000]
WINDOW_SIZES = [50, 100, 500]
REFERENCE_SIZES = [500, None]
N_REPS = 3
DRIFT_START_RATIO = 0.50


# ---------------------------------------------------------------------------
# Dataset loading with a custom split ratio (no file changes)
# ---------------------------------------------------------------------------

def _load_with_ratio(dataset: str, reference_ratio: float, random_state: int = 42) -> DatasetSplit:
    """Load a dataset with an arbitrary reference_ratio, overriding config.yaml."""
    cfg = load_config()
    cfg["split"]["reference_ratio"] = reference_ratio

    # We patch the config loader used inside the per-dataset loaders so they
    # pick up our modified config rather than re-reading the file.
    with patch.object(_loader_module, "load_config", return_value=cfg):
        from data.loader import _LOADERS
        return _LOADERS[dataset](cfg, random_state)


# ---------------------------------------------------------------------------
# Detector factory (same thresholds as main experiment)
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

def _load_done(path: Path, key_cols: list[str]) -> set[tuple]:
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
# Core monitoring function
# ---------------------------------------------------------------------------

def _monitor(detector, ref_arr: np.ndarray, stream_arr: np.ndarray,
             interval: int, window: int) -> dict:
    first_alarm = None
    alarm_count = 0
    n_checks = 0
    for idx in range(window, len(stream_arr), interval):
        w = stream_arr[idx - window: idx]
        res = detector.detect(ref_arr, w)
        n_checks += 1
        if res.is_drift:
            alarm_count += 1
            if first_alarm is None:
                first_alarm = idx
    return {
        "first_alarm_idx": first_alarm,
        "alarm_count": alarm_count,
        "n_checks": n_checks,
    }


# ---------------------------------------------------------------------------
# Main sensitivity run
# ---------------------------------------------------------------------------

def run_split_ratio(reference_ratio: float, no_resume: bool = False) -> Path:
    label = f"{int(reference_ratio * 100)}_{int((1 - reference_ratio) * 100)}"
    out = OUT_DIR / f"split_{label}.csv"
    key_cols = ["dataset", "detector", "drift_type", "monitoring_interval",
                "window_size", "reference_size", "repetition"]

    if no_resume and out.exists():
        out.unlink()
        logger.info("Removed %s (--no-resume)", out)

    done = _load_done(out, key_cols)
    logger.info("Split %s | already completed: %d configs", label, len(done))

    combos = [
        (ds, det, dt, iv, win, rs, rep)
        for ds in DATASETS
        for det in DETECTORS
        for dt in DRIFT_TYPES
        for iv in MONITORING_INTERVALS
        for win in WINDOW_SIZES
        for rs in REFERENCE_SIZES
        for rep in range(N_REPS)
    ]
    rs_label = lambda rs: rs if rs is not None else "full"
    pending = [
        c for c in combos
        if (c[0], c[1], c[2], c[3], c[4], rs_label(c[5]), c[6]) not in done
    ]
    logger.info("Split %s | remaining: %d / %d configs", label, len(pending), len(combos))

    for ds, det_name, drift_type, iv, win, rs, rep in tqdm(
            pending, desc=f"split_{label}", total=len(pending)):

        seed = 42 + rep
        rsl = rs_label(rs)
        try:
            split = _load_with_ratio(ds, reference_ratio, random_state=seed)
            features = MULTIVARIATE_FEATURES[ds]

            # Build reference array
            r_arr = split.X_reference[features].values
            if rs is not None and len(r_arr) > rs:
                r_arr = r_arr[-rs:]

            # Apply correlation drift to stream
            sim = CorrelationDriftSimulator(random_state=seed)
            drift_start = int(len(split.X_stream) * DRIFT_START_RATIO)
            X_drifted, spec = sim.apply_correlation_shift(
                split.X_stream.copy(), features, drift_start,
                correlation_strength=0.8, drift_type=drift_type,
            )
            s_arr = X_drifted[features].values

            detector = _make_detector(det_name)
            res = _monitor(detector, r_arr, s_arr, iv, win)

            detected = res["first_alarm_idx"] is not None
            row = {
                "reference_ratio": reference_ratio,
                "split_label": label,
                "dataset": ds,
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
                           ds, det_name, drift_type, iv, win, rsl, rep, e)

    logger.info("Split %s complete. Saved to %s", label, out)
    return out


# ---------------------------------------------------------------------------
# Summary: compare effect ordering across splits
# ---------------------------------------------------------------------------

def summarize(ratios: list[float]):
    logger.info("=== Effect-size comparison across split ratios ===")
    from scipy.stats import kruskal

    rows = []
    for ratio in ratios:
        label = f"{int(ratio * 100)}_{int((1 - ratio) * 100)}"
        p = OUT_DIR / f"split_{label}.csv"
        if not p.exists():
            logger.warning("Missing: %s", p)
            continue

        df = pd.read_csv(p)
        detected_df = df[df["detected"] == 1].copy()

        # Kruskal-Wallis epsilon-squared for each factor on detection delay
        factors = {
            "detector":             "detector",
            "window_size":          "window_size",
            "monitoring_interval":  "monitoring_interval",
        }

        for factor_name, col in factors.items():
            groups = [g["detection_delay"].dropna().values
                      for _, g in detected_df.groupby(col)
                      if len(g) > 0]
            if len(groups) < 2 or sum(len(g) for g in groups) < 3:
                continue
            try:
                H, p_val = kruskal(*groups)
                n = sum(len(g) for g in groups)
                eps2 = H / (n - 1)
                rows.append({
                    "split": f"{int(ratio*100)}/{int((1-ratio)*100)}",
                    "factor": factor_name,
                    "H": round(H, 2),
                    "p": f"{p_val:.2e}",
                    "epsilon_sq": round(eps2, 4),
                    "n_detected": n,
                })
            except Exception as e:
                logger.warning("Kruskal failed for %s %s: %s", label, factor_name, e)

        # Miss rate by detector class
        mi_dets = ["ks", "psi", "wasserstein", "hdddm"]
        jd_dets = ["mmd", "c2st"]
        mi_miss = df[df.detector.isin(mi_dets)]["miss"].mean()
        jd_miss = df[df.detector.isin(jd_dets)]["miss"].mean()
        rows.append({
            "split": f"{int(ratio*100)}/{int((1-ratio)*100)}",
            "factor": "miss_rate_marginal_info",
            "H": "", "p": "", "epsilon_sq": round(mi_miss, 4),
            "n_detected": "",
        })
        rows.append({
            "split": f"{int(ratio*100)}/{int((1-ratio)*100)}",
            "factor": "miss_rate_joint_dep",
            "H": "", "p": "", "epsilon_sq": round(jd_miss, 4),
            "n_detected": "",
        })

    if rows:
        summary_df = pd.DataFrame(rows)
        out = OUT_DIR / "split_sensitivity_summary.csv"
        summary_df.to_csv(out, index=False)
        print("\n=== SPLIT SENSITIVITY SUMMARY ===")
        print(summary_df.to_string(index=False))
        logger.info("Summary saved to %s", out)
    else:
        logger.warning("No data to summarize -- run the analyses first.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=None,
                   help="Run only this split ratio (0.5 or 0.7). Default: run both.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--summarize-only", action="store_true",
                   help="Skip running experiments, just recompute the summary.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ratio is not None:
        ratios = [args.ratio]
    else:
        ratios = [0.5, 0.7]

    if not args.summarize_only:
        for ratio in ratios:
            run_split_ratio(ratio, no_resume=args.no_resume)

    # Always summarize alongside the 60/40 baseline
    summarize([0.5, 0.6, 0.7])


if __name__ == "__main__":
    main()
