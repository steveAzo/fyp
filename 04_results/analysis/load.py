"""
Data loading and cleaning for analysis scripts.

All analysis scripts import from here so path resolution and label
cleaning happen in exactly one place.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Path to the raw experiment output — resolved relative to this file so
# the scripts run correctly regardless of working directory.
RESULTS_PATH = (
    Path(__file__).resolve().parents[2]
    / "02_implementation"
    / "results"
    / "results.csv"
)

PLOTS_DIR = Path(__file__).resolve().parent.parent / "plots"
TABLES_DIR = Path(__file__).resolve().parent.parent / "experiments"


def load_results(path: str | Path | None = None) -> pd.DataFrame:
    """
    Load results.csv and apply human-readable labels used across all plots.

    Returns a DataFrame with two extra columns:
      ref_label   : "Unlimited" | "100" | "500"
      detector_label : capitalised display name
    """
    src = Path(path) if path else RESULTS_PATH

    if not src.exists():
        raise FileNotFoundError(
            f"Results file not found at {src}.\n"
            "Run `python run_experiments.py --mode full` first."
        )

    df = pd.read_csv(src)

    # Human-readable reference size
    df["ref_label"] = df["reference_size"].apply(
        lambda x: "Unlimited" if int(x) == -1 else str(int(x))
    )
    df["ref_label"] = pd.Categorical(
        df["ref_label"], categories=["100", "500", "Unlimited"], ordered=True
    )

    # Capitalised detector names for plot legends
    _det_names = {"ks": "KS-Test", "psi": "PSI", "wasserstein": "Wasserstein"}
    df["detector_label"] = df["detector"].map(_det_names)

    # Capitalised drift types
    df["drift_label"] = df["drift_type"].str.capitalize()

    # Missed detection as percentage (for y-axes)
    df["missed_pct"] = df["missed_detection"].astype(float) * 100.0

    return df


def detected_only(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where drift was successfully detected."""
    return df[~df["missed_detection"]].copy()
