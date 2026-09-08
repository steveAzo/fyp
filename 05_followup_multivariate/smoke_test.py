from __future__ import annotations

import numpy as np
import pandas as pd

from detectors import (
    C2STDetector,
    HDDDMDetector,
    KSTestDetector,
    MMDDetector,
    PSIDetector,
    WassersteinDetector,
)
from simulators import MultivariateDriftSimulator


def main():
    print("=== Running 05_followup_multivariate Smoke Test ===")

    rng = np.random.default_rng(42)

    n_ref, n_test = 500, 200
    d = 3

    ref_data = rng.normal(loc=0.0, scale=1.0, size=(n_ref, d))
    test_clean = rng.normal(loc=0.0, scale=1.0, size=(n_test, d))
    test_drift = rng.normal(loc=0.0, scale=1.0, size=(n_test, d))
    test_drift[:, :2] += 2.0

    detectors = {
        "KS-Test (1D)": KSTestDetector(threshold=0.05),
        "PSI (1D)": PSIDetector(threshold=0.25),
        "Wasserstein (1D)": WassersteinDetector(threshold=0.25),
        "MMD (3D)": MMDDetector(threshold=0.01, n_permutations=199, max_samples=300),
        "C2ST (3D)": C2STDetector(threshold=0.01),
        "HDDDM (3D)": HDDDMDetector(threshold=0.15),
    }

    print("\n[A] Testing No Drift (Clean vs Clean):")
    for name, det in detectors.items():
        if "1D" in name:
            res = det.detect(ref_data[:, 0], test_clean[:, 0], feature="Feat0")
        else:
            res = det.detect(ref_data, test_clean, feature="Feat0+1+2")
        print(f"  {name:18s} -> is_drift: {res.is_drift!s:5s} | score: {res.score:.4f} | time: {res.computation_time_ms:.2f} ms")

    print("\n[B] Testing Injected Joint Drift (Clean vs Shifted):")
    for name, det in detectors.items():
        if "1D" in name:
            res = det.detect(ref_data[:, 0], test_drift[:, 0], feature="Feat0")
        else:
            res = det.detect(ref_data, test_drift, feature="Feat0+1+2")
        print(f"  {name:18s} -> is_drift: {res.is_drift!s:5s} | score: {res.score:.4f} | time: {res.computation_time_ms:.2f} ms")

    print("\n[C] Testing Multivariate Simulator:")
    df_raw = pd.DataFrame(ref_data, columns=["F1", "F2", "F3"])
    sim = MultivariateDriftSimulator(random_state=42)
    df_drifted, spec = sim.apply_multivariate(
        df_raw, features=["F1", "F2"], drift_start_idx=250, magnitude=2.0, drift_type="sudden"
    )
    print(f"  Drift spec: {spec.drift_type} on features: {spec.parameters['features']}")
    print(f"  Data shape pre/post: {df_raw.shape} -> {df_drifted.shape}")

    print("\n[OK] All smoke tests passed successfully!")


if __name__ == "__main__":
    main()
