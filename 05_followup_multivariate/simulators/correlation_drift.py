from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from .base import DriftSimulator, DriftSpec


class CorrelationDriftSimulator(DriftSimulator):
    """Marginal-preserving correlation-structure drift.

    MultivariateDriftSimulator shifts each feature's own distribution
    independently -- that is detectable by two independent per-feature tests
    (e.g. Bonferroni-corrected KS), so it cannot tell us whether a genuinely
    multivariate detector (MMD, C2ST, HDDDM) adds anything over a univariate
    baseline. This simulator instead changes ONLY the dependency between two
    features: the marginal distribution of each feature is exactly preserved
    (values are permuted among themselves, never altered or resampled), while
    their rank correlation is pushed toward the target `correlation_strength`.
    A test that only looks at marginals cannot detect this by construction;
    only a test sensitive to the joint distribution can.

    Mechanism: within an "active" index range, a `mix_fraction` of rows have
    their second feature's value swapped for the value that would put it in
    matching rank order with the first feature (partial rank-matching /
    Spearman-style copula perturbation via random-subset sorting). Because
    every substituted value already existed elsewhere in the same window
    (this is a permutation of the observed values, not interpolation), the
    marginal distribution of both features is unchanged up to sampling noise
    from which rows were selected.
    """

    def _rank_match(
        self,
        anchor: np.ndarray,
        companion: np.ndarray,
        mix_fraction: float,
        rng: np.random.Generator,
        negative: bool = False,
    ) -> np.ndarray:
        n = len(anchor)
        companion = companion.copy()
        k = int(round(np.clip(mix_fraction, 0.0, 1.0) * n))
        if k < 2:
            return companion

        subset_idx = rng.choice(n, size=k, replace=False)
        anchor_order = np.argsort(anchor[subset_idx])
        companion_sorted = np.sort(companion[subset_idx])
        if negative:
            companion_sorted = companion_sorted[::-1]

        matched = np.empty(k, dtype=companion.dtype)
        matched[anchor_order] = companion_sorted
        companion[subset_idx] = matched
        return companion

    def apply_correlation_shift(
        self,
        X: pd.DataFrame,
        features: list[str],
        drift_start_idx: int,
        correlation_strength: float = 0.8,
        drift_type: str = "sudden",
        transition_length: int = 500,
        negative: Optional[bool] = None,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        if len(features) != 2:
            raise ValueError("CorrelationDriftSimulator requires exactly 2 features")

        X = X.copy()
        n = len(X)
        rng = np.random.default_rng(self.random_state)
        f1, f2 = features
        col1, col2 = X.columns.get_loc(f1), X.columns.get_loc(f2)

        anchor_full = X.iloc[:, col1].to_numpy(copy=True)
        companion_full = X.iloc[:, col2].to_numpy(copy=True)

        if negative is None:
            # Auto-calibrate direction: push correlation away from whatever it
            # already is pre-drift, rather than always "more positive". Some
            # feature pairs start meaningfully correlated (e.g. telco's
            # Monthly/Total Charges, r~0.66, since Total Charges is roughly
            # cumulative billing) -- pushing further positive from there is a
            # much weaker distributional change than pushing toward
            # decorrelation. Pairs that start near-independent (credit,
            # maintenance) push positive as before.
            baseline_slice = slice(0, drift_start_idx)
            baseline_corr = np.corrcoef(anchor_full[baseline_slice], companion_full[baseline_slice])[0, 1]
            negative = bool(baseline_corr > 0.3)

        if drift_type == "sudden":
            seg = slice(drift_start_idx, n)
            companion_full[seg] = self._rank_match(
                anchor_full[seg], companion_full[seg], correlation_strength, rng, negative,
            )

        elif drift_type == "gradual":
            end_idx = min(drift_start_idx + transition_length, n)
            n_ramp = end_idx - drift_start_idx
            block = max(1, n_ramp // 20)
            for start in range(drift_start_idx, end_idx, block):
                stop = min(start + block, end_idx)
                progress = (start - drift_start_idx) / max(1, n_ramp)
                local_strength = correlation_strength * progress
                seg = slice(start, stop)
                companion_full[seg] = self._rank_match(
                    anchor_full[seg], companion_full[seg], local_strength, rng, negative,
                )
            if end_idx < n:
                seg = slice(end_idx, n)
                companion_full[seg] = self._rank_match(
                    anchor_full[seg], companion_full[seg], correlation_strength, rng, negative,
                )

        elif drift_type == "recurring":
            cycle_length = 200
            positions = np.arange(drift_start_idx, n)
            cycle_pos = (positions - drift_start_idx) % (2 * cycle_length)
            active_positions = positions[cycle_pos < cycle_length]
            if len(active_positions) > 0:
                seg = active_positions
                companion_full[seg] = self._rank_match(
                    anchor_full[seg], companion_full[seg], correlation_strength, rng, negative,
                )
        else:
            raise ValueError(f"Unknown drift_type {drift_type!r}")

        X.iloc[:, col2] = companion_full

        spec = DriftSpec(
            drift_type=f"correlation_{drift_type}",
            feature="+".join(features),
            drift_start_idx=drift_start_idx,
            parameters={
                "features": features,
                "correlation_strength": correlation_strength,
                "drift_type": drift_type,
                "negative": negative,
            },
        )
        return X, spec

    def apply_multivariate(
        self,
        X: pd.DataFrame,
        features: list[str],
        drift_start_idx: int,
        magnitude: float = 0.8,
        drift_type: str = "sudden",
        transition_length: int = 500,
        **kwargs,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        # `magnitude` here is interpreted as correlation_strength (a mixing
        # fraction in [0, 1]), NOT a std-multiple, to keep the call signature
        # compatible with how the runner dispatches to simulators.
        return self.apply_correlation_shift(
            X, features, drift_start_idx,
            correlation_strength=magnitude,
            drift_type=drift_type,
            transition_length=transition_length,
        )

    def apply(
        self,
        X: pd.DataFrame,
        feature: str,
        drift_start_idx: int,
        **kwargs,
    ) -> tuple[pd.DataFrame, DriftSpec]:
        features = [f.strip() for f in feature.split("+")]
        return self.apply_multivariate(X, features, drift_start_idx, **kwargs)
