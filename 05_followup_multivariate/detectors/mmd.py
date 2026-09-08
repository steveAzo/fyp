from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from .base import DetectionResult, DriftDetector

class MMDDetector(DriftDetector):
    def __init__(
        self,
        threshold: float = 0.01,
        gamma: float | None = None,
        n_permutations: int = 199,
        max_samples: int = 300,
        random_state: int = 42,
    ) -> None:
        super().__init__(threshold)
        # A permutation test's smallest achievable p-value is 1/(n+1) -- if
        # that floor is >= threshold, the test can NEVER reject the null no
        # matter how strong the real signal is. This silently produced a
        # 100% miss rate for every MMD run before it was caught, so it's
        # asserted here rather than left as an assumption someone has to
        # remember when tuning n_permutations or threshold in the future.
        min_achievable_p = 1.0 / (n_permutations + 1)
        if min_achievable_p >= threshold:
            raise ValueError(
                f"n_permutations={n_permutations} gives a minimum achievable "
                f"p-value of {min_achievable_p:.4f}, which is >= threshold="
                f"{threshold}. This detector could never report drift. "
                f"Increase n_permutations or raise threshold."
            )
        self.gamma = gamma
        self.n_permutations = n_permutations
        # MMD's exact-kernel permutation test is O(n^2) per permutation, so an
        # uncapped window_size (up to 1000 in the constraint grid) combined
        # with frequent monitoring makes the full grid computationally
        # infeasible on this hardware (~11.5s/call at window=1000,
        # benchmarked). Capping the sample budget fed to the kernel
        # computation is itself a resource constraint in the spirit of this
        # study, not just an implementation shortcut -- stated explicitly
        # here and in the methodology writeup. window_size still controls
        # what MMD is offered; max_samples caps what it actually computes on.
        self.max_samples = max_samples
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def _cap(self, x: np.ndarray) -> np.ndarray:
        if len(x) > self.max_samples:
            idx = self.rng.choice(len(x), size=self.max_samples, replace=False)
            return x[idx]
        return x

    def _compute_rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        dists_sq = cdist(X, Y, metric="sqeuclidean")
        return np.exp(-gamma * dists_sq)

    def _compute_mmd2_unbiased(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
        m = len(X)
        n = len(Y)
        if m <= 1 or n <= 1:
            return 0.0

        K_XX = self._compute_rbf_kernel(X, X, gamma)
        K_YY = self._compute_rbf_kernel(Y, Y, gamma)
        K_XY = self._compute_rbf_kernel(X, Y, gamma)

        np.fill_diagonal(K_XX, 0.0)
        np.fill_diagonal(K_YY, 0.0)

        term_XX = K_XX.sum() / (m * (m - 1))
        term_YY = K_YY.sum() / (n * (n - 1))
        term_XY = K_XY.mean()

        mmd2 = term_XX + term_YY - 2.0 * term_XY
        return float(max(0.0, mmd2))

    def detect(
        self,
        reference: np.ndarray,
        test: np.ndarray,
        feature: str = "",
    ) -> DetectionResult:
        ref = self._clean(self._to_array(reference))
        tst = self._clean(self._to_array(test))

        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if tst.ndim == 1:
            tst = tst.reshape(-1, 1)

        t0 = self._now_ms()

        ref = self._cap(ref)
        tst = self._cap(tst)

        min_n = min(len(ref), len(tst))
        ref_sub = ref[self.rng.choice(len(ref), size=min_n, replace=False)] if len(ref) > min_n else ref
        tst_sub = tst[self.rng.choice(len(tst), size=min_n, replace=False)] if len(tst) > min_n else tst

        pooled_raw = np.vstack([ref_sub, tst_sub])
        pooled_mean = np.mean(pooled_raw, axis=0)
        pooled_std = np.std(pooled_raw, axis=0)
        pooled_std[pooled_std == 0.0] = 1.0

        ref_norm = (ref_sub - pooled_mean) / pooled_std
        tst_norm = (tst_sub - pooled_mean) / pooled_std

        if self.gamma is None:
            combined = np.vstack([ref_norm, tst_norm])
            if len(combined) > 500:
                sub_idx = self.rng.choice(len(combined), size=500, replace=False)
                sub_data = combined[sub_idx]
            else:
                sub_data = combined
            
            pdist_sq = cdist(sub_data, sub_data, metric="sqeuclidean")
            median_sq = np.median(pdist_sq[pdist_sq > 0]) if np.any(pdist_sq > 0) else 1.0
            gamma = 1.0 / max(1e-5, float(median_sq))
        else:
            gamma = self.gamma

        observed_mmd2 = self._compute_mmd2_unbiased(ref_norm, tst_norm, gamma)

        m = len(ref_norm)
        pooled = np.vstack([ref_norm, tst_norm])
        n_pooled = len(pooled)

        count = 0
        for _ in range(self.n_permutations):
            perm = self.rng.permutation(n_pooled)
            perm_ref = pooled[perm[:m]]
            perm_tst = pooled[perm[m:]]
            perm_mmd2 = self._compute_mmd2_unbiased(perm_ref, perm_tst, gamma)
            if perm_mmd2 >= observed_mmd2:
                count += 1

        p_value = float((count + 1) / (self.n_permutations + 1))
        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=bool(p_value < self.threshold),
            score=float(observed_mmd2),
            detector_name="MMD",
            feature=feature,
            computation_time_ms=elapsed,
        )
