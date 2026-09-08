from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from .base import DetectionResult, DriftDetector

class C2STDetector(DriftDetector):
    def __init__(
        self,
        threshold: float = 0.01,
        test_size: float = 0.5,
        random_state: int = 42,
    ) -> None:
        super().__init__(threshold)
        self.test_size = test_size
        self.random_state = random_state

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

        min_n = min(len(ref), len(tst))
        if min_n < 5:
            return DetectionResult(
                is_drift=False,
                score=0.5,
                detector_name="C2ST",
                feature=feature,
                computation_time_ms=self._now_ms() - t0,
            )

        rng = np.random.default_rng(self.random_state)
        ref_sub = ref[rng.choice(len(ref), size=min_n, replace=False)] if len(ref) > min_n else ref
        tst_sub = tst[rng.choice(len(tst), size=min_n, replace=False)] if len(tst) > min_n else tst

        X = np.vstack([ref_sub, tst_sub])
        y = np.hstack([np.zeros(len(ref_sub)), np.ones(len(tst_sub))])

        X_train, X_eval, y_train, y_eval = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )

        clf = LogisticRegression(solver="lbfgs", max_iter=200, random_state=self.random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_eval)

        n_correct = int((y_pred == y_eval).sum())
        n_eval = len(y_eval)
        acc = float(n_correct / n_eval)

        b_res = binomtest(n_correct, n_eval, p=0.5, alternative="greater")
        p_value = float(b_res.pvalue)

        elapsed = self._now_ms() - t0

        return DetectionResult(
            is_drift=bool(p_value < self.threshold),
            score=acc,
            detector_name="C2ST",
            feature=feature,
            computation_time_ms=elapsed,
        )
