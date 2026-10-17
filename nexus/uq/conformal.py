"""Conformal prediction: finite-sample valid prediction sets and intervals.

Split (inductive) conformal prediction gives a distribution-free guarantee: if
calibration and test points are exchangeable, the prediction region covers the
true label with probability at least 1 - alpha, whatever the underlying model.

The guarantee is the point of this module, so it is tested directly:
benchmarks/validate_conformal.py measures empirical coverage against the
nominal level across datasets, models and alpha values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The (1-alpha) conformal quantile with the finite-sample correction.

    Returns the ceil((n+1)(1-alpha))-th smallest score, the order statistic
    that makes the coverage guarantee hold at finite n rather than only
    asymptotically. When that index exceeds n the region is unbounded, and the
    maximum observed score is returned.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    n = len(scores)
    if n == 0:
        return np.inf
    # The k-th smallest score, k = ceil((n+1)(1-alpha)). Taking the order
    # statistic directly matters: np.quantile at the fraction k/n interpolates
    # over n-1 intervals and returns a different value.
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float(np.max(scores))
    return float(np.partition(scores, k - 1)[k - 1])


# ------------------------------------------------------------- classification

@dataclass
class ConformalClassifier:
    """Split-conformal prediction sets for multiclass classification.

    score: "softmax" uses 1 - p(true class) (least ambiguous sets);
           "aps" uses adaptive prediction sets, which are better calibrated
           conditionally on the input at the cost of larger sets.
    """
    alpha: float = 0.1
    score: str = "softmax"
    mondrian: bool = False          # class-conditional calibration
    random_state: int = 0

    def fit(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> "ConformalClassifier":
        probs_cal = np.asarray(probs_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        scores = self._nonconformity(probs_cal, y_cal)
        self.n_classes_ = probs_cal.shape[1]
        if self.mondrian:
            self.qhat_ = {}
            for c in range(self.n_classes_):
                mask = y_cal == c
                self.qhat_[c] = (conformal_quantile(scores[mask], self.alpha)
                                 if mask.any() else np.inf)
        else:
            self.qhat_ = conformal_quantile(scores, self.alpha)
        return self

    def _nonconformity(self, probs: np.ndarray, y: np.ndarray) -> np.ndarray:
        rows = np.arange(len(y))
        if self.score == "softmax":
            return 1.0 - probs[rows, y]
        if self.score == "aps":
            order = np.argsort(-probs, axis=1)
            sorted_probs = np.take_along_axis(probs, order, axis=1)
            cumulative = np.cumsum(sorted_probs, axis=1)
            rank = np.argmax(order == y[:, None], axis=1)
            rng = np.random.default_rng(self.random_state)
            u = rng.random(len(y))
            return cumulative[rows, rank] - u * sorted_probs[rows, rank]
        raise ValueError(f"unknown score {self.score!r}")

    def predict_set(self, probs: np.ndarray) -> np.ndarray:
        """Boolean (n, n_classes) membership matrix of the prediction sets."""
        probs = np.asarray(probs, dtype=float)
        n, k = probs.shape
        include = np.zeros((n, k), dtype=bool)
        for c in range(k):
            labels = np.full(n, c)
            scores = self._nonconformity(probs, labels)
            q = self.qhat_[c] if self.mondrian else self.qhat_
            include[:, c] = scores <= q
        return include

    def coverage(self, probs: np.ndarray, y: np.ndarray) -> float:
        sets = self.predict_set(probs)
        return float(np.mean(sets[np.arange(len(y)), np.asarray(y, int)]))

    def average_set_size(self, probs: np.ndarray) -> float:
        return float(self.predict_set(probs).sum(axis=1).mean())


# ----------------------------------------------------------------- regression

@dataclass
class ConformalRegressor:
    """Split-conformal prediction intervals.

    normalize=True scales the interval by a per-point difficulty estimate
    (e.g. an ensemble standard deviation), giving intervals that adapt in width
    while keeping the marginal coverage guarantee.
    """
    alpha: float = 0.1
    normalize: bool = False
    eps: float = 1e-6

    def fit(self, pred_cal: np.ndarray, y_cal: np.ndarray,
            sigma_cal: np.ndarray | None = None) -> "ConformalRegressor":
        residual = np.abs(np.asarray(y_cal, float) - np.asarray(pred_cal, float))
        if self.normalize:
            if sigma_cal is None:
                raise ValueError("normalize=True requires sigma_cal")
            residual = residual / (np.asarray(sigma_cal, float) + self.eps)
        self.qhat_ = conformal_quantile(residual, self.alpha)
        return self

    def predict_interval(self, pred: np.ndarray,
                         sigma: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        pred = np.asarray(pred, dtype=float)
        width = self.qhat_
        if self.normalize:
            if sigma is None:
                raise ValueError("normalize=True requires sigma")
            width = self.qhat_ * (np.asarray(sigma, float) + self.eps)
        return pred - width, pred + width

    def coverage(self, pred: np.ndarray, y: np.ndarray,
                 sigma: np.ndarray | None = None) -> float:
        low, high = self.predict_interval(pred, sigma)
        y = np.asarray(y, float)
        return float(np.mean((y >= low) & (y <= high)))

    def average_width(self, pred: np.ndarray, sigma: np.ndarray | None = None) -> float:
        low, high = self.predict_interval(pred, sigma)
        return float(np.mean(high - low))


@dataclass
class ConformalQuantileRegressor:
    """Conformalized quantile regression (Romano et al. 2019).

    Takes lower/upper quantile predictions and calibrates them so the interval
    attains the nominal coverage, correcting a miscalibrated quantile model in
    either direction.
    """
    alpha: float = 0.1

    def fit(self, lower_cal: np.ndarray, upper_cal: np.ndarray,
            y_cal: np.ndarray) -> "ConformalQuantileRegressor":
        y = np.asarray(y_cal, float)
        scores = np.maximum(np.asarray(lower_cal, float) - y,
                            y - np.asarray(upper_cal, float))
        self.qhat_ = conformal_quantile(scores, self.alpha)
        return self

    def predict_interval(self, lower: np.ndarray,
                         upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(lower, float) - self.qhat_, np.asarray(upper, float) + self.qhat_

    def coverage(self, lower: np.ndarray, upper: np.ndarray, y: np.ndarray) -> float:
        low, high = self.predict_interval(lower, upper)
        y = np.asarray(y, float)
        return float(np.mean((y >= low) & (y <= high)))


# ------------------------------------------------------------- cross-conformal

def jackknife_plus_interval(
    predictions_loo: np.ndarray, residuals_loo: np.ndarray,
    pred_test: np.ndarray, alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Jackknife+ / CV+ intervals (Barber et al. 2021).

    predictions_loo : (n_folds, n_test) test predictions from each held-out fit
    residuals_loo   : (n_folds,) or (n_cal,) absolute out-of-fold residuals
    Gives coverage at least 1 - 2*alpha without a separate calibration split.
    """
    preds = np.atleast_2d(np.asarray(predictions_loo, float))
    res = np.asarray(residuals_loo, float).reshape(-1, 1)
    low = np.quantile(preds - res, alpha, axis=0, method="lower")
    high = np.quantile(preds + res, 1 - alpha, axis=0, method="higher")
    return low, high


def coverage_width_criterion(coverage: float, width: float, target: float,
                             penalty: float = 1.0) -> float:
    """Width penalised by any shortfall in coverage (lower is better)."""
    return float(width * (1.0 + penalty * max(0.0, target - coverage) / max(target, 1e-9)))
