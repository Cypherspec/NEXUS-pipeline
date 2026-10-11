"""Evaluation metrics for probabilistic multiclass predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from nexus.ensemble import log_loss_from_proba


def expected_calibration_error(y, proba, n_bins: int = 15) -> float:
    """Top-label ECE with equal-width confidence bins."""
    conf = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def roc_auc(y, proba) -> float:
    """Binary AUC, or macro one-vs-rest AUC over classes present in y."""
    k = proba.shape[1]
    if k == 2:
        return float(roc_auc_score(y, proba[:, 1]))
    present = np.unique(y)
    scores = [roc_auc_score(y == c, proba[:, c]) for c in present]
    return float(np.mean(scores))


def brier(y, proba) -> float:
    onehot = np.eye(proba.shape[1])[y]
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def all_metrics(y, proba) -> dict[str, float]:
    proba = np.asarray(proba, dtype=np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    return {
        "log_loss": log_loss_from_proba(y, proba),
        "roc_auc": roc_auc(y, proba),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": brier(y, proba),
        "ece": expected_calibration_error(y, proba),
    }


LOWER_IS_BETTER = {"log_loss": True, "roc_auc": False, "accuracy": False,
                   "balanced_accuracy": False, "brier": True, "ece": True}
