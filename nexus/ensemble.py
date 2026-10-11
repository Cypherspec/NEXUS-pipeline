"""Greedy forward ensemble selection with replacement (Caruana et al., 2004)."""

from __future__ import annotations

import numpy as np

EPS = 1e-15


def log_loss_from_proba(y: np.ndarray, proba: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y)), y], EPS, 1.0)
    return float(-np.mean(np.log(p)))


def mix(probas: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted average of a (models, n, k) probability stack."""
    return np.tensordot(weights, probas, axes=1)


def greedy_ensemble_selection(
    probas: np.ndarray, y: np.ndarray, n_iter: int = 50
) -> np.ndarray:
    """Return normalised model weights chosen greedily on held-out predictions.

    probas: (M, n, k) out-of-fold class probabilities for M models.
    y: (n,) integer labels in [0, k).
    Each step adds (with replacement) the model whose inclusion most lowers
    log loss of the running average; starting from the best single model.
    """
    M = probas.shape[0]
    counts = np.zeros(M)
    running = np.zeros_like(probas[0])
    for step in range(n_iter):
        candidates = (running * step + probas) / (step + 1)  # (M, n, k)
        losses = [log_loss_from_proba(y, c) for c in candidates]
        best = int(np.argmin(losses))
        counts[best] += 1
        running = candidates[best]
    return counts / counts.sum()
