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


def bagged_greedy_selection(
    probas: np.ndarray, y: np.ndarray, n_iter: int = 50, n_bags: int = 20,
    model_frac: float = 0.5, seed: int = 0,
) -> np.ndarray:
    """Caruana et al. (2006): average greedy weights over random model subsets."""
    rng = np.random.default_rng(seed)
    M = probas.shape[0]
    take = max(1, int(round(model_frac * M)))
    w = np.zeros(M)
    for _ in range(n_bags):
        idx = np.sort(rng.choice(M, take, replace=False))
        w[idx] += greedy_ensemble_selection(probas[idx], y, n_iter=n_iter)
    return w / w.sum()


def optimal_convex_weights(probas: np.ndarray, y: np.ndarray, l2: float = 0.0) -> np.ndarray:
    """Simplex weights minimising log loss of the mixture (softmax parametrised)."""
    from scipy.optimize import minimize

    M = probas.shape[0]
    p_true = np.clip(probas[:, np.arange(len(y)), y], EPS, 1.0)  # (M, n)

    def f(z):
        w = np.exp(z - z.max()); w /= w.sum()
        mixp = np.clip(w @ p_true, EPS, None)
        loss = -np.mean(np.log(mixp)) + l2 * np.sum(w ** 2)
        g_w = -(p_true / mixp).mean(axis=1) + 2 * l2 * w
        return loss, w * (g_w - w @ g_w)

    res = minimize(f, np.zeros(M), jac=True, method="L-BFGS-B")
    w = np.exp(res.x - res.x.max())
    return w / w.sum()
