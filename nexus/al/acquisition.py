"""Acquisition functions and batch selection for Bayesian optimisation."""

from __future__ import annotations

import numpy as np

SQRT2 = np.sqrt(2.0)
SQRT2PI = np.sqrt(2.0 * np.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    from math import erf
    return 0.5 * (1.0 + np.vectorize(erf)(z / SQRT2))


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z ** 2) / SQRT2PI


def expected_improvement(mean: np.ndarray, std: np.ndarray, best: float,
                         xi: float = 0.01) -> np.ndarray:
    """E[max(f - best - xi, 0)] for maximisation."""
    std = np.maximum(std, 1e-12)
    z = (mean - best - xi) / std
    return (mean - best - xi) * _normal_cdf(z) + std * _normal_pdf(z)


def probability_of_improvement(mean: np.ndarray, std: np.ndarray, best: float,
                               xi: float = 0.01) -> np.ndarray:
    std = np.maximum(std, 1e-12)
    return _normal_cdf((mean - best - xi) / std)


def upper_confidence_bound(mean: np.ndarray, std: np.ndarray,
                           beta: float = 2.0) -> np.ndarray:
    return mean + beta * std


def thompson_sample(mean: np.ndarray, std: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mean + std * rng.normal(size=len(mean))


def greedy(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return mean


def uncertainty(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Pure exploration: pick the least certain points."""
    return std


ACQUISITIONS = {
    "ei": lambda m, s, best, **kw: expected_improvement(m, s, best, **kw),
    "pi": lambda m, s, best, **kw: probability_of_improvement(m, s, best, **kw),
    "ucb": lambda m, s, best, **kw: upper_confidence_bound(m, s, **kw),
    "thompson": lambda m, s, best, **kw: thompson_sample(m, s, **kw),
    "greedy": lambda m, s, best, **kw: greedy(m, s),
    "uncertainty": lambda m, s, best, **kw: uncertainty(m, s),
    "random": lambda m, s, best, **kw: np.random.default_rng(
        kw.get("seed", 0)).random(len(m)),
}


def acquire(name: str, mean: np.ndarray, std: np.ndarray, best: float,
            **kwargs) -> np.ndarray:
    return ACQUISITIONS[name](mean, std, best, **kwargs)


def select_batch(scores: np.ndarray, batch_size: int,
                 features: np.ndarray | None = None,
                 diversity_weight: float = 0.0,
                 candidates: np.ndarray | None = None) -> np.ndarray:
    """Pick a batch, optionally penalising similarity to already-picked points.

    With diversity_weight = 0 this is simple top-k. Otherwise each subsequent
    pick maximises score minus a penalty for its greatest similarity to the
    batch so far, which avoids spending a whole batch on one chemical series.
    """
    pool = np.arange(len(scores)) if candidates is None else np.asarray(candidates)
    if batch_size >= len(pool):
        return pool
    if diversity_weight <= 0 or features is None:
        return pool[np.argsort(-scores[pool])[:batch_size]]

    from nexus.chem.fingerprint import bulk_tanimoto

    chosen: list[int] = []
    remaining = list(pool)
    scale = np.abs(scores[pool]).max() or 1.0
    while len(chosen) < batch_size and remaining:
        if not chosen:
            pick = max(remaining, key=lambda i: scores[i])
        else:
            sims = np.max([bulk_tanimoto(features[c], features[remaining])
                           for c in chosen], axis=0)
            adjusted = scores[remaining] - diversity_weight * scale * sims
            pick = remaining[int(np.argmax(adjusted))]
        chosen.append(pick)
        remaining.remove(pick)
    return np.array(chosen)
