"""Stein variational gradient descent.

SVGD transports a set of particles towards a target distribution by following
a kernelised gradient flow: each particle is pushed by the score function of
the target and repelled from its neighbours, so the particle set approximates
the full posterior rather than collapsing to a single mode.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def rbf_kernel_and_grad(particles: np.ndarray, bandwidth: float | None = None
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Kernel matrix and the gradient of the kernel w.r.t. the first argument.

    With `bandwidth=None` the median heuristic is used: h = median(d^2)/log(n),
    which keeps the repulsion well scaled as the particles move.
    """
    n = len(particles)
    diff = particles[:, None, :] - particles[None, :, :]
    sq = np.sum(diff ** 2, axis=-1)
    if bandwidth is None:
        median = np.median(sq)
        bandwidth = np.sqrt(0.5 * median / max(np.log(n + 1), 1e-9)) if median > 0 else 1.0
    h = max(bandwidth, 1e-6) ** 2
    K = np.exp(-sq / (2.0 * h))
    grad_K = -(K[:, :, None] * diff) / h          # d/dx_i k(x_i, x_j)
    return K, grad_K


def svgd(
    particles: np.ndarray,
    score_fn: Callable[[np.ndarray], np.ndarray],
    steps: int = 200,
    step_size: float = 0.1,
    bandwidth: float | None = None,
    adagrad: bool = True,
    callback: Callable[[int, np.ndarray], None] | None = None,
) -> np.ndarray:
    """Evolve particles towards the target distribution.

    score_fn(x) returns grad log p(x) evaluated at every particle, shape (n, d).
    """
    x = np.array(particles, dtype=np.float64, copy=True)
    historical = np.zeros_like(x)
    for t in range(steps):
        scores = np.asarray(score_fn(x), dtype=np.float64)
        K, grad_K = rbf_kernel_and_grad(x, bandwidth)
        # phi(x_i) = (1/n) sum_j [ k(x_j, x_i) score(x_j) + grad_{x_j} k(x_j, x_i) ]
        phi = (K @ scores + grad_K.sum(axis=0)) / len(x)
        if adagrad:
            historical = 0.9 * historical + 0.1 * phi ** 2 if t > 0 else phi ** 2
            x = x + step_size * phi / (1e-6 + np.sqrt(historical))
        else:
            x = x + step_size * phi
        if callback is not None:
            callback(t, x)
    return x


def gaussian_score(mean: np.ndarray, cov: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Score function of a multivariate normal, for testing and sampling."""
    precision = np.linalg.inv(np.atleast_2d(cov))

    def score(x: np.ndarray) -> np.ndarray:
        return -(x - mean) @ precision
    return score


def mixture_score(means: list[np.ndarray], covs: list[np.ndarray],
                  weights: list[float] | None = None
                  ) -> Callable[[np.ndarray], np.ndarray]:
    """Score of a Gaussian mixture; useful for checking multi-modal coverage."""
    k = len(means)
    weights = weights or [1.0 / k] * k
    precisions = [np.linalg.inv(np.atleast_2d(c)) for c in covs]
    dets = [np.linalg.det(np.atleast_2d(c)) for c in covs]

    def score(x: np.ndarray) -> np.ndarray:
        responsibilities = np.zeros((len(x), k))
        for j, (m, P, d, w) in enumerate(zip(means, precisions, dets, weights)):
            delta = x - m
            quad = np.einsum("ij,jk,ik->i", delta, P, delta)
            responsibilities[:, j] = w * np.exp(-0.5 * quad) / np.sqrt(max(d, 1e-300))
        total = responsibilities.sum(axis=1, keepdims=True)
        responsibilities /= np.maximum(total, 1e-300)
        out = np.zeros_like(x)
        for j, (m, P) in enumerate(zip(means, precisions)):
            out += responsibilities[:, j:j + 1] * (-(x - m) @ P)
        return out
    return score


def bayesian_linear_score(X: np.ndarray, y: np.ndarray, noise: float = 0.1,
                          prior_variance: float = 1.0
                          ) -> Callable[[np.ndarray], np.ndarray]:
    """Score of the posterior over linear-regression weights.

    Lets SVGD produce a weight ensemble whose predictive spread is a usable
    uncertainty estimate.
    """
    XtX = X.T @ X
    Xty = X.T @ y

    def score(w: np.ndarray) -> np.ndarray:
        likelihood = (Xty[None, :] - w @ XtX) / noise ** 2
        prior = -w / prior_variance
        return likelihood + prior
    return score
