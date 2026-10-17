"""Non-equilibrium free energy estimation.

The Jarzynski equality relates the free energy difference between two states to
an average over irreversible work:

    exp(-beta * dF) = < exp(-beta * W) >_forward

Crooks' fluctuation theorem relates forward and reverse work distributions, and
the Bennett acceptance ratio (BAR) is the minimum-variance estimator that uses
both directions. These are the estimators behind alchemical binding free energy
calculations.

Every estimator here is checked against systems whose free energy difference is
known in closed form (tests/test_physics.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def logsumexp(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    out = peak + np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True))
    if axis is None:
        return float(out.reshape(-1)[0])
    return np.squeeze(out, axis=axis)


def jarzynski_free_energy(work: np.ndarray, beta: float = 1.0) -> float:
    """dF from forward work values via the Jarzynski equality.

    Evaluated in log space: dF = -(1/beta) * [logsumexp(-beta*W) - log(N)].
    """
    work = np.asarray(work, dtype=float)
    work = work[np.isfinite(work)]
    if len(work) == 0:
        return float("nan")
    return float(-(logsumexp(-beta * work) - np.log(len(work))) / beta)


def jarzynski_cumulant_estimate(work: np.ndarray, beta: float = 1.0) -> float:
    """Second-order cumulant expansion: dF ~ <W> - beta*var(W)/2.

    Lower variance than the exponential average when the work distribution is
    near-Gaussian, but biased when it is not.
    """
    work = np.asarray(work, dtype=float)
    return float(work.mean() - beta * work.var() / 2.0)


def bar_free_energy(work_forward: np.ndarray, work_reverse: np.ndarray,
                    beta: float = 1.0, tol: float = 1e-10,
                    max_iter: int = 500) -> float:
    """Bennett acceptance ratio estimate of dF.

    Solves the BAR self-consistency condition by bisection on the
    implicit equation for dF; uses both work directions and is the
    minimum-variance estimator among those that do.
    """
    wf = np.asarray(work_forward, dtype=float)
    wr = np.asarray(work_reverse, dtype=float)
    wf, wr = wf[np.isfinite(wf)], wr[np.isfinite(wr)]
    if len(wf) == 0 or len(wr) == 0:
        return float("nan")
    nf, nr = len(wf), len(wr)
    offset = np.log(nf / nr)

    def residual(dF: float) -> float:
        """sum_f 1/(1 + (nf/nr) e^{beta(Wf - dF)}) - sum_r 1/(1 + (nr/nf) e^{beta(Wr + dF)}).

        This form is monotonically increasing in dF, running from -nr to +nf,
        so it has exactly one root and bisection is guaranteed to find it.
        """
        a = -np.logaddexp(0.0, offset + beta * (wf - dF))
        b = -np.logaddexp(0.0, -offset + beta * (wr + dF))
        return float(np.exp(logsumexp(a)) - np.exp(logsumexp(b)))

    low = min(wf.min(), -wr.max()) - 100.0 / beta
    high = max(wf.max(), -wr.min()) + 100.0 / beta
    if residual(low) > 0 or residual(high) < 0:      # degenerate input
        return float("nan")
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        if (high - low) < tol:
            return float(mid)
        if residual(mid) < 0:
            low = mid
        else:
            high = mid
    return float(0.5 * (low + high))


def crooks_free_energy(work_forward: np.ndarray, work_reverse: np.ndarray,
                       beta: float = 1.0, bins: int = 30) -> float:
    """dF from the crossing point of the forward and reverse work histograms.

    Crooks' theorem makes P_F(W) = P_R(-W) exactly at W = dF.
    """
    wf = np.asarray(work_forward, dtype=float)
    wr = np.asarray(work_reverse, dtype=float)
    lo = min(wf.min(), (-wr).min())
    hi = max(wf.max(), (-wr).max())
    edges = np.linspace(lo, hi, bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    pf, _ = np.histogram(wf, bins=edges, density=True)
    pr, _ = np.histogram(-wr, bins=edges, density=True)
    valid = (pf > 0) & (pr > 0)
    if valid.sum() < 2:
        return float("nan")
    difference = np.log(pf[valid]) - np.log(pr[valid])
    x = centres[valid]
    # The crossing is where log(P_F/P_R) = 0; interpolate the sign change.
    sign_change = np.where(np.sign(difference[:-1]) != np.sign(difference[1:]))[0]
    if len(sign_change) == 0:
        slope, intercept = np.polyfit(x, difference, 1)
        return float(-intercept / slope) if slope != 0 else float("nan")
    k = sign_change[0]
    x0, x1 = x[k], x[k + 1]
    d0, d1 = difference[k], difference[k + 1]
    return float(x0 - d0 * (x1 - x0) / (d1 - d0))


def work_distribution_overlap(work_forward: np.ndarray, work_reverse: np.ndarray,
                              bins: int = 30) -> float:
    """Histogram overlap of forward and reverse work, in [0, 1].

    A practical diagnostic: estimates are only trustworthy when the two
    distributions overlap appreciably.
    """
    wf = np.asarray(work_forward, dtype=float)
    wr = -np.asarray(work_reverse, dtype=float)
    lo, hi = min(wf.min(), wr.min()), max(wf.max(), wr.max())
    edges = np.linspace(lo, hi, bins + 1)
    pf, _ = np.histogram(wf, bins=edges, density=True)
    pr, _ = np.histogram(wr, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.sum(np.minimum(pf, pr)) * width)


def jarzynski_bootstrap_ci(work: np.ndarray, beta: float = 1.0,
                           n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the Jarzynski estimate."""
    work = np.asarray(work, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(work), size=(n_boot, len(work)))
    estimates = np.array([jarzynski_free_energy(work[i], beta) for i in idx])
    return (float(np.percentile(estimates, 100 * alpha / 2)),
            float(np.percentile(estimates, 100 * (1 - alpha / 2))))


# ------------------------------------------------------- switching simulations

@dataclass
class SwitchingProtocol:
    """A finite-time switching simulation under overdamped Langevin dynamics.

    energy(x, lam) is the potential at coupling parameter lam in [0, 1];
    dU/dlam is estimated by finite differences and integrated along the
    trajectory to give the accumulated work.
    """
    energy: Callable[[np.ndarray, float], np.ndarray]
    beta: float = 1.0
    n_steps: int = 200
    dt: float = 0.01
    friction: float = 1.0
    dlam: float = 1e-5

    def _force(self, x: np.ndarray, lam: float, eps: float = 1e-5) -> np.ndarray:
        """-dU/dx by central differences (works for any callable potential)."""
        grad = np.zeros_like(x)
        for d in range(x.shape[1]):
            step = np.zeros_like(x)
            step[:, d] = eps
            grad[:, d] = (self.energy(x + step, lam) - self.energy(x - step, lam)) / (2 * eps)
        return -grad

    def run(self, x0: np.ndarray, forward: bool = True,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Switch lambda from 0 to 1 (or back) and accumulate work per walker."""
        rng = np.random.default_rng(seed)
        x = np.array(x0, dtype=float, copy=True)
        work = np.zeros(len(x))
        lambdas = np.linspace(0.0, 1.0, self.n_steps + 1)
        if not forward:
            lambdas = lambdas[::-1]
        noise_scale = np.sqrt(2.0 * self.dt / (self.beta * self.friction))

        for k in range(self.n_steps):
            lam, lam_next = lambdas[k], lambdas[k + 1]
            # Work done by changing lambda at fixed configuration.
            work += self.energy(x, lam_next) - self.energy(x, lam)
            # Relax at the new lambda.
            force = self._force(x, lam_next)
            x = x + self.dt * force / self.friction + noise_scale * rng.normal(size=x.shape)
        return work, x


def equilibrium_sample(energy: Callable[[np.ndarray, float], np.ndarray],
                       lam: float, n_walkers: int, n_dim: int = 1,
                       beta: float = 1.0, n_steps: int = 2000, dt: float = 0.01,
                       seed: int = 0) -> np.ndarray:
    """Draw equilibrium configurations at fixed lambda by Langevin burn-in."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_walkers, n_dim))
    noise_scale = np.sqrt(2.0 * dt / beta)
    eps = 1e-5
    for _ in range(n_steps):
        grad = np.zeros_like(x)
        for d in range(n_dim):
            step = np.zeros_like(x)
            step[:, d] = eps
            grad[:, d] = (energy(x + step, lam) - energy(x - step, lam)) / (2 * eps)
        x = x - dt * grad + noise_scale * rng.normal(size=x.shape)
    return x
