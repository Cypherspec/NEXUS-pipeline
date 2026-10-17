"""Gaussian process regression with kernels suited to molecular data.

Exact inference by Cholesky factorisation, with marginal-likelihood
hyperparameter fitting. The Tanimoto kernel operates directly on binary
fingerprints and is the standard choice for molecular surrogates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def rbf_kernel(A: np.ndarray, B: np.ndarray, lengthscale: float = 1.0,
               variance: float = 1.0) -> np.ndarray:
    d2 = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :]
          - 2.0 * A @ B.T)
    return variance * np.exp(-0.5 * np.maximum(d2, 0.0) / lengthscale ** 2)


def matern52_kernel(A: np.ndarray, B: np.ndarray, lengthscale: float = 1.0,
                    variance: float = 1.0) -> np.ndarray:
    d2 = (np.sum(A ** 2, 1)[:, None] + np.sum(B ** 2, 1)[None, :]
          - 2.0 * A @ B.T)
    r = np.sqrt(np.maximum(d2, 0.0)) / lengthscale
    s = np.sqrt(5.0) * r
    return variance * (1.0 + s + s ** 2 / 3.0) * np.exp(-s)


def tanimoto_kernel(A: np.ndarray, B: np.ndarray, variance: float = 1.0) -> np.ndarray:
    """Tanimoto (MinMax) similarity as a positive-definite kernel."""
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    inner = A @ B.T
    na = np.sum(A ** 2, 1)[:, None]
    nb = np.sum(B ** 2, 1)[None, :]
    denom = na + nb - inner
    out = np.zeros_like(inner)
    np.divide(inner, denom, out=out, where=denom > 0)
    return variance * out


def linear_kernel(A: np.ndarray, B: np.ndarray, variance: float = 1.0) -> np.ndarray:
    return variance * (A @ B.T)


KERNELS = {"rbf": rbf_kernel, "matern52": matern52_kernel,
           "tanimoto": tanimoto_kernel, "linear": linear_kernel}


@dataclass
class GaussianProcess:
    """Exact GP regression with a Gaussian likelihood.

    kernel: "rbf", "matern52", "tanimoto" or "linear".
    """
    kernel: str = "rbf"
    lengthscale: float = 1.0
    variance: float = 1.0
    noise: float = 1e-2
    normalize_y: bool = True
    jitter: float = 1e-8

    def _k(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        fn = KERNELS[self.kernel]
        if self.kernel in ("tanimoto", "linear"):
            return fn(A, B, variance=self.variance)
        return fn(A, B, lengthscale=self.lengthscale, variance=self.variance)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcess":
        self.X_ = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.y_mean_ = float(y.mean()) if self.normalize_y else 0.0
        self.y_std_ = float(y.std()) if self.normalize_y else 1.0
        if self.y_std_ <= 0:
            self.y_std_ = 1.0
        self.y_ = (y - self.y_mean_) / self.y_std_
        K = self._k(self.X_, self.X_)
        K[np.diag_indices_from(K)] += self.noise + self.jitter
        self.L_ = np.linalg.cholesky(K)
        self.alpha_ = _cho_solve(self.L_, self.y_)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        X = np.asarray(X, dtype=np.float64)
        Ks = self._k(self.X_, X)
        mean = Ks.T @ self.alpha_ * self.y_std_ + self.y_mean_
        if not return_std:
            return mean
        v = np.linalg.solve(self.L_, Ks)
        diag = np.diag(self._k(X, X)) if len(X) < 4000 else np.array(
            [float(self._k(x[None, :], x[None, :])) for x in X])
        var = np.maximum(diag - np.sum(v ** 2, axis=0), 1e-12)
        return mean, np.sqrt(var) * self.y_std_

    def log_marginal_likelihood(self) -> float:
        n = len(self.y_)
        return float(-0.5 * self.y_ @ self.alpha_
                     - np.sum(np.log(np.diag(self.L_)))
                     - 0.5 * n * np.log(2 * np.pi))

    def optimize(self, X: np.ndarray, y: np.ndarray,
                 lengthscales=(0.25, 0.5, 1.0, 2.0, 4.0, 8.0),
                 noises=(1e-3, 1e-2, 1e-1), variances=(0.5, 1.0, 2.0)
                 ) -> "GaussianProcess":
        """Grid search on the exact log marginal likelihood."""
        best, best_score = None, -np.inf
        ls_grid = (1.0,) if self.kernel in ("tanimoto", "linear") else lengthscales
        for ls in ls_grid:
            for noise in noises:
                for var in variances:
                    self.lengthscale, self.noise, self.variance = ls, noise, var
                    try:
                        self.fit(X, y)
                        score = self.log_marginal_likelihood()
                    except np.linalg.LinAlgError:
                        continue
                    if score > best_score:
                        best, best_score = (ls, noise, var), score
        if best is not None:
            self.lengthscale, self.noise, self.variance = best
            self.fit(X, y)
        return self

    def sample(self, X: np.ndarray, n_samples: int = 1, seed: int = 0) -> np.ndarray:
        """Draw functions from the posterior at X."""
        X = np.asarray(X, dtype=np.float64)
        Ks = self._k(self.X_, X)
        mean = Ks.T @ self.alpha_
        v = np.linalg.solve(self.L_, Ks)
        cov = self._k(X, X) - v.T @ v
        cov[np.diag_indices_from(cov)] += 1e-8
        rng = np.random.default_rng(seed)
        draws = rng.multivariate_normal(mean, cov, size=n_samples, method="eigh")
        return draws * self.y_std_ + self.y_mean_


def _cho_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


@dataclass
class MultiFidelityGP:
    """Linear auto-regressive multi-fidelity GP (Kennedy & O'Hagan 2000).

    The high-fidelity function is modelled as rho * f_low(x) + delta(x): a cheap
    surrogate is corrected by a discrepancy GP fitted on the points where both
    fidelities are available. Cheap evaluations (docking) therefore inform
    expensive ones (free-energy calculation or assay).
    """
    kernel: str = "rbf"
    lengthscale: float = 1.0
    noise: float = 1e-2
    rho: float = 1.0
    fit_rho: bool = True

    def fit(self, X_low: np.ndarray, y_low: np.ndarray,
            X_high: np.ndarray, y_high: np.ndarray) -> "MultiFidelityGP":
        self.low_ = GaussianProcess(kernel=self.kernel, lengthscale=self.lengthscale,
                                    noise=self.noise).fit(X_low, y_low)
        low_at_high = self.low_.predict(X_high, return_std=False)
        if self.fit_rho and np.std(low_at_high) > 1e-12:
            design = np.c_[low_at_high, np.ones(len(low_at_high))]
            coef, *_ = np.linalg.lstsq(design, np.asarray(y_high).ravel(), rcond=None)
            self.rho = float(coef[0])
        residual = np.asarray(y_high).ravel() - self.rho * low_at_high
        self.delta_ = GaussianProcess(kernel=self.kernel, lengthscale=self.lengthscale,
                                      noise=self.noise).fit(X_high, residual)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        low = self.low_.predict(X, return_std=True)
        delta = self.delta_.predict(X, return_std=True)
        mean = self.rho * low[0] + delta[0]
        if not return_std:
            return mean
        std = np.sqrt((self.rho * low[1]) ** 2 + delta[1] ** 2)
        return mean, std
