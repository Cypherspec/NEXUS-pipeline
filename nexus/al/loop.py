"""Closed-loop active learning for virtual screening.

Each round: fit a surrogate on everything measured so far, score the remaining
library with an acquisition function, pick a batch, "measure" it, repeat. A
retrospective run over a library with known labels measures how much faster
actives are found than by random picking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from nexus.al.acquisition import acquire, select_batch
from nexus.al.gp import GaussianProcess


@dataclass
class ScreeningResult:
    selected: list[np.ndarray] = field(default_factory=list)
    hits_per_round: list[int] = field(default_factory=list)
    cumulative_hits: list[int] = field(default_factory=list)
    best_value: list[float] = field(default_factory=list)
    n_screened: list[int] = field(default_factory=list)

    @property
    def total_hits(self) -> int:
        return self.cumulative_hits[-1] if self.cumulative_hits else 0

    def enrichment_over_random(self, n_library: int, n_actives: int) -> float:
        """Hits found divided by the hits random selection would be expected
        to find after screening the same number of compounds."""
        screened = self.n_screened[-1] if self.n_screened else 0
        expected = n_actives * screened / n_library if n_library else 0.0
        return float(self.total_hits / expected) if expected > 0 else float("nan")


def active_learning_screen(
    features: np.ndarray,
    oracle: Callable[[np.ndarray], np.ndarray],
    n_rounds: int = 10,
    batch_size: int = 50,
    initial_size: int = 50,
    acquisition: str = "ei",
    surrogate: str = "gp",
    kernel: str = "tanimoto",
    hit_threshold: float | None = None,
    diversity_weight: float = 0.0,
    seed: int = 0,
) -> ScreeningResult:
    """Run a retrospective screening campaign.

    oracle(indices) returns measured values for those library members; nothing
    else about the labels is visible to the loop.
    """
    rng = np.random.default_rng(seed)
    n = len(features)
    measured = rng.choice(n, min(initial_size, n), replace=False)
    values = np.asarray(oracle(measured), dtype=float)

    result = ScreeningResult()
    is_hit = (lambda v: v >= hit_threshold) if hit_threshold is not None else (lambda v: v > 0.5)
    result.selected.append(measured)
    result.hits_per_round.append(int(np.sum(is_hit(values))))
    result.cumulative_hits.append(result.hits_per_round[-1])
    result.best_value.append(float(values.max()))
    result.n_screened.append(len(measured))

    for round_index in range(n_rounds):
        remaining = np.setdiff1d(np.arange(n), measured)
        if len(remaining) == 0:
            break
        if acquisition == "random":
            picks = rng.choice(remaining, min(batch_size, len(remaining)), replace=False)
        else:
            model = _fit_surrogate(surrogate, kernel, features[measured], values)
            mean, std = model.predict(features[remaining])
            scores = acquire(acquisition, mean, std, float(values.max()),
                             **({"seed": seed + round_index}
                                if acquisition == "thompson" else {}))
            local = select_batch(scores, min(batch_size, len(remaining)),
                                 features=features[remaining],
                                 diversity_weight=diversity_weight)
            picks = remaining[local]
        new_values = np.asarray(oracle(picks), dtype=float)
        measured = np.concatenate([measured, picks])
        values = np.concatenate([values, new_values])

        result.selected.append(picks)
        result.hits_per_round.append(int(np.sum(is_hit(new_values))))
        result.cumulative_hits.append(result.cumulative_hits[-1] + result.hits_per_round[-1])
        result.best_value.append(float(values.max()))
        result.n_screened.append(len(measured))
    return result


def _fit_surrogate(kind: str, kernel: str, X: np.ndarray, y: np.ndarray):
    if kind == "gp":
        return GaussianProcess(kernel=kernel, noise=1e-2).fit(X, y)
    if kind == "rf":
        return _RandomForestSurrogate().fit(X, y)
    raise ValueError(f"unknown surrogate {kind!r}")


class _RandomForestSurrogate:
    """Bagged-tree surrogate whose ensemble spread serves as the uncertainty.

    Used when the library is too large for an exact GP (cubic in the number of
    measured points).
    """

    def __init__(self, n_estimators: int = 100, seed: int = 0):
        self.n_estimators = n_estimators
        self.seed = seed

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import RandomForestRegressor
        self.model_ = RandomForestRegressor(
            n_estimators=self.n_estimators, n_jobs=1, random_state=self.seed)
        self.model_.fit(X, y)
        return self

    def predict(self, X: np.ndarray, return_std: bool = True):
        per_tree = np.stack([t.predict(X) for t in self.model_.estimators_])
        mean = per_tree.mean(axis=0)
        return (mean, per_tree.std(axis=0)) if return_std else mean
