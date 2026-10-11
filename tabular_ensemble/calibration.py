"""Post-hoc temperature scaling applied to ensemble probabilities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from tabular_ensemble.ensemble import EPS, log_loss_from_proba


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, EPS, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Temperature T in [0.05, 20] minimising log loss of p**(1/T)."""
    res = minimize_scalar(
        lambda log_t: log_loss_from_proba(y, apply_temperature(proba, np.exp(log_t))),
        bounds=(np.log(0.05), np.log(20.0)),
        method="bounded",
    )
    return float(np.exp(res.x))
