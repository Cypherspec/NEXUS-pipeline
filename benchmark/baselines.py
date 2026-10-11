"""Single-model baselines evaluated alongside NEXUS."""

from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from nexus.classifier import _LocalLabelModel
from nexus.ensemble import log_loss_from_proba
from nexus.preprocessing import dense_view, split_columns, tree_view
from nexus.zoo import DEFAULT_ZOO

N_TUNING_CONFIGS = 20


def fit_single(spec, X_tr, y_tr, n_classes, seed):
    numeric, categorical = split_columns(X_tr)
    view = tree_view(numeric, categorical) if spec.view == "tree" else dense_view(numeric, categorical)
    return _LocalLabelModel(make_pipeline(view, spec.make(seed)), n_classes).fit(X_tr, y_tr)


def lgbm_search_space(rng: np.random.Generator) -> dict:
    return {
        "n_estimators": int(rng.choice([100, 200, 400, 800])),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.3)))),
        "num_leaves": int(rng.choice([7, 15, 31, 63, 127])),
        "min_child_samples": int(rng.choice([5, 10, 20, 50])),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "subsample_freq": 1,
        "colsample_bytree": float(rng.uniform(0.4, 1.0)),
        "reg_lambda": float(np.exp(rng.uniform(np.log(1e-3), np.log(10.0)))),
    }


def fit_tuned_lightgbm(X_tr, y_tr, n_classes, seed):
    """Random search (20 configs) scored by log loss on a 20% stratified holdout,
    then refit of the best config on all training data."""
    numeric, categorical = split_columns(X_tr)
    view = tree_view(numeric, categorical)
    rng = np.random.default_rng(seed)
    configs = [lgbm_search_space(rng) for _ in range(N_TUNING_CONFIGS)]
    strat = y_tr if np.bincount(y_tr).min() >= 2 else None
    a, b = train_test_split(np.arange(len(y_tr)), test_size=0.2, random_state=seed, stratify=strat)
    best, best_loss = None, np.inf
    for cfg in configs:
        est = LGBMClassifier(**cfg, random_state=seed, n_jobs=1, verbose=-1)
        m = _LocalLabelModel(make_pipeline(clone(view), est), n_classes).fit(X_tr.iloc[a], y_tr[a])
        loss = log_loss_from_proba(y_tr[b], m.predict_proba(X_tr.iloc[b]))
        if loss < best_loss:
            best, best_loss = cfg, loss
    est = LGBMClassifier(**best, random_state=seed, n_jobs=1, verbose=-1)
    return _LocalLabelModel(make_pipeline(clone(view), est), n_classes).fit(X_tr, y_tr), best


SINGLE_MODELS = [s.name for s in DEFAULT_ZOO]
