"""Gradient-boosting estimators with early stopping on an internal holdout.

Placed after the preprocessing view, so it receives numeric arrays. A
stratified 10% holdout *of the training data it is given* drives early
stopping; the caller's validation fold is never touched.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

HOLDOUT = 0.1
MIN_HOLDOUT = 20
PATIENCE = 100


def _holdout_split(y, seed):
    rng = np.random.default_rng(seed)
    hold = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n = int(np.floor(HOLDOUT * len(idx)))
        if n:
            hold.extend(rng.choice(idx, n, replace=False))
    hold = np.array(sorted(hold), dtype=int)
    train = np.setdiff1d(np.arange(len(y)), hold)
    return train, hold


class EarlyStoppedBoosting(ClassifierMixin, BaseEstimator):
    def __init__(self, library="lightgbm", params=None, max_rounds=3000, random_state=0):
        self.library = library
        self.params = params
        self.max_rounds = max_rounds
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        tr, ho = _holdout_split(y, self.random_state)
        use_es = len(ho) >= MIN_HOLDOUT
        if not use_es:
            tr = np.arange(len(y))
        p = dict(self.params or {})
        lib, seed = self.library, self.random_state
        if lib == "lightgbm":
            import lightgbm as lgb
            m = lgb.LGBMClassifier(n_estimators=self.max_rounds if use_es else 300, random_state=seed,
                                   n_jobs=1, verbose=-1, **p)
            kw = {"eval_set": [(X[ho], y[ho])], "callbacks": [lgb.early_stopping(PATIENCE, verbose=False)]} if use_es else {}
            m.fit(X[tr], y[tr], **kw)
        elif lib == "xgboost":
            from xgboost import XGBClassifier
            m = XGBClassifier(n_estimators=self.max_rounds if use_es else 300, random_state=seed, n_jobs=1,
                              tree_method="hist", early_stopping_rounds=PATIENCE if use_es else None, **p)
            kw = {"eval_set": [(X[ho], y[ho])], "verbose": False} if use_es else {}
            m.fit(X[tr], y[tr], **kw)
        elif lib == "catboost":
            from catboost import CatBoostClassifier
            m = CatBoostClassifier(iterations=self.max_rounds if use_es else 500, random_seed=seed,
                                   thread_count=1, verbose=0, allow_writing_files=False, **p)
            kw = {"eval_set": (X[ho], y[ho]), "early_stopping_rounds": PATIENCE, "use_best_model": True} if use_es else {}
            m.fit(X[tr], y[tr], **kw)
        else:
            raise ValueError(lib)
        self.model_ = m
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = np.zeros((len(X), len(self.classes_)))
        out[:, np.searchsorted(self.classes_, self.model_.classes_)] = self.model_.predict_proba(X)
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]
