"""NexusClassifier: bagged model zoo + greedy ensemble selection + temperature."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

from nexus.calibration import apply_temperature, fit_temperature
from nexus.ensemble import greedy_ensemble_selection, mix
from nexus.preprocessing import dense_view, split_columns, to_frame, tree_view
from nexus.zoo import ModelSpec, resolve_zoo


class _LocalLabelModel:
    """Fits on whatever subset of classes a fold contains; predicts all k."""

    def __init__(self, pipeline, n_classes: int):
        self.pipeline = pipeline
        self.n_classes = n_classes

    def fit(self, X, y):
        self.present_ = np.unique(y)
        if len(self.present_) == 1:
            return self
        local = np.searchsorted(self.present_, y)
        self.pipeline.fit(X, local)
        return self

    def predict_proba(self, X):
        out = np.zeros((len(X), self.n_classes))
        if len(self.present_) == 1:
            out[:, self.present_[0]] = 1.0
            return out
        out[:, self.present_] = self.pipeline.predict_proba(X)
        return out


def _fit_one(spec: ModelSpec, view, X_tr, y_tr, X_va, n_classes, seed):
    t0 = time.perf_counter()
    model = _LocalLabelModel(make_pipeline(clone(view), spec.make(seed)), n_classes)
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    pred = model.predict_proba(X_va) if X_va is not None else None
    return model, pred, fit_time


def _make_folds(y, n_folds, seed):
    """Stratified folds, shrinking K when the rarest class is too small."""
    min_count = int(np.bincount(y).min())
    if min_count >= 2:
        k = max(2, min(n_folds, min_count))
        return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(y, y))
    return list(KFold(n_folds, shuffle=True, random_state=seed).split(y))


@dataclass
class FitReport:
    oof_proba: dict[str, np.ndarray] = field(default_factory=dict)
    fit_seconds: dict[str, float] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0


class NexusClassifier(ClassifierMixin, BaseEstimator):
    """Cross-validated ensemble selection over a diverse model zoo.

    Each base model is trained with K-fold cross-fitting; its K fold models
    are kept and averaged at prediction time (bagging), and their out-of-fold
    predictions are used to (1) choose ensemble weights by greedy forward
    selection minimising log loss and (2) fit a single temperature that
    calibrates the final mixture. No data outside the training set is used.

    Parameters
    ----------
    models : "default" or sequence of model names / ModelSpec
    n_folds : int, number of cross-fitting folds
    n_repeats : int, independent repetitions of K-fold cross-fitting
    stack_models : None or sequence of model names; layer-2 models trained on
        the raw features plus all layer-1 out-of-fold probabilities
    ensemble_size : int, greedy selection iterations
    calibrate : bool, apply temperature scaling to the mixture
    random_state : int
    n_jobs : int, parallel (model, fold) fits via joblib
    """

    def __init__(self, models="default", n_folds=5, n_repeats=1, stack_models=None,
                 ensemble_size=50, calibrate=True, random_state=0, n_jobs=1):
        self.models = models
        self.n_folds = n_folds
        self.n_repeats = n_repeats
        self.stack_models = stack_models
        self.ensemble_size = ensemble_size
        self.calibrate = calibrate
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _crossfit(self, zoo, X, y, k, prefix, report):
        """Bagged cross-fitting of every spec in `zoo` on frame X.

        Repetition r re-shuffles the folds; averaging a point's R out-of-fold
        predictions mimics the bagged predictor used at test time.
        """
        numeric, categorical = split_columns(X)
        views = {"tree": tree_view(numeric, categorical),
                 "dense": dense_view(numeric, categorical)}
        jobs = []
        for r in range(self.n_repeats):
            folds = _make_folds(y, self.n_folds, self.random_state + 1000 * r)
            jobs += [(spec, f + 1000 * r, tr, va) for spec in zoo
                     for f, (tr, va) in enumerate(folds)]
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_one)(spec, views[spec.view], X.iloc[tr], y[tr],
                              X.iloc[va], k, self.random_state + f)
            for spec, f, tr, va in jobs
        )
        models = {prefix + spec.name: [] for spec in zoo}
        for spec in zoo:
            report.oof_proba[prefix + spec.name] = np.zeros((len(X), k))
            report.fit_seconds[prefix + spec.name] = 0.0
        for (spec, _, _, va), (model, pred, secs) in zip(jobs, results):
            name = prefix + spec.name
            models[name].append(model)
            report.oof_proba[name][va] += pred / self.n_repeats
            report.fit_seconds[name] += secs
        return models

    @staticmethod
    def _stack_features(X, l1_probas: list[np.ndarray]) -> pd.DataFrame:
        P = np.concatenate(l1_probas, axis=1)
        cols = [f"__nexus_l1_{i}" for i in range(P.shape[1])]
        return pd.concat([X, pd.DataFrame(P, columns=cols, index=X.index)], axis=1)

    def fit(self, X, y):
        X = to_frame(X).reset_index(drop=True)
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(np.asarray(y))
        self.classes_ = self.label_encoder_.classes_
        k = len(self.classes_)
        self.zoo_ = resolve_zoo(self.models)
        self.stack_zoo_ = resolve_zoo(self.stack_models) if self.stack_models else ()

        report = FitReport()
        self.fold_models_ = self._crossfit(self.zoo_, X, y_enc, k, "", report)
        self.l1_names_ = [s.name for s in self.zoo_]
        if self.stack_zoo_:
            # Layer 2 sees the raw features plus every layer-1 OOF probability.
            X2 = self._stack_features(X, [report.oof_proba[n] for n in self.l1_names_])
            self.fold_models_.update(self._crossfit(self.stack_zoo_, X2, y_enc, k, "L2:", report))

        names = list(self.fold_models_)
        stack = np.stack([report.oof_proba[n] for n in names])
        w = greedy_ensemble_selection(stack, y_enc, n_iter=self.ensemble_size)
        report.weights = dict(zip(names, w))
        oof_mix = mix(stack, w)
        report.temperature = fit_temperature(oof_mix, y_enc) if self.calibrate else 1.0

        self.weights_ = report.weights
        self.temperature_ = report.temperature
        self.report_ = report
        self.y_encoded_ = y_enc
        return self

    def predict_proba_members(self, X) -> dict[str, np.ndarray]:
        """Bagged (fold-averaged) class probabilities for every member."""
        X = to_frame(X).reset_index(drop=True)
        bag = lambda models, Z: np.mean([m.predict_proba(Z) for m in models], axis=0)
        out = {n: bag(self.fold_models_[n], X) for n in self.l1_names_}
        if self.stack_zoo_:
            X2 = self._stack_features(X, [out[n] for n in self.l1_names_])
            for n, models in self.fold_models_.items():
                if n.startswith("L2:"):
                    out[n] = bag(models, X2)
        return out

    def predict_proba(self, X):
        members = self.predict_proba_members(X)
        names = list(members)
        w = np.array([self.weights_[n] for n in names])
        active = w > 0
        p = mix(np.stack([members[n] for n, a in zip(names, active) if a]), w[active])
        return apply_temperature(p, self.temperature_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
