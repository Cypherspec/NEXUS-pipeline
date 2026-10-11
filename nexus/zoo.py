"""The default model zoo.

Hyperparameters are fixed a priori (no per-dataset tuning) and are the same
configurations used for the single-model baselines in the benchmark, so any
gain NEXUS shows over them comes from bagging, selection and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    view: str  # "tree" or "dense"
    make: Callable[[int], object]  # seed -> unfitted estimator


def _lgbm(seed):
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8,
        subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=1,
        verbose=-1,
    )


def _xgb(seed):
    return XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", random_state=seed, n_jobs=1,
    )


def _cat(seed):
    return CatBoostClassifier(
        iterations=500, learning_rate=0.08, depth=6, random_seed=seed,
        thread_count=1, verbose=0, allow_writing_files=False,
    )


def _rf(seed):
    return RandomForestClassifier(
        n_estimators=300, max_features="sqrt", random_state=seed, n_jobs=1
    )


def _et(seed):
    return ExtraTreesClassifier(
        n_estimators=300, max_features="sqrt", random_state=seed, n_jobs=1
    )


def _logreg(seed):
    return LogisticRegression(C=1.0, max_iter=3000)


def _knn(seed):
    return KNeighborsClassifier(n_neighbors=15, weights="distance")


def _mlp(seed):
    return MLPClassifier(
        hidden_layer_sizes=(256, 128), alpha=1e-4, early_stopping=True,
        max_iter=300, random_state=seed,
    )


DEFAULT_ZOO: tuple[ModelSpec, ...] = (
    ModelSpec("lightgbm", "gbdt", "tree", _lgbm),
    ModelSpec("xgboost", "gbdt", "tree", _xgb),
    ModelSpec("catboost", "gbdt", "tree", _cat),
    ModelSpec("random_forest", "bagged_trees", "tree", _rf),
    ModelSpec("extra_trees", "bagged_trees", "tree", _et),
    ModelSpec("logreg", "linear", "dense", _logreg),
    ModelSpec("knn", "neighbors", "dense", _knn),
    ModelSpec("mlp", "neural", "dense", _mlp),
)

ZOO_BY_NAME = {m.name: m for m in DEFAULT_ZOO}


def resolve_zoo(models) -> tuple[ModelSpec, ...]:
    if models is None or models == "default":
        return DEFAULT_ZOO
    return tuple(ZOO_BY_NAME[m] if isinstance(m, str) else m for m in models)
