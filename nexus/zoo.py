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
    return tuple(ALL_SPECS[m] if isinstance(m, str) else m for m in models)


# ---------------------------------------------------------------------------
# Extended portfolio (NEXUS v2 candidates). Configurations are chosen a priori
# to add diversity along axes known to matter (tree size, randomisation,
# learning rate with early stopping); they are not tuned on the benchmark.

def _es(library, **params):
    from nexus.early_stopping import EarlyStoppedBoosting
    return lambda seed: EarlyStoppedBoosting(library, params, random_state=seed)


def _lgbm_xt(seed):
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, extra_trees=True,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed,
        n_jobs=1, verbose=-1,
    )


def _rf_entropy(seed):
    return RandomForestClassifier(n_estimators=300, criterion="entropy", max_features="sqrt",
                                  random_state=seed, n_jobs=1)


def _et_entropy(seed):
    return ExtraTreesClassifier(n_estimators=300, criterion="entropy", max_features="sqrt",
                                random_state=seed, n_jobs=1)


def _knn5(seed):
    return KNeighborsClassifier(n_neighbors=5, weights="distance")


def _mlp_wide(seed):
    return MLPClassifier(hidden_layer_sizes=(512,), alpha=1e-3, early_stopping=True,
                         max_iter=300, random_state=seed)


EXTRA_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("lightgbm_es", "gbdt", "tree", _es("lightgbm", learning_rate=0.03, num_leaves=31,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8)),
    ModelSpec("lightgbm_es_large", "gbdt", "tree", _es("lightgbm", learning_rate=0.03, num_leaves=128,
              min_child_samples=5, subsample=0.8, subsample_freq=1, colsample_bytree=0.6)),
    ModelSpec("lightgbm_es_small", "gbdt", "tree", _es("lightgbm", learning_rate=0.03, num_leaves=7,
              min_child_samples=20, colsample_bytree=0.8, reg_lambda=1.0)),
    ModelSpec("lightgbm_xt", "gbdt", "tree", _lgbm_xt),
    ModelSpec("xgboost_es", "gbdt", "tree", _es("xgboost", learning_rate=0.03, max_depth=6,
              subsample=0.8, colsample_bytree=0.8)),
    ModelSpec("xgboost_es_deep", "gbdt", "tree", _es("xgboost", learning_rate=0.03, max_depth=10,
              min_child_weight=2, subsample=0.8, colsample_bytree=0.6)),
    ModelSpec("catboost_es", "gbdt", "tree", _es("catboost", learning_rate=0.05, depth=6)),
    ModelSpec("catboost_es_d8", "gbdt", "tree", _es("catboost", learning_rate=0.05, depth=8, l2_leaf_reg=5)),
    ModelSpec("catboost_es_d4", "gbdt", "tree", _es("catboost", learning_rate=0.08, depth=4)),
    ModelSpec("random_forest_entropy", "bagged_trees", "tree", _rf_entropy),
    ModelSpec("extra_trees_entropy", "bagged_trees", "tree", _et_entropy),
    ModelSpec("knn5", "neighbors", "dense", _knn5),
    ModelSpec("mlp_wide", "neural", "dense", _mlp_wide),
)

ALL_SPECS = {m.name: m for m in DEFAULT_ZOO + EXTRA_SPECS}
