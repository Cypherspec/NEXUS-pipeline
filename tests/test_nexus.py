import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from nexus import NexusClassifier, apply_temperature, fit_temperature, greedy_ensemble_selection
from nexus.ensemble import log_loss_from_proba
from nexus.metrics import all_metrics, expected_calibration_error

FAST = ["lightgbm", "logreg", "knn"]


def _data(n=300, k=2, seed=0):
    X, y = make_classification(n_samples=n, n_features=8, n_informative=5, n_classes=k,
                               n_clusters_per_class=1, random_state=seed)
    return X, y


def test_greedy_selection_prefers_good_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    good = np.where(np.eye(2)[y] == 1, 0.9, 0.1)
    bad = np.full((500, 2), 0.5)
    w = greedy_ensemble_selection(np.stack([bad, good]), y, n_iter=20)
    assert w.sum() == pytest.approx(1.0)
    assert w[1] > 0.9


def test_greedy_selection_never_worse_than_best_single_on_fit_data():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 400)
    probas = rng.dirichlet(np.ones(3), size=(5, 400))
    probas[2] = 0.6 * np.eye(3)[y] + 0.4 * probas[2]
    w = greedy_ensemble_selection(probas, y, n_iter=30)
    mixed = np.tensordot(w, probas, axes=1)
    assert log_loss_from_proba(y, mixed) <= min(log_loss_from_proba(y, p) for p in probas) + 1e-12


def test_temperature_identity_and_fit():
    rng = np.random.default_rng(2)
    p = rng.dirichlet(np.ones(4), size=200)
    np.testing.assert_allclose(apply_temperature(p, 1.0), p, atol=1e-12)
    y = p.argmax(1)
    overconfident = apply_temperature(p, 0.2)
    t = fit_temperature(overconfident, rng.integers(0, 4, 200))
    assert t > 1.0  # random labels: should soften an overconfident model


def test_ece_perfect_calibration_is_zero():
    y = np.array([0, 1, 0, 1])
    p = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


@pytest.mark.parametrize("k", [2, 3])
def test_fit_predict_numpy(k):
    X, y = _data(k=k)
    clf = NexusClassifier(models=FAST, n_folds=3, stack_models=None).fit(X[:240], y[:240])
    p = clf.predict_proba(X[240:])
    assert p.shape == (60, k)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
    assert sum(clf.weights_.values()) == pytest.approx(1.0)
    assert all_metrics(y[240:], p)["accuracy"] > 0.7


def test_mixed_types_missing_values_and_string_labels():
    rng = np.random.default_rng(3)
    n = 240
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": pd.Categorical(rng.choice(["x", "y", "z"], n)),
    })
    X.loc[rng.choice(n, 20, replace=False), "a"] = np.nan
    X.loc[rng.choice(n, 20, replace=False), "c"] = np.nan
    y = np.where(X["b"] + (X["c"] == "x") > 0.3, "pos", "neg")
    clf = NexusClassifier(models=["lightgbm", "random_forest", "logreg"], n_folds=3).fit(X, y)
    assert set(clf.predict(X)) <= {"pos", "neg"}
    assert (clf.predict(X) == y).mean() > 0.8


def test_rare_class_missing_from_some_folds():
    X, y = _data(n=200, k=3)
    y[:3] = 3  # a class with only 3 members, fewer than n_folds=5
    clf = NexusClassifier(models=FAST, n_folds=5).fit(X, y)
    assert clf.predict_proba(X).shape == (200, 4)


def test_deterministic():
    X, y = _data()
    a = NexusClassifier(models=FAST, n_folds=3, random_state=7).fit(X, y).predict_proba(X)
    b = NexusClassifier(models=FAST, n_folds=3, random_state=7).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(a, b)


def test_sklearn_clone_and_params():
    from sklearn.base import clone
    clf = NexusClassifier(models=FAST, n_folds=4, ensemble_size=10)
    c = clone(clf)
    assert c.get_params()["n_folds"] == 4


def test_repeats_one_matches_v1_and_more_repeats_bag_more_models():
    X, y = _data()
    a = NexusClassifier(models=FAST, n_folds=3, random_state=1).fit(X, y)
    b = NexusClassifier(models=FAST, n_folds=3, n_repeats=1, random_state=1).fit(X, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))
    c = NexusClassifier(models=FAST, n_folds=3, n_repeats=2, random_state=1).fit(X, y)
    assert len(c.fold_models_["lightgbm"]) == 6
    np.testing.assert_allclose(c.report_.oof_proba["lightgbm"].sum(1), 1.0, atol=1e-9)


def test_two_layer_stacking():
    X, y = _data(k=3)
    clf = NexusClassifier(models=FAST, stack_models=["lightgbm", "logreg"], n_folds=3).fit(X[:240], y[:240])
    members = clf.predict_proba_members(X[240:])
    assert {"L2:lightgbm", "L2:logreg"} <= set(members)
    assert sum(clf.weights_.values()) == pytest.approx(1.0)
    p = clf.predict_proba(X[240:])
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
    assert (clf.predict(X[240:]) == y[240:]).mean() > 0.6
