"""Conformal prediction: the coverage guarantee is the specification."""

import numpy as np
import pytest

from nexus.uq.conformal import (
    ConformalClassifier,
    ConformalQuantileRegressor,
    ConformalRegressor,
    conformal_quantile,
    coverage_width_criterion,
    jackknife_plus_interval,
)

RNG = np.random.default_rng(0)


def test_conformal_quantile_finite_sample_correction():
    scores = np.arange(1.0, 11.0)          # n = 10
    # ceil(11 * 0.9)/10 = 1.0 -> the maximum
    assert conformal_quantile(scores, 0.1) == 10.0
    # ceil(11 * 0.5)/10 = 0.6 -> the 6th smallest
    assert conformal_quantile(scores, 0.5) == 6.0
    assert conformal_quantile(np.array([]), 0.1) == np.inf


def test_conformal_quantile_is_monotone_in_alpha():
    scores = RNG.normal(size=200)
    qs = [conformal_quantile(scores, a) for a in (0.01, 0.05, 0.1, 0.2, 0.5)]
    assert all(a >= b for a, b in zip(qs, qs[1:]))


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_regression_coverage_matches_nominal(alpha):
    n = 8000
    x = RNG.normal(size=n)
    y = 2 * x + RNG.normal(0, 1, n)
    pred = 2 * x
    cal, test = slice(0, n // 2), slice(n // 2, n)
    model = ConformalRegressor(alpha=alpha).fit(pred[cal], y[cal])
    empirical = model.coverage(pred[test], y[test])
    # Three standard errors of a binomial proportion at n/2 test points.
    se = np.sqrt((1 - alpha) * alpha / (n // 2))
    assert abs(empirical - (1 - alpha)) < 4 * se


def test_regression_coverage_holds_for_a_biased_model():
    """The guarantee is model-free: a badly biased predictor still gets coverage."""
    n = 6000
    y = RNG.normal(size=n)
    pred = np.full(n, 5.0)                 # wildly wrong, but exchangeable
    cal, test = slice(0, n // 2), slice(n // 2, n)
    model = ConformalRegressor(alpha=0.1).fit(pred[cal], y[cal])
    assert model.coverage(pred[test], y[test]) > 0.88


def test_regression_coverage_over_many_trials():
    """Averaged over repeated draws, coverage must sit at the nominal level."""
    alpha, trials, n = 0.1, 200, 400
    covered = []
    for t in range(trials):
        rng = np.random.default_rng(t)
        y = rng.normal(size=n)
        pred = np.zeros(n)
        half = n // 2
        model = ConformalRegressor(alpha=alpha).fit(pred[:half], y[:half])
        covered.append(model.coverage(pred[half:], y[half:]))
    mean_coverage = float(np.mean(covered))
    assert mean_coverage >= 1 - alpha - 0.01
    assert mean_coverage < 1 - alpha + 0.05


def test_normalized_regression_widths_adapt():
    n = 4000
    sigma = RNG.uniform(0.2, 3.0, n)
    y = RNG.normal(0, sigma)
    pred = np.zeros(n)
    cal, test = slice(0, n // 2), slice(n // 2, n)
    plain = ConformalRegressor(alpha=0.1).fit(pred[cal], y[cal])
    norm = ConformalRegressor(alpha=0.1, normalize=True).fit(
        pred[cal], y[cal], sigma_cal=sigma[cal])
    assert norm.coverage(pred[test], y[test], sigma[test]) > 0.86
    low, high = norm.predict_interval(pred[test], sigma[test])
    widths = high - low
    assert widths.std() > 0            # adaptive, unlike the fixed-width version
    assert np.corrcoef(widths, sigma[test])[0, 1] > 0.99


@pytest.mark.parametrize("score", ["softmax", "aps"])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_classification_coverage_matches_nominal(score, alpha):
    n, k = 8000, 4
    logits = RNG.normal(size=(n, k))
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([RNG.choice(k, p=row) for row in probs])
    cal, test = slice(0, n // 2), slice(n // 2, n)
    model = ConformalClassifier(alpha=alpha, score=score).fit(probs[cal], y[cal])
    empirical = model.coverage(probs[test], y[test])
    se = np.sqrt((1 - alpha) * alpha / (n // 2))
    assert abs(empirical - (1 - alpha)) < 4 * se


def test_classification_sets_shrink_as_alpha_grows():
    n, k = 4000, 5
    probs = RNG.dirichlet(np.ones(k), size=n)
    y = np.array([RNG.choice(k, p=row) for row in probs])
    cal, test = slice(0, n // 2), slice(n // 2, n)
    sizes = []
    for alpha in (0.05, 0.2, 0.4):
        model = ConformalClassifier(alpha=alpha).fit(probs[cal], y[cal])
        sizes.append(model.average_set_size(probs[test]))
    assert sizes[0] > sizes[1] > sizes[2]


def test_mondrian_gives_per_class_coverage():
    """Class-conditional calibration protects the rare class."""
    n, k = 6000, 2
    y = (RNG.random(n) < 0.1).astype(int)          # 10% positives
    probs = np.zeros((n, k))
    noise = RNG.normal(0, 1.2, n)
    p1 = 1 / (1 + np.exp(-(y * 2 - 1 + noise)))
    probs[:, 1], probs[:, 0] = p1, 1 - p1
    cal, test = slice(0, n // 2), slice(n // 2, n)
    mondrian = ConformalClassifier(alpha=0.1, mondrian=True).fit(probs[cal], y[cal])
    sets = mondrian.predict_set(probs[test])
    y_test = y[test]
    rare_coverage = sets[np.arange(len(y_test)), y_test][y_test == 1].mean()
    assert rare_coverage > 0.82          # the minority class is still covered


def test_conformalized_quantile_regression_fixes_miscalibration():
    n = 6000
    y = RNG.normal(size=n)
    # Deliberately far too narrow an interval.
    lower, upper = np.full(n, -0.05), np.full(n, 0.05)
    cal, test = slice(0, n // 2), slice(n // 2, n)
    model = ConformalQuantileRegressor(alpha=0.1).fit(lower[cal], upper[cal], y[cal])
    assert model.coverage(lower[test], upper[test], y[test]) > 0.87


def test_jackknife_plus_interval_covers():
    n_test, n_folds = 300, 40
    preds = RNG.normal(0, 0.1, size=(n_folds, n_test))
    residuals = np.abs(RNG.normal(0, 1, n_folds))
    y = RNG.normal(0, 1, n_test)
    low, high = jackknife_plus_interval(preds, residuals, preds.mean(axis=0), alpha=0.1)
    assert np.all(high >= low)
    assert np.mean((y >= low) & (y <= high)) > 0.75      # guarantee is 1 - 2*alpha


def test_coverage_width_criterion_penalises_undercoverage():
    good = coverage_width_criterion(0.90, 2.0, 0.90)
    bad = coverage_width_criterion(0.70, 2.0, 0.90)
    assert bad > good == 2.0
