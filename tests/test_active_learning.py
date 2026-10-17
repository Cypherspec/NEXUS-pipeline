"""Gaussian processes, acquisition functions, SVGD and the screening loop."""

import numpy as np
import pytest

from nexus.al.acquisition import (
    acquire,
    expected_improvement,
    probability_of_improvement,
    select_batch,
    thompson_sample,
    upper_confidence_bound,
)
from nexus.al.gp import (
    GaussianProcess,
    MultiFidelityGP,
    linear_kernel,
    matern52_kernel,
    rbf_kernel,
    tanimoto_kernel,
)
from nexus.al.loop import active_learning_screen
from nexus.al.svgd import (
    bayesian_linear_score,
    gaussian_score,
    mixture_score,
    rbf_kernel_and_grad,
    svgd,
)

RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ kernels

@pytest.mark.parametrize("fn", [rbf_kernel, matern52_kernel, linear_kernel,
                                tanimoto_kernel])
def test_kernels_are_symmetric_and_psd(fn):
    X = RNG.random((12, 4))
    K = fn(X, X)
    np.testing.assert_allclose(K, K.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(K)) > -1e-8


def test_rbf_peaks_at_zero_distance():
    x = np.zeros((1, 3))
    assert rbf_kernel(x, x)[0, 0] == pytest.approx(1.0)
    far = rbf_kernel(x, np.full((1, 3), 10.0))[0, 0]
    assert far < 1e-6


def test_tanimoto_matches_the_set_definition():
    a = np.array([[1, 1, 0, 0]], dtype=float)
    b = np.array([[1, 0, 1, 0]], dtype=float)
    # |intersection| / |union| = 1/3
    assert tanimoto_kernel(a, b)[0, 0] == pytest.approx(1 / 3)
    assert tanimoto_kernel(a, a)[0, 0] == pytest.approx(1.0)


# ----------------------------------------------------------- Gaussian process

def test_gp_interpolates_noiseless_training_data():
    X = RNG.uniform(-3, 3, (15, 1))
    y = np.sin(X).ravel()
    gp = GaussianProcess(kernel="rbf", lengthscale=1.0, noise=1e-8).fit(X, y)
    mean, std = gp.predict(X)
    np.testing.assert_allclose(mean, y, atol=1e-4)
    assert np.all(std < 1e-2)


def test_gp_uncertainty_grows_away_from_data():
    X = RNG.uniform(-1, 1, (20, 1))
    gp = GaussianProcess(kernel="rbf", lengthscale=0.5).fit(X, np.sin(X).ravel())
    _, near = gp.predict(np.array([[0.0]]))
    _, far = gp.predict(np.array([[20.0]]))
    assert far[0] > 5 * near[0]


def test_gp_recovers_a_smooth_function():
    X = RNG.uniform(-3, 3, (40, 1))
    y = np.sin(X).ravel() + 0.05 * RNG.normal(size=40)
    gp = GaussianProcess(kernel="rbf").optimize(X, y)
    Xs = np.linspace(-3, 3, 100)[:, None]
    mean = gp.predict(Xs, return_std=False)
    assert np.sqrt(np.mean((mean - np.sin(Xs).ravel()) ** 2)) < 0.1


def test_gp_optimize_improves_marginal_likelihood():
    X = RNG.uniform(-3, 3, (30, 1))
    y = np.sin(X).ravel()
    default = GaussianProcess(kernel="rbf", lengthscale=0.05).fit(X, y)
    before = default.log_marginal_likelihood()
    tuned = GaussianProcess(kernel="rbf").optimize(X, y)
    assert tuned.log_marginal_likelihood() >= before


def test_gp_posterior_samples_have_right_shape_and_spread():
    X = RNG.uniform(-2, 2, (10, 1))
    gp = GaussianProcess(kernel="rbf").fit(X, np.sin(X).ravel())
    Xs = np.linspace(-2, 2, 25)[:, None]
    draws = gp.sample(Xs, n_samples=8, seed=0)
    assert draws.shape == (8, 25)
    assert draws.std() > 0


def test_multi_fidelity_beats_high_fidelity_alone():
    Xl = RNG.uniform(-3, 3, (60, 1))
    yl = 0.8 * np.sin(Xl).ravel() + 0.3          # cheap, biased
    Xh = RNG.uniform(-3, 3, (12, 1))
    yh = np.sin(Xh).ravel()                       # expensive, true
    Xs = np.linspace(-3, 3, 100)[:, None]
    truth = np.sin(Xs).ravel()
    mf = MultiFidelityGP().fit(Xl, yl, Xh, yh)
    only_high = GaussianProcess(kernel="rbf").optimize(Xh, yh)
    err_mf = np.sqrt(np.mean((mf.predict(Xs, return_std=False) - truth) ** 2))
    err_hi = np.sqrt(np.mean((only_high.predict(Xs, return_std=False) - truth) ** 2))
    assert err_mf <= err_hi
    assert mf.rho == pytest.approx(1.25, abs=0.2)    # true scaling is 1/0.8


# ------------------------------------------------------------- acquisitions

def test_expected_improvement_is_non_negative():
    m = np.linspace(-5, 5, 50)
    assert np.all(expected_improvement(m, np.ones(50), best=0.0) >= 0)


def test_expected_improvement_rewards_promise_and_uncertainty():
    ei = expected_improvement(np.array([0.0, 2.0]), np.array([1.0, 1.0]), best=1.0)
    assert ei[1] > ei[0]                                   # higher mean wins
    ei2 = expected_improvement(np.array([0.0, 0.0]), np.array([0.1, 2.0]), best=1.0)
    assert ei2[1] > ei2[0]                                 # more uncertainty wins


def test_probability_of_improvement_bounds():
    p = probability_of_improvement(np.linspace(-5, 5, 20), np.ones(20), best=0.0)
    assert np.all((p >= 0) & (p <= 1))


def test_ucb_increases_with_beta():
    m, s = np.array([1.0]), np.array([2.0])
    assert upper_confidence_bound(m, s, 3.0) > upper_confidence_bound(m, s, 1.0)


def test_thompson_sampling_is_random_but_centred():
    m, s = np.zeros(2000), np.ones(2000)
    draws = thompson_sample(m, s, seed=0)
    assert abs(draws.mean()) < 0.1 and 0.9 < draws.std() < 1.1


def test_acquire_dispatches_all_registered_names():
    m, s = RNG.normal(size=10), np.abs(RNG.normal(size=10))
    for name in ("ei", "pi", "ucb", "greedy", "uncertainty"):
        out = acquire(name, m, s, best=0.0)
        assert out.shape == (10,) and np.isfinite(out).all()


def test_select_batch_takes_top_k_without_diversity():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert sorted(select_batch(scores, 2)) == [1, 3]


def test_select_batch_with_diversity_avoids_duplicates():
    # Two identical fingerprints with the top scores; diversity should force
    # the batch to reach for a different one.
    features = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]], dtype=float)
    scores = np.array([1.0, 0.99, 0.5])
    picked = select_batch(scores, 2, features=features, diversity_weight=1.0)
    assert 2 in picked


# ------------------------------------------------------------------- SVGD

def test_svgd_recovers_gaussian_moments():
    target_mean = np.array([2.0, -1.0])
    target_cov = np.array([[1.0, 0.6], [0.6, 1.5]])
    particles = RNG.normal(0, 3, (200, 2))
    out = svgd(particles, gaussian_score(target_mean, target_cov),
               steps=600, step_size=0.3)
    np.testing.assert_allclose(out.mean(axis=0), target_mean, atol=0.3)
    np.testing.assert_allclose(np.cov(out.T), target_cov, atol=0.5)


def test_svgd_covers_both_modes():
    score = mixture_score([np.array([-3.0, 0.0]), np.array([3.0, 0.0])],
                          [np.eye(2) * 0.5] * 2)
    out = svgd(RNG.normal(0, 1, (200, 2)), score, steps=800, step_size=0.3)
    left = float((out[:, 0] < 0).mean())
    assert 0.3 < left < 0.7


def test_svgd_particles_do_not_collapse():
    out = svgd(RNG.normal(0, 3, (100, 2)), gaussian_score(np.zeros(2), np.eye(2)),
               steps=300, step_size=0.2)
    assert out.std(axis=0).min() > 0.4


def test_svgd_kernel_gradient_matches_finite_difference():
    x = RNG.normal(size=(5, 2))
    K, grad = rbf_kernel_and_grad(x, bandwidth=1.0)
    eps = 1e-6
    for i in range(5):
        for d in range(2):
            plus = x.copy(); plus[i, d] += eps
            minus = x.copy(); minus[i, d] -= eps
            kp = rbf_kernel_and_grad(plus, bandwidth=1.0)[0]
            km = rbf_kernel_and_grad(minus, bandwidth=1.0)[0]
            numeric = (kp[i] - km[i]) / (2 * eps)
            np.testing.assert_allclose(grad[i, :, d], numeric, atol=1e-5)


def test_bayesian_linear_score_vanishes_at_the_posterior_mean():
    X = RNG.normal(size=(50, 3))
    w_true = np.array([1.0, -2.0, 0.5])
    y = X @ w_true
    noise, prior = 0.1, 1.0
    score = bayesian_linear_score(X, y, noise=noise, prior_variance=prior)
    A = X.T @ X / noise ** 2 + np.eye(3) / prior
    mode = np.linalg.solve(A, X.T @ y / noise ** 2)
    np.testing.assert_allclose(score(mode[None, :])[0], np.zeros(3), atol=1e-6)


# ----------------------------------------------------- active learning loop

def test_screening_finds_more_hits_than_random():
    """On a library where activity is a clear function of the features,
    an informed strategy must beat random selection."""
    rng = np.random.default_rng(0)
    n, d = 600, 16
    features = (rng.random((n, d)) < 0.3).astype(float)
    weights = rng.normal(size=d)
    scores = features @ weights
    y = (scores > np.quantile(scores, 0.9)).astype(float)   # 10% actives

    informed = active_learning_screen(
        features, oracle=lambda i: y[i], n_rounds=5, batch_size=30,
        initial_size=30, acquisition="greedy", surrogate="gp",
        kernel="tanimoto", hit_threshold=0.5, seed=0)
    random_run = active_learning_screen(
        features, oracle=lambda i: y[i], n_rounds=5, batch_size=30,
        initial_size=30, acquisition="random", hit_threshold=0.5, seed=0)
    assert informed.total_hits > random_run.total_hits


def test_screening_bookkeeping_is_consistent():
    rng = np.random.default_rng(1)
    features = (rng.random((200, 8)) < 0.4).astype(float)
    y = (features.sum(axis=1) > 4).astype(float)
    res = active_learning_screen(
        features, oracle=lambda i: y[i], n_rounds=3, batch_size=20,
        initial_size=20, acquisition="ucb", surrogate="gp", kernel="tanimoto",
        hit_threshold=0.5, seed=0)
    assert res.n_screened == [20, 40, 60, 80]
    assert res.cumulative_hits == list(np.cumsum(res.hits_per_round))
    picked = np.concatenate(res.selected)
    assert len(picked) == len(np.unique(picked))       # never re-measured
    assert all(b >= a for a, b in zip(res.best_value, res.best_value[1:]))


def test_random_strategy_enrichment_is_about_one():
    rng = np.random.default_rng(2)
    n = 1500
    features = (rng.random((n, 10)) < 0.3).astype(float)
    y = (rng.random(n) < 0.1).astype(float)            # activity unrelated to features
    res = active_learning_screen(
        features, oracle=lambda i: y[i], n_rounds=4, batch_size=50,
        initial_size=50, acquisition="random", hit_threshold=0.5, seed=0)
    enrichment = res.enrichment_over_random(n, int(y.sum()))
    assert 0.4 < enrichment < 1.8
