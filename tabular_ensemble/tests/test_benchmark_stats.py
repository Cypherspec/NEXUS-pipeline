import numpy as np
import pandas as pd
import pytest

from tabular_ensemble_benchmark.analyze import bootstrap_ci, friedman_nemenyi, holm


def test_holm_matches_reference():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    np.testing.assert_allclose(holm(p), [0.03, 0.06, 0.06, 0.02])


def test_nemenyi_cd_matches_demsar_table():
    # Demšar (2006): q_0.05 for k=5 is 2.728; CD = q*sqrt(k(k+1)/(6N))
    ranks = pd.DataFrame(np.tile(np.arange(1, 6), (30, 1)), columns=list("abcde"))
    out = friedman_nemenyi(ranks)
    assert out["nemenyi_cd"] == pytest.approx(2.728 * np.sqrt(5 * 6 / (6 * 30)), rel=2e-3)
    assert out["p_value"] < 1e-6


def test_bootstrap_ci_contains_mean():
    x = np.random.default_rng(0).normal(1.0, 1.0, 200)
    lo, hi = bootstrap_ci(x)
    assert lo < x.mean() < hi
