import numpy as np
import pytest

from omnilearned.sbi.fidelity import (
    equivalent_shift, global_uniformity, mean_relevance, permutation_test, setting_verdict)


def _obs(rng, n, shift=0.0):
    return {"x": rng.normal(shift, 1.0, n), "y": rng.normal(0, 1, (n, 4))}


def test_permutation_null_is_calibrated():
    rng = np.random.default_rng(1)
    ps = [permutation_test(_obs(rng, 300), _obs(rng, 300), ["x", "y"], n_perm=99,
                           seed=s, particle_stride=1)["p_family"] for s in range(60)]
    g = global_uniformity(ps)
    assert g["ks_p"] > 0.01 and 0.3 < np.mean(ps) < 0.7


def test_permutation_detects_shift_and_relevance_needs_size():
    rng = np.random.default_rng(2)
    a, b = _obs(rng, 4000), _obs(rng, 4000, shift=0.15)
    perm = permutation_test(a, b, ["x", "y"], n_perm=99, particle_stride=1)
    assert perm["p_family"] < 1e-4 and perm["p_method"] == "gumbel_tail"
    assert perm["p"]["x"] <= 0.02
    # 0.15 sd shift is 4.7 standard errors at N=1000: relevant at tol 0.5
    rel = mean_relevance(a, b, ["x"], n_events=1000)
    assert rel["x"]["relevant"] and setting_verdict(perm, rel, 0.05)["fail"]
    # the same shift is irrelevant for an analysis of N=4 events
    rel4 = mean_relevance(a, b, ["x"], n_events=4)
    assert not rel4["x"]["relevant"] and not setting_verdict(perm, rel4, 0.05)["fail"]


def test_jitter_does_not_fail():
    rng = np.random.default_rng(3)
    a, b = _obs(rng, 500), _obs(rng, 500)
    perm = permutation_test(a, b, ["x", "y"], n_perm=99, particle_stride=1)
    rel = mean_relevance(a, b, ["x"], n_events=1000)
    assert not setting_verdict(perm, rel, 0.05 / 20)["fail"]


def test_equivalent_shift_recovers_parameter_offset():
    G = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    C = np.diag([1.0, 2.0, 3.0])
    dth = np.array([0.1, -0.05])
    est, cov = equivalent_shift(G, C, G @ dth, np.eye(3) * 1e-4)
    np.testing.assert_allclose(est, dth, atol=1e-12)
    assert cov.shape == (2, 2) and np.all(np.diag(cov) > 0)


def test_global_uniformity_counts():
    g = global_uniformity([0.001, 0.5, 0.9, 0.2])
    assert g["n_below_0.01"] == 1 and g["binomial_p_at_least"] == pytest.approx(1 - 0.99 ** 4)
