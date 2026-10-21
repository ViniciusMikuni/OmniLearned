"""Support-following scan on a fine grid (thin curved degeneracy)."""

import numpy as np

from omnilearned.sbi.scan import (
    Grid,
    ScanResult,
    dilate,
    joint_scan,
    seed_mask,
    support_scan,
)


def ridge_fn(width=0.03, curv=0.6, depth=2000.0):
    """Per-event log densities whose sum is a thin parabolic ridge in 3D.

    Two events; their sum is -0.5 ((y - curv x^2) / width)^2 - 0.5 (z/0.4)^2 times
    a constant, so the posterior is a curved tube spanning the x range.
    """
    def fn(sl, cand):
        x, y, z = cand.T
        r = -0.5 * ((y - curv * x ** 2) / width) ** 2 - 0.5 * (z / 0.4) ** 2
        n = sl.stop - sl.start
        return np.repeat((r / 2)[:, None], n, 1)
    return fn


def grid3(n=(41, 61, 9)):
    return Grid(["x", "y", "z"], [np.linspace(-1, 1, n[0]), np.linspace(-0.2, 0.8, n[1]),
                                  np.linspace(-1, 1, n[2])])


def test_seed_mask_and_dilate():
    m = seed_mask((7, 5), (3, 2))
    assert np.flatnonzero(m.any(1)).tolist() == [0, 3, 6]
    assert np.flatnonzero(m.any(0)).tolist() == [0, 2, 4]
    a = np.zeros((5, 5), bool)
    a[0, 0] = True
    d = dilate(a)
    assert d.sum() == 4 and d[1, 1] and not d[2, 2]


def test_support_scan_matches_dense_grid_and_saves_work():
    g = grid3()
    fn = ridge_fn()
    dense = joint_scan(fn, 2, g, keep_per_event=False)
    sup = support_scan(fn, 2, g, seed_stride=(5, 6, 4), delta_logpost=12.0,
                       keep_per_event=False)
    d = sup.diagnostics
    assert d["support_closed"] and d["complete"]
    assert d["evaluated"] < 0.5 * g.size
    a, b = dense.summary(), sup.summary()
    for n in g.names:
        sd = a["posterior_std"][n]
        assert abs(a["posterior_mean"][n] - b["posterior_mean"][n]) < 1e-3 * sd
        assert abs(a["posterior_std"][n] - b["posterior_std"][n]) < 1e-3 * sd
    assert a["mle"] == b["mle"]
    # every supported node has all neighbours evaluated
    lp = sup.log_posterior()
    act = sup.evaluated & (lp >= np.nanmax(lp) - 12.0)
    assert not (dilate(act) & ~sup.evaluated).any()
    assert np.isnan(sup.loglik[~sup.evaluated]).all()


def test_support_scan_with_log_norm_matches_dense():
    g = grid3((21, 31, 5))
    fn = ridge_fn(width=0.06)
    norm = lambda c: 0.01 * c[:, 0] - 0.02 * c[:, 2]
    dense = joint_scan(fn, 2, g, log_norm_fn=norm)
    sup = support_scan(fn, 2, g, seed_stride=(4, 5, 2), delta_logpost=30.0, log_norm_fn=norm)
    ev = sup.evaluated
    np.testing.assert_allclose(sup.loglik[ev], dense.loglik[ev])
    np.testing.assert_allclose(sup.per_event.reshape(g.shape + (2,))[ev],
                               dense.per_event.reshape(g.shape + (2,))[ev])


def test_support_scan_save_load_roundtrip(tmp_path):
    g = grid3((11, 11, 3))
    sup = support_scan(ridge_fn(width=0.1), 2, g, seed_stride=(3, 3, 1))
    sup.save(str(tmp_path / "r"))
    back = ScanResult.load(str(tmp_path / "r"))
    assert (back.evaluated == sup.evaluated).all()
    np.testing.assert_array_equal(np.isnan(back.loglik), np.isnan(sup.loglik))


def test_support_scan_time_limit_reports_incomplete():
    g = grid3((11, 11, 3))
    sup = support_scan(ridge_fn(), 2, g, seed_stride=(2, 2, 1), max_seconds=0.0)
    assert not sup.diagnostics["complete"] and not sup.diagnostics["support_closed"]
