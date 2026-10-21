"""toy-quenching-v1 simulator checks (geometry, moments, records, selection).

Ported from the omni-sbi-mc application tests; needs FastJet through hepyy.
"""

import numpy as np
import pytest

from omnilearned.sbi.toy import simulator as toy
from omnilearned.sbi.toy.config import array_sha256, load_toy_config, sample_setting
from omnilearned.sbi.toy.design import (
    STREAMS,
    check_reservation,
    design_settings,
    reserved_settings,
    rng_for,
)
from omnilearned.sbi.toy.reco import cluster_events, four_vectors

CFG, C, S = load_toy_config()


def test_reproducible_and_seed_dependent():
    a, ma = toy.sample_toy((4.5, 0.4, 0.65), 200, rng_for(1, 2, 3), C, S, stream=3)
    b, mb = toy.sample_toy((4.5, 0.4, 0.65), 200, rng_for(1, 2, 3), C, S, stream=3)
    c, _ = toy.sample_toy((4.5, 0.4, 0.65), 200, rng_for(1, 2, 4), C, S, stream=3)
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["event_id"], b["event_id"])
    assert ma == mb
    assert not np.array_equal(a["x"], c["x"])


def test_production_points_inside_overlap_and_symmetric():
    rng = np.random.default_rng(0)
    R = C.geometry_radius_fm
    for b in [0.0, 4.5, 11.0]:
        x, y = toy.sample_production_points(b, 20000, rng, R)
        fp, fm = toy.overlap_factors(x, y, b, R)
        assert np.all(fp > 0) and np.all(fm > 0)
        assert abs(x.mean()) < 0.05 and abs(y.mean()) < 0.05
    # b = 0: density proportional to (1 - r^2/R^2)^2 -> <r^2> = R^2 / 4.
    x, y = toy.sample_production_points(0.0, 200000, rng, R)
    assert abs(np.mean(x ** 2 + y ** 2) / R ** 2 - 0.25) < 0.005
    with pytest.raises(ValueError):
        toy.sample_production_points(13.0, 10, rng, R)


def test_exit_distance_geometry():
    R = C.geometry_radius_fm
    L = toy.exit_distance(np.array([0.0]), np.array([0.0]), np.array([1.0]),
                          np.array([0.0]), 0.0, R)
    assert np.isclose(L[0], R)
    # b = 4: from the origin along +x the left disk (centre -2) is exited at x = R - 2.
    L = toy.exit_distance(np.array([0.0]), np.array([0.0]), np.array([1.0]),
                          np.array([0.0]), 4.0, R)
    assert np.isclose(L[0], R - 2.0)
    # Roundoff at the boundary does not produce NaN.
    L = toy.exit_distance(np.array([R - 2.0 - 1e-15]), np.array([0.0]),
                          np.array([1.0]), np.array([0.0]), 4.0, R)
    assert np.isfinite(L[0]) and L[0] >= 0


def test_moment_refinement_and_limits():
    rng = np.random.default_rng(5)
    th = (4.5, 0.40, 0.65)
    R = C.geometry_radius_fm
    x0, y0 = toy.sample_production_points(th[0], 3000, rng, R)
    psi = rng.uniform(0, 2 * np.pi, 3000)
    nx, ny = np.cos(psi), np.sin(psi)
    L = toy.exit_distance(x0, y0, nx, ny, th[0], R)
    I64 = toy.path_moments(x0, y0, nx, ny, L, th, C, 64)
    I128 = toy.path_moments(x0, y0, nx, ny, L, th, C, 128)
    I256 = toy.path_moments(x0, y0, nx, ny, L, th, C, 256)
    d64 = np.abs(I64 - I256).mean(0)
    d128 = np.abs(I128 - I256).mean(0)
    assert np.all(d128 <= d64 + 1e-12)
    assert np.all(d64 / np.maximum(I256.mean(0), 1e-12) < 0.01)
    # Below T_cut everywhere: no interaction.
    I = toy.path_moments(x0, y0, nx, ny, L, (4.5, 0.15, 0.65), C, 64)
    assert np.all(I == 0)
    # A path shorter than tau_init has zero moments.
    I = toy.path_moments(np.zeros(1), np.zeros(1), np.ones(1), np.zeros(1),
                         np.array([0.3]), (0.0, 0.5, 0.65), C, 64)
    assert np.all(I == 0)


def test_records_units_four_vectors_and_membership():
    rec, man = toy.sample_toy((1.5, 0.5, 0.35), 300, rng_for(7, 1), C, S, stream=6)
    assert man["accepted"] == 300 and man["complete"]
    assert man["attempted"] <= man["attempts_generated"]
    x = rec["x"]
    assert x.shape == (300, 32, 3)
    assert np.all(np.abs(x[..., 0]) < 2.5) and np.all(np.abs(x[..., 1]) < 1.2)
    pt = np.exp(x[..., 2])
    assert np.all(pt > 0)
    p4 = four_vectors(x[..., 0], x[..., 1], pt)
    np.testing.assert_allclose(p4, rec["p4"], rtol=1e-12, atol=1e-12)
    m2 = p4[..., 3] ** 2 - (p4[..., :3] ** 2).sum(-1)
    assert np.all(np.abs(m2) < 1e-8 * p4[..., 3] ** 2)
    assert np.all(rec["membership"] == 0)
    assert np.all((rec["jet_pt"] >= 60) & (rec["jet_pt"] <= 200))
    assert np.all(np.abs(rec["jet_eta"]) < 1.0)
    r = cluster_events(rec["p4"], 0.4)
    np.testing.assert_allclose(r["jet_pt"], rec["jet_pt"])
    ids = rec["event_id"]
    assert np.unique(ids).size == ids.size
    assert np.all((ids >> 48) == 6)
    # Hidden diagnostics are aligned with records but are a separate store.
    assert set(rec["diag"]) >= {"P0", "I1", "I2", "I3", "x0", "y0", "component"}
    assert np.all(rec["diag"]["component"].sum(1) == C.soft_count)


def test_acceptance_bookkeeping():
    rec, man = toy.sample_toy((11.0, 0.25, 1.2), 100, rng_for(9), C, S)
    total_rej = sum(man["rejections"].values())
    assert total_rej + man["accepted_generated"] == man["attempts_generated"]
    assert 0 < man["acceptance"] < 1


def test_reservation_and_design():
    for prof in ["demonstration", "dense"]:
        d = CFG["design"][prof]
        placement = check_reservation(d, reserved_settings(CFG))
        p = placement["primary_test"]
        assert all(p["strictly_between_axis_values"].values()) and p["inside_bounds"]
        assert all(all(v["strictly_between_axis_values"].values()) for v in placement.values())
        assert design_settings(d).shape[0] == {"demonstration": 80, "dense": 1430}[prof]
    bad = dict(CFG["design"]["demonstration"])
    bad["b"] = [0.0, 4.5, 11.0]
    bad["T_ref"] = [0.25, 0.40, 0.55]
    bad["tau_init"] = [0.2, 0.65, 1.2]
    with pytest.raises(ValueError):
        check_reservation(bad, reserved_settings(CFG))


def test_streams_distinct():
    assert len(set(STREAMS.values())) == len(STREAMS)


def test_cross_host_reproducibility():
    """Reference from the GPU node (2026-10-21): same arrays on any host."""
    rec, man = sample_setting((6.0, 0.40, 0.65), 300, 424242)
    assert man["accepted"] == 300 and man["attempted"] == 7760
    assert array_sha256(rec) == "63a9cf13542863418c643d8c377ee26e617c415bd91efba316f84dc1998d24d4"
