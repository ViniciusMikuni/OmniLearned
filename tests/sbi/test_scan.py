"""Gate A: joint scans, posterior measures and local-variable integration."""

import numpy as np
import pytest
import torch
from scipy.stats import norm

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.flow import FlowSurrogate
from omnilearned.sbi.likelihood import LikelihoodOptions
from omnilearned.sbi.observations import read_observations, write_observations
from omnilearned.sbi.parameters import ConditionLayout
from omnilearned.sbi.scan import (
    Grid,
    joint_scan,
    local_variable_loglik,
    normalized_log_weights,
    refine_axes,
    trapezoid_weights,
    uniform_log_prior,
)
from omnilearned.sbi.transforms import FeatureTransform

from fields import TranslationField


# Correlated 2D Gaussian model: x_i ~ N(A theta, I).
A = np.array([[1.0, 0.95], [0.95, 1.0]])


def gaussian_events(theta, n, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 2)) + A @ np.asarray(theta)


def logp_fn_factory(x):
    def fn(sl, cand):
        mu = cand @ A.T  # [m, 2]
        d = x[sl][None] - mu[:, None]
        return -0.5 * (d ** 2).sum(-1) - np.log(2 * np.pi)
    return fn


def test_trapezoid_weights_integrate():
    a = np.array([0.0, 0.3, 1.0, 1.5, 3.0])
    w = trapezoid_weights(a)
    assert np.isclose(w.sum(), 3.0)
    assert np.isclose((w * a).sum(), 4.5)  # linear functions exact


def test_joint_scan_finds_correlated_optimum_where_slices_fail():
    truth = [1.2, -0.9]
    x = gaussian_events(truth, 2000, 0)
    grid = Grid(["t1", "t2"], [np.linspace(-2, 2, 81), np.linspace(-2, 2, 81)])
    res = joint_scan(logp_fn_factory(x), len(x), grid, event_chunk=500)
    mle = res.summary()["mle"]
    analytic = np.linalg.solve(A, x.mean(0))
    assert abs(mle["t1"] - analytic[0]) <= 0.05 + 1e-9
    assert abs(mle["t2"] - analytic[1]) <= 0.05 + 1e-9
    # Legacy-style independent coordinate scans from a default point.
    theta0 = np.array([0.0, 0.0])
    fn = logp_fn_factory(x)
    best = theta0.copy()
    for d in range(2):
        cands = np.repeat(theta0[None], 81, 0)
        cands[:, d] = grid.axes[d]
        prof = fn(slice(0, len(x)), cands).sum(1)
        best[d] = grid.axes[d][np.argmax(prof)]
    # Coordinate-wise optimum is far from the joint optimum.
    assert np.linalg.norm(best - analytic) > 1.0
    # Strong anti-correlation retained in the joint posterior.
    corr = np.array(res.summary()["posterior_correlation"])
    assert corr[0, 1] < -0.9


def test_chunking_and_candidate_batching_invariance():
    x = gaussian_events([0.1, 0.2], 333, 1)
    grid = Grid(["t1", "t2"], [np.linspace(-1, 1, 9), np.linspace(-1, 1, 7)])
    a = joint_scan(logp_fn_factory(x), len(x), grid, event_chunk=1, candidate_chunk=1)
    b = joint_scan(logp_fn_factory(x), len(x), grid, event_chunk=1000, candidate_chunk=63)
    c = joint_scan(logp_fn_factory(x), len(x), grid, event_chunk=17, candidate_chunk=5)
    np.testing.assert_allclose(a.loglik, b.loglik, rtol=0, atol=1e-9)
    np.testing.assert_allclose(a.loglik, c.loglik, rtol=0, atol=1e-9)
    # One-event evaluations summed equal the full-sample sum.
    one = sum(logp_fn_factory(x)(slice(i, i + 1), grid.points())[:, 0]
              for i in range(len(x)))
    np.testing.assert_allclose(a.loglik.reshape(-1), one, atol=1e-8)


def test_posterior_normalization_nonuniform_grid_and_prior():
    # 1D Gaussian likelihood N(theta; 0.3, 0.2^2) with Gaussian prior N(0, 0.5^2).
    axis = np.concatenate([np.linspace(-3, -0.5, 15), np.linspace(-0.45, 1.2, 400),
                           np.linspace(1.25, 3, 10)])
    grid = Grid(["t"], [axis])
    loglik = norm.logpdf(axis, 0.3, 0.2)
    logprior = norm.logpdf(axis, 0.0, 0.5)
    from omnilearned.sbi.scan import ScanResult

    res = ScanResult(grid, loglik, logprior, n_events=1)
    mass = res.posterior_mass()
    assert np.isclose(mass.sum(), 1.0)
    dens = res.posterior_density()
    assert np.isclose((dens * grid.cell_volumes()).sum(), 1.0)
    prec = 1 / 0.2 ** 2 + 1 / 0.5 ** 2
    mean = (0.3 / 0.2 ** 2) / prec
    s = res.summary()
    assert abs(s["posterior_mean"]["t"] - mean) < 2e-3
    assert abs(s["posterior_std"]["t"] - prec ** -0.5) < 2e-3
    lo, hi = s["credible_intervals"]["t"]["0.68"]
    assert abs(lo - norm.ppf(0.16, mean, prec ** -0.5)) < 5e-3
    assert abs(hi - norm.ppf(0.84, mean, prec ** -0.5)) < 5e-3
    plo, phi_ = s["profile_intervals"]["t"]["delta_2nll_1"][:2]
    assert abs(plo - 0.1) < 5e-3 and abs(phi_ - 0.5) < 5e-3


def test_uniform_prior_density_and_marginals():
    grid = Grid(["a", "b"], [np.linspace(0, 2, 5), np.linspace(-1, 1, 3)])
    lp = uniform_log_prior(grid)
    assert np.allclose(np.exp(lp), 1 / 4.0)
    from omnilearned.sbi.scan import ScanResult

    res = ScanResult(grid, np.zeros(grid.shape), lp)
    ax, dens, m = res.marginal("a")
    assert np.isclose((dens * trapezoid_weights(ax)).sum(), 1.0)
    assert np.allclose(dens, 0.5)


def test_local_variable_integration_analytic():
    # q(x|b) = N(x; b, 1); g(b) = N(b; mu, s^2) -> marginal N(x; mu, 1 + s^2).
    mu_grid = np.array([-0.5, 0.0, 0.7])
    s = 0.8
    rng = np.random.default_rng(3)
    x = rng.normal(0.2, np.sqrt(1 + s ** 2), size=50)
    b = np.linspace(-8, 8, 801)
    per_event = norm.logpdf(x[None, None, :], b[:, None, None], 1.0)
    per_event = np.broadcast_to(per_event, (len(b), len(mu_grid), len(x)))
    dens = norm.pdf(b[:, None], mu_grid[None], s)
    lw = normalized_log_weights(b, dens)
    ll, per, post = local_variable_loglik(per_event, lw)
    exact = norm.logpdf(x[None], mu_grid[:, None], np.sqrt(1 + s ** 2)).sum(-1)
    np.testing.assert_allclose(ll, exact, atol=1e-6)
    np.testing.assert_allclose(post.sum(0), 1.0)
    # Event-local posterior mean of b: (x/1 + mu/s^2)/(1 + 1/s^2).
    pm = (b[:, None] * post[:, 1, :]).sum(0)
    np.testing.assert_allclose(pm, (x + 0 / s ** 2) / (1 + 1 / s ** 2), atol=1e-4)
    # Wrong order (one shared b for the whole sample) gives a different answer.
    wrong = np.log((np.exp(per_event.sum(-1) - per_event.sum(-1).max(0))
                    * np.exp(lw)).sum(0)) + per_event.sum(-1).max(0)
    assert np.abs(wrong - exact).max() > 1.0


def test_refine_axes_without_truth():
    x = gaussian_events([0.3, 0.1], 5000, 4)
    grid = Grid(["t1", "t2"], [np.linspace(-3, 3, 13), np.linspace(-3, 3, 13)])
    res = joint_scan(logp_fn_factory(x), len(x), grid)
    fine = refine_axes(res, delta_logpost=4.5, points=[41, 41])
    assert fine.axes[0][0] >= -3 and fine.axes[0][-1] <= 3
    assert fine.axes[0][-1] - fine.axes[0][0] < 6
    res2 = joint_scan(logp_fn_factory(x), len(x), fine)
    an = np.linalg.solve(A, x.mean(0))
    m = res2.summary()["mle"]
    step = np.diff(fine.axes[0])[0]
    assert abs(m["t1"] - an[0]) <= step and abs(m["t2"] - an[1]) <= step


def test_selection_normalization_term():
    x = gaussian_events([0.0, 0.0], 10, 5)
    grid = Grid(["t1", "t2"], [np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)])
    a = joint_scan(logp_fn_factory(x), len(x), grid)
    b = joint_scan(logp_fn_factory(x), len(x), grid,
                   log_norm_fn=lambda p: np.log(0.5 + 0.1 * p[:, 0]))
    expected = a.loglik.reshape(-1) - 10 * np.log(0.5 + 0.1 * grid.points()[:, 0])
    np.testing.assert_allclose(b.loglik.reshape(-1), expected)


def _surrogate():
    model = TranslationField([0.0, 0.0, 0.0], use_cond=True)
    layout = ConditionLayout(["mu"], ["mu"], {"mu": 0.0}, {"mu": 1.0})
    tr = FeatureTransform(["a", "b", "c"], [None, None, None], [0, 0, 0], [1, 1, 1])
    return FlowSurrogate(model, GaussianBase.standard(3), tr, layout)


def test_measured_input_fixture_full_fit_path(tmp_path):
    """Observation file without truth labels runs read -> likelihood -> scan."""
    rng = np.random.default_rng(6)
    mu_true = 0.35
    # Translation field v = mu: z1 = x + mu, base N(0,1) => x ~ N(-mu, 1).
    x = rng.normal(-mu_true, 1.0, size=(400, 4, 3)).astype(np.float32)
    mask = np.ones((400, 4), dtype=bool)
    path = tmp_path / "measured.h5"
    write_observations(path, x, mask, np.arange(400) + 7000, ["a", "b", "c"])
    obs = read_observations(path)
    sur = _surrogate()
    grid = Grid(["mu"], [np.linspace(-0.2, 0.8, 101)])
    o = LikelihoodOptions(solver="euler", nsteps=4, divergence="hutchinson",
                          num_probes=1)

    def fn(sl, cand):
        lp, _ = sur.log_prob(obs.x[sl], obs.mask[sl], cand, obs.event_id[sl], o)
        return lp

    res = joint_scan(fn, len(obs), grid, event_chunk=128, candidate_chunk=11)
    doc = res.save(str(tmp_path / "fit"))
    est = doc["summary"]["mle"]["mu"]
    assert abs(est - (-x.mean())) < 0.011
    sd = doc["summary"]["posterior_std"]["mu"]
    assert abs(sd - 1 / np.sqrt(x.size)) < 0.003


def test_observation_reader_rejects_truth(tmp_path):
    import h5py

    path = tmp_path / "bad.h5"
    write_observations(path, np.zeros((2, 3, 3)), np.ones((2, 3), bool), [1, 2],
                       ["a", "b", "c"])
    with h5py.File(path, "a") as f:
        f.create_dataset("truth_theta", data=np.zeros(3))
    with pytest.raises(ValueError, match="truth"):
        read_observations(path)
    with pytest.raises(ValueError):
        write_observations(tmp_path / "x.h5", np.zeros((2, 3, 3)),
                           np.ones((2, 3), bool), [1, 2], ["a", "b", "c"],
                           attrs={"theta_star": 1.0})


def test_sbi_fit_cli_with_checkpoint(tmp_path):
    """omnilearned sbi-fit: checkpoint + truth-free observations -> grids."""
    from typer.testing import CliRunner

    from omnilearned.cli import app
    from omnilearned.sbi.scan import ScanResult
    from omnilearned.sbi.velocity import SetFlowVelocity

    torch.manual_seed(0)
    layout = ConditionLayout(["mu"], ["mu"], {"mu": 0.0}, {"mu": 1.0})
    tr = FeatureTransform(["a", "b", "c"], [None, None, None], [0, 0, 0], [1, 1, 1])
    sur = FlowSurrogate(SetFlowVelocity(3, 1, dim=16, depth=1, heads=2),
                        GaussianBase.standard(3), tr, layout)
    ck = tmp_path / "ck.pt"
    sha = sur.save(ck)
    rng = np.random.default_rng(0)
    write_observations(tmp_path / "obs.h5", rng.normal(size=(20, 4, 3)),
                       np.ones((20, 4), bool), np.arange(20), ["a", "b", "c"])
    cfg = tmp_path / "fit.toml"
    cfg.write_text(f"""
schema = "omnilearned-sbi-fit-v1"
checkpoint = "{ck}"
checkpoint_sha256 = "{sha}"
observations = "{tmp_path / 'obs.h5'}"
output_prefix = "{tmp_path / 'out' / 'fit'}"
device = "cpu"
mode = "fixed_vector"
[grid]
mu = {{min = -1.0, max = 1.0, n = 5}}
[likelihood]
solver = "euler"
nsteps = 2
num_probes = 1
[scan]
refine = {{rounds = 1, delta_logpost = 50.0, points = [5]}}
""")
    res = CliRunner().invoke(app, ["sbi-fit", "--config", str(cfg)])
    assert res.exit_code == 0, res.output + repr(res.exception)
    loaded = ScanResult.load(str(tmp_path / "out" / "fit"))
    assert loaded.per_event.shape == (5, 20)
    bad = tmp_path / "bad.toml"
    bad.write_text(cfg.read_text() + "\n[holdout]\nprimary_test = [1.0]\n")
    res = CliRunner().invoke(app, ["sbi-fit", "--config", str(bad)])
    assert res.exit_code != 0


def test_surrogate_product_and_paired_layouts_agree():
    from omnilearned.sbi.velocity import SetFlowVelocity

    torch.manual_seed(1)
    model = SetFlowVelocity(3, 2, dim=16, depth=1, heads=2)
    torch.nn.init.normal_(model.out.weight, std=0.2)
    layout = ConditionLayout.from_bounds(["p", "q"], [0, 0], [1, 1])
    tr = FeatureTransform(["a", "b", "c"], [None, None, None], [0, 0, 0], [1, 1, 1])
    sur = FlowSurrogate(model, GaussianBase.standard(3), tr, layout)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4, 3))
    mask = np.ones((5, 4), bool)
    cands = np.array([[0.1, 0.2], [0.7, 0.4], [0.5, 0.9]])
    o = LikelihoodOptions(solver="midpoint", nsteps=4, num_probes=2)
    prod, _ = sur.log_prob(x, mask, cands, np.arange(5), o)
    assert prod.shape == (3, 5)
    for m in range(3):
        rows = np.repeat(cands[m:m + 1], 5, 0)
        paired, _ = sur.log_prob_paired(x, mask, rows, np.arange(5), o)
        np.testing.assert_allclose(paired, prod[m], atol=1e-5)
