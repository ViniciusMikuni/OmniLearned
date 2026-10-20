"""Gate A: flow log densities against exact analytic results."""

import numpy as np
import pytest
import torch

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.likelihood import (
    LikelihoodOptions,
    frozen_for_inference,
    log_prob,
    stable_probes,
)
from omnilearned.sbi.transforms import FeatureTransform
from omnilearned.sbi.velocity import SetFlowVelocity

from fields import (
    LinearField,
    TranslationField,
    ZeroField,
    gaussian_logpdf,
    linear_exact_logp,
)

torch.set_default_dtype(torch.float32)

MEANS = [0.3, -1.0, 2.0]
STDS = [0.5, 2.0, 1.5]
BASE = GaussianBase(MEANS, STDS)


def make_cloud(n=6, p=5, f=3, n_real=None, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, f, generator=g, dtype=dtype)
    mask = torch.ones(n, p, dtype=torch.bool)
    if n_real is not None:
        for i, r in enumerate(n_real):
            mask[i, r:] = False
    return x * mask.unsqueeze(-1), mask


def opts(**kw):
    d = dict(solver="rk4", nsteps=32, divergence="exact", max_nfe=100000)
    d.update(kw)
    return LikelihoodOptions(**d)


def test_zero_field_is_base_density():
    x, mask = make_cloud(n_real=[5, 3, 1, 5, 2, 4])
    lp, diag = log_prob(ZeroField(), x, mask, BASE, options=opts())
    exact = (gaussian_logpdf(x.numpy(), np.array(MEANS), np.array(STDS))
             * mask.numpy()[..., None]).reshape(6, -1).sum(-1)
    np.testing.assert_allclose(lp.numpy(), exact, rtol=0, atol=1e-10)
    assert diag["nonfinite"] == 0


def test_translation_field():
    c = [0.7, -0.2, 1.1]
    x, mask = make_cloud(n_real=[5, 2, 5, 4, 1, 3])
    lp, _ = log_prob(TranslationField(c), x, mask, BASE, options=opts())
    z1 = x.numpy() + np.array(c)
    exact = (gaussian_logpdf(z1, np.array(MEANS), np.array(STDS))
             * mask.numpy()[..., None]).reshape(6, -1).sum(-1)
    np.testing.assert_allclose(lp.numpy(), exact, atol=1e-9)


@pytest.mark.parametrize("divergence", ["exact", "hutchinson"])
def test_diagonal_linear_field(divergence):
    A = np.diag([0.4, -0.3, 0.8])
    x, mask = make_cloud(n_real=[5, 3, 1, 5, 2, 4])
    field = LinearField(A, alpha=1.0, beta=0.5).double()
    o = opts(divergence=divergence, num_probes=1, nsteps=64)
    lp, _ = log_prob(field, x, mask, BASE, options=o, event_ids=np.arange(6))
    exact = linear_exact_logp(x.numpy(), mask.numpy(), A, 1.0, 0.5, MEANS, STDS)
    # Rademacher probes are exact for diagonal Jacobians.
    np.testing.assert_allclose(lp.numpy(), exact, atol=1e-8)


def test_full_linear_field_and_refinement():
    A = np.array([[0.2, 0.5, -0.1], [-0.3, 0.1, 0.4], [0.25, -0.2, -0.3]])
    x, mask = make_cloud(n_real=[5, 4, 3, 2, 1, 5])
    exact = linear_exact_logp(x.numpy(), mask.numpy(), A, 0.8, 0.6, MEANS, STDS)
    field = LinearField(A, alpha=0.8, beta=0.6).double()
    errs = {}
    for solver in ["euler", "midpoint", "rk4"]:
        errs[solver] = []
        for n in [4, 8, 16, 32]:
            lp, _ = log_prob(field, x, mask, BASE, options=opts(solver=solver, nsteps=n))
            errs[solver].append(np.abs(lp.numpy() - exact).max())
        # Error must shrink under refinement at the expected order.
        e = np.array(errs[solver])
        assert np.all(np.diff(e) < 0), (solver, e)
    assert errs["euler"][-1] / errs["euler"][-2] < 0.6
    assert errs["midpoint"][-1] / errs["midpoint"][-2] < 0.3
    assert errs["rk4"][-1] < 1e-6
    lp_a, _ = log_prob(field, x, mask, BASE,
                       options=opts(solver="dopri5", rtol=1e-9, atol=1e-9))
    np.testing.assert_allclose(lp_a.numpy(), exact, atol=1e-6)


def test_hutchinson_unbiased_for_full_jacobian():
    A = np.array([[0.2, 0.5, -0.1], [-0.3, 0.1, 0.4], [0.25, -0.2, -0.3]])
    x, mask = make_cloud(n=4, n_real=[5, 4, 3, 2])
    field = LinearField(A, 0.8, 0.0).double()
    exact = linear_exact_logp(x.numpy(), mask.numpy(), A, 0.8, 0.0, MEANS, STDS)
    est = []
    for seed in range(200):
        o = opts(divergence="hutchinson", num_probes=1, probe_seed=seed, nsteps=16)
        est.append(log_prob(field, x, mask, BASE, options=o,
                            event_ids=np.arange(4))[0].numpy())
    est = np.array(est)
    err = est.mean(0) - exact
    sem = est.std(0) / np.sqrt(len(est))
    assert np.all(np.abs(err) < 4 * sem + 1e-6), (err, sem)
    # More probes reduce the spread.
    o = opts(divergence="hutchinson", num_probes=16, probe_seed=999, nsteps=16)
    many = log_prob(field, x, mask, BASE, options=o, event_ids=np.arange(4))[0].numpy()
    assert np.all(np.abs(many - exact) < 4 * est.std(0) / 4 + 1e-6)


def test_padding_and_batch_invariance():
    torch.manual_seed(3)
    model = SetFlowVelocity(3, 2, dim=32, depth=2, heads=4).double()
    torch.nn.init.normal_(model.out.weight, std=0.3)
    for blk in model.blocks:
        torch.nn.init.normal_(blk.mod.weight, std=0.1)
    x, mask = make_cloud(n=8, p=5, n_real=[5, 4, 3, 5, 2, 1, 5, 3], seed=1)
    cond = torch.randn(8, 2, dtype=torch.float64)
    ids = np.arange(100, 108)
    o = opts(divergence="hutchinson", num_probes=2, solver="midpoint", nsteps=16)
    full, _ = log_prob(model, x, mask, BASE, cond=cond, options=o, event_ids=ids)
    parts = [log_prob(model, x[s:s + 3], mask[s:s + 3], BASE, cond=cond[s:s + 3],
                      options=o, event_ids=ids[s:s + 3])[0] for s in range(0, 8, 3)]
    np.testing.assert_allclose(full.numpy(), torch.cat(parts).numpy(), atol=1e-10)
    # Extra padding columns change nothing.
    xp = torch.cat([x, torch.zeros(8, 3, 3, dtype=x.dtype)], 1)
    mp = torch.cat([mask, torch.zeros(8, 3, dtype=torch.bool)], 1)
    padded, _ = log_prob(model, xp, mp, BASE, cond=cond, options=o, event_ids=ids)
    np.testing.assert_allclose(full.numpy(), padded.numpy(), atol=1e-10)


def test_permutation_invariance_exact_divergence():
    torch.manual_seed(4)
    model = SetFlowVelocity(3, 1, dim=32, depth=2, heads=2).double()
    torch.nn.init.normal_(model.out.weight, std=0.3)
    x, mask = make_cloud(n=3, p=6, seed=2)
    cond = torch.randn(3, 1, dtype=torch.float64)
    perm = torch.randperm(6)
    o = opts(nsteps=16)
    a, _ = log_prob(model, x, mask, BASE, cond=cond, options=o)
    b, _ = log_prob(model, x[:, perm], mask[:, perm], BASE, cond=cond, options=o)
    np.testing.assert_allclose(a.numpy(), b.numpy(), atol=1e-9)


def test_exact_vs_hutchinson_network():
    torch.manual_seed(5)
    model = SetFlowVelocity(3, 1, dim=32, depth=2, heads=2).double()
    torch.nn.init.normal_(model.out.weight, std=0.3)
    x, mask = make_cloud(n=3, p=4, seed=7)
    cond = torch.zeros(3, 1, dtype=torch.float64)
    ex, _ = log_prob(model, x, mask, BASE, cond=cond, options=opts(nsteps=8))
    est = np.array([
        log_prob(model, x, mask, BASE, cond=cond,
                 options=opts(nsteps=8, divergence="hutchinson", num_probes=4,
                              probe_seed=s), event_ids=np.arange(3))[0].numpy()
        for s in range(40)
    ])
    sem = est.std(0) / np.sqrt(len(est))
    assert np.all(np.abs(est.mean(0) - ex.numpy()) < 4 * sem + 1e-8)


def test_model_state_restored_and_weights_unchanged():
    field = LinearField(np.eye(3) * 0.1).double()
    field.train()
    before = {k: v.clone() for k, v in field.state_dict().items()}
    x, mask = make_cloud(n=2)
    with frozen_for_inference(field):
        assert not field.training
        assert not field.weight.requires_grad
        log_prob(field, x, mask, BASE, options=opts(nsteps=4))
    assert field.training and field.weight.requires_grad
    for k, v in field.state_dict().items():
        assert torch.equal(v, before[k])


def test_stable_probes_independent_of_padding_and_order():
    a = stable_probes([5, 9], 4, 3, 2, seed=11, kind="rademacher")
    b = stable_probes([9, 5], 7, 3, 2, seed=11, kind="rademacher")
    assert torch.equal(a[:, 0], b[:, 1, :4])
    assert torch.equal(a[:, 1], b[:, 0, :4])
    c = stable_probes([5], 4, 3, 2, seed=12, kind="rademacher")
    assert not torch.equal(a[:, 0], c[:, 0])


def test_physical_coordinate_jacobian():
    """Density through a bounded transform equals transformed density + log|J|."""
    tr = FeatureTransform(["eta", "phi", "logpt"], [2.5, 1.2, None],
                          [0.1, -0.05, 2.0], [0.8, 0.6, 1.3])
    g = torch.Generator().manual_seed(0)
    x = torch.stack([2.4 * torch.tanh(torch.randn(4, 5, generator=g)),
                     1.1 * torch.tanh(torch.randn(4, 5, generator=g)),
                     torch.randn(4, 5, generator=g) + 2], -1).double()
    mask = torch.ones(4, 5, dtype=torch.bool)
    mask[1, 3:] = False
    x = x * mask.unsqueeze(-1)
    base = GaussianBase.standard(3)
    lp, _ = log_prob(ZeroField(), x, mask, base, options=opts(nsteps=2),
                     input_transform=tr)
    u, _ = tr.forward(x, mask)
    # Numerical Jacobian of the elementwise map.
    h = 1e-6
    num = torch.zeros(4)
    for f in range(3):
        xp = x.clone(); xp[..., f] += h
        xm = x.clone(); xm[..., f] -= h
        d = (tr.forward(xp, mask)[0][..., f] - tr.forward(xm, mask)[0][..., f]) / (2 * h)
        num += (torch.log(d.abs().clamp(min=1e-300)) * mask).sum(-1)
    expected = base.log_prob(u, mask) + num
    np.testing.assert_allclose(lp.numpy(), expected.numpy(), atol=1e-6)
    back = tr.inverse(u, mask)
    np.testing.assert_allclose(back.numpy(), x.numpy(), atol=1e-12)


def test_exchangeable_base_density_and_sampling():
    from scipy.stats import multivariate_normal

    from omnilearned.sbi.base import ExchangeableGaussianBase, base_from_dict

    base = ExchangeableGaussianBase([0.2, -0.5, 1.0], [0.3, 0.6, 1.1], [0.9, 0.0, 0.4])
    x, mask = make_cloud(n=5, p=6, n_real=[6, 4, 1, 3, 6], seed=3)
    lp = base.log_prob(x, mask).numpy()
    for i in range(5):
        n = int(mask[i].sum())
        exact = 0.0
        for f in range(3):
            cov = base.within_stds[f] ** 2 * np.eye(n) + base.common_stds[f] ** 2 * np.ones((n, n))
            exact += multivariate_normal(np.full(n, base.means[f]), cov).logpdf(
                x[i, :n, f].numpy())
        assert abs(lp[i] - exact) < 1e-9
    # Permutation invariance and padding invariance.
    perm = torch.randperm(6)
    np.testing.assert_allclose(base.log_prob(x[:, perm], mask[:, perm]).numpy(), lp)
    # Sampling reproduces the within/common structure.
    g = torch.Generator().manual_seed(0)
    s = base.sample_like(torch.zeros(20000, 8, 3, dtype=torch.float64), None, g)
    within = s.std(1).pow(2).mean(0).sqrt().numpy()
    centre = s.mean(1).std(0).numpy()
    np.testing.assert_allclose(within, base.within_stds, rtol=0.03)
    expect = np.sqrt(np.array(base.common_stds) ** 2 + np.array(base.within_stds) ** 2 / 8)
    np.testing.assert_allclose(centre, expect, rtol=0.03)
    # Fitting recovers parameters; round trip through checkpoint metadata.
    fitted = ExchangeableGaussianBase.fit(s.numpy(), np.ones((20000, 8), bool))
    np.testing.assert_allclose(fitted.within_stds, base.within_stds, rtol=0.03)
    np.testing.assert_allclose(fitted.common_stds[0], 0.9, rtol=0.03)
    again = base_from_dict(base.to_dict())
    np.testing.assert_allclose(again.log_prob(x, mask).numpy(), lp)
    # Flow with zero field returns the exchangeable base density.
    lpf, _ = log_prob(ZeroField(), x, mask, base, options=opts(nsteps=2))
    np.testing.assert_allclose(lpf.numpy(), lp, atol=1e-12)
