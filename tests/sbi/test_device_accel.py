"""Accelerator (MPS, CUDA) execution checked against CPU references.

Each test runs once per available accelerator and is skipped otherwise.
"""

import numpy as np
import pytest
import torch

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.device import resolve_device, mps_available
from omnilearned.sbi.flow import TrainOptions, flow_matching_loss, train_flow
from omnilearned.sbi.likelihood import LikelihoodOptions, log_prob
from omnilearned.sbi.velocity import SetFlowVelocity

needs_mps = pytest.mark.skipif(not mps_available(), reason="MPS not available")
ACCELERATORS = [
    pytest.param("mps", marks=needs_mps, id="mps"),
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                  reason="CUDA not available"),
                 id="cuda"),
]
accelerators = pytest.mark.parametrize("dev", ACCELERATORS)


def test_resolve_device_rules():
    assert resolve_device("cpu").device == "cpu"
    info = resolve_device("auto")
    assert info.device in ("cpu", "mps", "cuda:0")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            resolve_device("cuda")
        fb = resolve_device("cuda", allow_fallback=True)
        assert fb.fallback and fb.device == "cpu"
    with pytest.raises(ValueError):
        resolve_device("tpu")


def _model(seed=0):
    torch.manual_seed(seed)
    m = SetFlowVelocity(3, 2, dim=64, depth=3, heads=4)
    torch.nn.init.normal_(m.out.weight, std=0.2)
    for blk in m.blocks:
        torch.nn.init.normal_(blk.mod.weight, std=0.05)
    return m


@needs_mps
def test_mps_to_cpu_float64_conversion():
    """Guard against the fused MPS->CPU float64 conversion returning zeros."""
    from omnilearned.sbi.device import to_cpu_float64

    a = torch.arange(1, 6, dtype=torch.float32, device="mps")
    np.testing.assert_array_equal(to_cpu_float64(a).numpy(), [1, 2, 3, 4, 5])
    fused = a.to("cpu", torch.float64)
    if not torch.equal(fused, torch.arange(1, 6, dtype=torch.float64)):
        # Known torch/MPS defect (observed with torch 2.14.1); recorded, not hidden.
        print(f"fused MPS->CPU float64 conversion is broken in torch {torch.__version__}")


@accelerators
def test_likelihood_matches_cpu(dev):
    model = _model()
    g = torch.Generator().manual_seed(1)
    x = torch.randn(64, 32, 3, generator=g)
    mask = torch.ones(64, 32, dtype=torch.bool)
    mask[::3, 20:] = False
    x = x * mask.unsqueeze(-1)
    cond = torch.randn(64, 2, generator=g)
    base = GaussianBase([0.1, -0.2, 0.3], [1.0, 0.9, 1.2])
    o = LikelihoodOptions(solver="midpoint", nsteps=16, divergence="hutchinson",
                          num_probes=2)
    ids = np.arange(64)
    cpu, _ = log_prob(model, x, mask, base, cond=cond, options=o, event_ids=ids)
    m = model.to(dev)
    acc, d = log_prob(m, x.to(dev), mask.to(dev), base, cond=cond.to(dev),
                      options=o, event_ids=ids)
    assert d["device"].startswith(dev) and d["dtype"] == "torch.float32"
    assert acc.dtype == torch.float64 and acc.device.type == "cpu"
    # float32 on both sides; differences are roundoff in a 96-dim density.
    np.testing.assert_allclose(acc.numpy(), cpu.numpy(), atol=2e-3, rtol=0)
    # Likelihood differences between candidates agree more tightly.
    cond2 = cond + 0.1
    cpu2, _ = log_prob(model.cpu(), x, mask, base, cond=cond2, options=o, event_ids=ids)
    acc2, _ = log_prob(model.to(dev), x.to(dev), mask.to(dev), base,
                       cond=cond2.to(dev), options=o, event_ids=ids)
    np.testing.assert_allclose((acc2 - acc).numpy(), (cpu2 - cpu).numpy(), atol=1e-3)


@accelerators
def test_input_derivatives_match_cpu(dev):
    model = _model(2)
    x = torch.randn(8, 16, 3)
    mask = torch.ones(8, 16, dtype=torch.bool)
    t = torch.rand(8)
    cond = torch.randn(8, 2)
    eps = torch.randn(8, 16, 3)
    grads = []
    for d in ["cpu", dev]:
        m = model.to(d)
        xi = x.to(d).requires_grad_(True)
        v = m(xi, t.to(d), cond.to(d), mask.to(d))
        (g,) = torch.autograd.grad(v, xi, grad_outputs=eps.to(d))
        grads.append(g.cpu())
    np.testing.assert_allclose(grads[1].numpy(), grads[0].numpy(), atol=1e-4)


@accelerators
def test_training_steps(dev):
    g = torch.Generator().manual_seed(0)
    n = 512
    cond = torch.rand(n, 2, generator=g) * 2 - 1
    u = torch.randn(n, 8, 3, generator=g) * 0.5 + cond[:, None, :1]
    mask = torch.ones(n, 8, dtype=torch.bool)
    data = {"u": u, "mask": mask, "cond": cond}
    model = SetFlowVelocity(3, 2, dim=32, depth=2, heads=2)
    opts = TrainOptions(steps=60, batch_size=64, val_every=30, warmup_steps=5,
                        device=dev, lr=1e-3)
    best, hist = train_flow(model, GaussianBase.standard(3), data, data, opts,
                            log=lambda s: None)
    assert hist["device"]["device"] == dev
    assert hist["val_loss"][-1] < hist["val_loss"][0] + 1e-6
    assert all(np.isfinite(hist["train_loss"]))
    # Loss of the returned model evaluates on CPU and the accelerator consistently.
    gg = torch.Generator().manual_seed(3)
    t = torch.rand(128, generator=gg)
    eps = torch.randn(128, 8, 3, generator=gg)
    l_cpu = flow_matching_loss(best, u[:128], mask[:128], cond[:128],
                               GaussianBase.standard(3), t=t, eps=eps)
    bm = best.to(dev)
    l_acc = flow_matching_loss(bm, u[:128].to(dev), mask[:128].to(dev),
                               cond[:128].to(dev), GaussianBase.standard(3),
                               t=t.to(dev), eps=eps.to(dev))
    assert abs(float(l_cpu) - float(l_acc)) < 1e-4


@accelerators
def test_flow_surrogate_sample_and_density(tmp_path, dev):
    from omnilearned.sbi.flow import FlowSurrogate
    from omnilearned.sbi.parameters import ConditionLayout
    from omnilearned.sbi.transforms import FeatureTransform

    layout = ConditionLayout.from_bounds(["a", "b"], [0, 0], [1, 2])
    tr = FeatureTransform(["eta", "phi", "lpt"], [2.5, 1.2, None], [0, 0, 1], [0.5, 0.4, 1])
    sur = FlowSurrogate(_model(4), GaussianBase.standard(3), tr, layout)
    path = tmp_path / "s.pt"
    sur.save(path)
    res = {}
    for d in ["cpu", dev]:
        s = FlowSurrogate.load(path, device=d)
        x = s.sample([0.5, 1.0], 16, 32, seed=3, nsteps=8)
        assert x.dtype == np.float32 and np.isfinite(x).all()
        lp, _ = s.log_prob(x.astype(np.float64), np.ones((16, 32), bool),
                           np.array([[0.5, 1.0], [0.2, 1.5]]), np.arange(16),
                           LikelihoodOptions(solver="midpoint", nsteps=8, num_probes=1))
        res[d] = (x, lp)
    np.testing.assert_allclose(res[dev][0], res["cpu"][0], atol=2e-3)
    np.testing.assert_allclose(res[dev][1], res["cpu"][1], atol=5e-2)
