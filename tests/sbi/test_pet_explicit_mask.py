"""Upstream PET2 generator with an explicit validity mask."""

import numpy as np
import torch

from omnilearned.diffusion import network_wrapper
from omnilearned.network import PET2
from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.likelihood import LikelihoodOptions, log_prob
from omnilearned.sbi.velocity import PETVelocity, build_velocity


def small_pet(use_local=True):
    torch.manual_seed(0)
    return PET2(
        input_dim=3, mode="generator", conditional=True, cond_dim=2, num_classes=1,
        base_dim=32, num_transformers=2, num_transformers_head=1, num_heads=2,
        num_tokens=2, K=4, num_coord=2, use_local=use_local, use_int=False,
    ).eval()


def test_explicit_mask_matches_legacy_when_no_zero_crossing():
    pet = small_pet()
    x = torch.randn(4, 10, 3)
    x[:, :, 2] = x[:, :, 2].abs() + 0.1  # feature 2 nonzero for real particles
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[1, 6:] = False
    x = x * mask.unsqueeze(-1)
    t = torch.rand(4)
    cond = torch.randn(4, 2)
    with torch.no_grad():
        legacy = network_wrapper(pet, x, cond, None, None, None, t)
        explicit = PETVelocity(pet)(x, t, cond, mask)
    np.testing.assert_allclose(explicit.numpy(), legacy.numpy(), atol=1e-6)


def test_explicit_mask_keeps_real_particle_with_zero_feature():
    pet = small_pet()
    x = torch.randn(2, 6, 3)
    x[0, 2, 2] = 0.0  # real particle whose third feature crosses zero
    mask = torch.ones(2, 6, dtype=torch.bool)
    t = torch.rand(2)
    cond = torch.randn(2, 2)
    with torch.no_grad():
        legacy = network_wrapper(pet, x, cond, None, None, None, t)
        explicit = PETVelocity(pet)(x, t, cond, mask)
    assert torch.all(legacy[0, 2] == 0)  # legacy drops it as padding
    assert torch.any(explicit[0, 2] != 0)


def test_pet_velocity_likelihood_runs_and_is_finite():
    model = build_velocity({"architecture": "pet2", "input_dim": 3, "cond_dim": 2,
                            "base_dim": 32, "num_transformers": 1,
                            "num_transformers_head": 1, "num_heads": 2,
                            "num_tokens": 1, "K": 3, "num_coord": 2,
                            "use_int": False})
    x = torch.randn(3, 8, 3)
    mask = torch.ones(3, 8, dtype=torch.bool)
    o = LikelihoodOptions(solver="euler", nsteps=4, divergence="exact")
    lp, d = log_prob(model.eval(), x, mask, GaussianBase.standard(3),
                     cond=torch.zeros(3, 2), options=o)
    assert torch.isfinite(lp).all() and d["nfe"] == 4
