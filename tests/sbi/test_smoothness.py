"""Smoothness diagnostics discriminate smooth fields from hard-kNN fields."""

import numpy as np
import torch
import torch.nn as nn

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.smoothness import (
    finite_difference_divergence,
    knn_switches,
    logp_refinement,
    normalization,
    permutation_gap,
    reconstruction_error,
    trajectory,
)
from omnilearned.sbi.velocity import SetFlowVelocity

from fields import LinearField


class HardKNNField(nn.Module):
    """v_i = mean of the k nearest neighbours minus x_i (hard top-k selection)."""

    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def forward(self, x, t, cond, mask):
        d = torch.cdist(x[..., :2], x[..., :2])
        idx = torch.topk(-d, self.k + 1, dim=-1).indices[..., 1:]
        nb = torch.gather(x.unsqueeze(1).expand(-1, x.shape[1], -1, -1), 2,
                          idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))
        return 0.8 * (nb.mean(2) - x)


def smooth_model():
    torch.manual_seed(0)
    m = SetFlowVelocity(3, 1, dim=16, depth=2, heads=2).double()
    nn.init.normal_(m.out.weight, std=0.3)
    return m.eval()


def test_smooth_fields_pass_diagnostics():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(6, 4, 3, generator=g, dtype=torch.float64)
    mask = torch.ones(6, 4, dtype=torch.bool)
    cond = torch.zeros(6, 1, dtype=torch.float64)
    base = GaussianBase.standard(3)
    for field in [LinearField(np.eye(3) * 0.3).double(), smooth_model()]:
        rec = reconstruction_error(field, x, mask, cond)
        assert rec[64] < rec[8] and rec[64] < 1e-4
        ref = logp_refinement(field, x, mask, cond, base, steps=(8, 16, 32, 128))
        assert ref[32] < ref[8] / 8
        fd = finite_difference_divergence(field, x, mask, cond, h=1e-5)
        assert fd["max_abs_diff"] < 1e-5
        assert permutation_gap(field, x, mask, cond, base) < 1e-9


def test_hard_knn_field_shows_switches():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(32, 6, 3, generator=g, dtype=torch.float64)
    mask = torch.ones(32, 6, dtype=torch.bool)
    field = HardKNNField()
    traj = trajectory(field, x, mask, None, nsteps=32)
    assert knn_switches(traj, k=2) > 0


def test_normalization_of_smooth_model():
    m = smooth_model()
    res = normalization(m, GaussianBase.standard(3), 2, 3, torch.zeros(1, 1), n=4000,
                        nsteps=16, batch=2000)
    assert abs(res["integral"] - 1) < 4 * res["stderr"] + 0.02, res
