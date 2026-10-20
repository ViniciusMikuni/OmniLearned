"""Analytic velocity fields with exact flow densities for tests."""

import math

import numpy as np
import torch
import torch.nn as nn


class ZeroField(nn.Module):
    def forward(self, x, t, cond, mask):
        return torch.zeros_like(x)


class TranslationField(nn.Module):
    """v = c (+ cond[:, 0] on every coordinate when use_cond)."""

    def __init__(self, c, use_cond=False):
        super().__init__()
        self.register_buffer("c", torch.as_tensor(c, dtype=torch.float32))
        self.use_cond = use_cond

    def forward(self, x, t, cond, mask):
        v = self.c.to(x.dtype).expand_as(x).clone()
        if self.use_cond:
            v = v + cond[:, :1, None].to(x.dtype)
        return v


class LinearField(nn.Module):
    """v = (alpha + beta t) * (x @ A^T), applied per particle."""

    def __init__(self, A, alpha=1.0, beta=0.0):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
        self.alpha, self.beta = alpha, beta
        self.weight = nn.Parameter(torch.ones(1))  # for state-restoration tests

    def forward(self, x, t, cond, mask):
        s = (self.alpha + self.beta * t).view(-1, 1, 1).to(x.dtype)
        return s * (x @ self.A.to(x.dtype).T) * self.weight


def gaussian_logpdf(z, means, stds):
    return (-0.5 * ((z - means) / stds) ** 2 - np.log(stds) - 0.5 * math.log(2 * math.pi))


def linear_exact_logp(x, mask, A, alpha, beta, means, stds):
    """Exact log density for LinearField: z1 = expm(k A) x with k=alpha+beta/2."""
    from scipy.linalg import expm

    k = alpha + 0.5 * beta
    E = expm(k * np.asarray(A, dtype=np.float64))
    z1 = np.asarray(x, dtype=np.float64) @ E.T
    lp = gaussian_logpdf(z1, np.asarray(means), np.asarray(stds))
    lp = (lp * mask[..., None]).reshape(x.shape[0], -1).sum(-1)
    return lp + mask.sum(-1) * k * np.trace(A)
