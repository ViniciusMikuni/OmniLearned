"""Diagnostics for whether a velocity field supports the flow density identity.

The continuous change-of-variables formula assumes a velocity field that is
(locally Lipschitz) continuous in x. Hard top-k neighbour selection and
value-derived masks can make the field piecewise, so these checks compare a
field's numerical behaviour against what a smooth field must show:

* forward/reverse reconstruction error shrinking under step refinement;
* log-density step refinement at the solver's order;
* autograd divergence agreeing with a central finite difference;
* how often k-nearest-neighbour sets change along trajectories;
* permutation invariance of the log density (exact divergence);
* normalization of the implied density by importance sampling (small clouds).
"""

import math

import numpy as np
import torch

from omnilearned.sbi.likelihood import LikelihoodOptions, log_prob, sample


def _integrate(velocity, x, mask, cond, nsteps, forward=True):
    """Midpoint integration data->base (forward) or base->data."""
    mask_f = mask.to(x.dtype).unsqueeze(-1)
    h = 1.0 / nsteps
    z = x.clone()
    with torch.no_grad():
        for i in range(nsteps):
            t = i * h if forward else 1.0 - i * h
            s = h if forward else -h

            def f(tt, zz):
                tv = torch.full((zz.shape[0],), tt, device=zz.device, dtype=zz.dtype)
                return velocity(zz * mask_f, tv, cond, mask) * mask_f

            k1 = f(t, z)
            z = z + s * f(t + 0.5 * s, z + 0.5 * s * k1)
    return z


def reconstruction_error(velocity, x, mask, cond, steps=(8, 16, 32, 64)):
    out = {}
    for n in steps:
        z = _integrate(velocity, x, mask, cond, n, True)
        xr = _integrate(velocity, z, mask, cond, n, False)
        err = ((xr - x) * mask.unsqueeze(-1)).abs().amax(dim=(1, 2))
        out[n] = float(err.max())
    return out


def logp_refinement(velocity, x, mask, cond, base, steps=(8, 16, 32, 64, 128)):
    vals = {}
    for n in steps:
        lp, _ = log_prob(velocity, x, mask, base, cond=cond,
                         options=LikelihoodOptions(solver="midpoint", nsteps=n,
                                                   divergence="exact", max_nfe=10 ** 6))
        vals[n] = lp.numpy()
    ref = vals[steps[-1]]
    return {n: float(np.abs(v - ref).max()) for n, v in vals.items() if n != steps[-1]}


def finite_difference_divergence(velocity, x, mask, cond, t=0.5, h=1e-3, npoints=None):
    """Autograd vs central-difference divergence at time t (float64 advised)."""
    mask_f = mask.to(x.dtype).unsqueeze(-1)
    tv = torch.full((x.shape[0],), t, dtype=x.dtype, device=x.device)
    xi = x.clone().requires_grad_(True)
    v = velocity(xi * mask_f, tv, cond, mask) * mask_f
    auto = torch.zeros(x.shape[0], dtype=x.dtype)
    fd = torch.zeros(x.shape[0], dtype=x.dtype)
    P, F = x.shape[1], x.shape[2]
    idx = [(p, f) for p in range(P) for f in range(F)]
    if npoints:
        idx = idx[:npoints]
    for p, f in idx:
        (g,) = torch.autograd.grad(v[:, p, f].sum(), xi, retain_graph=True)
        auto += (g[:, p, f] * mask[:, p]).detach().cpu()
        with torch.no_grad():
            xp, xm = x.clone(), x.clone()
            xp[:, p, f] += h
            xm[:, p, f] -= h
            vp = velocity(xp * mask_f, tv, cond, mask)[:, p, f]
            vm = velocity(xm * mask_f, tv, cond, mask)[:, p, f]
            fd += ((vp - vm) / (2 * h) * mask[:, p]).cpu()
    return {"max_abs_diff": float((auto - fd).abs().max()),
            "max_rel_diff": float(((auto - fd).abs() / (fd.abs() + 1e-6)).max())}


def knn_switches(x_traj, k, num_coord=2, mask=None):
    """Number of k-NN set changes along a trajectory [T, N, P, F]."""
    changes = 0
    prev = None
    for xt in x_traj:
        pts = xt[..., :num_coord]
        d = torch.cdist(pts, pts)
        if mask is not None:
            d = d + 999.0 * (~mask).float().unsqueeze(1)
        nn = torch.topk(-d, k + 1, dim=-1).indices[..., 1:].sort(-1).values
        if prev is not None:
            changes += int((nn != prev).any(-1).sum())
        prev = nn
    return changes


def trajectory(velocity, x, mask, cond, nsteps=64):
    mask_f = mask.to(x.dtype).unsqueeze(-1)
    h = 1.0 / nsteps
    z = x.clone()
    out = [z.clone()]
    with torch.no_grad():
        for i in range(nsteps):
            t = i * h
            tv = torch.full((z.shape[0],), t + 0.5 * h, dtype=z.dtype, device=z.device)
            t0 = torch.full((z.shape[0],), t, dtype=z.dtype, device=z.device)
            k1 = velocity(z * mask_f, t0, cond, mask) * mask_f
            z = z + h * velocity((z + 0.5 * h * k1) * mask_f, tv, cond, mask) * mask_f
            out.append(z.clone())
    return torch.stack(out)


def permutation_gap(velocity, x, mask, cond, base, nsteps=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[1], generator=g)
    o = LikelihoodOptions(solver="midpoint", nsteps=nsteps, divergence="exact", max_nfe=10 ** 6)
    a, _ = log_prob(velocity, x, mask, base, cond=cond, options=o)
    b, _ = log_prob(velocity, x[:, perm], mask[:, perm], base, cond=cond, options=o)
    return float((a - b).abs().max())


def normalization(velocity, base, num_particles, num_features, cond_row, n=20000,
                  nsteps=32, seed=0, batch=2000):
    """Importance-sampling estimate of the integral of the model density.

    Proposal: Gaussian matched to model samples (inflated by 1.3). Returns
    the estimate and its standard error; a valid density integrates to 1.
    """
    mask = torch.ones(min(batch, n), num_particles, dtype=torch.bool)
    g = torch.Generator().manual_seed(seed)
    dtype = next(velocity.parameters()).dtype if any(True for _ in velocity.parameters()) \
        else torch.float64
    cond_row = cond_row.to(dtype)
    cond = cond_row.expand(mask.shape[0], -1)
    noise = base.sample_like(torch.zeros(mask.shape[0], num_particles, num_features,
                                         dtype=dtype), mask, g)
    xs = sample(velocity, base, noise.shape, mask, cond=cond, nsteps=nsteps,
                noise=noise).double()
    flat = xs.reshape(xs.shape[0], -1)
    mu = flat.mean(0)
    cov = torch.cov(flat.T) * 1.3 ** 2 + 1e-6 * torch.eye(flat.shape[1], dtype=flat.dtype)
    prop = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
    ratios = []
    o = LikelihoodOptions(solver="rk4", nsteps=nsteps, divergence="exact", max_nfe=10 ** 6)
    for s in range(0, n, batch):
        m = min(batch, n - s)
        y = prop.sample((m,))
        lq, _ = log_prob(velocity, y.reshape(m, num_particles, num_features).to(dtype),
                         torch.ones(m, num_particles, dtype=torch.bool), base,
                         cond=cond_row.expand(m, -1), options=o)
        ratios.append((lq - prop.log_prob(y)).exp())
    r = torch.cat(ratios).numpy()
    return {"integral": float(r.mean()), "stderr": float(r.std() / math.sqrt(r.size)),
            "samples": int(r.size)}
