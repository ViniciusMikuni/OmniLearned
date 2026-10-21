import math

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint

from omnilearned.data_spec import multiplicity_counts, noise_parameters
from omnilearned.packing import padded


class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return x.unsqueeze(-1) * y.to(x.dtype)


def get_eps(x, mask, dataset):
    mean, std = noise_parameters(x, dataset)
    return (torch.randn_like(x) * std + mean) * mask


def get_logsnr_alpha_sigma(time, shift=1.0):
    alpha = (1.0 - time)[:, None, None]
    sigma = time[:, None, None]
    logsnr = -2 * torch.log(sigma / (alpha + 1e-6))
    return logsnr, alpha, sigma


def get_ad_eps_hl(x):
    means = torch.tensor([6.40028, 5.0057, 0.4544, 0.6861], device=x.device)
    stds = torch.tensor(
        [0.16999, 0.318289, 0.139043, 0.193016],
        device=x.device,
    )

    if x.shape[-1] != 4:
        return torch.randn_like(x)
    means = means.view(1, -1)
    stds = stds.view(1, -1)
    eps = torch.randn_like(x) * stds + means

    return eps


def perturb(x, time, dataset, mask=None):
    mask = x[:, :, 2:3] != 0 if mask is None else mask
    eps = get_eps(x, mask, dataset)
    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time)
    z = alpha * x + sigma * eps
    return z, eps - x, torch.ones_like(x)


def perturb_hl(x, time):
    eps = get_ad_eps_hl(x)
    alpha = 1.0 - time[:, None]
    sigma = time[:, None]
    z = alpha * x + sigma * eps

    return z, eps - x, torch.ones_like(x)


def network_wrapper(model, z, condition, pid, add_info, y, time, mask=None):
    base = model.module if hasattr(model, "module") else model
    x, body_mask = base.body(
        z, condition, pid, add_info, time, mask=mask, return_mask=True
    )
    return base.generator(x, y, mask=body_mask)


@torch.no_grad()
def generate(
    model,
    y,
    shape,
    cond=None,
    pid=None,
    add_info=None,
    nsteps=256,
    multiplicity=None,
    dataset="top",
    device="cuda",
):
    if nsteps < 2:
        raise ValueError("nsteps must be at least 2")
    b, maximum, features = shape
    counts = (
        torch.full((b,), maximum, device=device, dtype=torch.long)
        if multiplicity is None
        else multiplicity_counts(multiplicity.to(device), maximum)
    )
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    state = torch.empty((int(offsets[-1]), features), device=device)
    mean, std = noise_parameters(state, dataset)
    state = torch.randn_like(state) * std + mean
    width = max(1, int(counts.max()))

    def rhs(t, packed):
        dense, mask = padded(packed, offsets, width)
        v = network_wrapper(
            model,
            dense,
            cond,
            pid[:, :width] if pid is not None else None,
            add_info[:, :width] if add_info is not None else None,
            y,
            t.expand(b),
            mask=mask,
        )
        return v[mask.squeeze(-1)]

    if state.numel():
        state = odeint(
            rhs,
            state,
            state.new_tensor([1.0, 0.0]),
            method="midpoint",
            options={"step_size": 1.0 / (nsteps - 1)},
        )[-1]
    return padded(state, offsets, maximum)[0]


@torch.no_grad()
def generate_hl(model, shape, cond=None, nsteps=512, device="cuda"):
    if nsteps < 2:
        raise ValueError("nsteps must be at least 2")
    state = get_ad_eps_hl(torch.empty(shape, device=device))

    def rhs(t, x):
        return model(x, t.expand(shape[0]), cond)

    return odeint(
        rhs,
        state,
        state.new_tensor([1.0, 0.0]),
        method="midpoint",
        options={"step_size": 1.0 / (nsteps - 1)},
    )[-1]


def estimate_max_logp(
    model,
    x,
    y,
    num_param=2,
    cond=None,
    pid=None,
    add_info=None,
    multiplicity=None,
    nsteps=32,
    seed=None,
    device="cuda",
    scan_ranges=None,
    scan_points=7,
    coarse_steps=8,
    refinements=2,
    probes=1,
    candidate_batch=4,
    dataset="top",
):
    """Independent scans with fixed other parameters and batched refinement.

    Uses your existing network_wrapper; tensor auxiliary inputs must be
    event-batched (leading dimension N). Assumes data at t=0, noise at t=1.
    scan_points: points per scan; refinements: additional narrowed scans.
    probes: fixed Hutchinson probes during refinement (coarse scan uses one).
    axes/profiles contain the final local scan, not the full original range.
    max_logp is evaluated at the combined estimates, as in the original.
    """
    if cond is None or multiplicity is None or scan_ranges is None:
        raise ValueError("cond, multiplicity and scan_ranges are required")
    counts = [scan_points] * num_param if isinstance(scan_points, int) else scan_points
    if len(counts) != num_param or len(scan_ranges) != num_param:
        raise ValueError("Provide one range and point count per parameter")
    if (
        min(counts) < 3
        or min(nsteps, coarse_steps, probes, candidate_batch) < 1
        or refinements < 1
        or any(lo >= hi for lo, hi in scan_ranges)
    ):
        raise ValueError("Invalid scan settings")

    model.eval()
    x = x.detach().to(device)
    n, m, f = x.shape
    cond = cond.detach().to(device)
    if cond.ndim != 2 or cond.shape[0] != n or cond.shape[1] < num_param + 1:
        raise ValueError("cond must have shape (N, fixed + num_param + 1)")
    theta0 = cond[:, -num_param - 1 : -1].clone()
    nparts = multiplicity_counts(multiplicity.to(device), m).reshape(n, 1)
    mask = (torch.arange(m, device=device)[None] < nparts).to(x.dtype)[..., None]
    x = x * mask
    mean, std = noise_parameters(x, dataset)
    gen = None if seed is None else torch.Generator(device=device).manual_seed(seed)
    eps = (
        2 * torch.randint(2, (probes, n, m, f), device=device, generator=gen) - 1
    ).to(x.dtype) * mask
    times = x.new_tensor([0.0, 1.0])

    def repeat(a, k):
        return None if a is None else a.to(device).repeat((k,) + (1,) * (a.ndim - 1))

    @torch.no_grad()
    def logp(theta, steps, q):
        values = []
        for th in theta.split(candidate_batch):  # (candidates, events, parameters)
            k = len(th)
            c = repeat(cond, k)
            c[:, -num_param - 1 : -1] = th.reshape(k * n, num_param)
            xx, mm = repeat(x, k), repeat(mask, k)
            yy, pp, aa = (repeat(a, k) for a in (y, pid, add_info))
            ee = eps[:q].repeat(1, k, 1, 1)

            def rhs(t, state, mm=mm, c=c, pp=pp, aa=aa, yy=yy, k=k, ee=ee):
                with torch.enable_grad():
                    z = state[0].detach().requires_grad_(True)
                    v = (
                        network_wrapper(
                            model,
                            z * mm,
                            c,
                            pp,
                            aa,
                            yy,
                            t.expand(k * n),
                            mask=mm.bool(),
                        )
                        * mm
                    )
                    div = z.new_zeros(k * n)
                    for j, e in enumerate(ee):
                        g = torch.autograd.grad(
                            (v * e).sum(), z, retain_graph=j < q - 1
                        )[0]
                        div = div + (g * e).sum((1, 2)) / q
                return v.detach(), div.detach()

            z, delta = odeint(
                rhs,
                (xx, xx.new_zeros(k * n)),
                times,
                method="rk4",
                options={"step_size": 1.0 / steps},
            )
            base = (
                -0.5 * ((z[-1] - mean) / std).square()
                - std.log()
                - 0.5 * math.log(2 * math.pi)
            )
            values.append(((base * mm).sum((1, 2)) + delta[-1]).reshape(k, n).sum(1))
        return torch.cat(values)

    best, axes, profiles = theta0.clone(), [], []
    for d, ((lo, hi), count) in enumerate(zip(scan_ranges, counts, strict=False)):
        grid = torch.linspace(lo, hi, count, device=device, dtype=cond.dtype)
        for level in range(refinements + 1):
            theta = theta0.expand(len(grid), -1, -1).clone()
            theta[:, :, d] = grid[:, None]
            prof = logp(
                theta,
                coarse_steps if level == 0 else nsteps,
                1 if level == 0 else probes,
            )
            if not torch.isfinite(prof).all():
                raise RuntimeError(
                    "Non-finite likelihood; check the model and integration"
                )
            peak = prof.argmax().item()
            if level < refinements:
                # Include the current maximum exactly, even at a boundary.
                fine = torch.linspace(
                    grid[max(0, peak - 1)].item(),
                    grid[min(len(grid) - 1, peak + 1)].item(),
                    count,
                    device=device,
                    dtype=grid.dtype,
                )
                grid = torch.cat((fine, grid[peak : peak + 1])).unique(sorted=True)
        best[:, d] = grid[peak]
        axes.append(grid.cpu())
        profiles.append(prof.cpu())

    return {
        "theta": best,
        "max_logp": logp(best[None], nsteps, probes).item(),
        "profiles": profiles,
        "axes": axes,
    }
