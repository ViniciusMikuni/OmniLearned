import torch
import torch.nn as nn
import numpy as np
from torchdiffeq import odeint


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


def get_logsnr_alpha_sigma(time, shift=1.0):
    alpha = (1.0 - time)[:, None, None]
    sigma = time[:, None, None]
    logsnr = -2 * torch.log(sigma / (alpha + 1e-6))
    return logsnr, alpha, sigma


def get_ad_eps(x, mask):
    means = torch.tensor(
        [0.00027097, 0.0000285, 0.90671477, 1.29942334], device=x.device
    )
    stds = torch.tensor(
        [0.30882706, 0.30609399, 1.23984054, 1.28986600], device=x.device
    )

    # Shape: (1, 1, 4) so it can broadcast across (B, N, 4)
    means = means.view(1, 1, 4)
    stds = stds.view(1, 1, 4)

    eps = (torch.randn_like(x) * stds + means) * mask
    return eps


def perturb(x, time, noise=1e-4):
    mask = x[:, :, 2:3] != 0
    eps = torch.randn_like(x) * mask  # eps ~ N(0, 1)
    # eps = get_ad_eps(x, mask)

    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time)
    z = alpha * x + sigma * eps
    return z, eps - x


def network_wrapper(model, z, condition, pid, add_info, y, time):
    base_model = model.module if hasattr(model, "module") else model
    x = base_model.body(z, condition, pid, add_info, time)
    x = base_model.generator(x, y)
    return x


def generate(
    model, y, shape, cond=None, pid=None, add_info=None, nsteps=64, device="cuda"
) -> torch.Tensor:
    x = torch.randn(*shape).to(device)  # x_T ~ N(0, 1)
    nsample = x.shape[0]
    # Let's create the mask for the zero-padded particles
    nparts = (100 * cond[:, -1]).int().view((-1, 1)).to(device)
    max_part = x.shape[1]
    mask = torch.tile(
        torch.arange(max_part).to(device), (nparts.shape[0], 1)
    ) < torch.tile(nparts, (1, max_part))

    # x_0 = x*mask.float().unsqueeze(-1)
    x_0 = get_ad_eps(x, mask.float().unsqueeze(-1))

    def ode_wrapper(t, x_t):
        time = t * torch.ones((nsample,)).to(device)
        v = network_wrapper(model, x_t, cond, pid, add_info, y, time)
        return v

    x_t = odeint(
        func=ode_wrapper,
        y0=x_0,
        t=torch.tensor(np.linspace(1, 0, nsteps)).to(device, dtype=x_0.dtype),
        method="midpoint",
    )
    return x_t[-1]
