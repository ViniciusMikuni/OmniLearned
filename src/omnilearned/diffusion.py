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
        [0.0, 0.0, 1.11398, 1.43384],
        device=x.device
    )
    stds = torch.tensor(
        [0.270949, 0.27426, 1.30273, 1.33559],
        device=x.device,
    )

    means = means.view(1, 1, 4)
    stds = stds.view(1, 1, 4)
    eps = torch.randn_like(x) * stds + means

    return eps * mask

def get_ad_eps_hl(x):
    means = torch.tensor(
        [6.40028, 5.0057, 0.6861],
        device=x.device
    )
    stds = torch.tensor(
        [0.16999, 0.318289, 0.193016],
        device=x.device,
    )

    means = means.view(1, -1)
    stds = stds.view(1, -1)
    eps = torch.randn_like(x) * stds + means

    return eps


def perturb(x, time):
    mask = x[:, :, 2:3] != 0
    eps = get_ad_eps(x, mask)
    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time)
    z = alpha * x + sigma * eps
    return z, eps - x, torch.ones_like(x)


def perturb_hl(x, time):
    eps = get_ad_eps_hl(x)
    alpha = 1.0 - time[:,None]
    sigma = time[:,None]        
    z = alpha * x + sigma * eps

    return z, eps - x, torch.ones_like(x)



def network_wrapper(model, z, condition, pid, add_info, y, time):
    base_model = model.module if hasattr(model, "module") else model
    x = base_model.body(z, condition, pid, add_info, time)
    x = base_model.generator(x, y)
    return x


def generate(
    model, y, shape, cond=None, pid=None, add_info=None, nsteps=128, device="cuda"
) -> torch.Tensor:
    x = torch.randn(*shape).to(device)  # x_T ~ N(0, 1)
    nsample = x.shape[0]
    # Let's create the mask for the zero-padded particles
    nparts = (100 * cond[:, -1]).int().view((-1, 1)).to(device)
    max_part = x.shape[1]
    mask = torch.tile(
        torch.arange(max_part).to(device), (nparts.shape[0], 1)
    ) < torch.tile(nparts, (1, max_part))


    x_0 = get_ad_eps(x, mask.float().unsqueeze(-1))

    def ode_wrapper(t, x_t):
        time = t * torch.ones((nsample,)).to(device)
        x_t = x_t * mask.float().unsqueeze(-1)
        v = network_wrapper(model, x_t, cond, pid, add_info, y, time)
        return v

    x_t = odeint(
        func=ode_wrapper,
        y0=x_0,
        t=torch.tensor(np.linspace(1, 0, nsteps)).to(device, dtype=x_0.dtype),
        method="midpoint",
    )
    return x_t[-1]

def generate_hl(
    model, shape, cond=None, nsteps=512, device="cuda"
) -> torch.Tensor:
    
    x = torch.randn(*shape).to(device)  # x_T ~ N(0, 1)
    nsample = x.shape[0]
    
    x_0 = get_ad_eps_hl(x)

    def ode_wrapper(t, x_t):
        time = t * torch.ones((nsample,)).to(device)
        v = model(x_t,time,cond)
        return v

    x_t = odeint(
        func=ode_wrapper,
        y0=x_0,
        t=torch.tensor(np.linspace(1, 0, nsteps)).to(device, dtype=x_0.dtype),
        method="midpoint",
    )
    return x_t[-1]
