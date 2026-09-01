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


def get_eps(x, mask, dataset):
    if dataset == "jetnet":
        return get_jetnet_eps(x, mask)
    elif dataset == "lhco":
        return get_ad_eps(x, mask)
    else:
        return torch.randn_like(x) * mask


def get_logsnr_alpha_sigma(time, shift=1.0):
    alpha = (1.0 - time)[:, None, None]
    sigma = time[:, None, None]
    logsnr = -2 * torch.log(sigma / (alpha + 1e-6))
    return logsnr, alpha, sigma


def get_ad_eps(x, mask):
    means = torch.tensor([0.0, 0.0, 1.11398, 1.43384], device=x.device)
    stds = torch.tensor(
        [0.270949, 0.27426, 1.30273, 1.33559],
        device=x.device,
    )

    if x.shape[-1] == 4:
        means = means.view(1, 1, 4)
        stds = stds.view(1, 1, 4)
        eps = torch.randn_like(x) * stds + means
    else:
        eps = torch.randn_like(x)

    return eps * mask


def get_jetnet_eps(x, mask):
    means = torch.tensor([0.0, 0.0, 2.92904309e00, 3.14843288e00], device=x.device)
    stds = torch.tensor(
        [0.11580968, 0.11667726, 0.95442127, 0.98245158],
        device=x.device,
    )

    if x.shape[-1] == 4:
        means = means.view(1, 1, 4)
        stds = stds.view(1, 1, 4)
        eps = torch.randn_like(x) * stds + means
    else:
        eps = torch.randn_like(x)

    return eps * mask


def get_ad_eps_hl(x):
    means = torch.tensor([6.40028, 5.0057, 0.4544, 0.6861], device=x.device)
    stds = torch.tensor(
        [0.16999, 0.318289, 0.139043, 0.193016],
        device=x.device,
    )

    means = means.view(1, -1)
    stds = stds.view(1, -1)
    eps = torch.randn_like(x) * stds + means

    return eps


def perturb(x, time, dataset):
    mask = x[:, :, 2:3] != 0
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


def network_wrapper(model, z, condition, pid, add_info, y, time):
    base_model = model.module if hasattr(model, "module") else model
    x = base_model.body(z, condition, pid, add_info, time)
    x = base_model.generator(x, y)
    return x


def generate(
    model,
    y,
    shape,
    cond=None,
    pid=None,
    add_info=None,
    nsteps=128,
    multiplicity=None,
    dataset="top",
    device="cuda",
) -> torch.Tensor:
    x = torch.randn(*shape).to(device)  # x_T ~ N(0, 1)
    nsample = x.shape[0]
    # Let's create the mask for the zero-padded particles
    nparts = (100 * multiplicity).int().view((-1, 1)).to(device)

    max_part = x.shape[1]
    mask = torch.tile(
        torch.arange(max_part).to(device), (nparts.shape[0], 1)
    ) < torch.tile(nparts, (1, max_part))

    x_0 = get_eps(x, mask.float().unsqueeze(-1), dataset)

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


def generate_hl(model, shape, cond=None, nsteps=512, device="cuda") -> torch.Tensor:
    x = torch.randn(*shape).to(device)  # x_T ~ N(0, 1)
    nsample = x.shape[0]

    x_0 = get_ad_eps_hl(x)

    def ode_wrapper(t, x_t):
        time = t * torch.ones((nsample,)).to(device)
        v = model(x_t, time, cond)
        return v

    x_t = odeint(
        func=ode_wrapper,
        y0=x_0,
        t=torch.tensor(np.linspace(1, 0, nsteps)).to(device, dtype=x_0.dtype),
        method="midpoint",
    )
    return x_t[-1]


# def estimate_max_logp(
#     model, x, y, num_param=2, cond=None, pid=None, add_info=None,
#     multiplicity=None, nsteps=32, seed=None, device="cuda",
#     scan_ranges=None, scan_points=21, n_time_bins=20, time_chunk=None,
#     # ---- new, defaulted to preserve original behavior ----
#     n_noise=1, coord_passes=1, parts_per_unit=100,
# ):
#     """
#     Scan `num_param` conditioning params independently and pick the values that
#     minimize the flow-matching loss (returned as a negated 'score').

#     NOTE: 'max_logp' is the NEGATIVE avg FM loss, not a real log-likelihood;
#     the key name is kept only for backward compatibility.

#     Defaults (n_noise=1, coord_passes=1, parts_per_unit=100) reproduce the
#     original behavior. n_noise>1 averages over several shared eps draws to cut
#     variance; coord_passes>1 enables coordinate descent for correlated params.
#     """
#     assert cond is not None, "cond must contain the params to scan."
#     assert scan_ranges is not None, "scan_ranges required: list of (min,max)."
#     assert multiplicity is not None, "multiplicity required (builds the mask)."

#     return _run(
#         model, x, y, num_param, cond, pid, add_info, multiplicity, seed,
#         device, scan_ranges, scan_points, n_time_bins, time_chunk,
#         n_noise, coord_passes, parts_per_unit,
#     )


# @torch.no_grad()
# def _run(model, x, y, num_param, cond, pid, add_info, multiplicity, seed,
#          device, scan_ranges, scan_points, n_time_bins, time_chunk,
#          n_noise, coord_passes, parts_per_unit):

#     x = x.to(device)
#     nsample, max_part, nfeat = x.shape

#     # mask (same as generate())
#     nparts = (parts_per_unit * multiplicity).int().view(-1, 1).to(device)
#     ar = torch.arange(max_part, device=device)
#     mask_f = (ar.unsqueeze(0) < nparts).float().unsqueeze(-1)   # [N,P,1]
#     x = x * mask_f
#     n_real = torch.clamp(mask_f.sum() * nfeat, min=1.0)

#     # eps: [K,N,P,F], drawn once, shared across all t and thetas
#     if seed is not None:
#         with torch.random.fork_rng(devices=[device] if str(device) != "cpu" else []):
#             torch.manual_seed(seed)
#             if str(device) != "cpu":
#                 torch.cuda.manual_seed_all(seed)
#             eps = torch.stack([get_jetnet_eps(x, mask_f) for _ in range(n_noise)])
#     else:
#         eps = torch.stack([get_jetnet_eps(x, mask_f) for _ in range(n_noise)])
#     K = eps.shape[0]
#     target = (eps - x) * mask_f                                 # [K,N,P,F]

#     # split cond: [...fixed..., theta(num_param), mult(1)]
#     cond = cond.to(device)
#     assert cond.shape[-1] >= num_param + 1, "cond last dim too small."
#     mult_col   = cond[..., -1:].detach()
#     theta0     = cond[..., -(num_param + 1):-1].detach()
#     cond_fixed = cond[..., :-(num_param + 1)].detach()

#     def assemble(theta):
#         parts = ([cond_fixed] if cond_fixed.shape[-1] else []) + [theta, mult_col]
#         return torch.cat(parts, dim=-1)

#     t_grid = (torch.arange(n_time_bins, device=device, dtype=x.dtype) + 0.5) / n_time_bins
#     tc = time_chunk or n_time_bins

#     def expand(t, reps):
#         if not torch.is_tensor(t):
#             return t
#         return t.repeat_interleave(reps, dim=0) if False else \
#                t.unsqueeze(0).expand(reps, *t.shape).reshape(reps * t.shape[0], *t.shape[1:])

#     def score(theta):
#         cond_full = assemble(theta)                            # [N,C]
#         loss = torch.zeros((), device=device, dtype=x.dtype)
#         for s in range(0, n_time_bins, tc):
#             tb = t_grid[s:s + tc]
#             Tb = tb.shape[0]
#             reps = Tb * K
#             # z = (1-t)x + t*eps over (Tb,K,N)
#             tcol = tb.view(Tb, 1, 1, 1, 1)
#             z = ((1 - tcol) * x + tcol * eps.unsqueeze(0)) * mask_f      # [Tb,K,N,P,F]
#             z = z.reshape(reps * nsample, max_part, nfeat)

#             time  = tb.view(Tb, 1, 1).expand(Tb, K, nsample).reshape(-1)
#             condb = cond_full.expand(Tb, K, nsample, -1).reshape(reps * nsample, -1)
#             maskb = mask_f.expand(Tb, K, nsample, max_part, 1).reshape(reps * nsample, max_part, 1)
#             tgtb  = target.expand(Tb, K, nsample, max_part, nfeat).reshape(reps * nsample, max_part, nfeat)

#             v = network_wrapper(model, z, condb,
#                                 expand(pid, reps), expand(add_info, reps),
#                                 expand(y, reps), time).float()
#             loss = loss + (((v - tgtb) ** 2) * maskb).sum()
#         return -(loss / (n_time_bins * K * n_real))

#     if isinstance(scan_points, int):
#         scan_points = [scan_points] * num_param
#     axes = [torch.linspace(lo, hi, n, device=device)
#             for (lo, hi), n in zip(scan_ranges, scan_points)]

#     best = theta0.clone()
#     profiles = [None] * num_param
#     for _ in range(coord_passes):
#         for d in range(num_param):
#             prof = torch.empty(len(axes[d]), device=device)
#             for i, val in enumerate(axes[d]):
#                 th = best.clone()
#                 th[..., d] = val
#                 prof[i] = score(th)
#             profiles[d] = prof
#             best[..., d] = axes[d][prof.argmax()]

#     return {
#         "theta": best,
#         "max_logp": score(best).item(),
#         "profiles": [p.cpu() for p in profiles],
#         "axes": [a.cpu() for a in axes],
#     }


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
    scan_ranges=None,  # list of (min, max), one per scanned parameter
    scan_points=21,  # int (same for all) or list[int] per parameter
):
    """
    Estimate the maximum log-likelihood of an observed point cloud `x` by
    scanning `num_param` conditioning parameters INDEPENDENTLY.

    For each scanned parameter, a 1D grid of `scan_points` values is evaluated
    while the other scanned parameters are held at their default (theta0).
    This costs sum(scan_points) evaluations instead of a full product grid.

    Layout of `cond` (last dim):
        [ ...fixed columns... , theta (num_param) , multiplicity (1) ]
    The multiplicity column is always last and is NOT scanned.

    The base distribution matches get_jetnet_eps: a per-feature diagonal
    Gaussian with the given means/stds when nfeat == 4, otherwise a standard
    normal. The -log(std) Jacobian term is included for correct normalization.

    Only likelihood VALUES are needed, so everything runs under no_grad with
    the fast attention kernel (no double-backward).

    Returns a dict with the best-fit parameters, the max log-likelihood,
    per-parameter 1D log-likelihood profiles, and the grid axes.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    x = x.to(device)
    nsample = x.shape[0]
    max_part = x.shape[1]
    nfeat = x.shape[-1]

    # ---- rebuild the mask exactly as in generate() ------------------------
    nparts = (100 * multiplicity).int().view((-1, 1)).to(device)
    mask = torch.tile(
        torch.arange(max_part, device=device), (nparts.shape[0], 1)
    ) < torch.tile(nparts, (1, max_part))
    mask_f = mask.float().unsqueeze(-1)
    x = x * mask_f

    # ---- base distribution parameters (must match get_eps) ---------
    if nfeat == 4:
        base_means = torch.tensor(
            [0.0, 0.0, 2.92904309e00, 3.14843288e00], device=device
        ).view(1, 1, 4)
        base_stds = torch.tensor(
            [0.11580968, 0.11667726, 0.95442127, 0.98245158], device=device
        ).view(1, 1, 4)
    else:
        base_means = torch.zeros(1, 1, nfeat, device=device)
        base_stds = torch.ones(1, 1, nfeat, device=device)

    # ---- freeze the Hutchinson probe (Rademacher) -------------------------
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        eps = torch.randint(0, 2, x.shape, generator=gen, device=device).float()
    else:
        eps = torch.randint(0, 2, x.shape, device=device).float()
    eps = (eps * 2.0 - 1.0) * mask_f

    # ---- split cond: [ ...fixed... , theta (num_param) , multiplicity ] ---
    if cond is None:
        raise ValueError("cond must contain the parameters to be scanned.")
    cond = cond.to(device)
    if cond.shape[-1] < num_param + 1:
        raise ValueError(f"cond last dim ({cond.shape[-1]}) must be >= num_param + 1.")

    mult_col = cond[..., -1:].detach()
    theta0 = cond[..., -(num_param + 1) : -1].detach()  # default theta values
    cond_fixed = cond[..., : -(num_param + 1)].detach()

    def assemble_cond(theta):
        parts = []
        if cond_fixed.shape[-1] > 0:
            parts.append(cond_fixed)
        parts.append(theta)
        parts.append(mult_col)
        return torch.cat(parts, dim=-1)

    # ---- likelihood evaluation (value only) -------------------------------
    t_span = torch.tensor([0.0, 1.0], device=device, dtype=x.dtype)
    step_size = 1.0 / nsteps
    xnum = x.numel()

    def logp_value(theta):
        """Return summed log p(x | theta) over the batch."""
        cond_full = assemble_cond(theta)

        def velocity(t, x_t):
            time = t * torch.ones((nsample,), device=device)
            x_t = x_t * mask_f
            return network_wrapper(model, x_t, cond_full, pid, add_info, y, time)

        def augmented(t, state):
            x_t = state[:xnum].view_as(x)
            with torch.enable_grad():
                x_in = x_t.requires_grad_(True)
                v = velocity(t, x_in)
                (grad,) = torch.autograd.grad((v * eps).sum(), x_in, create_graph=False)
            div = ((grad * eps) * mask_f).sum(dim=tuple(range(1, x_t.dim())))
            return torch.cat([v.reshape(-1), div.reshape(-1)])

        state0 = torch.cat([x.reshape(-1), torch.zeros(nsample, device=device)])
        sol = odeint(
            augmented,
            state0,
            t_span,
            method="rk4",
            options={"step_size": step_size},
        )
        final = sol[-1]
        x_1 = final[:xnum].view_as(x)
        delta_logp = final[xnum:]

        # diagonal-Gaussian base log density, summed over real points only
        log_pz = (
            -0.5 * ((x_1 - base_means) / base_stds) ** 2
            - torch.log(base_stds)
            - 0.5 * np.log(2.0 * np.pi)
        )
        log_pz = (log_pz * mask_f).sum(dim=tuple(range(1, x_1.dim())))

        return (log_pz + delta_logp).sum()

    # ---- build per-parameter 1D grids -------------------------------------
    if scan_ranges is None:
        raise ValueError("scan_ranges must be provided: list of (min, max) per param.")
    if isinstance(scan_points, int):
        scan_points = [scan_points] * num_param

    axes = [
        torch.linspace(lo, hi, n, device=device)
        for (lo, hi), n in zip(scan_ranges, scan_points)
    ]

    # ---- scan each parameter independently --------------------------------
    profiles = []  # 1D logp profile per parameter
    best_theta = theta0.clone()

    with torch.no_grad():
        for d in range(num_param):
            grid_d = axes[d]
            prof = torch.empty(grid_d.shape[0], device=device)
            for i, val in enumerate(grid_d):
                theta_i = theta0.clone()
                theta_i[..., d] = val  # vary only parameter d
                prof[i] = logp_value(theta_i)
            profiles.append(prof)
            best_theta[..., d] = grid_d[torch.argmax(prof)]

    # ---- final likelihood at the combined best-fit ------------------------
    with torch.no_grad():
        max_logp = logp_value(best_theta).item()

    return {
        "theta": best_theta,  # best-fit scanned params
        "max_logp": max_logp,  # LL at combined best-fit
        "profiles": [p.cpu() for p in profiles],  # 1D LL profile per param
        "axes": [a.cpu() for a in axes],  # grid axis per param
    }
