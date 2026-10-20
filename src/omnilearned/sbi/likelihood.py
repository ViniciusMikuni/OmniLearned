"""Per-event conditional log densities of flow-matching generators.

Convention (shared with ``omnilearned.diffusion.generate``): the network
predicts ``v = d z_t / dt`` for ``z_t = (1 - t) x + t eps``; generation runs
from base noise at ``t = 1`` to data at ``t = 0``. Integrating a data point
forward from ``t = 0`` to ``t = 1`` gives the continuous change of variables

    log p_data(x) = log p_base(z_1) + integral_0^1 div v(z_t, t) dt.

Only real (unmasked) particle coordinates are random variables; padded
entries are held at zero and excluded from velocity, divergence and base
density. The returned values are one per input row and are never summed
here: ensemble aggregation belongs to :mod:`omnilearned.sbi.scan`.
"""

import contextlib
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from omnilearned.sbi.device import synchronize, to_cpu_float64


FIXED_STEP_SOLVERS = ("euler", "midpoint", "rk4")
ADAPTIVE_SOLVERS = ("dopri5", "bosh3", "adaptive_heun")
NFE_PER_STEP = {"euler": 1, "midpoint": 2, "rk4": 4}


@dataclass
class LikelihoodOptions:
    solver: str = "rk4"
    nsteps: int = 64
    rtol: float = 1e-5
    atol: float = 1e-5
    divergence: str = "hutchinson"  # "hutchinson" or "exact"
    num_probes: int = 4
    probe: str = "rademacher"  # "rademacher" or "gaussian"
    probe_seed: int = 0
    max_nfe: int = 4096

    def validate(self):
        if self.solver not in FIXED_STEP_SOLVERS + ADAPTIVE_SOLVERS:
            raise ValueError(f"Unknown solver '{self.solver}'")
        if self.divergence not in ("hutchinson", "exact"):
            raise ValueError(f"Unknown divergence '{self.divergence}'")
        if self.probe not in ("rademacher", "gaussian"):
            raise ValueError(f"Unknown probe distribution '{self.probe}'")
        if self.divergence == "hutchinson" and self.num_probes < 1:
            raise ValueError("num_probes must be >= 1")
        if self.solver in FIXED_STEP_SOLVERS:
            nfe = self.nsteps * NFE_PER_STEP[self.solver]
            if nfe > self.max_nfe:
                raise ValueError(
                    f"{self.solver} with {self.nsteps} steps needs {nfe} "
                    f"evaluations > max_nfe={self.max_nfe}"
                )
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


@contextlib.contextmanager
def frozen_for_inference(model):
    """Put ``model`` in eval mode with frozen parameters, then restore.

    Weights are never modified; module training flags and parameter
    ``requires_grad`` flags are restored exactly on exit.
    """
    if model is None:
        yield model
        return
    modes = [(m, m.training) for m in model.modules()]
    grads = [(p, p.requires_grad) for p in model.parameters()]
    try:
        model.eval()
        for p, _ in grads:
            p.requires_grad_(False)
        yield model
    finally:
        for m, mode in modes:
            m.training = mode
        for p, flag in grads:
            p.requires_grad_(flag)


def stable_probes(event_ids, num_particles, num_features, num_probes, seed, kind):
    """Probe vectors [K, N, P, F] derived only from (seed, event id).

    Particle ``p`` of an event always receives the same probe values, so the
    probes do not depend on batch composition, chunking or padding length.
    """
    ids = np.asarray(event_ids, dtype=np.int64).reshape(-1)
    out = np.empty((ids.shape[0], num_particles, num_probes, num_features))
    for i, eid in enumerate(ids):
        u = int(eid) & 0xFFFFFFFFFFFFFFFF
        rng = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence([int(seed), u >> 32, u & 0xFFFFFFFF]))
        )
        if kind == "rademacher":
            draw = rng.integers(0, 2, size=(num_particles, num_probes, num_features))
            out[i] = 2.0 * draw - 1.0
        else:
            out[i] = rng.standard_normal((num_particles, num_probes, num_features))
    return torch.from_numpy(out).permute(2, 0, 1, 3).contiguous()


def _divergence(v, x, mask_f, options, probes):
    """Divergence of v w.r.t. x per row, restricted to real coordinates."""
    if not v.requires_grad:  # field independent of x (e.g. constant)
        return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    if options.divergence == "exact":
        div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        flat_v = v.reshape(x.shape[0], -1)
        flat_m = mask_f.expand_as(x).reshape(x.shape[0], -1)
        for j in range(flat_v.shape[1]):
            if not bool(flat_m[:, j].any()):
                continue
            (g,) = torch.autograd.grad(
                flat_v[:, j].sum(), x, retain_graph=True, allow_unused=True
            )
            if g is None:
                continue
            div = div + g.reshape(x.shape[0], -1)[:, j] * flat_m[:, j]
        return div

    div = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    nprobe = probes.shape[0]
    for k in range(nprobe):
        eps = probes[k]
        (g,) = torch.autograd.grad(
            v, x, grad_outputs=eps, retain_graph=k < nprobe - 1, allow_unused=True
        )
        if g is None:
            continue
        div = div + (g * eps * mask_f).reshape(x.shape[0], -1).sum(-1)
    return div / nprobe


def _rhs_factory(velocity, cond, mask, mask_f, options, probes, counter):
    def rhs(t, x):
        counter[0] += 1
        with torch.enable_grad():
            x_in = (x * mask_f).detach().requires_grad_(True)
            tvec = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
            v = velocity(x_in, tvec, cond, mask) * mask_f
            div = _divergence(v, x_in, mask_f, options, probes)
        return v.detach(), div.detach()

    return rhs


def _fixed_step(rhs, x0, solver, nsteps):
    h = 1.0 / nsteps
    x = x0
    acc = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
    for i in range(nsteps):
        t = i * h
        if solver == "euler":
            v, d = rhs(t, x)
            x = x + h * v
            acc = acc + h * d
        elif solver == "midpoint":
            v1, _ = rhs(t, x)
            v2, d2 = rhs(t + 0.5 * h, x + 0.5 * h * v1)
            x = x + h * v2
            acc = acc + h * d2
        else:
            v1, d1 = rhs(t, x)
            v2, d2 = rhs(t + 0.5 * h, x + 0.5 * h * v1)
            v3, d3 = rhs(t + 0.5 * h, x + 0.5 * h * v2)
            v4, d4 = rhs(t + h, x + h * v3)
            x = x + (h / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4)
            acc = acc + (h / 6.0) * (d1 + 2 * d2 + 2 * d3 + d4)
    return x, acc


def _adaptive(rhs, x0, options):
    from torchdiffeq import odeint

    n = x0.numel()

    def func(t, state):
        v, d = rhs(t, state[:n].view_as(x0))
        return torch.cat([v.reshape(-1), d])

    state0 = torch.cat(
        [x0.reshape(-1), torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)]
    )
    sol = odeint(
        func,
        state0,
        torch.tensor([0.0, 1.0], device=x0.device, dtype=x0.dtype),
        method=options.solver,
        rtol=options.rtol,
        atol=options.atol,
        options={"max_num_steps": max(1, options.max_nfe // 6)},
    )
    final = sol[-1]
    return final[:n].view_as(x0), final[n:]


def log_prob(
    velocity,
    x,
    mask,
    base,
    cond=None,
    options=None,
    event_ids=None,
    probes=None,
    input_transform=None,
):
    """Per-row log density of ``x`` under the flow defined by ``velocity``.

    Args:
        velocity: callable ``(x_t, t, cond, mask) -> v`` with explicit mask.
        x: [N, P, F] observations in the model coordinates, or physical
            coordinates when ``input_transform`` is given.
        mask: [N, P] boolean validity mask (fixed; never inferred from x).
        base: :class:`~omnilearned.sbi.base.GaussianBase` of the checkpoint.
        cond: [N, C] conditioning rows (candidate parameters already assembled).
        options: :class:`LikelihoodOptions`.
        event_ids: [N] stable ids that select the Hutchinson probes.
        probes: optional precomputed probes [K, N, P, F] (overrides ids).
        input_transform: optional transform with ``forward(x, mask) ->
            (u, log_abs_det)``; its per-event log Jacobian is added so the
            result is a density in the physical coordinates of ``x``.

    Returns:
        (logp, diagnostics): logp is a float64 CPU tensor [N].
    """
    options = (options or LikelihoodOptions()).validate()
    device = x.device
    mask = mask.to(device=device, dtype=torch.bool)
    log_jac = torch.zeros(x.shape[0], dtype=torch.float64)
    if input_transform is not None:
        x, lj = input_transform.forward(x, mask)
        log_jac = to_cpu_float64(lj)
    mask_f = mask.to(x.dtype).unsqueeze(-1)
    x0 = x.detach() * mask_f

    if options.divergence == "hutchinson" and probes is None:
        if event_ids is None:
            raise ValueError("Hutchinson divergence needs event_ids or probes")
        # Each distinct event is drawn once (rows repeat events across candidates).
        uniq, inverse = np.unique(np.asarray(event_ids, dtype=np.int64), return_inverse=True)
        probes = stable_probes(
            uniq,
            x.shape[1],
            x.shape[2],
            options.num_probes,
            options.probe_seed,
            options.probe,
        )[:, torch.from_numpy(inverse.reshape(-1))]
    if probes is not None:
        probes = probes.to(device=device, dtype=x.dtype)

    counter = [0]
    rhs = _rhs_factory(velocity, cond, mask, mask_f, options, probes, counter)
    synchronize(device)
    start = time.perf_counter()
    if options.solver in FIXED_STEP_SOLVERS:
        z1, delta = _fixed_step(rhs, x0, options.solver, options.nsteps)
    else:
        z1, delta = _adaptive(rhs, x0, options)
    synchronize(device)
    elapsed = time.perf_counter() - start

    base_lp = base.log_prob(z1, mask).detach()
    base_lp = to_cpu_float64(base_lp)
    delta = to_cpu_float64(delta)
    logp = base_lp + delta + log_jac
    finite = torch.isfinite(logp)
    diagnostics = {
        "solver": options.solver,
        "nsteps": options.nsteps,
        "divergence": options.divergence,
        "num_probes": options.num_probes if options.divergence == "hutchinson" else 0,
        "nfe": counter[0],
        "rows": int(x.shape[0]),
        "nonfinite": int((~finite).sum()),
        "seconds": elapsed,
        "device": str(device),
        "dtype": str(x.dtype),
        "base_logp": base_lp,
        "delta_logp": delta,
        "log_jacobian": log_jac,
        "z1_rms": to_cpu_float64(
            (z1.detach() * mask_f)
            .pow(2)
            .reshape(x.shape[0], -1)
            .sum(-1)
            .div(mask_f.reshape(x.shape[0], -1).sum(-1).clamp(min=1) * x.shape[-1])
            .sqrt()
        ),
    }
    return logp, diagnostics


def sample(velocity, base, shape, mask, cond=None, nsteps=64, solver="midpoint",
           generator=None, device="cpu", noise=None):
    """Generate by integrating from base noise at t=1 to data at t=0."""
    mask = mask.to(device=device, dtype=torch.bool)
    mask_f = mask.float().unsqueeze(-1)
    if noise is None:
        template = torch.zeros(shape, device=device, dtype=torch.float32)
        if generator is not None and generator.device.type != torch.device(device).type:
            noise = base.sample_like(template.cpu(), mask.cpu(), generator).to(device)
        else:
            noise = base.sample_like(template, mask, generator)
    x = noise.to(device) * mask_f
    h = 1.0 / nsteps
    with torch.no_grad():
        for i in range(nsteps):
            t = 1.0 - i * h

            def f(tt, xx):
                tv = torch.full((xx.shape[0],), tt, device=xx.device, dtype=xx.dtype)
                return velocity(xx * mask_f, tv, cond, mask) * mask_f

            if solver == "euler":
                x = x - h * f(t, x)
            elif solver == "midpoint":
                k1 = f(t, x)
                x = x - h * f(t - 0.5 * h, x - 0.5 * h * k1)
            elif solver == "rk4":
                k1 = f(t, x)
                k2 = f(t - 0.5 * h, x - 0.5 * h * k1)
                k3 = f(t - 0.5 * h, x - 0.5 * h * k2)
                k4 = f(t - h, x - h * k3)
                x = x - (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError(f"Unknown solver '{solver}'")
    return x * mask_f
