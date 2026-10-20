"""Single-process conditional flow-matching training and checkpoint bundle.

Training, generation and likelihood evaluation share the checkpoint's base
specification, feature transform, condition layout and explicit masks. The
trainer runs on one explicitly selected device (CPU, CUDA or MPS) without
distributed setup; model tensors are float32.
"""

import copy
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass, asdict

import numpy as np
import torch

from omnilearned.sbi.base import base_from_dict
from omnilearned.sbi.device import resolve_device, synchronize
from omnilearned.sbi.likelihood import (
    frozen_for_inference,
    log_prob,
    sample,
)
from omnilearned.sbi.parameters import ConditionLayout
from omnilearned.sbi.transforms import FeatureTransform
from omnilearned.sbi.velocity import build_velocity, velocity_config


CHECKPOINT_SCHEMA = "omnilearned-sbi-flow-v1"


@dataclass
class TrainOptions:
    steps: int = 2000
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    val_every: int = 250
    val_batches: int = 8
    seed: int = 0
    val_seed: int = 1
    device: str = "auto"
    allow_device_fallback: bool = False

    def to_dict(self):
        return asdict(self)


def flow_matching_loss(model, x, mask, cond, base, generator=None, t=None, eps=None):
    """Masked conditional flow-matching loss (mean over real coordinates)."""
    mask_f = mask.to(x.dtype).unsqueeze(-1)
    if t is None:
        t = torch.rand(x.shape[0], device=x.device, generator=generator)
    if eps is None:
        eps = base.sample_like(x, mask, generator)
    tt = t.view(-1, 1, 1)
    z = ((1 - tt) * x + tt * eps) * mask_f
    target = (eps - x) * mask_f
    v = model(z, t, cond, mask)
    err = ((v - target) ** 2 * mask_f).reshape(x.shape[0], -1).sum(-1)
    ncoord = mask_f.reshape(x.shape[0], -1).sum(-1) * x.shape[-1]
    return (err / ncoord.clamp(min=1)).mean()


class FlowSurrogate:
    """A conditional generator O with everything needed to use it."""

    def __init__(self, model, base, transform, layout, metadata=None, device="cpu"):
        self.model = model
        self.base = base
        self.transform = transform
        self.layout = layout
        self.metadata = dict(metadata or {})
        self.device = torch.device(device)

    # ---- density --------------------------------------------------------
    def log_prob(self, x_phys, mask, candidates, event_ids, options,
                 candidate_names=None, context=None):
        """Per-(event, candidate) physical-coordinate log densities.

        Args:
            x_phys: [N, P, F] float numpy/tensor observations.
            candidates: [M, n_unknown] candidate parameter values.
        Returns:
            (float64 array [M, N], diagnostics)
        """
        x = torch.as_tensor(np.asarray(x_phys), dtype=torch.float32)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        u, logj = self.transform.forward(x.double(), mask_t)
        u = u.float()
        candidates = np.atleast_2d(candidates)
        M, N = candidates.shape[0], x.shape[0]
        ctx = None
        if context is not None:
            ctx = np.repeat(np.asarray(context)[None], M, 0).reshape(M * N, -1)
        cond = self.layout.assemble(
            np.repeat(candidates, N, axis=0), candidate_names, context=ctx
        )
        xs = u.repeat(M, 1, 1).to(self.device)
        ms = mask_t.repeat(M, 1).to(self.device)
        ids = np.tile(np.asarray(event_ids), M)
        with frozen_for_inference(self.model):
            lp, diag = log_prob(
                self.model,
                xs,
                ms,
                self.base,
                cond=torch.from_numpy(cond).to(self.device),
                options=options,
                event_ids=ids,
            )
        lp = lp + logj.repeat(M)
        return lp.numpy().reshape(M, N), diag

    def log_prob_paired(self, x_phys, mask, rows, event_ids, options,
                        candidate_names=None, context=None):
        """Log density of event i at its own parameter row ``rows[i]`` -> [N]."""
        x = torch.as_tensor(np.asarray(x_phys), dtype=torch.float32)
        mask_t = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        u, logj = self.transform.forward(x.double(), mask_t)
        cond = self.layout.assemble(np.asarray(rows), candidate_names, context=context)
        with frozen_for_inference(self.model):
            lp, diag = log_prob(self.model, u.float().to(self.device),
                                mask_t.to(self.device), self.base,
                                cond=torch.from_numpy(cond).to(self.device),
                                options=options, event_ids=np.asarray(event_ids))
        return (lp + logj).numpy(), diag

    # ---- generation -----------------------------------------------------
    def sample(self, candidate, n, num_particles, seed, nsteps=64, solver="midpoint",
               candidate_names=None, batch=4096):
        """Physical-coordinate samples at one candidate (fixed multiplicity)."""
        F = self.base.num_features
        gen = torch.Generator().manual_seed(int(seed))
        out = []
        with frozen_for_inference(self.model):
            for s in range(0, n, batch):
                m = min(batch, n - s)
                mask = torch.ones(m, num_particles, dtype=torch.bool)
                noise = self.base.sample_like(
                    torch.zeros(m, num_particles, F), mask, gen
                )
                cond = self.layout.assemble_torch(
                    np.repeat(np.atleast_2d(candidate), m, 0), self.device,
                    candidate_names=candidate_names,
                )
                u = sample(self.model, self.base, (m, num_particles, F), mask,
                           cond=cond, nsteps=nsteps, solver=solver,
                           device=self.device, noise=noise)
                x = self.transform.inverse(u.cpu().double(), mask)
                out.append(x.float().numpy())
        return np.concatenate(out)

    # ---- persistence ----------------------------------------------------
    def state(self):
        return {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": velocity_config(self.model),
            "model_state": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "base": self.base.to_dict(),
            "transform": self.transform.to_dict(),
            "condition_layout": self.layout.to_dict(),
            "metadata": self.metadata,
        }

    def save(self, path):
        buf = io.BytesIO()
        torch.save(self.state(), buf)
        data = buf.getvalue()
        with open(path, "wb") as f:
            f.write(data)
        return hashlib.sha256(data).hexdigest()

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError(f"{path} is not a {CHECKPOINT_SCHEMA} checkpoint")
        model = build_velocity(state["architecture"])
        model.load_state_dict(state["model_state"])
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        meta = dict(state["metadata"])
        meta["checkpoint_sha256"] = sha
        return cls(
            model,
            base_from_dict(state["base"]),
            FeatureTransform.from_dict(state["transform"]),
            ConditionLayout.from_dict(state["condition_layout"]),
            meta,
            device,
        )


def _lr_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    return f


def train_flow(model, base, train, val, options, log=print, on_best=None):
    """Train a velocity model by flow matching; select by validation loss only.

    Args:
        model: velocity module ``(x, t, cond, mask) -> v``.
        train, val: dicts with tensors ``u`` [N, P, F] (model coordinates),
            ``mask`` [N, P] and ``cond`` [N, C].
        on_best: callback ``(ema_model, step, val_loss)`` when validation
            improves (e.g. to save a checkpoint).
    Returns:
        (best EMA model on CPU, history dict)
    """
    info = resolve_device(options.device, options.allow_device_fallback)
    device = info.torch_device
    torch.manual_seed(options.seed)
    gen = torch.Generator(device="cpu").manual_seed(options.seed)
    model = model.to(device).float()
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    tr = {k: v.to(device) for k, v in train.items()}
    va = {k: v.to(device) for k, v in val.items()}
    ntr, nva = tr["u"].shape[0], va["u"].shape[0]

    # Fixed validation noise/time so the selection metric is comparable.
    vgen = torch.Generator(device="cpu").manual_seed(options.val_seed)
    vb = min(options.batch_size * 4, nva)
    val_sets = []
    for _ in range(options.val_batches):
        idx = torch.randint(0, nva, (vb,), generator=vgen)
        t = torch.rand(vb, generator=vgen)
        eps = base.sample_like(va["u"][idx].cpu(), va["mask"][idx].cpu(), vgen)
        val_sets.append((idx.to(device), t.to(device), eps.to(device)))

    opt = torch.optim.AdamW(model.parameters(), lr=options.lr,
                            weight_decay=options.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(options.warmup_steps, options.steps)
    )
    history = {"step": [], "train_loss": [], "val_loss": [], "seconds": [],
               "device": info.to_dict()}
    best = (float("inf"), -1)
    best_state = None
    running = 0.0
    start = time.time()
    for step in range(options.steps):
        model.train()
        idx = torch.randint(0, ntr, (options.batch_size,), generator=gen).to(device)
        t = torch.rand(options.batch_size, generator=gen).to(device)
        # Noise is drawn on CPU for device-independent streams; padded
        # entries are masked inside the loss.
        eps = base.sample_like(
            torch.zeros((options.batch_size,) + tuple(tr["u"].shape[1:])), None, gen
        ).to(device)
        loss = flow_matching_loss(model, tr["u"][idx], tr["mask"][idx],
                                  tr["cond"][idx], base, t=t, eps=eps)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), options.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(options.ema_decay).add_(pm.detach(), alpha=1 - options.ema_decay)
        running += float(loss.detach())
        if (step + 1) % options.val_every == 0 or step + 1 == options.steps:
            with torch.no_grad():
                vl = np.mean([
                    float(flow_matching_loss(ema, va["u"][i], va["mask"][i],
                                             va["cond"][i], base, t=t_, eps=e_))
                    for i, t_, e_ in val_sets
                ])
            synchronize(device)
            ntrain = (step + 1) % options.val_every or options.val_every
            history["step"].append(step + 1)
            history["train_loss"].append(running / ntrain)
            history["val_loss"].append(float(vl))
            history["seconds"].append(time.time() - start)
            running = 0.0
            log(f"step {step + 1:6d} train {history['train_loss'][-1]:.5f} "
                f"val(ema) {vl:.5f} lr {sched.get_last_lr()[0]:.2e} "
                f"{time.time() - start:.0f}s")
            if vl < best[0]:
                best = (float(vl), step + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in ema.state_dict().items()}
                if on_best is not None:
                    on_best(ema, step + 1, float(vl))
    history["best_val_loss"], history["best_step"] = best
    model_cpu = copy.deepcopy(ema).cpu()
    if best_state is not None:
        model_cpu.load_state_dict(best_state)
    return model_cpu, history


def json_dumps(obj):
    return json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
