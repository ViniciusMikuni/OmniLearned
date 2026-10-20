"""Conditional velocity fields for flow-matching likelihoods.

``SetFlowVelocity`` is the density-compatible variant: a smooth,
permutation-equivariant transformer with global attention over real particles
only and an explicit validity mask. Its output is a differentiable function
of every coordinate, unlike hard top-k neighbour selection, so the
continuous-flow density identity applies.

``PETVelocity`` adapts the upstream ``PET2`` generator (unchanged
architecture) to the same ``(x, t, cond, mask)`` interface with an explicit
mask, for comparison studies.
"""

import math

import torch
import torch.nn as nn


class FourierTime(nn.Module):
    def __init__(self, dim, max_freq=64.0):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(torch.linspace(0.0, math.log(max_freq), half))
        self.register_buffer("freqs", freqs * 2 * math.pi)

    def forward(self, t):
        a = t[:, None].float() * self.freqs[None]
        return torch.cat([torch.sin(a), torch.cos(a)], -1)


class MaskedSelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, h, mask):
        B, P, D = h.shape
        q, k, v = self.qkv(h).reshape(B, P, 3, self.heads, D // self.heads).unbind(2)
        q, k, v = (a.transpose(1, 2) for a in (q, k, v))  # [B, H, P, d]
        scores = q @ k.transpose(-1, -2) / math.sqrt(D // self.heads)
        key_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # rows of fully padded events
        out = (attn @ v).transpose(1, 2).reshape(B, P, D)
        return self.proj(out)


class ModulatedBlock(nn.Module):
    """Pre-norm transformer block with adaptive (FiLM) conditioning."""

    def __init__(self, dim, heads, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = MaskedSelfAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim), nn.GELU(), nn.Linear(mlp_ratio * dim, dim)
        )
        self.mod = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(self, h, c, mask, mask_f):
        s1, b1, g1, s2, b2, g2 = self.mod(c)[:, None].chunk(6, -1)
        a = self.norm1(h) * (1 + s1) + b1
        h = h + (1 + g1) * self.attn(a, mask) * mask_f
        a = self.norm2(h) * (1 + s2) + b2
        h = h + (1 + g2) * self.mlp(a) * mask_f
        return h


class SetFlowVelocity(nn.Module):
    """Smooth permutation-equivariant conditional velocity field."""

    def __init__(self, num_features, cond_dim, dim=128, depth=4, heads=4,
                 mlp_ratio=2, time_dim=64, pool_context=True, cond_embed=False,
                 cond_to_particles=False):
        """
        cond_embed: embed the condition with its own MLP and add it to the
            time context (instead of only concatenating it with time features).
        cond_to_particles: also add a projection of the condition to every
            particle token at the input, so each block sees it directly.
        """
        super().__init__()
        self.config = dict(num_features=num_features, cond_dim=cond_dim, dim=dim,
                           depth=depth, heads=heads, mlp_ratio=mlp_ratio,
                           time_dim=time_dim, pool_context=pool_context,
                           cond_embed=cond_embed, cond_to_particles=cond_to_particles)
        self.pool_context = pool_context
        self.cond_mlp = (
            nn.Sequential(nn.Linear(cond_dim, dim), nn.GELU(), nn.Linear(dim, dim))
            if cond_embed or cond_to_particles else None
        )
        self.cond_in = nn.Linear(dim, dim) if cond_to_particles else None
        self.split_cond = cond_embed
        self.embed = nn.Sequential(
            nn.Linear(num_features, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.time = FourierTime(time_dim)
        ctx_in = time_dim + (0 if cond_embed else cond_dim)
        self.context = nn.Sequential(
            nn.Linear(ctx_in, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [ModulatedBlock(dim, heads, mlp_ratio) for _ in range(depth)]
        )
        self.pool = nn.Linear(dim, dim) if pool_context else None
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.out = nn.Linear(dim, num_features)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, cond, mask):
        mask = mask.bool()
        mask_f = mask.to(x.dtype).unsqueeze(-1)
        parts = [self.time(t).to(x.dtype)]
        if cond is not None and not self.split_cond:
            parts.append(cond.to(x.dtype))
        c = self.context(torch.cat(parts, -1))
        h = self.embed(x)
        if self.cond_mlp is not None:
            ce = self.cond_mlp(cond.to(x.dtype))
            if self.split_cond:
                c = c + ce
            if self.cond_in is not None:
                h = h + self.cond_in(ce)[:, None]
        h = h * mask_f
        for i, blk in enumerate(self.blocks):
            if self.pool is not None and i == len(self.blocks) // 2:
                pooled = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1.0)
                c = c + self.pool(pooled)
            h = blk(h, c, mask, mask_f)
        return self.out(self.norm(h)) * mask_f


class PETVelocity(nn.Module):
    """Explicit-mask velocity adapter around an upstream PET2 generator."""

    def __init__(self, pet, y=None, pid=None, add_info=None):
        super().__init__()
        self.pet = pet
        self.y = y
        self.pid = pid
        self.add_info = add_info
        self.config = {"architecture": "pet2"}

    def forward(self, x, t, cond, mask):
        B, P = mask.shape
        h = self.pet.body(x, cond, self.pid, self.add_info, t, mask=mask)
        extra = h.shape[1] - P
        full = torch.cat(
            [torch.ones(B, extra, 1, dtype=torch.bool, device=x.device),
             mask.bool().view(B, P, 1)], 1
        )
        y = self.y
        if y is not None and y.shape[0] != B:
            y = y[:1].expand(B)
        return self.pet.generator(h, y, mask=full)


def build_velocity(config):
    """Create a velocity model from a checkpoint architecture config."""
    config = dict(config)
    arch = config.pop("architecture", "set_transformer")
    if arch == "set_transformer":
        return SetFlowVelocity(**config)
    if arch == "pet2":
        from omnilearned.network import PET2

        pet = PET2(mode="generator", conditional=True, num_classes=1, **config)
        model = PETVelocity(pet)
        model.config = {"architecture": "pet2", **config}
        return model
    raise ValueError(f"Unknown velocity architecture '{arch}'")


def velocity_config(model):
    cfg = dict(model.config)
    cfg.setdefault("architecture", "set_transformer")
    return cfg
