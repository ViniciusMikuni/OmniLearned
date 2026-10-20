"""Invertible per-feature transforms with explicit log Jacobians.

A feature may be mapped from a finite interval (-B, B) to the real line with
``atanh(x / B)`` and is then affinely standardized. The per-event
log|det du/dx| over real particles is returned so densities can be reported
in physical coordinates. Constants that do not depend on candidate
parameters cancel in likelihood ratios but are still applied consistently.
"""

import numpy as np
import torch


class FeatureTransform:
    def __init__(self, names, bounds, shift, scale):
        self.names = list(names)
        self.bounds = [None if b in (None, 0, 0.0) else float(b) for b in bounds]
        self.shift = [float(s) for s in shift]
        self.scale = [float(s) for s in scale]
        n = len(self.names)
        if not (len(self.bounds) == len(self.shift) == len(self.scale) == n):
            raise ValueError("names, bounds, shift and scale must align")

    @classmethod
    def fit(cls, names, bounds, x, mask):
        """Fit the affine standardization on (training) data only."""
        x = np.asarray(x, dtype=np.float64)
        mask = np.asarray(mask, dtype=bool)
        raw = _bounded_forward(x, bounds)
        sel = raw[mask]
        shift = sel.mean(0)
        scale = sel.std(0)
        scale[scale <= 0] = 1.0
        return cls(names, bounds, shift, scale)

    def to_dict(self):
        return {"type": "bounded_atanh_affine", "names": self.names,
                "bounds": self.bounds, "shift": self.shift, "scale": self.scale}

    @classmethod
    def from_dict(cls, d):
        return cls(d["names"], d["bounds"], d["shift"], d["scale"])

    def _vec(self, values, like):
        return torch.tensor(values, dtype=like.dtype, device=like.device)

    def forward(self, x, mask):
        """Physical -> model coordinates; returns (u, log|du/dx| per event)."""
        mask_f = mask.to(x.dtype).unsqueeze(-1)
        u = x.clone()
        logj = torch.zeros_like(x)
        for f, bound in enumerate(self.bounds):
            if bound is not None:
                r = x[..., f] / bound
                if bool(((r.abs() >= 1) & mask.bool()).any()):
                    raise ValueError(
                        f"feature {self.names[f]} outside its bounded interval"
                    )
                r = torch.where(mask.bool(), r, torch.zeros_like(r))
                u[..., f] = torch.atanh(r)
                logj[..., f] = -np.log(bound) - torch.log1p(-r * r)
        shift = self._vec(self.shift, x)
        scale = self._vec(self.scale, x)
        u = (u - shift) / scale
        logj = logj - torch.log(scale)
        u = u * mask_f
        logdet = (logj * mask_f).reshape(x.shape[0], -1).sum(-1)
        return u, logdet

    def inverse(self, u, mask):
        mask_f = mask.to(u.dtype).unsqueeze(-1)
        x = u * self._vec(self.scale, u) + self._vec(self.shift, u)
        for f, bound in enumerate(self.bounds):
            if bound is not None:
                x[..., f] = bound * torch.tanh(x[..., f])
        return x * mask_f

    def forward_numpy(self, x, mask):
        u, logdet = self.forward(
            torch.as_tensor(np.asarray(x), dtype=torch.float64),
            torch.as_tensor(np.asarray(mask), dtype=torch.bool),
        )
        return u.numpy(), logdet.numpy()


def _bounded_forward(x, bounds):
    out = np.array(x, dtype=np.float64, copy=True)
    for f, bound in enumerate(bounds):
        if bound not in (None, 0, 0.0):
            out[..., f] = np.arctanh(np.clip(out[..., f] / bound, -1 + 1e-15, 1 - 1e-15))
    return out
