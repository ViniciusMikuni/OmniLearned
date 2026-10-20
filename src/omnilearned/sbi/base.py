"""Checkpoint-aware base distribution shared by generation, training and density.

The flow-matching convention in OmniLearned is ``z_t = (1 - t) x + t eps`` with
data at ``t = 0`` and base noise ``eps`` at ``t = 1``. Every place that draws
``eps`` or evaluates its density must use the same specification, so it is a
single object that is stored in checkpoint metadata.
"""

import math

import numpy as np
import torch


# Legacy dataset-specific bases used by upstream OmniLearned (4 features).
LEGACY_PRESETS = {
    "jetnet": (
        [0.0, 0.0, 2.92904309e00, 3.14843288e00],
        [0.11580968, 0.11667726, 0.95442127, 0.98245158],
    ),
    "lhco": (
        [0.0, 0.0, 1.11398, 1.43384],
        [0.270949, 0.27426, 1.30273, 1.33559],
    ),
}


class GaussianBase:
    """Diagonal Gaussian over particle features, independent per particle.

    Padded particles (``mask == 0``) are not random coordinates: they are set
    to zero when sampling and excluded from the log density.
    """

    def __init__(self, means, stds, name="gaussian"):
        self.means = [float(m) for m in means]
        self.stds = [float(s) for s in stds]
        if len(self.means) != len(self.stds):
            raise ValueError("means and stds must have the same length")
        if any(s <= 0 for s in self.stds):
            raise ValueError("base standard deviations must be positive")
        self.name = name

    @property
    def num_features(self):
        return len(self.means)

    @classmethod
    def standard(cls, num_features):
        return cls([0.0] * num_features, [1.0] * num_features, name="standard")

    @classmethod
    def from_dataset(cls, dataset, num_features):
        """Reproduce the upstream ``get_eps`` choice for ``dataset``."""
        if dataset in LEGACY_PRESETS and num_features == 4:
            means, stds = LEGACY_PRESETS[dataset]
            return cls(means, stds, name=dataset)
        return cls.standard(num_features)

    def to_dict(self):
        return {"type": "diag_gaussian", "name": self.name, "means": self.means,
                "stds": self.stds}

    @classmethod
    def from_dict(cls, d):
        if d.get("type", "diag_gaussian") != "diag_gaussian":
            raise ValueError(f"Unsupported base type {d.get('type')}")
        return cls(d["means"], d["stds"], name=d.get("name", "gaussian"))

    def _params(self, like):
        shape = (1,) * (like.dim() - 1) + (self.num_features,)
        means = torch.tensor(self.means, device=like.device, dtype=like.dtype)
        stds = torch.tensor(self.stds, device=like.device, dtype=like.dtype)
        return means.view(shape), stds.view(shape)

    def check(self, x):
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Base has {self.num_features} features, input has {x.shape[-1]}"
            )

    def sample_like(self, x, mask=None, generator=None):
        """Draw base noise with the shape of ``x``; ``mask`` is [..., P] or [..., P, 1]."""
        self.check(x)
        means, stds = self._params(x)
        noise = torch.randn(
            x.shape, generator=generator, device=x.device, dtype=x.dtype
        )
        eps = noise * stds + means
        if mask is not None:
            eps = eps * _feature_mask(mask, x)
        return eps

    def log_prob(self, x, mask=None):
        """Per-event log density, summed over real particles and features."""
        self.check(x)
        means, stds = self._params(x)
        logp = (
            -0.5 * ((x - means) / stds) ** 2
            - torch.log(stds)
            - 0.5 * math.log(2.0 * math.pi)
        )
        if mask is not None:
            logp = logp * _feature_mask(mask, x)
        return logp.reshape(x.shape[0], -1).sum(-1)

    def fingerprint(self):
        return np.round(np.array(self.means + self.stds), 10).tobytes().hex()


class ExchangeableGaussianBase:
    """Exchangeable Gaussian over the particles of each event, per feature.

    For feature f and the n real particles of an event,
    ``x_kf = m_f + c_f + s_f xi_kf`` with a shared offset ``c_f ~ N(0, tau_f^2)``
    and ``xi_kf ~ N(0, 1)``: covariance ``s_f^2 I + tau_f^2 1 1^T``. The density
    is permutation invariant and depends only on the real particles, so it is
    a valid base for exchangeable variable-multiplicity clouds. It matches
    clouds whose particles share an event-level location (e.g. a jet axis).
    """

    def __init__(self, means, within_stds, common_stds, name="exchangeable"):
        self.means = [float(m) for m in means]
        self.within_stds = [float(s) for s in within_stds]
        self.common_stds = [float(s) for s in common_stds]
        if not (len(self.means) == len(self.within_stds) == len(self.common_stds)):
            raise ValueError("parameter lists must have equal length")
        if any(s <= 0 for s in self.within_stds) or any(s < 0 for s in self.common_stds):
            raise ValueError("within stds must be positive and common stds >= 0")
        self.name = name

    @property
    def num_features(self):
        return len(self.means)

    @classmethod
    def fit(cls, u, mask, floor=1e-3):
        """Moment estimates from (training) data in model coordinates."""
        u = np.asarray(u, dtype=np.float64)
        m = np.asarray(mask, dtype=bool)
        n = m.sum(1)
        keep = n > 1
        u, m, n = u[keep], m[keep], n[keep]
        mf = m[..., None].astype(np.float64)
        ev_mean = (u * mf).sum(1) / n[:, None]
        within = (((u - ev_mean[:, None]) ** 2) * mf).sum(1) / (n[:, None] - 1)
        s2 = within.mean(0)
        tau2 = ev_mean.var(0) - (s2 / n[:, None]).mean(0)
        means = ev_mean.mean(0)
        return cls(means, np.sqrt(np.maximum(s2, floor ** 2)),
                   np.sqrt(np.maximum(tau2, 0.0)))

    def to_dict(self):
        return {"type": "exchangeable_gaussian", "name": self.name, "means": self.means,
                "within_stds": self.within_stds, "common_stds": self.common_stds}

    @classmethod
    def from_dict(cls, d):
        return cls(d["means"], d["within_stds"], d["common_stds"], d.get("name", "exchangeable"))

    def check(self, x):
        if x.shape[-1] != self.num_features:
            raise ValueError(
                f"Base has {self.num_features} features, input has {x.shape[-1]}"
            )

    def _vec(self, values, like):
        shape = (1,) * (like.dim() - 1) + (self.num_features,)
        return torch.tensor(values, device=like.device, dtype=like.dtype).view(shape)

    def sample_like(self, x, mask=None, generator=None):
        self.check(x)
        xi = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
        c = torch.randn((x.shape[0],) + (1,) * (x.dim() - 2) + (x.shape[-1],),
                        generator=generator, device=x.device, dtype=x.dtype)
        eps = (self._vec(self.means, x) + c * self._vec(self.common_stds, x)
               + xi * self._vec(self.within_stds, x))
        if mask is not None:
            eps = eps * _feature_mask(mask, x)
        return eps

    def log_prob(self, x, mask=None):
        self.check(x)
        if mask is None:
            mask_f = torch.ones(x.shape[:-1] + (1,), device=x.device, dtype=x.dtype)
        else:
            mask_f = _feature_mask(mask, x)
        y = (x - self._vec(self.means, x)) * mask_f
        s2 = self._vec(self.within_stds, x).reshape(-1) ** 2
        t2 = self._vec(self.common_stds, x).reshape(-1) ** 2
        n = mask_f.reshape(x.shape[0], -1).sum(-1)[:, None]  # [B, 1]
        flat = y.reshape(x.shape[0], -1, x.shape[-1])
        sq = (flat ** 2).sum(1)  # [B, F]
        tot = flat.sum(1)
        denom = s2 + n * t2
        quad = sq / s2 - t2 * tot ** 2 / (s2 * denom)
        logdet = (n - 1) * torch.log(s2) + torch.log(denom)
        lp = -0.5 * (quad + logdet + n * math.log(2 * math.pi))
        return torch.where(n > 0, lp, torch.zeros_like(lp)).sum(-1)


def base_from_dict(d):
    """Construct the base specification stored in a checkpoint."""
    kind = d.get("type", "diag_gaussian")
    if kind == "diag_gaussian":
        return GaussianBase.from_dict(d)
    if kind == "exchangeable_gaussian":
        return ExchangeableGaussianBase.from_dict(d)
    raise ValueError(f"Unsupported base type {kind}")


def _feature_mask(mask, x):
    mask = mask.to(dtype=x.dtype, device=x.device)
    if mask.dim() == x.dim() - 1:
        mask = mask.unsqueeze(-1)
    return mask
