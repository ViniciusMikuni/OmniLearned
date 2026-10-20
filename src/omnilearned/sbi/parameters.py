"""Named mapping from candidate parameters and known context to conditions.

The checkpoint declares its condition layout: an ordered list of column
names, each either an *unknown* inferred parameter or a *known* context
column measured with the observation, plus a fixed affine normalization.
Unknown columns are filled only from candidate values; observations never
supply them.
"""

import numpy as np
import torch


class ConditionLayout:
    def __init__(self, columns, unknown, center, scale, units=None):
        self.columns = list(columns)
        self.unknown = list(unknown)
        missing = [u for u in self.unknown if u not in self.columns]
        if missing:
            raise ValueError(f"unknown parameters {missing} not in layout columns")
        self.known = [c for c in self.columns if c not in self.unknown]
        self.center = {k: float(v) for k, v in dict(center).items()}
        self.scale = {k: float(v) for k, v in dict(scale).items()}
        for c in self.columns:
            self.center.setdefault(c, 0.0)
            self.scale.setdefault(c, 1.0)
        self.units = dict(units or {})

    @classmethod
    def from_bounds(cls, unknown, lower, upper, known=(), units=None):
        """Normalize each unknown parameter to [-1, 1] over its design bounds."""
        center = {n: 0.5 * (lo + hi) for n, lo, hi in zip(unknown, lower, upper)}
        scale = {n: 0.5 * (hi - lo) if hi > lo else 1.0
                 for n, lo, hi in zip(unknown, lower, upper)}
        return cls(list(unknown) + list(known), unknown, center, scale, units)

    def to_dict(self):
        return {"columns": self.columns, "unknown": self.unknown,
                "center": self.center, "scale": self.scale, "units": self.units}

    @classmethod
    def from_dict(cls, d):
        return cls(d["columns"], d["unknown"], d["center"], d["scale"],
                   d.get("units"))

    @property
    def dim(self):
        return len(self.columns)

    def normalize_table(self, values, names):
        values = np.asarray(values, dtype=np.float64)
        out = np.empty_like(values)
        for j, n in enumerate(names):
            out[..., j] = (values[..., j] - self.center[n]) / self.scale[n]
        return out

    def assemble(self, candidates, candidate_names=None, context=None,
                 context_names=None):
        """Build normalized condition rows.

        Args:
            candidates: [M, n_unknown] candidate parameter values.
            candidate_names: order of the candidate columns (default: layout).
            context: optional [M, n_known] measured context aligned with rows.
        Returns:
            float32 numpy array [M, dim].
        """
        candidate_names = list(candidate_names or self.unknown)
        if sorted(candidate_names) != sorted(self.unknown):
            raise ValueError(
                f"candidates {candidate_names} must cover unknowns {self.unknown}"
            )
        candidates = np.atleast_2d(np.asarray(candidates, dtype=np.float64))
        rows = candidates.shape[0]
        out = np.zeros((rows, self.dim), dtype=np.float64)
        for j, name in enumerate(candidate_names):
            col = self.columns.index(name)
            out[:, col] = (candidates[:, j] - self.center[name]) / self.scale[name]
        if self.known:
            if context is None:
                raise ValueError(f"known context {self.known} must be supplied")
            context = np.atleast_2d(np.asarray(context, dtype=np.float64))
            context_names = list(context_names or self.known)
            for name in self.known:
                col = self.columns.index(name)
                src = context_names.index(name)
                out[:, col] = (context[:, src] - self.center[name]) / self.scale[name]
        return out.astype(np.float32)

    def assemble_torch(self, candidates, device, **kwargs):
        return torch.from_numpy(self.assemble(candidates, **kwargs)).to(device)
