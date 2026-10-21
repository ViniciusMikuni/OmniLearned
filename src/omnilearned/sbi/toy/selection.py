"""Selected-distribution normalization of the surrogate.

The surrogate is trained on selected jets, but its support is unbounded in
the transformed coordinates (the fiducial |eta|, |phi| cuts are enforced by
the atanh transform; jet-level cuts are not). The selected density is

    q_sel(G | theta) = q_raw(G | theta) 1[G in S] / Pr_q(S | theta).

Pr_q(S | theta) is estimated by sampling the surrogate with common random
numbers (identical base noise for every candidate, so differences between
candidates are smooth) and applying the same FastJet selection. It depends
only on the checkpoint, never on observations or truth, and is tabulated once
per checkpoint on a fixed grid and interpolated linearly in log Pr.
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from omnilearned.sbi.toy.reco import four_vectors
from omnilearned.sbi.toy.simulator import PARAMETERS, apply_selection


def select_clouds(x_phys, sel):
    """Apply the toy selection to physical-coordinate clouds [N, 32, 3]."""
    ev = {"eta": x_phys[..., 0], "phi": x_phys[..., 1], "pt": np.exp(x_phys[..., 2]),
          "valid": np.all(np.isfinite(x_phys), axis=(1, 2))}
    ok, reason, r = apply_selection(ev, sel)
    return ok, reason, r


def acceptance_table(surrogate, sel, axes, n_samples, seed, nsteps=32, solver="midpoint",
                     log=print):
    """Pr_q(S | theta) on the product grid ``axes`` (PARAMETERS order)."""
    shape = tuple(len(a) for a in axes)
    acc = np.zeros(shape)
    reasons = np.zeros(shape + (5,), dtype=np.int64)
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, len(axes))
    for i, th in enumerate(pts):
        x = surrogate.sample(th, n_samples, sel.fixed_constituent_count, seed=seed,
                             nsteps=nsteps, solver=solver, candidate_names=list(PARAMETERS))
        ok, reason, _ = select_clouds(x.astype(np.float64), sel)
        idx = np.unravel_index(i, shape)
        acc[idx] = ok.mean()
        for k in range(5):
            reasons[idx + (k,)] = int((reason == k).sum())
        if (i + 1) % max(1, len(pts) // 10) == 0:
            log(f"  acceptance table {i + 1}/{len(pts)}: Pr={acc[idx]:.4f}")
    return acc, reasons


def save_table(path, axes, acc, reasons, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, acceptance=acc, reasons=reasons,
             **{f"axis_{k}": a for k, a in enumerate(axes)})
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=1)


class TableLogNorm:
    """log Pr_q(S | theta) interpolated from a saved table."""

    def __init__(self, path):
        d = np.load(path)
        self.axes = [d[f"axis_{k}"] for k in range(len(PARAMETERS))]
        acc = d["acceptance"]
        if np.any(acc <= 0):
            raise ValueError("acceptance table has empty cells")
        self.log_acc = np.log(acc)
        self.interp = RegularGridInterpolator(self.axes, self.log_acc, method="linear",
                                              bounds_error=False, fill_value=None)
        self.description = f"surrogate selection acceptance table {Path(path).name}"

    def __call__(self, points, names=PARAMETERS):
        pts = np.atleast_2d(points)
        if list(names) != list(PARAMETERS):
            order = [list(names).index(p) for p in PARAMETERS]
            pts = pts[:, order]
        return self.interp(pts)
