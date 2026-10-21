"""Joint parameter scans and ensemble likelihood mathematics.

Two fit modes are supported:

* fixed vector: every observation shares one parameter vector ``theta``;
  ``log L(theta) = sum_i log q(G_i | theta)``.
* shared + local: a shared vector ``phi`` and an event-local variable ``b``
  with a normalized law ``g(b | phi)``;
  ``log L(phi) = sum_i log integral q(G_i | b, phi) g(b | phi) db``.
  The integral is done per event (log-sum-exp over quadrature nodes) before
  the sum over events.

All grids keep the full joint result. Posterior grids use cell volumes from
trapezoid weights so they integrate to one over the declared domain. Slices,
profiles and marginals are returned separately and labelled.
"""

import json
import math
import time
import warnings
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Grids and integration measures
# ---------------------------------------------------------------------------


def trapezoid_weights(axis):
    """Integration weights of a sorted 1D node set over [axis[0], axis[-1]]."""
    a = np.asarray(axis, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("axis must be a non-empty 1D array")
    if a.size == 1:
        return np.ones(1)
    if np.any(np.diff(a) <= 0):
        raise ValueError("axis must be strictly increasing")
    w = np.zeros_like(a)
    d = np.diff(a)
    w[:-1] += 0.5 * d
    w[1:] += 0.5 * d
    return w


class Grid:
    """Cartesian product grid over named axes (C order)."""

    def __init__(self, names, axes):
        self.names = list(names)
        self.axes = [np.asarray(a, dtype=np.float64) for a in axes]
        if len(self.names) != len(self.axes):
            raise ValueError("names and axes must align")
        self.weights = [trapezoid_weights(a) for a in self.axes]

    @classmethod
    def from_spec(cls, spec):
        """spec: ordered mapping name -> {min, max, n} or explicit list."""
        names, axes = [], []
        for name, s in spec.items():
            names.append(name)
            if isinstance(s, dict):
                axes.append(np.linspace(s["min"], s["max"], int(s["n"])))
            else:
                axes.append(np.asarray(s, dtype=np.float64))
        return cls(names, axes)

    @property
    def shape(self):
        return tuple(len(a) for a in self.axes)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def points(self):
        mesh = np.meshgrid(*self.axes, indexing="ij")
        return np.stack([m.reshape(-1) for m in mesh], -1)

    def cell_volumes(self):
        vol = np.ones(self.shape)
        for k, w in enumerate(self.weights):
            shape = [1] * len(self.axes)
            shape[k] = len(w)
            vol = vol * w.reshape(shape)
        return vol

    def to_dict(self):
        return {"names": self.names, "axes": [a.tolist() for a in self.axes]}


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


@dataclass
class ScanResult:
    grid: Grid
    loglik: np.ndarray  # joint log likelihood on the grid
    logprior: np.ndarray
    per_event: np.ndarray = None  # [M, N] per-event log densities (optional)
    log_norm: np.ndarray = None  # per-candidate selection normalization
    n_events: int = 0
    mode: str = "fixed_vector"
    diagnostics: dict = field(default_factory=dict)
    evaluated: np.ndarray = None  # support scans: nodes actually computed

    # ---- posterior ------------------------------------------------------
    def log_posterior(self):
        return self.loglik + self.logprior

    def posterior_mass(self):
        """Normalized posterior probability mass per grid cell."""
        lp = self.log_posterior()
        finite = np.isfinite(lp)
        if not finite.any():
            raise ValueError("no finite posterior values on grid")
        m = np.where(finite, np.exp(lp - lp[finite].max()), 0.0)
        m = m * self.grid.cell_volumes()
        return m / m.sum()

    def posterior_density(self):
        return self.posterior_mass() / self.grid.cell_volumes()

    def marginal(self, name):
        """Posterior marginal density along axis ``name`` (integrated)."""
        k = self.grid.names.index(name)
        mass = self.posterior_mass()
        others = tuple(i for i in range(mass.ndim) if i != k)
        m = mass.sum(axis=others)
        return self.grid.axes[k], m / self.grid.weights[k], m

    def profile(self, name):
        """Profile log likelihood: maximize over the other axes."""
        k = self.grid.names.index(name)
        others = tuple(i for i in range(self.loglik.ndim) if i != k)
        with warnings.catch_warnings():  # all-nan slices of support scans
            warnings.simplefilter("ignore", RuntimeWarning)
            return self.grid.axes[k], np.nanmax(self.loglik, axis=others)

    def slice(self, name, at=None):
        """Conditional slice of log L through ``at`` (default: grid MLE)."""
        k = self.grid.names.index(name)
        idx = list(self.mle_index() if at is None else at)
        idx[k] = slice(None)
        return self.grid.axes[k], self.loglik[tuple(idx)]

    def mle_index(self):
        return np.unravel_index(np.nanargmax(self.loglik), self.loglik.shape)

    def map_index(self):
        return np.unravel_index(np.nanargmax(self.log_posterior()), self.loglik.shape)

    def summary(self, levels=(0.68, 0.95)):
        pts = self.grid.points().reshape(self.grid.shape + (len(self.grid.names),))
        mass = self.posterior_mass()
        flat_mass = mass.reshape(-1)
        flat_pts = pts.reshape(-1, len(self.grid.names))
        mean = flat_mass @ flat_pts
        cov = (flat_pts - mean).T @ ((flat_pts - mean) * flat_mass[:, None])
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(std, std)
        mle = self.mle_index()
        out = {
            "mode": self.mode,
            "n_events": self.n_events,
            "parameters": self.grid.names,
            "mle": {n: float(pts[mle][k]) for k, n in enumerate(self.grid.names)},
            "max_loglik": float(np.nanmax(self.loglik)),
            "posterior_mean": dict(zip(self.grid.names, mean.tolist())),
            "posterior_std": dict(zip(self.grid.names, std.tolist())),
            "posterior_correlation": corr.tolist(),
            "credible_intervals": {},
            "profile_intervals": {},
            "boundary": {},
        }
        for k, name in enumerate(self.grid.names):
            axis, dens, m = self.marginal(name)
            out["credible_intervals"][name] = {
                str(lv): equal_tailed_interval(axis, dens, lv) for lv in levels
            }
            ax, prof = self.profile(name)
            out["profile_intervals"][name] = {
                "delta_2nll_1": profile_interval(ax, prof, 1.0),
                "delta_2nll_3.84": profile_interval(ax, prof, 3.84),
            }
            edge_mass = float(m[0] + m[-1]) if len(m) > 1 else 0.0
            out["boundary"][name] = {
                "mle_on_edge": bool(mle[k] in (0, len(axis) - 1)) and len(axis) > 1,
                "edge_node_mass": edge_mass,
            }
        return out

    # ---- persistence ----------------------------------------------------
    def save(self, path_prefix, extra=None):
        arrays = {
            "loglik": self.loglik,
            "logprior": self.logprior,
            "cell_volumes": self.grid.cell_volumes(),
            "posterior_mass": self.posterior_mass(),
        }
        for k, (n, a) in enumerate(zip(self.grid.names, self.grid.axes)):
            arrays[f"axis_{k}"] = a
        if self.per_event is not None:
            arrays["per_event"] = self.per_event
        if self.log_norm is not None:
            arrays["log_norm"] = self.log_norm
        if self.evaluated is not None:
            arrays["evaluated"] = self.evaluated
        np.savez_compressed(f"{path_prefix}.npz", **arrays)
        doc = {
            "grid": self.grid.to_dict(),
            "integration_measure": "trapezoid cell volumes over the grid domain",
            "summary": self.summary(),
            "diagnostics": _jsonable(self.diagnostics),
        }
        if extra:
            doc.update(_jsonable(extra))
        with open(f"{path_prefix}.json", "w") as f:
            json.dump(doc, f, indent=1)
        return doc

    @classmethod
    def load(cls, path_prefix):
        with open(f"{path_prefix}.json") as f:
            doc = json.load(f)
        arr = np.load(f"{path_prefix}.npz")
        grid = Grid(doc["grid"]["names"], doc["grid"]["axes"])
        return cls(
            grid=grid,
            loglik=arr["loglik"],
            logprior=arr["logprior"],
            per_event=arr["per_event"] if "per_event" in arr else None,
            log_norm=arr["log_norm"] if "log_norm" in arr else None,
            n_events=doc["summary"]["n_events"],
            mode=doc["summary"]["mode"],
            diagnostics=doc.get("diagnostics", {}),
            evaluated=arr["evaluated"] if "evaluated" in arr else None,
        )


def equal_tailed_interval(axis, density, level):
    """Equal-tailed interval of a 1D density sampled at nodes (piecewise linear)."""
    axis = np.asarray(axis, dtype=np.float64)
    density = np.clip(np.asarray(density, dtype=np.float64), 0, None)
    if axis.size < 2:
        return [float(axis[0]), float(axis[0])]
    seg = 0.5 * (density[1:] + density[:-1]) * np.diff(axis)
    cdf = np.concatenate([[0.0], np.cumsum(seg)])
    if cdf[-1] <= 0:
        return [float("nan"), float("nan")]
    cdf /= cdf[-1]
    lo = float(np.interp(0.5 * (1 - level), cdf, axis))
    hi = float(np.interp(0.5 * (1 + level), cdf, axis))
    return [lo, hi]


def profile_interval(axis, profile, delta_2nll):
    """Range where -2 (profile - max) <= delta, linearly interpolated."""
    axis = np.asarray(axis, dtype=np.float64)
    q = -2.0 * (np.asarray(profile, dtype=np.float64) - np.nanmax(profile))
    inside = q <= delta_2nll
    idx = np.where(inside)[0]
    if idx.size == 0:
        return [float("nan"), float("nan")]
    i0, i1 = idx[0], idx[-1]
    lo, hi = axis[i0], axis[i1]
    if i0 > 0:
        lo = np.interp(delta_2nll, [q[i0], q[i0 - 1]], [axis[i0], axis[i0 - 1]])
    if i1 < axis.size - 1:
        hi = np.interp(delta_2nll, [q[i1], q[i1 + 1]], [axis[i1], axis[i1 + 1]])
    return [float(lo), float(hi), bool(i0 == 0), bool(i1 == axis.size - 1)]


def uniform_log_prior(grid, bounds=None):
    """Uniform density over the box; -inf outside optional ``bounds``."""
    lp = np.zeros(grid.shape)
    vol = 1.0
    for k, a in enumerate(grid.axes):
        lo, hi = (a[0], a[-1]) if bounds is None else bounds[k]
        vol *= (hi - lo) if hi > lo else 1.0
        if bounds is not None:
            shape = [1] * len(grid.axes)
            shape[k] = len(a)
            out = ((a < lo) | (a > hi)).reshape(shape)
            lp = np.where(out, -np.inf, lp)
    return lp - math.log(vol)


def joint_scan(
    logp_fn,
    n_events,
    grid,
    event_chunk=256,
    candidate_chunk=8,
    log_prior=None,
    log_norm_fn=None,
    keep_per_event=True,
    progress=None,
    max_seconds=None,
):
    """Fixed-vector joint scan.

    Args:
        logp_fn: ``(event_slice, candidates[m, P]) -> array [m, n]`` of
            per-event log densities (float64) for those events/candidates.
        n_events: total number of observations.
        grid: :class:`Grid` over the fitted parameters.
        log_norm_fn: optional ``(candidates[M, P]) -> [M]`` log selection
            probability of the surrogate; ``-n_events * log_norm`` is added
            (selected-distribution normalization).
        max_seconds: abort with an incomplete result when exceeded.
    """
    pts = grid.points()
    M = pts.shape[0]
    total = np.zeros(M, dtype=np.float64)
    per_event = np.full((M, n_events), np.nan) if keep_per_event else None
    nonfinite = np.zeros(M, dtype=np.int64)
    start = time.time()
    complete = True
    for c0 in range(0, M, candidate_chunk):
        cand = pts[c0 : c0 + candidate_chunk]
        for e0 in range(0, n_events, event_chunk):
            sl = slice(e0, min(n_events, e0 + event_chunk))
            vals = np.asarray(logp_fn(sl, cand), dtype=np.float64)
            if vals.shape != (cand.shape[0], sl.stop - sl.start):
                raise ValueError(f"logp_fn returned shape {vals.shape}")
            bad = ~np.isfinite(vals)
            nonfinite[c0 : c0 + cand.shape[0]] += bad.sum(1)
            total[c0 : c0 + cand.shape[0]] += np.where(bad, 0.0, vals).sum(1)
            if per_event is not None:
                per_event[c0 : c0 + cand.shape[0], sl] = vals
        if progress is not None:
            progress(min(M, c0 + candidate_chunk), M, time.time() - start)
        if max_seconds is not None and time.time() - start > max_seconds:
            complete = c0 + candidate_chunk >= M
            if not complete:
                break
    loglik = np.where(nonfinite > 0, np.nan, total)
    log_norm = None
    if log_norm_fn is not None:
        log_norm = np.asarray(log_norm_fn(pts), dtype=np.float64)
        loglik = loglik - n_events * log_norm
    if not complete:
        done = min(M, c0 + candidate_chunk)
        loglik[done:] = np.nan
    lp = uniform_log_prior(grid) if log_prior is None else log_prior
    return ScanResult(
        grid=grid,
        loglik=loglik.reshape(grid.shape),
        logprior=np.broadcast_to(lp, grid.shape).copy(),
        per_event=per_event,
        log_norm=None if log_norm is None else log_norm.reshape(grid.shape),
        n_events=n_events,
        mode="fixed_vector",
        diagnostics={
            "seconds": time.time() - start,
            "complete": complete,
            "nonfinite_event_values": int(nonfinite.sum()),
            "candidates_with_nonfinite": int((nonfinite > 0).sum()),
            "accumulation_dtype": "float64",
            "event_chunk": event_chunk,
            "candidate_chunk": candidate_chunk,
        },
    )


def seed_mask(shape, stride):
    """Sub-lattice of a grid: every ``stride[k]``-th node per axis plus the last."""
    mask = np.ones(shape, dtype=bool)
    for k, (n, s) in enumerate(zip(shape, stride)):
        keep = np.zeros(n, dtype=bool)
        keep[:: max(1, int(s))] = True
        keep[-1] = True
        sh = [1] * len(shape)
        sh[k] = n
        mask = mask & keep.reshape(sh)
    return mask


def dilate(mask):
    """Boolean dilation over the full 3^d - 1 neighbourhood (no wrap-around)."""
    d = mask.ndim
    pad = np.pad(mask, 1)
    out = np.zeros_like(mask)
    for off in np.ndindex(*(3,) * d):
        out |= pad[tuple(slice(o, o + n) for o, n in zip(off, mask.shape))]
    return out


def support_scan(
    logp_fn,
    n_events,
    grid,
    seed_stride,
    delta_logpost=10.0,
    event_chunk=256,
    candidate_chunk=8,
    log_prior=None,
    log_norm_fn=None,
    keep_per_event=True,
    progress=None,
    max_seconds=None,
):
    """Fixed-vector scan that evaluates the supported region of a fine grid.

    For posteriors much narrower than the domain (thin, curved degeneracies)
    where a dense grid fine enough to resolve them is unaffordable. The seed
    sub-lattice (:func:`seed_mask`) is evaluated first; then every unevaluated
    node adjacent (3^d - 1 neighbourhood) to an evaluated node with log
    posterior >= max - ``delta_logpost`` is evaluated, repeatedly, until no
    such node remains ("closed"). Unevaluated nodes keep ``loglik = nan`` and
    zero posterior mass, so the result equals the dense-grid result up to the
    mass of nodes below max - delta that are not connected to the seeds'
    supported region. A supported region not touched by any seed node with
    log posterior >= max - delta is missed: the seed lattice must be fine
    enough to intersect every relevant mode. Returns a :class:`ScanResult`
    whose ``evaluated`` mask records which nodes were computed.
    """
    pts = grid.points()
    shape = grid.shape
    M = pts.shape[0]
    lp_prior = uniform_log_prior(grid) if log_prior is None else log_prior
    lp_prior = np.broadcast_to(lp_prior, shape).reshape(-1)
    loglik = np.full(M, np.nan)
    log_norm = np.full(M, np.nan) if log_norm_fn is not None else None
    per_event = np.full((M, n_events), np.nan) if keep_per_event else None
    evaluated = np.zeros(M, dtype=bool)
    nonfinite = np.zeros(M, dtype=np.int64)
    start = time.time()
    complete = True
    passes = []

    def evaluate(idx):
        nonlocal complete
        for c0 in range(0, len(idx), candidate_chunk):
            sel = idx[c0 : c0 + candidate_chunk]
            cand = pts[sel]
            total = np.zeros(len(sel))
            for e0 in range(0, n_events, event_chunk):
                sl = slice(e0, min(n_events, e0 + event_chunk))
                vals = np.asarray(logp_fn(sl, cand), dtype=np.float64)
                if vals.shape != (len(sel), sl.stop - sl.start):
                    raise ValueError(f"logp_fn returned shape {vals.shape}")
                bad = ~np.isfinite(vals)
                nonfinite[sel] += bad.sum(1)
                total += np.where(bad, 0.0, vals).sum(1)
                if per_event is not None:
                    per_event[sel, sl] = vals
            if log_norm is not None:
                log_norm[sel] = np.asarray(log_norm_fn(cand), dtype=np.float64)
                total = total - n_events * log_norm[sel]
            loglik[sel] = np.where(nonfinite[sel] > 0, np.nan, total)
            evaluated[sel] = True
            if progress is not None:
                progress(int(evaluated.sum()), M, time.time() - start)
            if max_seconds is not None and time.time() - start > max_seconds:
                complete = False
                return

    todo = np.flatnonzero(seed_mask(shape, seed_stride).reshape(-1))
    while todo.size and complete:
        passes.append(int(todo.size))
        evaluate(todo)
        lp = loglik + lp_prior
        if not np.isfinite(lp).any():
            break
        active = evaluated & (lp >= np.nanmax(lp) - delta_logpost)
        todo = np.flatnonzero(dilate(active.reshape(shape)).reshape(-1) & ~evaluated)
    lp = loglik + lp_prior
    finite = np.isfinite(lp)
    if finite.any():
        active = evaluated & (lp >= np.nanmax(lp) - delta_logpost)
        closed = bool(complete and not (dilate(active.reshape(shape)).reshape(-1)
                                        & ~evaluated).any())
    else:
        active, closed = np.zeros(M, dtype=bool), False
    return ScanResult(
        grid=grid,
        loglik=loglik.reshape(shape),
        logprior=lp_prior.reshape(shape).copy(),
        per_event=per_event,
        log_norm=None if log_norm is None else log_norm.reshape(shape),
        n_events=n_events,
        mode="fixed_vector",
        evaluated=evaluated.reshape(shape),
        diagnostics={
            "seconds": time.time() - start,
            "complete": complete,
            "scan": "support",
            "support_closed": closed,
            "delta_logpost": float(delta_logpost),
            "seed_stride": [int(s) for s in seed_stride],
            "evaluated": int(evaluated.sum()),
            "grid_size": int(M),
            "passes": passes,
            "supported_nodes": int(active.sum()),
            "nonfinite_event_values": int(nonfinite.sum()),
            "candidates_with_nonfinite": int((nonfinite > 0).sum()),
            "accumulation_dtype": "float64",
            "event_chunk": event_chunk,
            "candidate_chunk": candidate_chunk,
        },
    )


def combine_event_partitions(results):
    """Sum disjoint event partitions evaluated on the same grid."""
    base = results[0]
    loglik = sum(r.loglik for r in results)
    per = None
    if all(r.per_event is not None for r in results):
        per = np.concatenate([r.per_event for r in results], axis=1)
    return ScanResult(base.grid, loglik, base.logprior, per, None,
                      sum(r.n_events for r in results), base.mode)


def local_variable_loglik(per_event, log_weights):
    """Per-event marginalization over a local variable.

    Args:
        per_event: [Nb, M, N] log q(G_i | b_j, phi_m).
        log_weights: [Nb, M] log of quadrature weight times g(b_j | phi_m),
            normalized so that ``sum_j exp(log_weights[j, m]) = 1``.
    Returns:
        (loglik [M], per_event_marginal [M, N], local_posterior [Nb, M, N]).
    """
    per_event = np.asarray(per_event, dtype=np.float64)
    lw = np.asarray(log_weights, dtype=np.float64)[:, :, None]
    a = per_event + lw
    amax = np.max(a, axis=0, keepdims=True)
    amax = np.where(np.isfinite(amax), amax, 0.0)
    lse = amax[0] + np.log(np.sum(np.exp(a - amax), axis=0))
    post = np.exp(a - lse[None])
    return lse.sum(-1), lse, post


def normalized_log_weights(b_nodes, density_values):
    """log(w_j g(b_j)) with trapezoid weights, normalized to unit sum.

    ``density_values``: [Nb, M] unnormalized g(b_j | phi_m) >= 0.
    """
    w = trapezoid_weights(b_nodes)[:, None] * np.asarray(density_values, dtype=np.float64)
    tot = w.sum(0, keepdims=True)
    with np.errstate(divide="ignore"):
        return np.log(w) - np.log(tot)


def refine_axes(result, delta_logpost=12.0, points=None, pad_cells=1):
    """New axes covering the region with log posterior > max - delta.

    Uses only the scan result (never truth). The box is padded by
    ``pad_cells`` original cells and clipped to the original domain.
    """
    lp = result.log_posterior()
    keep = lp >= np.nanmax(lp) - delta_logpost
    axes = []
    for k, a in enumerate(result.grid.axes):
        others = tuple(i for i in range(lp.ndim) if i != k)
        idx = np.where(keep.any(axis=others))[0]
        lo = max(0, idx[0] - pad_cells)
        hi = min(len(a) - 1, idx[-1] + pad_cells)
        n = len(a) if points is None else points[k]
        axes.append(np.linspace(a[lo], a[hi], n) if hi > lo else np.array([a[lo]]))
    return Grid(result.grid.names, axes)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return obj
