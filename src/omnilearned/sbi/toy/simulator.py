"""toy-quenching-v1: fast invented jet-quenching simulator.

Implements the toy-quenching-v1 specification (PLAN_NLE_PERLMUTTER.md, "Toy"). Canonical coordinates are
``(b [fm], T_ref [GeV], tau_init [fm/c])``. All constants are illustrative
toy constants read from the versioned study configuration; they are not a
JEWEL calibration. Natural units, c = 1.

Observed records contain only constituent (eta, phi, log pT) features plus
four-vectors and FastJet membership. Production geometry, hard scale, path
moments and component labels go to a separate diagnostic record that the
fitting path never reads.
"""

from dataclasses import dataclass, asdict, fields

import numpy as np

from omnilearned.sbi.toy import reco

BACKEND = "toy-quenching-v1"
PARAMETERS = ("b", "T_ref", "tau_init")
UNITS = ("fm", "GeV", "fm/c")


@dataclass(frozen=True)
class ToyConstants:
    geometry_radius_fm: float = 6.5
    temperature_cut_GeV: float = 0.17
    temperature_scale_GeV: float = 0.35
    length_scale_fm: float = 1.0
    integration_steps: int = 64
    power_law_exponent: float = 5.0
    hard_pt_min_GeV: float = 40.0
    hard_pt_max_GeV: float = 500.0
    loss_coefficient: float = 0.12
    loss_lognormal_sigma: float = 0.30
    core_count: int = 24
    soft_count: int = 8
    dirichlet_core_alpha: float = 0.7
    dirichlet_soft_alpha: float = 0.9
    soft_fraction_baseline: float = 0.03
    soft_fraction_amplitude: float = 0.17
    soft_fraction_rate: float = 0.8
    vacuum_width_rad: float = 0.08
    vacuum_width_pt_exponent: float = -0.15
    broadening_width_rad: float = 0.04
    soft_width_ratio: float = 2.2
    pt_scale_GeV: float = 100.0
    center_eta_range: tuple = (-0.5, 0.5)
    center_phi_range_rad: tuple = (-0.6, 0.6)

    @classmethod
    def from_config(cls, toy):
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in toy.items()
              if k in names}
        unknown = set(toy) - names
        if unknown:
            raise ValueError(f"unknown toy constants {sorted(unknown)}")
        return cls(**kw)

    @property
    def n_particles(self):
        return self.core_count + self.soft_count


@dataclass(frozen=True)
class Selection:
    jet_radius: float = 0.4
    jet_pt_range_GeV: tuple = (60.0, 200.0)
    jet_abs_eta_max: float = 1.0
    constituent_abs_eta_max: float = 2.5
    constituent_abs_phi_max_rad: float = 1.2
    constituent_pt_min_GeV: float = 0.0
    require_all_particles_in_one_jet: bool = True
    fixed_constituent_count: int = 32
    max_attempts_per_requested_event: int = 1000

    @classmethod
    def from_config(cls, sel):
        names = {f.name for f in fields(cls)}
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in sel.items()
              if k in names}
        if sel.get("jet_algorithm", "antikt") != "antikt":
            raise ValueError("only anti-kt is implemented")
        return cls(**kw)


REJECTION_REASONS = (
    "invalid_weights",
    "fiducial_constituent",
    "multiple_jets",
    "jet_pt",
    "jet_eta",
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def overlap_factors(x, y, b, R):
    fp = np.maximum(0.0, 1.0 - ((x - b / 2) ** 2 + y ** 2) / R ** 2)
    fm = np.maximum(0.0, 1.0 - ((x + b / 2) ** 2 + y ** 2) / R ** 2)
    return fp, fm


def sample_production_points(b, n, rng, R, max_rounds=10000):
    """Rejection-sample points with density proportional to F+ F- (= g_b^2).

    ``b`` is a scalar or a per-event array of length ``n``.
    """
    scalar = np.ndim(b) == 0
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), (n,))
    if np.any(b < 0) or np.any(b >= 2 * R):
        raise ValueError("b outside [0, 2 R_geom)")
    xmax = R - b / 2
    ymax = np.sqrt(np.maximum(R ** 2 - b ** 2 / 4, 0.0))
    fmax = (1 - b ** 2 / (4 * R ** 2)) ** 2  # attained at the origin
    if scalar:
        xs = np.empty(n)
        ys = np.empty(n)
        filled = 0
        rounds = 0
        while filled < n:
            rounds += 1
            if rounds > max_rounds:
                raise RuntimeError("production-point rejection sampling did not converge")
            m = max(2 * (n - filled), 64)
            x = rng.uniform(-xmax[0], xmax[0], m)
            y = rng.uniform(-ymax[0], ymax[0], m)
            fp, fm = overlap_factors(x, y, b[0], R)
            acc = rng.uniform(0, fmax[0], m) < fp * fm
            take = np.flatnonzero(acc)[: n - filled]
            xs[filled: filled + take.size] = x[take]
            ys[filled: filled + take.size] = y[take]
            filled += take.size
        return xs, ys
    xs = np.zeros(n)
    ys = np.zeros(n)
    todo = np.arange(n)
    for _ in range(max_rounds):
        if todo.size == 0:
            return xs, ys
        x = rng.uniform(-1, 1, todo.size) * xmax[todo]
        y = rng.uniform(-1, 1, todo.size) * ymax[todo]
        fp, fm = overlap_factors(x, y, b[todo], R)
        acc = rng.uniform(0, 1, todo.size) * fmax[todo] < fp * fm
        xs[todo[acc]] = x[acc]
        ys[todo[acc]] = y[acc]
        todo = todo[~acc]
    raise RuntimeError("production-point rejection sampling did not converge")


def exit_distance(x, y, nx, ny, b, R):
    """Forward distance to leave the overlap (minimum over both disks)."""
    out = np.full(x.shape, np.inf)
    for cx in (b / 2, -b / 2):
        dx, dy = x - cx, y
        dn = dx * nx + dy * ny
        disc = dn ** 2 + R ** 2 - (dx ** 2 + dy ** 2)
        disc = np.maximum(disc, 0.0)  # roundoff at the boundary
        out = np.minimum(out, -dn + np.sqrt(disc))
    return np.maximum(out, 0.0)


def path_moments(x0, y0, nx, ny, L, theta, c, steps=None):
    """I_m (m = 1, 2, 3) by the midpoint rule on [tau_init, L_exit].

    The integrand vanishes for ell < tau_init, so integrating from tau_init
    places the first discontinuity on a node instead of inside a cell.
    """
    b, T_ref, tau = theta
    if np.ndim(b):
        b = np.asarray(b)[:, None]
    steps = steps or c.integration_steps
    R = c.geometry_radius_fm
    lo = np.full(L.shape, tau)
    span = np.maximum(L - lo, 0.0)
    h = span / steps
    k = (np.arange(steps) + 0.5)[None, :]
    ell = lo[:, None] + h[:, None] * k
    px = x0[:, None] + ell * nx[:, None]
    py = y0[:, None] + ell * ny[:, None]
    fp, fm = overlap_factors(px, py, b, R)
    g = np.sqrt(fp * fm)
    T = T_ref * np.cbrt(g) * np.cbrt(tau / ell)
    active = T > c.temperature_cut_GeV
    r = np.where(active, T / c.temperature_scale_GeV, 0.0)
    w = (h / c.length_scale_fm)[:, None]
    return np.stack([(r ** m * w).sum(1) for m in (1, 2, 3)], -1)


# ---------------------------------------------------------------------------
# Event generation
# ---------------------------------------------------------------------------


def _dirichlet(alpha, k, n, rng):
    g = rng.standard_gamma(alpha, size=(n, k))
    s = g.sum(1, keepdims=True)
    return g / s, g


def generate_attempts(theta, n, rng, c, b_per_event=None):
    """Raw (unselected) toy events at one parameter vector.

    With ``b_per_event`` (array [n]) the impact parameter is event-local and
    ``theta[0]`` is ignored (shared medium settings with local geometry).
    """
    theta = tuple(float(v) for v in theta)
    b, T_ref, tau = theta
    if b_per_event is not None:
        b = np.asarray(b_per_event, dtype=np.float64)
        theta = (b, T_ref, tau)
    if tau <= 0:
        raise ValueError("tau_init must be positive")
    R = c.geometry_radius_fm
    x0, y0 = sample_production_points(b, n, rng, R)
    psi = rng.uniform(0, 2 * np.pi, n)
    nx, ny = np.cos(psi), np.sin(psi)
    L = exit_distance(x0, y0, nx, ny, b, R)
    I = path_moments(x0, y0, nx, ny, L, theta, c)
    b_event = np.broadcast_to(b, (n,)).astype(np.float64)

    nexp = c.power_law_exponent
    lo, hi = c.hard_pt_min_GeV ** (1 - nexp), c.hard_pt_max_GeV ** (1 - nexp)
    u = rng.uniform(0, 1, n)
    P0 = (lo + u * (hi - lo)) ** (1 / (1 - nexp))

    s = c.loss_lognormal_sigma
    Z = np.exp(s * rng.standard_normal(n) - s ** 2 / 2)
    P_vis = P0 * np.exp(-c.loss_coefficient * I[:, 2] * Z)

    w_core, g_core = _dirichlet(c.dirichlet_core_alpha, c.core_count, n, rng)
    w_soft, g_soft = _dirichlet(c.dirichlet_soft_alpha, c.soft_count, n, rng)
    f_soft = c.soft_fraction_baseline + c.soft_fraction_amplitude * (
        1 - np.exp(-c.soft_fraction_rate * I[:, 0]))
    weights = np.concatenate([w_core * (1 - f_soft)[:, None],
                              w_soft * f_soft[:, None]], 1)
    pt = P_vis[:, None] * weights

    ratio = P0 / c.pt_scale_GeV
    sig_vac = c.vacuum_width_rad * ratio ** c.vacuum_width_pt_exponent
    sig_core = np.sqrt(sig_vac ** 2 + c.broadening_width_rad ** 2 * I[:, 1] / ratio)
    sig_soft = c.soft_width_ratio * sig_core
    sig = np.concatenate([np.repeat(sig_core[:, None], c.core_count, 1),
                          np.repeat(sig_soft[:, None], c.soft_count, 1)], 1)
    deta = rng.standard_normal(sig.shape) * sig
    dphi = rng.standard_normal(sig.shape) * sig
    eta0 = rng.uniform(*c.center_eta_range, n)
    phi0 = rng.uniform(*c.center_phi_range_rad, n)
    eta = eta0[:, None] + deta
    phi = phi0[:, None] + dphi

    component = np.concatenate([np.zeros((n, c.core_count), np.int8),
                                np.ones((n, c.soft_count), np.int8)], 1)
    perm = np.argsort(rng.uniform(size=pt.shape), axis=1)
    take = lambda a: np.take_along_axis(a, perm, 1)  # noqa: E731
    eta, phi, pt, component = take(eta), take(phi), take(pt), take(component)

    valid = (np.all(np.isfinite(pt), 1) & np.all(pt > 0, 1)
             & np.all(np.isfinite(np.log(np.where(pt > 0, pt, 1.0))), 1)
             & (g_core.sum(1) > 0) & (g_soft.sum(1) > 0))
    return {
        "eta": eta, "phi": phi, "pt": pt, "valid": valid,
        "diag": {
            "x0": x0, "y0": y0, "psi": psi, "L_exit": L, "I1": I[:, 0],
            "I2": I[:, 1], "I3": I[:, 2], "P0": P0, "Z_loss": Z, "P_vis": P_vis,
            "f_soft": f_soft, "sigma_core": sig_core, "eta0": eta0, "phi0": phi0,
            "component": component, "b": b_event,
        },
    }


def apply_selection(ev, sel):
    """Selection of the fixed-count benchmark; returns (accepted, reason, reco)."""
    n = ev["pt"].shape[0]
    reason = np.full(n, -1, dtype=np.int8)
    ok = ev["valid"].copy()
    reason[~ok] = 0
    fid = (np.all(np.abs(ev["eta"]) < sel.constituent_abs_eta_max, 1)
           & np.all(np.abs(ev["phi"]) < sel.constituent_abs_phi_max_rad, 1)
           & np.all(ev["pt"] > sel.constituent_pt_min_GeV, 1))
    reason[ok & ~fid] = 1
    ok &= fid
    p4 = reco.four_vectors(ev["eta"], ev["phi"], ev["pt"])
    p4 = np.where(ok[:, None, None], p4, 0.0)
    r = reco.cluster_events(p4, sel.jet_radius)
    one = r["njets"] == 1
    if sel.require_all_particles_in_one_jet:
        one &= r["jet_nconst"] == ev["pt"].shape[1]
    reason[ok & ~one] = 2
    ok &= one
    ptok = (r["jet_pt"] >= sel.jet_pt_range_GeV[0]) & (r["jet_pt"] <= sel.jet_pt_range_GeV[1])
    reason[ok & ~ptok] = 3
    ok &= ptok
    etaok = np.abs(r["jet_eta"]) < sel.jet_abs_eta_max
    reason[ok & ~etaok] = 4
    ok &= etaok
    r["p4"] = p4
    return ok, reason, r


def make_event_id(stream, index, attempt):
    """Stable int64 id: 8-bit stream, 16-bit setting/ensemble, 32-bit attempt."""
    attempt = np.asarray(attempt, dtype=np.int64)
    return (np.int64(stream) << 48) | (np.int64(index) << 32) | attempt


def sample_toy(theta, count, rng, constants, selection, stream=0, index=0,
               batch=None, attempt_cap=None, b_sampler=None):
    """Generate ``count`` accepted events at ``theta``.

    Returns (records, manifest). Records hold observed features, four-vectors,
    FastJet membership and event ids; ``records['diagnostics']`` is the hidden
    diagnostic store (never part of fitting inputs).
    """
    sel = selection
    if constants.n_particles != sel.fixed_constituent_count:
        raise ValueError("core+soft count must equal the fixed constituent count")
    cap = attempt_cap or sel.max_attempts_per_requested_event * max(count, 1)
    batch = batch or max(256, min(20000, 2 * count))
    acc_parts = []
    attempts = 0
    reasons = np.zeros(len(REJECTION_REASONS), dtype=np.int64)
    n_acc = 0
    while n_acc < count and attempts < cap:
        m = int(min(batch, cap - attempts))
        bvec = None if b_sampler is None else b_sampler(m, rng)
        ev = generate_attempts(theta, m, rng, constants, b_per_event=bvec)
        ok, reason, r = apply_selection(ev, sel)
        for k in range(len(REJECTION_REASONS)):
            reasons[k] += int((reason == k).sum())
        idx = np.flatnonzero(ok)
        att = attempts + idx
        part = {
            "eta": ev["eta"][idx], "phi": ev["phi"][idx], "pt": ev["pt"][idx],
            "p4": r["p4"][idx], "membership": r["membership"][idx],
            "jet_pt": r["jet_pt"][idx], "jet_eta": r["jet_eta"][idx],
            "jet_phi": r["jet_phi"][idx],
            "event_id": make_event_id(stream, index, att),
            "diag": {k: v[idx] for k, v in ev["diag"].items()},
        }
        acc_parts.append(part)
        n_acc += idx.size
        attempts += m
    rec = _concat(acc_parts, count)
    accepted = rec["event_id"].shape[0]
    manifest = {
        "backend": BACKEND,
        "theta": dict(zip(PARAMETERS, map(float, theta))),
        "local_b": b_sampler is not None,
        "requested": int(count),
        "accepted": int(accepted),
        # Attempts up to and including the last kept event.
        "attempted": _attempts_used(rec, stream, index),
        # Whole generated batches (used for the acceptance estimate).
        "attempts_generated": int(attempts),
        "accepted_generated": int(n_acc),
        "acceptance": float(n_acc / attempts) if attempts else 0.0,
        "rejections": dict(zip(REJECTION_REASONS, reasons.tolist())),
        "complete": bool(accepted >= count),
        "attempt_cap": int(cap),
        "stream": int(stream),
        "index": int(index),
    }
    return rec, manifest


def _attempts_used(rec, stream, index):
    if rec["event_id"].size == 0:
        return 0
    return int((rec["event_id"][-1] & 0xFFFFFFFF) + 1)


def _concat(parts, count):
    out = {}
    for k in parts[0]:
        if k == "diag":
            out[k] = {d: np.concatenate([p[k][d] for p in parts])[:count]
                      for d in parts[0][k]}
        else:
            out[k] = np.concatenate([p[k] for p in parts])[:count]
    out["x"] = np.stack([out["eta"], out["phi"], np.log(out["pt"])], -1)
    return out


def constants_dict(c):
    return asdict(c)
