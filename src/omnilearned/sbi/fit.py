"""Configuration-driven Route A fit (``omnilearned sbi-fit``).

A fit configuration contains only what a measured-data fit may know: the
observation file, the fixed checkpoint, the candidate domain and prior, and
numerical options. It has no truth section. Example (TOML)::

    schema = "omnilearned-sbi-fit-v1"
    checkpoint = "runs/x/checkpoint.pt"
    checkpoint_sha256 = "..."
    observations = "runs/x/observations.h5"
    observations_sha256 = "..."
    output_prefix = "runs/x/fit/result"
    device = "mps"
    mode = "fixed_vector"            # or "local_variable"

    [grid]                            # order defines the parameter vector
    b = {min = 0.0, max = 11.0, n = 7}
    T_ref = {min = 0.25, max = 0.55, n = 7}

    [prior]
    type = "uniform"
    bounds = {b = [0.0, 11.0], T_ref = [0.25, 0.55]}

    [likelihood]                      # LikelihoodOptions fields
    solver = "midpoint"
    nsteps = 32

    [scan]
    event_chunk = 256
    candidate_chunk = 8
    max_seconds = 900
    refine = {rounds = 1, delta_logpost = 12.0, points = [9, 9]}

    [local]                           # only for mode = "local_variable"
    name = "b"
    nodes = {min = 0.0, max = 11.0, n = 17}

Application code may pass ``log_norm_fn`` (selected-distribution
normalization) and ``local_log_weight_fn`` (normalized local-variable law)
callables; the configuration records their identity.
"""

import json
import os
import time

import numpy as np

from omnilearned.sbi.flow import FlowSurrogate
from omnilearned.sbi.device import resolve_device
from omnilearned.sbi.likelihood import LikelihoodOptions
from omnilearned.sbi.observations import file_sha256, read_observations
from omnilearned.sbi.scan import (
    Grid,
    ScanResult,
    joint_scan,
    local_variable_loglik,
    refine_axes,
    uniform_log_prior,
)


FIT_SCHEMA = "omnilearned-sbi-fit-v1"
FORBIDDEN_KEYS = ("truth", "holdout", "theta_star", "test_settings")


def load_fit_config(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    check_fit_config(cfg)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_config_sha256"] = file_sha256(path)
    return cfg


def check_fit_config(cfg):
    if cfg.get("schema") != FIT_SCHEMA:
        raise ValueError(f"fit config schema must be {FIT_SCHEMA}")

    def walk(obj, path=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if any(f in str(k).lower() for f in FORBIDDEN_KEYS):
                    raise ValueError(f"fit config must not contain '{path}{k}'")
                walk(v, f"{path}{k}.")

    walk(cfg)


def _grid(spec):
    return Grid.from_spec(spec)


def _prior(cfg, grid):
    prior = cfg.get("prior", {"type": "uniform"})
    if prior.get("type", "uniform") != "uniform":
        raise ValueError("only uniform priors are configurable from TOML")
    bounds = prior.get("bounds")
    if bounds is not None:
        bounds = [tuple(bounds[n]) for n in grid.names]
    return uniform_log_prior(grid, bounds)


def make_logp_fn(surrogate, obs, options, names, fixed=None):
    """(event_slice, candidates) -> [m, n] per-event log densities.

    ``fixed`` maps parameter names held fixed (e.g. a local node value) when
    ``names`` covers only part of the condition layout.
    """
    diag = {"seconds": 0.0, "nfe": 0, "rows": 0, "nonfinite": 0}

    def fn(sl, cand):
        cand = np.atleast_2d(cand)
        full_names = list(names)
        if fixed:
            extra = np.array([[fixed[k] for k in fixed]] * cand.shape[0])
            cand = np.concatenate([cand, extra], 1)
            full_names += list(fixed)
        ctx = None if obs.context is None else obs.context[sl]
        lp, d = surrogate.log_prob(obs.x[sl], obs.mask[sl], cand, obs.event_id[sl],
                                   options, candidate_names=full_names, context=ctx)
        diag["seconds"] += d["seconds"]
        diag["nfe"] += d["nfe"]
        diag["rows"] += d["rows"]
        diag["nonfinite"] += d["nonfinite"]
        return lp

    return fn, diag


def run_fit(cfg, log_norm_fn=None, local_log_weight_fn=None, log=print,
            surrogate=None, observations=None, extra_provenance=None):
    """Execute a fit configuration; returns the result document."""
    check_fit_config(cfg)
    start = time.time()
    dev = resolve_device(cfg.get("device", "auto"), cfg.get("allow_device_fallback", False))
    if surrogate is None:
        if cfg.get("checkpoint_sha256") and file_sha256(cfg["checkpoint"]) != cfg["checkpoint_sha256"]:
            raise ValueError("checkpoint hash does not match fit configuration")
        surrogate = FlowSurrogate.load(cfg["checkpoint"], device=dev.device)
    obs = observations or read_observations(cfg["observations"])
    if cfg.get("observations_sha256") and obs.sha256 != cfg["observations_sha256"]:
        raise ValueError("observation hash does not match fit configuration")
    options = LikelihoodOptions.from_dict(cfg.get("likelihood", {}))
    scan_cfg = cfg.get("scan", {})
    mode = cfg.get("mode", "fixed_vector")
    out_prefix = cfg["output_prefix"]
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    provenance = {
        "fit_config_sha256": cfg.get("_config_sha256"),
        "checkpoint_sha256": surrogate.metadata.get("checkpoint_sha256"),
        "observations_sha256": obs.sha256,
        "n_events": len(obs),
        "device": dev.to_dict(),
        "likelihood_options": options.to_dict(),
        "measure": "physical-coordinate conditional density of the observed "
                   "constituent features given the selection (fixed multiplicity)",
        "log_norm": None if log_norm_fn is None else getattr(log_norm_fn, "description", "callable"),
    }
    provenance.update(extra_provenance or {})

    def progress(done, total, sec):
        log(f"  candidates {done}/{total} {sec:.0f}s")

    if mode == "fixed_vector":
        grid = _grid(cfg["grid"])
        rounds = []
        refine = scan_cfg.get("refine", {})
        for r in range(1 + int(refine.get("rounds", 0))):
            fn, diag = make_logp_fn(surrogate, obs, options, grid.names)
            res = joint_scan(fn, len(obs), grid,
                             event_chunk=scan_cfg.get("event_chunk", 256),
                             candidate_chunk=scan_cfg.get("candidate_chunk", 8),
                             log_prior=_prior(cfg, grid), log_norm_fn=log_norm_fn,
                             keep_per_event=True, progress=progress,
                             max_seconds=scan_cfg.get("max_seconds"))
            res.diagnostics.update({"likelihood": diag, "round": r})
            prefix = f"{out_prefix}_round{r}"
            res.save(prefix, extra={"provenance": provenance})
            rounds.append(prefix)
            log(f"round {r}: grid {grid.shape} mle {res.summary()['mle']} "
                f"{res.diagnostics['seconds']:.0f}s complete={res.diagnostics['complete']}")
            if not res.diagnostics["complete"]:
                break
            if r < int(refine.get("rounds", 0)):
                grid = refine_axes(res, refine.get("delta_logpost", 12.0),
                                   refine.get("points"))
        doc = res.save(out_prefix, extra={"provenance": provenance, "rounds": rounds})
    elif mode == "local_variable":
        doc = _run_local(cfg, surrogate, obs, options, scan_cfg, local_log_weight_fn,
                         log_norm_fn, provenance, out_prefix, progress, log)
    else:
        raise ValueError(f"unknown fit mode {mode}")
    doc["wall_seconds"] = time.time() - start
    with open(f"{out_prefix}.json", "w") as f:
        json.dump(doc, f, indent=1, default=_default)
    return doc


def _run_local(cfg, surrogate, obs, options, scan_cfg, weight_fn, log_norm_fn,
               provenance, out_prefix, progress, log):
    """Shared parameters on a grid; local variable integrated per event."""
    if weight_fn is None:
        raise ValueError("local_variable mode needs local_log_weight_fn")
    loc = cfg["local"]
    name = loc["name"]
    nodes = np.linspace(loc["nodes"]["min"], loc["nodes"]["max"], int(loc["nodes"]["n"]))
    grid = _grid(cfg["grid"])  # shared parameters only
    full = Grid([name] + grid.names, [nodes] + grid.axes)
    fn, diag = make_logp_fn(surrogate, obs, options, full.names)
    # log_norm_fn (if any) acts per (local, shared) node and per event.
    res = joint_scan(fn, len(obs), full, event_chunk=scan_cfg.get("event_chunk", 256),
                     candidate_chunk=scan_cfg.get("candidate_chunk", 8),
                     keep_per_event=True, progress=progress,
                     max_seconds=scan_cfg.get("max_seconds"))
    per = res.per_event.reshape(len(nodes), grid.size, len(obs))
    if log_norm_fn is not None:
        ln = np.asarray(log_norm_fn(full.points()), dtype=np.float64)
        per = per - ln.reshape(len(nodes), grid.size)[:, :, None]
    log_w = np.asarray(weight_fn(nodes, grid.points()), dtype=np.float64)
    ll, per_marg, local_post = local_variable_loglik(per, log_w)
    shared = ScanResult(grid, ll.reshape(grid.shape), _prior(cfg, grid),
                        per_event=per_marg, n_events=len(obs), mode="local_variable",
                        diagnostics={"likelihood": diag, **res.diagnostics,
                                     "local_name": name, "local_nodes": nodes.tolist()})
    np.savez_compressed(f"{out_prefix}_local.npz", nodes=nodes, log_weights=log_w,
                        per_event_node_logp=per, local_posterior=local_post)
    doc = shared.save(out_prefix, extra={"provenance": provenance,
                                         "local_variable": {"name": name,
                                                            "nodes": nodes.tolist()}})
    log(f"local fit: mle {doc['summary']['mle']} {res.diagnostics['seconds']:.0f}s")
    return doc


def _default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)
