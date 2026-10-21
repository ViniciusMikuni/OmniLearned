"""Parameter designs, reserved settings, seeds and stream codes."""

import itertools

import numpy as np

from omnilearned.sbi.toy.simulator import PARAMETERS

# Event-id stream codes (8 bits): identify the split that produced an event.
STREAMS = {
    "train": 1,
    "validation": 2,
    "test": 3,
    "development": 4,
    "geometry_acceptance": 5,
    "sensitivity": 6,
    "local_test": 7,
    "predictive": 8,
    "jewel_train": 16,
    "jewel_validation": 17,
    "jewel_test": 18,
}


def design_settings(design):
    """Cartesian training design as an array [S, 3] in PARAMETERS order."""
    axes = [design[p] for p in PARAMETERS]
    return np.array(list(itertools.product(*axes)), dtype=np.float64)


def design_bounds(design):
    return [(float(min(design[p])), float(max(design[p]))) for p in PARAMETERS]


def reserved_settings(cfg):
    """Holdout settings, read only by the simulation/report driver."""
    h = cfg["holdout"]
    out = {"primary_test": np.array(h["primary_test"], dtype=np.float64)}
    for i, s in enumerate(h.get("additional_test_settings", [])):
        out[f"additional_test_{i}"] = np.array(s, dtype=np.float64)
    out["development_validation"] = np.array(h["development_validation"], dtype=np.float64)
    return out


def check_reservation(design, reserved, tol=1e-9):
    """No training tuple may equal a reserved setting; report axis placement."""
    settings = design_settings(design)
    report = {}
    for name, th in reserved.items():
        clash = np.any(np.all(np.abs(settings - th) < tol, 1))
        if clash:
            raise ValueError(f"reserved setting {name}={th} is a training tuple")
        between = {}
        for k, p in enumerate(PARAMETERS):
            ax = np.array(sorted(design[p]))
            between[p] = bool(ax[0] < th[k] < ax[-1] and not np.any(np.isclose(ax, th[k])))
        report[name] = {"theta": th.tolist(), "strictly_between_axis_values": between,
                        "inside_bounds": all(lo <= v <= hi for v, (lo, hi)
                                             in zip(th, design_bounds(design)))}
    return report


def rng_for(seed, *keys):
    """Independent generator for (seed, keys...) via SeedSequence spawning keys."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), *map(int, keys)])))
