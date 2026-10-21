"""Toy configuration loading and reproducible single-setting draws."""

import hashlib
from pathlib import Path

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from omnilearned.sbi.toy import simulator
from omnilearned.sbi.toy.design import STREAMS, rng_for

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "toy-quenching-v1.toml"


def load_toy_config(path=DEFAULT_CONFIG):
    """(config dict, ToyConstants, Selection) from a toy TOML file."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if cfg.get("backend", simulator.BACKEND) != simulator.BACKEND:
        raise ValueError(f"config backend {cfg.get('backend')} != {simulator.BACKEND}")
    return (cfg, simulator.ToyConstants.from_config(cfg["toy"]),
            simulator.Selection.from_config(cfg["selection"]))


def sample_setting(theta, n, seed, stream="predictive", index=0, path=DEFAULT_CONFIG):
    """Selected direct toy jets at one setting; generator (seed, stream, index)."""
    _, c, s = load_toy_config(path)
    rng = rng_for(seed, STREAMS[stream], index)
    return simulator.sample_toy(np.asarray(theta, dtype=np.float64), int(n), rng, c, s,
                                stream=STREAMS[stream], index=index)


def array_sha256(rec):
    """Platform-independent hash of the observed records (features and ids)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(rec["x"], dtype="<f8").tobytes())
    h.update(np.ascontiguousarray(rec["event_id"], dtype="<i8").tobytes())
    return h.hexdigest()
