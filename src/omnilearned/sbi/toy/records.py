"""Event record files.

* Training/validation sets (simulation only): features, mask, event ids and
  the generating parameter labels, plus four-vectors and FastJet membership.
* Observation files: framework ``omnilearned.sbi.observations`` format with
  features, mask and event ids only.
* Diagnostic stores: hidden generator history (positions, hard scale, path
  moments, component labels). Never read by fitting code.
* Truth manifests: JSON under a separate ``truth/`` directory, opened only by
  the report after fit outputs are sealed.
"""

import json
from pathlib import Path

import h5py
import numpy as np

from omnilearned.sbi.observations import file_sha256, write_observations

FEATURES = ("eta", "phi", "log_pt_over_GeV")


def write_training_set(path, parts):
    """parts: list of (theta, records) for one split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.concatenate([r["x"] for _, r in parts])
    theta = np.concatenate([np.repeat(np.asarray(t)[None], r["x"].shape[0], 0)
                            for t, r in parts])
    with h5py.File(path, "w") as f:
        f["x"] = x
        f["mask"] = np.ones(x.shape[:2], dtype=bool)
        f["event_id"] = np.concatenate([r["event_id"] for _, r in parts])
        f["theta"] = theta
        f["setting_index"] = np.concatenate(
            [np.full(r["x"].shape[0], i) for i, (_, r) in enumerate(parts)])
        f["p4"] = np.concatenate([r["p4"] for _, r in parts]).astype(np.float32)
        f["membership"] = np.concatenate([r["membership"] for _, r in parts]).astype(np.int8)
        f["jet_pt"] = np.concatenate([r["jet_pt"] for _, r in parts])
        f["jet_eta"] = np.concatenate([r["jet_eta"] for _, r in parts])
        f.attrs["feature_names"] = list(FEATURES)
        f.attrs["parameter_names"] = ["b", "T_ref", "tau_init"]
    return file_sha256(path)


def read_training_set(path):
    with h5py.File(path, "r") as f:
        return {k: f[k][...] for k in ("x", "mask", "event_id", "theta", "setting_index",
                                         "jet_pt", "jet_eta")}


def write_diagnostics(path, parts):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f["event_id"] = np.concatenate([r["event_id"] for _, r in parts])
        for k in parts[0][1]["diag"]:
            f[k] = np.concatenate([r["diag"][k] for _, r in parts])
        f.attrs["access"] = "diagnostic only; never an input to fitting"
    return file_sha256(path)


def write_observation_file(path, records, attrs=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask = records.get("mask")
    if mask is None:
        mask = np.ones(records["x"].shape[:2], dtype=bool)
    x = np.where(mask[..., None], records["x"], 0.0).astype(np.float64)
    return write_observations(path, x, mask, records["event_id"], FEATURES, attrs=attrs)


def write_truth(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return file_sha256(path)
