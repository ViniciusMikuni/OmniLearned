"""Observation input path shared by closure pseudo data and measured data.

An observation file is HDF5 with only these datasets:

* ``x``        [N, P, F] float constituent features in physical coordinates
* ``mask``     [N, P] bool validity mask (explicit; no feature is a sentinel)
* ``event_id`` [N] int64 stable identifiers (select Hutchinson probes)
* ``context``  optional [N, C] known measured context, attribute ``names``

Attributes ``feature_names`` and ``record_schema`` describe the content.
Truth parameters, generator histories and diagnostics must live elsewhere;
the reader refuses files that carry them so measured-data and closure fits
use an identical path.
"""

import hashlib
from dataclasses import dataclass

import h5py
import numpy as np


ALLOWED = ("x", "mask", "event_id", "context")
FORBIDDEN_PREFIXES = ("truth", "theta", "param", "label", "diagnostic", "history")
RECORD_SCHEMA = "omnilearned-sbi-observations-v1"


@dataclass
class Observations:
    x: np.ndarray
    mask: np.ndarray
    event_id: np.ndarray
    context: np.ndarray = None
    context_names: tuple = ()
    feature_names: tuple = ()
    sha256: str = ""
    path: str = ""

    def __len__(self):
        return self.x.shape[0]

    def subset(self, index):
        return Observations(
            self.x[index],
            self.mask[index],
            self.event_id[index],
            None if self.context is None else self.context[index],
            self.context_names,
            self.feature_names,
            self.sha256,
            self.path,
        )


def file_sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_observations(path, x, mask, event_id, feature_names, context=None,
                       context_names=(), attrs=None):
    x = np.asarray(x)
    if x.dtype != np.float64:  # float64 is kept (no rounding onto bounds)
        x = x.astype(np.float32)
    mask = np.asarray(mask, dtype=bool)
    event_id = np.asarray(event_id, dtype=np.int64)
    if x.ndim != 3 or mask.shape != x.shape[:2] or event_id.shape != x.shape[:1]:
        raise ValueError("inconsistent observation shapes")
    if np.unique(event_id).size != event_id.size:
        raise ValueError("event ids must be unique")
    if np.any(x[~mask] != 0):
        raise ValueError("padded entries must be zero")
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=x, compression="gzip")
        f.create_dataset("mask", data=mask, compression="gzip")
        f.create_dataset("event_id", data=event_id)
        if context is not None:
            ds = f.create_dataset("context", data=np.asarray(context, np.float64))
            ds.attrs["names"] = list(context_names)
        f.attrs["feature_names"] = list(feature_names)
        f.attrs["record_schema"] = RECORD_SCHEMA
        for k, v in (attrs or {}).items():
            if k.lower().startswith(FORBIDDEN_PREFIXES):
                raise ValueError(f"attribute '{k}' not allowed in observations")
            f.attrs[k] = v
    return file_sha256(path)


def read_observations(path):
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        bad = [k for k in keys if k.lower().startswith(FORBIDDEN_PREFIXES)]
        bad += [k for k in f.attrs if k.lower().startswith(FORBIDDEN_PREFIXES)]
        if bad:
            raise ValueError(
                f"observation file {path} carries truth/diagnostic fields {bad}; "
                "fitting inputs must contain observations only"
            )
        extra = [k for k in keys if k not in ALLOWED]
        if extra:
            raise ValueError(f"unexpected datasets {extra} in {path}")
        if f.attrs.get("record_schema") != RECORD_SCHEMA:
            raise ValueError(f"{path} is not a {RECORD_SCHEMA} file")
        x = f["x"][...]
        mask = f["mask"][...].astype(bool)
        eid = f["event_id"][...].astype(np.int64)
        context, cnames = None, ()
        if "context" in f:
            context = f["context"][...]
            cnames = tuple(f["context"].attrs.get("names", []))
        fnames = tuple(f.attrs.get("feature_names", []))
    return Observations(x, mask, eid, context, cnames, fnames, file_sha256(path),
                        str(path))
