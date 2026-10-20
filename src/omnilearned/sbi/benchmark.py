"""Runtime and memory of per-event likelihood evaluation.

Times one ``log_prob`` call for combinations of particle count, Hutchinson
probe count and batch rows (event x candidate pairs) on a given device and
reports seconds per row and per row-function-evaluation, and peak device
memory where the backend exposes it. Scan cost then follows as
rows = events x candidates x refinement rounds.
"""

import time

import numpy as np
import torch

from omnilearned.sbi.device import synchronize
from omnilearned.sbi.likelihood import LikelihoodOptions, log_prob


def _memory(device):
    device = torch.device(device)
    if device.type == "mps":
        return {"current_allocated_MB": torch.mps.current_allocated_memory() / 2 ** 20,
                "driver_allocated_MB": torch.mps.driver_allocated_memory() / 2 ** 20}
    if device.type == "cuda":
        return {"peak_allocated_MB": torch.cuda.max_memory_allocated(device) / 2 ** 20}
    return {}


def benchmark_log_prob(model, base, cond_dim, device, particle_counts=(8, 16, 32, 64),
                       probes=(1, 4, 16), rows=(256, 1024, 4096), solver="midpoint",
                       nsteps=8, seed=0):
    device = torch.device(device)
    out = []
    g = torch.Generator().manual_seed(seed)
    for P in particle_counts:
        for K in probes:
            for R in rows:
                x = torch.randn(R, P, base.num_features, generator=g)
                mask = torch.ones(R, P, dtype=torch.bool)
                cond = torch.randn(R, cond_dim, generator=g)
                o = LikelihoodOptions(solver=solver, nsteps=nsteps, num_probes=K,
                                      divergence="hutchinson")
                if device.type == "mps":
                    torch.mps.empty_cache()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                args = (model, x.to(device), mask.to(device), base)
                log_prob(*args, cond=cond.to(device), options=o, event_ids=np.arange(R))
                synchronize(device)
                t0 = time.perf_counter()
                _, d = log_prob(*args, cond=cond.to(device), options=o, event_ids=np.arange(R))
                synchronize(device)
                sec = time.perf_counter() - t0
                out.append({"particles": P, "probes": K, "rows": R, "solver": solver,
                            "nsteps": nsteps, "nfe": d["nfe"], "seconds": sec,
                            "ms_per_row": 1e3 * sec / R,
                            "us_per_row_nfe": 1e6 * sec / (R * d["nfe"]),
                            **_memory(device)})
    return out
