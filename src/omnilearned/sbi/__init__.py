"""Likelihood-based simulation inference (Route A) for OmniLearned generators.

Modules:
    device        explicit CPU/CUDA/MPS selection with recorded fallback
    base          checkpoint-aware Gaussian base distribution
    transforms    bounded/affine feature transforms with log Jacobians
    parameters    named condition layout (unknown parameters vs known context)
    likelihood    per-event flow log densities (exact or Hutchinson divergence)
    scan          joint grids, posterior/profile/marginal, local-variable fits
    observations  truth-free observation records (closure and measured data)
    velocity      density-compatible set transformer and PET2 adapter
    flow          single-process flow-matching training and checkpoint bundle
    fit           configuration-driven fit entry point (``omnilearned sbi-fit``)
"""

from omnilearned.sbi.base import GaussianBase
from omnilearned.sbi.device import resolve_device, DeviceInfo
from omnilearned.sbi.likelihood import (
    LikelihoodOptions,
    log_prob,
    sample,
    frozen_for_inference,
    stable_probes,
)
from omnilearned.sbi.parameters import ConditionLayout
from omnilearned.sbi.scan import (
    Grid,
    ScanResult,
    joint_scan,
    local_variable_loglik,
    normalized_log_weights,
    refine_axes,
    trapezoid_weights,
    uniform_log_prior,
)
from omnilearned.sbi.transforms import FeatureTransform
from omnilearned.sbi.observations import (
    Observations,
    read_observations,
    write_observations,
)

__all__ = [
    "GaussianBase",
    "resolve_device",
    "DeviceInfo",
    "LikelihoodOptions",
    "log_prob",
    "sample",
    "frozen_for_inference",
    "stable_probes",
    "ConditionLayout",
    "Grid",
    "ScanResult",
    "joint_scan",
    "local_variable_loglik",
    "normalized_log_weights",
    "refine_axes",
    "trapezoid_weights",
    "uniform_log_prior",
    "FeatureTransform",
    "Observations",
    "read_observations",
    "write_observations",
]
