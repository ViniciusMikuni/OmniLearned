"""Explicit single-process device selection.

Upstream training chooses CUDA or CPU implicitly. Likelihood inference and
small conditional generators also need Apple Silicon MPS, which is never
selected silently here: callers ask for a device and receive a record of what
was actually used, including any explicitly permitted fallback.
"""

from dataclasses import dataclass, asdict

import torch


SUPPORTED = ("auto", "cpu", "cuda", "mps")


@dataclass
class DeviceInfo:
    requested: str
    device: str
    fallback: bool = False
    reason: str = ""
    torch_version: str = torch.__version__

    @property
    def torch_device(self):
        return torch.device(self.device)

    def to_dict(self):
        return asdict(self)


def mps_available():
    return bool(
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )


def resolve_device(requested="auto", allow_fallback=False):
    """Return a DeviceInfo for ``requested``.

    ``auto`` prefers CUDA, then MPS, then CPU. An explicitly requested
    accelerator that is unavailable raises unless ``allow_fallback`` is set, in
    which case CPU is used and the fallback is recorded.
    """
    requested = (requested or "auto").lower()
    base = requested.split(":")[0]
    if base not in SUPPORTED:
        raise ValueError(f"Unsupported device '{requested}', choose from {SUPPORTED}")

    if base == "auto":
        if torch.cuda.is_available():
            return DeviceInfo(requested, "cuda:0", reason="auto: cuda available")
        if mps_available():
            return DeviceInfo(requested, "mps", reason="auto: mps available")
        return DeviceInfo(requested, "cpu", reason="auto: no accelerator")

    if base == "cpu":
        return DeviceInfo(requested, "cpu")

    available = torch.cuda.is_available() if base == "cuda" else mps_available()
    if available:
        return DeviceInfo(requested, requested if base == "cuda" else "mps")
    if not allow_fallback:
        raise RuntimeError(
            f"Requested device '{requested}' is not available and fallback "
            "was not allowed"
        )
    return DeviceInfo(
        requested, "cpu", fallback=True, reason=f"{requested} unavailable"
    )


def model_dtype(device):
    """Model tensors are float32 on every device (MPS has no float64)."""
    return torch.float32


def accumulation_device(device):
    """Device for float64 ensemble accumulation (CPU unless CUDA)."""
    device = torch.device(device)
    return device if device.type == "cuda" else torch.device("cpu")


def to_cpu_float64(t):
    """Move a tensor to CPU, then widen to float64.

    The two steps are deliberate: with torch 2.14.1 on MPS the fused
    ``t.to("cpu", torch.float64)`` silently returns zeros for float32 input.
    """
    return t.detach().cpu().to(torch.float64)


def autocast_device_type(device):
    return torch.device(device).type


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
