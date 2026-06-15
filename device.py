"""
Device helpers for autoresearch.

The training code is intentionally small, but keeping device selection in one
place avoids scattering cuda/mps/cpu checks through the experiment loop.
"""

import contextlib
import os

import torch


def get_default_device():
    """Select the device requested by AUTORESEARCH_DEVICE, or auto-detect one."""
    requested = os.environ.get("AUTORESEARCH_DEVICE", "auto").lower()
    if requested in {"cuda", "gpu"}:
        if not torch.cuda.is_available():
            raise RuntimeError("AUTORESEARCH_DEVICE=cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("AUTORESEARCH_DEVICE=mps was requested, but MPS is not available.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if requested != "auto":
        raise RuntimeError(f"Unknown AUTORESEARCH_DEVICE={requested!r}; expected auto, cuda, mps, or cpu.")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cuda_capability(device):
    if device.type != "cuda":
        return None
    return torch.cuda.get_device_capability(device)


def get_device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"


def get_total_memory_gb(device):
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return props.total_memory / 1024**3
    return 0.0


def get_autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device.type == "mps":
        return torch.amp.autocast(device_type="mps", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def get_parameter_dtype(device):
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def get_peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    if device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0.0


def should_compile(device):
    requested = os.environ.get("AUTORESEARCH_COMPILE", "auto").lower()
    if requested in {"0", "false", "no", "off"}:
        return False
    if requested in {"1", "true", "yes", "on"}:
        return True
    return device.type == "cuda"
