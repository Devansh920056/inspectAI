"""Compute-device selection with graceful CPU fallback."""

from __future__ import annotations

import torch

VALID_PREFERENCES = ("auto", "cpu", "mps", "cuda")


def mps_available() -> bool:
    """True if this PyTorch build has MPS and the hardware/OS supports it."""
    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_built() and backend.is_available())


def select_device(preference: str = "auto") -> torch.device:
    """Return the device to compute on.

    ``auto`` prefers Apple MPS, then CUDA (only if present), then CPU.
    An explicit preference that is unavailable raises ``RuntimeError`` instead
    of silently falling back, so misconfiguration is visible.
    """
    if preference not in VALID_PREFERENCES:
        raise ValueError(f"preference must be one of {VALID_PREFERENCES}, got {preference!r}")

    if preference == "cpu":
        return torch.device("cpu")
    if preference == "mps":
        if not mps_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")

    if mps_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
