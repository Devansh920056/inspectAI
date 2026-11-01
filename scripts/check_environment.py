#!/usr/bin/env python3
"""Verify the InspectAI development environment.

Reports OS / chip / RAM / disk / Python, checks PyTorch and the selected
compute device, then runs a real forward + backward pass (and a short
optimisation loop) on that device, cross-checking results against CPU.

Exit code 0 = all required checks passed, 1 = a required check failed.
Usage: python scripts/check_environment.py [--device auto|cpu|mps|cuda]
"""

from __future__ import annotations

import argparse
import copy
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Let unsupported MPS ops fall back to CPU instead of crashing. Must be set
# before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REQUIRED_PYTHON = (3, 11)
results: list[tuple[str, bool, str]] = []  # (name, passed, detail) for required checks
warnings: list[str] = []


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def chip_name() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor() or "unknown"
    return platform.processor() or platform.machine() or "unknown"


def ram_gb() -> float | None:
    if sys.platform == "darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        return int(out) / 1024**3 if out.isdigit() else None
    try:
        with open("/proc/meminfo") as f:
            return int(f.readline().split()[1]) / 1024**2
    except (OSError, ValueError, IndexError):
        return None


def under_rosetta() -> bool:
    return sys.platform == "darwin" and _run(["sysctl", "-n", "sysctl.proc_translated"]) == "1"


def check(name: str, passed: bool, detail: str = "") -> bool:
    results.append((name, passed, detail))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    return passed


def section(title: str) -> None:
    print(f"\n== {title} ==")


def report_system() -> None:
    section("System")
    if sys.platform == "darwin":
        print(f"  OS            : macOS {platform.mac_ver()[0]}")
    else:
        print(f"  OS            : {platform.platform()}")
    print(f"  Machine arch  : {platform.machine()}")
    print(f"  Chip          : {chip_name()}")
    ram = ram_gb()
    print(f"  RAM           : {ram:.1f} GB" if ram else "  RAM           : unknown")
    usage = shutil.disk_usage(REPO_ROOT)
    print(f"  Disk (repo)   : {usage.free / 1024**3:.0f} GB free of {usage.total / 1024**3:.0f} GB")
    print(f"  Python        : {platform.python_version()} ({sys.executable})")
    is_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    # A conda env is not a venv (prefix == base_prefix) but is an isolated env
    # unless it is conda's "base".
    is_conda_env = (
        os.environ.get("CONDA_PREFIX") == sys.prefix
        and os.environ.get("CONDA_DEFAULT_ENV", "base") != "base"
    )
    in_venv = is_venv or is_conda_env
    print(f"  Virtualenv    : {'yes (' + sys.prefix + ')' if in_venv else 'no'}")

    if under_rosetta():
        warnings.append("Python is running under Rosetta (x86_64 emulation); MPS will be unavailable. "
                        "Recreate the venv with a native arm64 Python.")
    if sys.version_info[:2] != REQUIRED_PYTHON:
        warnings.append(f"Python {platform.python_version()} in use; project targets "
                        f"{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}.")
    if not in_venv:
        warnings.append("Not running inside a virtual environment (or non-base conda env).")


def import_torch():
    section("PyTorch")
    try:
        import numpy
        import torch
    except ImportError as exc:
        check("import torch/numpy", False, f"{exc} -- run: pip install -r requirements.txt")
        return None
    check("import torch/numpy", True, f"torch {torch.__version__}, numpy {numpy.__version__}")
    from inspectai.device import mps_available

    print(f"  MPS built     : {torch.backends.mps.is_built()}")
    print(f"  MPS available : {mps_available()}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  CPU threads   : {torch.get_num_threads()}")
    return torch


def tensor_tests(torch, device) -> None:
    import torch.nn as nn
    import torch.nn.functional as F

    section(f"Forward/backward test on '{device}'")

    def sync():
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # 1. Plain autograd: d/da sum(a^2) = 2a
    a = torch.randn(8, 8, device=device, requires_grad=True)
    (a * a).sum().backward()
    sync()
    check("autograd d/da sum(a^2) == 2a", torch.allclose(a.grad, 2 * a.detach(), atol=1e-5))

    # 2. Small MLP: compare device result against a CPU reference
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1))
    x, y = torch.randn(64, 16), torch.randn(64, 1)

    def fwd_bwd(dev):
        m = copy.deepcopy(base).to(dev)
        out = m(x.to(dev))
        loss = F.mse_loss(out, y.to(dev))
        loss.backward()
        if dev.type == "mps":
            torch.mps.synchronize()
        return out.detach().cpu(), loss.item(), [p.grad.detach().cpu() for p in m.parameters()]

    cpu_out, cpu_loss, cpu_grads = fwd_bwd(torch.device("cpu"))
    dev_out, dev_loss, dev_grads = fwd_bwd(device)
    check("forward output finite", bool(torch.isfinite(dev_out).all()), f"loss={dev_loss:.6f}")
    check("gradients finite and non-zero",
          all(bool(torch.isfinite(g).all()) and g.abs().sum() > 0 for g in dev_grads),
          f"{len(dev_grads)} tensors")
    check("matches CPU reference (atol 1e-4)",
          torch.allclose(dev_out, cpu_out, atol=1e-4)
          and all(torch.allclose(d, c, atol=1e-4) for d, c in zip(dev_grads, cpu_grads)),
          f"|loss diff|={abs(dev_loss - cpu_loss):.2e}")

    # 3. Short optimisation loop: loss must drop
    model = copy.deepcopy(base).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    xd, yd = x.to(device), y.to(device)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = F.mse_loss(model(xd), yd)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    check("SGD loss decreases over 30 steps", losses[-1] < losses[0],
          f"{losses[0]:.4f} -> {losses[-1]:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

    print("InspectAI environment check")
    report_system()
    torch = import_torch()
    if torch is not None:
        from inspectai.device import select_device

        try:
            device = select_device(args.device)
        except RuntimeError as exc:
            check("select device", False, str(exc))
        else:
            check("select device", True, f"{device} (requested: {args.device})")
            try:
                tensor_tests(torch, device)
            except Exception as exc:  # report any backend failure as a check failure
                check("tensor tests ran", False, f"{type(exc).__name__}: {exc}")

    section("Summary")
    for w in warnings:
        print(f"  [WARN] {w}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"  {len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("  FAILED: " + ", ".join(failed))
        return 1
    print("  Environment OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
