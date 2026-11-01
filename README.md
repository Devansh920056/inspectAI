# InspectAI

Stage 0: reproducible development environment (Python 3.11, PyTorch, Apple MPS with CPU fallback).

## Requirements

- macOS on Apple Silicon (MPS) — Intel Macs and Linux work on CPU
- Python **3.11** (native arm64 build, not Rosetta)
- Git

## Setup

```bash
# 1. Install Python 3.11 if you don't have it (either option)
brew install python@3.11
#   or: uv python install 3.11

# 2. Create and activate the virtual environment
python3.11 -m venv .venv          # or: uv venv --python 3.11
source .venv/bin/activate

# Conda alternative (if you use Anaconda/Miniconda):
#   conda create -p ./.venv python=3.11 -y
#   conda activate ./.venv
# Note: conda warns when the project path contains spaces; it works, but a
# space-free path is safer.

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Verify
python scripts/check_environment.py
python -m unittest discover tests
```

`check_environment.py` prints hardware/OS/Python details, whether MPS is available,
the selected device, and runs a forward + backward pass (cross-checked against CPU)
plus a short SGD loop. It exits non-zero if any required check fails.

Force a device with `--device cpu|mps|cuda|auto` (default `auto`: MPS, then CUDA if present, then CPU).
`PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the script so ops unsupported on MPS fall back to CPU.

## Layout

```
inspectai/          Python package (device.py: device selection)
scripts/            Utility scripts (check_environment.py)
tests/              unittest tests
requirements.txt    Dependencies
pyproject.toml      Package metadata
```

`data/`, `checkpoints/`, `runs/`, model files and virtual environments are git-ignored — never commit them.

## Troubleshooting

- **MPS unavailable on Apple Silicon**: the script warns if Python runs under Rosetta
  (`platform.machine()` is `x86_64`). Recreate the venv with a native arm64 Python.
- **Wrong Python version**: the script warns if it is not 3.11.
