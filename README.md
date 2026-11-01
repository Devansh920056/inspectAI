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

## Stage 1: dataset (MVTec AD)

Download and place MVTec AD (`bottle`, `hazelnut`, `carpet`) under `data/mvtec_ad/` — see [docs/dataset.md](docs/dataset.md).

```bash
python -m pip install -r requirements.txt            # torch, torchvision, pillow, numpy
python -m pip install -r requirements-notebooks.txt  # matplotlib, ipykernel (for notebooks)
python scripts/dataset_report.py --figures           # validate + reports/dataset_statistics.{json,md}
python -m unittest discover tests                    # tests/test_real_data.py runs once data is present
python -m ipykernel install --user --name inspectai  # then pick this kernel in Jupyter/VS Code
```

Notebooks (in `notebooks/`): `01_dataset_exploration.ipynb`, `02_image_preprocessing.ipynb`.

## Layout

```
inspectai/          Python package
  device.py           device selection (MPS / CUDA / CPU)
  data/               MVTec AD loading, validation, statistics, transforms, Dataset/DataLoader
scripts/            check_environment.py, dataset_report.py
notebooks/          exploration and preprocessing notebooks
docs/               dataset documentation
reports/            generated dataset statistics (JSON + Markdown, figures)
tests/              unittest tests (synthetic fixtures + real-data checks)
requirements.txt    Dependencies
pyproject.toml      Package metadata
```

`data/`, `checkpoints/`, `runs/`, model files and virtual environments are git-ignored — never commit them.

## Troubleshooting

- **MPS unavailable on Apple Silicon**: the script warns if Python runs under Rosetta
  (`platform.machine()` is `x86_64`). Recreate the venv with a native arm64 Python.
- **Wrong Python version**: the script warns if it is not 3.11.
