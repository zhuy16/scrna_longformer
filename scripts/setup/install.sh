#!/usr/bin/env bash
set -euo pipefail

# scrna-longformer preparation script (mamba-first)
# Usage: bash preparation/INSTALL.sh

ENV_NAME="scrna_fixed"
PYTHON_VERSION="3.10"

echo "== scrna-longformer installation (mamba variant) =="

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Please install Miniconda/Anaconda first." >&2
  exit 1
fi

# ensure conda is initialized in this shell
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
  echo "Creating conda env '${ENV_NAME}' with python=${PYTHON_VERSION}..."
  conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

echo "Ensuring mamba is available (installed into base)..."
if ! command -v mamba >/dev/null 2>&1; then
  echo "Installing mamba into base environment..."
  conda install -n base -c conda-forge mamba -y
fi

echo "Activating environment '${ENV_NAME}'"
conda activate "${ENV_NAME}"

echo "Installing scientific dependencies via mamba (conda-forge)..."
mamba install -n "${ENV_NAME}" -c conda-forge scanpy anndata umap-learn python-igraph leidenalg -y

echo "Installing PyTorch (pytorch + torchvision + torchaudio)..."
# let conda/mamba choose the best compatible build (MPS/CPU on macOS)
mamba install -n "${ENV_NAME}" -c pytorch -c conda-forge pytorch torchvision torchaudio -y

echo "Installing PyYAML..."
mamba install -n "${ENV_NAME}" -c conda-forge pyyaml -y

echo "Installing any remaining pip packages from requirements (fallback)..."
# activate to ensure pip targets the right env
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r ../requirements.txt || true

echo "\nInstallation complete. To use the environment run:\n  conda activate ${ENV_NAME}\n"
echo "Then verify with: python scripts/check_requirements.py"

exit 0
