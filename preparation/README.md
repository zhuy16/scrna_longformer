Preparation: environment and dependency installation
===============================================

This document collects the exact commands we used and recommended options to prepare the `scrna-longformer` development environment.

Principles
- Prefer conda/mamba for binary scientific packages (scanpy, anndata, python-igraph). It's faster and avoids build issues on macOS.
- Use pip only as a fallback or for smaller pure-Python packages.

1) Create and activate an environment (recommended)

```zsh
conda create -n scrna python=3.10 -y
conda activate scrna
```

2) Install mamba (optional but recommended)

```zsh
conda install -c conda-forge mamba -y
```

3) Fast (recommended): install core scientific deps with mamba

```zsh
# installs scanpy, anndata, umap-learn, igraph and leiden
mamba install -c conda-forge scanpy anndata umap-learn python-igraph leidenalg -y
```

4) Conda-only alternative (if you don't install mamba)

```zsh
conda install -c conda-forge scanpy anndata umap-learn python-igraph leidenalg -y
```

5) PyTorch (pick this after the steps above)

- Recommended via conda; this will select the right build for macOS (MPS) or CPU:

```zsh
conda install -c pytorch -c conda-forge pytorch torchvision torchaudio -y
```

- Pip fallback (CPU wheel):

```zsh
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

6) PyYAML (small and safe via either manager)

```zsh
# conda
conda install -c conda-forge pyyaml -y

# or pip
python -m pip install pyyaml
```

7) Verify installation (quick script included)

From the repo root (after activating `scrna`):

```zsh
python scripts/check_requirements.py
```

It prints OK or MISS lines for each package (and versions when importable).

8) Typical zsh notes
- "zsh: command not found: mamba" → just install mamba (see step 2) or use `conda install` instead.
- "zsh: number expected" is usually harmless when copy/pasting multi-line blocks; re-run the command normally.

9) Next steps after dependencies are installed

- Prepare PBMC3k HVG and kNN mask:

```zsh
python scripts/prepare_pbmc3k.py --k 16 --hvg 2000
```

- Train the classifier (uses Leiden pseudo-labels):

```zsh
python scripts/train_classifier.py --config configs/default.yaml
```

- Plot UMAP of embeddings:

```zsh
python scripts/plot_umap.py --emb data/pbmc3k_emb_cls.npy --labels data/pbmc3k_labels.npy
```

Troubleshooting
- If `scanpy` import fails after installation, try `pip uninstall scanpy anndata` and reinstall via conda. Mixing pip/conda can cause conflicts.
- If `python-igraph` is missing or fails to build with pip, prefer conda-forge.
- For GPU/CUDA, install a CUDA-enabled PyTorch from the official selector: https://pytorch.org/get-started/locally/

If you want, I can add a one-line `preparation/INSTALL.sh` that runs the conda commands you prefer. Tell me whether you want the `mamba` or `conda` variant.
