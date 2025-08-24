# scrna-longformer (MVP)

A tiny, reproducible starter for a kNN-masked gene-token transformer:
- Long sequence = ~2,000 **gene tokens**
- Local attention via **data-driven gene kNN** mask (cosine similarity)
- Outputs a **cell embedding** and a **cell-type classifier** head

## Quickstart

**⚠️ CRITICAL: Always use the working environment setup first:**

```bash
# 1) Setup working environment (handles leiden/igraph fixes automatically)
./setup_environment.sh

# 2) Verify environment is correct
./verify_environment.sh

# 3) Generate real PBMC data with leiden clustering
PYTHONPATH=./src python scripts/prepare_pbmc3k.py --out data/pbmc3k_hvg_knn_leiden.npz

# 4) Run fast cross-validation test
PYTHONPATH=./src python scripts/run_cv.py --folds 2 --config configs/real_leiden_top500.yaml

# 5) Visualize embeddings (optional)
PYTHONPATH=./src python scripts/plot_umap.py --emb data/pbmc3k_emb_cls.npy --labels data/pbmc3k_labels.npy
```

**Alternative wrapper method:**
```bash
./run_with_env.sh python scripts/run_cv.py --folds 2 --config configs/real_leiden_top500.yaml
```

Notes

Uses MPS on Apple Silicon automatically when available, else CPU.

This MVP focuses on classification + embeddings. A masked-gene loss can be added later.
# scrna_longformer

## Data preparation notes

### Gene scaling and z-scoring

**Important**: Starting from v0.1.1, the data preparation pipeline includes proper gene scaling:

1. **During preparation**: Genes are scaled to zero mean and unit variance across cells using `sc.pp.scale(adata, max_value=10)` before PCA and clustering
2. **For transformer input**: The saved expression matrix `X` is already properly scaled 
3. **Z-scoring config**: The `data.zscore: true` option applies additional z-scoring using pre-scaling gene statistics, which is mainly useful for MLM experiments on original log1p data

**Recommendation**: Use `configs/default_scaled.yaml` (with `zscore: false`) for classification tasks, since the data is already optimally scaled for the transformer.

### HVG selection warnings

When running the fast prepare script (`python scripts/prepare_pbmc3k.py --fast`), you may see a
warning like:

```
UserWarning: `flavor='seurat_v3'` expects raw count data, but non-integers were found.
```

Why: `seurat_v3` HVG selection calls LOESS (from `skmisc`) and expects raw integer counts. Our
fast pipeline normalizes and log-transforms before HVG selection to keep the code simple and fast,
so Scanpy emits a warning.

Is it a problem? For quick development/debugging: no — the resulting `data/pbmc3k_hvg_knn.npz` is
usable. For strict parity with Seurat v3, compute HVGs on raw counts or install `scikit-misc`
so the exact `seurat_v3` path runs.

Quick checks to validate `data/pbmc3k_hvg_knn.npz`:

1) Inspect shapes and types

```python
import numpy as np
data = np.load('data/pbmc3k_hvg_knn.npz')
print(data['X'].shape, data['y'].shape, data['A'].shape)
print(data['X'].dtype, data['y'].dtype, data['A'].dtype)
```

Expected: X=(n_cells, n_genes) float32, y=(n_cells,) int64, A=(n_genes,n_genes) bool.

2) Confirm mask properties

```python
A = data['A']
assert A.shape[0] == A.shape[1]
assert np.all(np.diag(A))  # diagonal True
assert np.array_equal(A, A.T)  # symmetric
```

3) Sanity-check embeddings after a short run

Run a 1-epoch training with `scripts/train_classifier.py --fast` (not implemented by default,
but you can set `--epochs 1` in the script) and check that saved embeddings are finite and
reasonable (use UMAP to inspect clusters visually).

If you'd like, I can add an automated validation script that runs these checks and reports a
short summary.

Mask usage examples
-------------------

The attention layers accept several mask shapes. Here are examples you can copy/paste:

```python
import torch
from scrna_longformer.layers import LocalGraphAttention

# x: (B,G,D)
B, G, D, H = 2, 256, 64, 4
x = torch.randn(B, G, D)
attn = LocalGraphAttention(d_model=D, n_heads=H)

# 1) Global mask (G,G)
mask_global = torch.ones(G, G, dtype=torch.bool)
out = attn(x, mask_global)

# 2) Per-batch mask (B,G,G)
mask_batch = torch.stack([mask_global for _ in range(B)], dim=0)
out = attn(x, mask_batch)

# 3) Per-head mask (H,G,G)
mask_head = torch.stack([mask_global for _ in range(H)], dim=0)
out = attn(x, mask_head)

# 4) Per-batch-per-head (B,H,G,G)
mask_full = mask_batch.unsqueeze(1).expand(B, H, G, G)
out = attn(x, mask_full)
```
