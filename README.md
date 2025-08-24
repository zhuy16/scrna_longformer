# scrna-longformer (MVP)

A tiny, reproducible starter for a kNN-masked gene-token transformer:
- Long sequence = ~2,000 **gene tokens**
- Local attention via **data-driven gene kNN** mask (cosine similarity)
- Outputs a **cell embedding** and a **cell-type classifier** head

## Quickstart
```
python -m venv .venv && source .venv/bin/activate  # or conda
pip install -r requirements.txt

# 1) Prepare PBMC3k data, HVGs, and gene kNN mask
python scripts/prepare_pbmc3k.py --k 16 --hvg 2000

# 2) Train classifier (Leiden labels as supervision)
python scripts/train_classifier.py --config configs/default.yaml

# 3) Visualize embeddings
python scripts/plot_umap.py --emb data/pbmc3k_emb_cls.npy --labels data/pbmc3k_labels.npy
```

Notes

Uses MPS on Apple Silicon automatically when available, else CPU.

This MVP focuses on classification + embeddings. A masked-gene loss can be added later.
# scrna_longformer

## Data preparation notes

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
