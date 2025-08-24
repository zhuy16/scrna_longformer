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
