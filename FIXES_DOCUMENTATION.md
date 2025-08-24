# Environment Setup and Fixes Documentation

## Summary

This document records the critical fixes made to get the scrna-longformer project working with real PBMC data and leiden clustering on macOS.

## Environment Issue & Solution

### Problem
- Original `scrna` environment had leiden/igraph OpenMP conflicts on macOS arm64
- System would fall back to base conda environment without required packages
- Leiden clustering would fail, resulting in single-class labels

### Solution
Created a fresh environment `scrna_fixed` with proper package versions:

```bash
conda create -n scrna_fixed python=3.10 -y
conda activate scrna_fixed
mamba install -c conda-forge scanpy anndata umap-learn python-igraph leidenalg libomp numpy pandas scikit-learn matplotlib pyyaml -y
conda install -c pytorch -c conda-forge pytorch torchvision torchaudio -y
```

## Data Processing Fixes

### 1. Added Gene Scaling Before PCA/Clustering
**Problem**: Missing `sc.pp.scale()` step before PCA
**Fix**: Added `sc.pp.scale(adata, max_value=10)` before PCA in both `load_pbmc3k_hvg()` and `prepare_pbmc3k_fast()`

**Code changes in `src/scrna_longformer/data.py`:**
```python
# Before PCA:
sc.pp.scale(adata, max_value=10)  # Scale genes to zero mean, unit variance
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=1.0)
```

### 2. Compute Gene Stats Before Scaling
**Problem**: Gene mean/std were computed after scaling (always ~0 and ~1)
**Fix**: Compute `gene_mean` and `gene_std` before scaling for MLM use cases

### 3. Memory Optimization
**Problem**: Full 2700×2000 gene matrix causes MPS memory overflow
**Fix**: Created smaller datasets with top 500 most variable genes

## Current Working Configuration

### Environment
- **Name**: `scrna_fixed`
- **Python**: 3.10
- **Key packages**: scanpy, leidenalg, pytorch with MPS support

### Data Files
- `data/pbmc3k_hvg_knn_leiden_top500.npz`: Real PBMC with 9 leiden clusters, top 500 genes
- Config: `configs/real_leiden_top500.yaml`

### Performance Results
- **Real leiden clusters**: Accuracy ~37%, F1 ~0.06
- **9 balanced clusters vs single-class**: Major improvement
- **Memory efficient**: Trains quickly without system freeze

## Key Learnings

1. **Leiden clustering works** with proper environment setup
2. **Gene scaling is critical** for PCA/clustering quality  
3. **Class imbalance** affects model evaluation (model predicts majority class)
4. **Memory management** needed for larger gene sets

## Next Steps for Future Work

1. **Address class imbalance** with weighted loss or balanced sampling
2. **Experiment with leiden resolution** for more balanced clusters
3. **Scale to full gene set** with gradient checkpointing or CPU training
4. **Add biological validation** by mapping clusters to known cell types

## Always-Use Commands

```bash
# Activate working environment
conda activate scrna_fixed

# Verify environment
python -c "import scanpy as sc, leidenalg; print('Environment OK')"

# Generate working data
PYTHONPATH=./src python scripts/prepare_pbmc3k.py --out data/pbmc3k_hvg_knn_leiden.npz

# Run fast CV test
PYTHONPATH=./src python scripts/run_cv.py --folds 2 --config configs/real_leiden_top500.yaml
```
