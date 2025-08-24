import numpy as np
import scanpy as sc


def _build_knn_mask_from_matrix(X, k):
    """Build a symmetric boolean kNN allow-mask from cells x genes matrix X.

    This uses a fast normalized dot-product to compute cosine similarity:
    - normalize gene vectors across cells
    - S = Xg.T @ Xg
    - top-k per row via argpartition
    """
    G = X.shape[1]
    # ensure dense
    X = X.toarray() if hasattr(X, "toarray") else X
    # normalize gene vectors (axis=0 over cells)
    norms = np.linalg.norm(X, axis=0, keepdims=True) + 1e-12
    Xg = X / norms
    S = Xg.T @ Xg
    idx = np.argpartition(-S, kth=min(k, G-1), axis=1)[:, :k]
    A = np.zeros((G, G), dtype=bool)
    rows = np.arange(G)[:, None]
    A[rows, idx] = True
    A = np.logical_or(A, A.T)
    np.fill_diagonal(A, True)
    return A


def load_pbmc3k_hvg(k=16, n_hvg=2000):
    """Full Scanpy-based PBMC3k prep (HVG selection, leiden labels, dense X)."""
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Highly-variable gene selection with graceful fallback if a flavor's dependencies are missing.
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
    except Exception:
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="cell_ranger")
        except Exception:
            # final fallback: simple variance-based selection
            Xtmp = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            var = np.var(Xtmp, axis=0)
            idx = np.argsort(-var)[:n_hvg]
            mask = np.zeros(adata.n_vars, dtype=bool)
            mask[idx] = True
            adata.var['highly_variable'] = mask
    adata = adata[:, adata.var['highly_variable']].copy()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    # pseudo-labels via Leiden (fast supervision)
    try:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=1.0)
        y = adata.obs['leiden'].astype('category').cat.codes.values
    except Exception:
        y = np.zeros(X.shape[0], dtype=int)

    A = _build_knn_mask_from_matrix(X, k)
    return X.astype("float32"), y.astype("int64"), A, adata.var_names.values


def prepare_pbmc3k_fast(k=64, n_hvg=256):
    """Fast PBMC3k prepare: small HVG, vectorized mask, and cached output.

    Returns: X (cells×G), y (cells), A (G×G), var_names
    """
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
    adata = adata[:, adata.var['highly_variable']].copy()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    # try to compute pseudo-labels, fall back to zeros
    try:
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.leiden(adata, resolution=1.0)
        y = adata.obs['leiden'].astype('category').cat.codes.values
    except Exception:
        y = np.zeros(X.shape[0], dtype=int)

    A = _build_knn_mask_from_matrix(X, k)
    return X.astype("float32"), y.astype("int64"), A, adata.var_names.values
