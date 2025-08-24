import numpy as np, scanpy as sc
from sklearn.metrics.pairwise import cosine_similarity

def load_pbmc3k_hvg(k=16, n_hvg=2000):
    adata = sc.datasets.pbmc3k()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
    adata = adata[:, adata.var['highly_variable']].copy()
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    # pseudo-labels via Leiden (fast supervision)
    sc.pp.pca(adata); sc.pp.neighbors(adata); sc.tl.leiden(adata, resolution=1.0)
    y = adata.obs['leiden'].astype('category').cat.codes.values

    # gene-gene cosine kNN (allow mask)
    G = X.shape[1]
    S = cosine_similarity(X.T)  # (G,G)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    A = np.zeros_like(S, dtype=bool)
    rows = np.arange(G)[:, None]
    A[rows, idx] = True
    A = np.logical_or(A, A.T)
    np.fill_diagonal(A, True)

    return X.astype("float32"), y.astype("int64"), A, adata.var_names.values
