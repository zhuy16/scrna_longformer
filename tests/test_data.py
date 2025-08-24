import os
import sys
import numpy as np

# ensure `src/` is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrna_longformer.data import _build_knn_mask_from_matrix


def test_build_knn_mask_identity():
    # Create cells=G identity matrix so gene vectors are orthogonal -> cosine matrix is identity
    G = 6
    X = np.eye(G, dtype=float)
    k = 1
    A = _build_knn_mask_from_matrix(X, k)
    assert A.shape == (G, G)
    # diagonal should be True, off-diagonal False
    assert np.all(np.diag(A))
    off = A.copy()
    np.fill_diagonal(off, False)
    assert not off.any()
    # symmetric
    assert np.array_equal(A, A.T)


def test_build_knn_mask_random_symmetry_and_diag():
    rng = np.random.RandomState(0)
    cells = 20
    G = 10
    X = rng.randn(cells, G)
    k = 3
    A = _build_knn_mask_from_matrix(X, k)
    assert A.shape == (G, G)
    # diagonal True
    assert np.all(np.diag(A))
    # symmetric
    assert np.array_equal(A, A.T)
    # each row should have at least k True (possibly more due to symmetrization)
    row_counts = A.sum(axis=1)
    assert np.all(row_counts >= k)
