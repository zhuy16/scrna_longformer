#!/usr/bin/env python3
"""Validate a prepared PBMC3k .npz produced by scripts/prepare_pbmc3k.py.

Checks performed:
- file exists and loads
- contains keys: X, y, A, var_names
- X: 2D numeric, non-empty, finite
- y: 1D, length == n_cells, integer
- A: square boolean mask of shape (n_genes, n_genes), symmetric, diagonal True
- var_names length == n_genes

Exit code 0 = pass, 1 = fail
"""
import sys, os, argparse
import numpy as np


def run_checks(path: str) -> int:
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        return 1
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"ERROR: failed to load npz: {e}")
        return 1

    required = {"X", "y", "A", "var_names"}
    keys = set(data.files)
    missing = required - keys
    if missing:
        print(f"ERROR: missing keys in npz: {missing}")
        return 1

    X = data["X"]
    y = data["y"]
    A = data["A"]
    var_names = data["var_names"]

    # X checks
    if not isinstance(X, np.ndarray):
        print("ERROR: X is not a numpy array")
        return 1
    if X.ndim != 2:
        print(f"ERROR: X must be 2D (cells x genes), got ndim={X.ndim}")
        return 1
    n_cells, n_genes = X.shape
    if n_cells == 0 or n_genes == 0:
        print("ERROR: X is empty")
        return 1
    if not np.issubdtype(X.dtype, np.floating):
        try:
            X = X.astype(float)
            print("WARN: X was not float, casted to float for checks")
        except Exception:
            print("ERROR: X is not numeric and cannot be cast to float")
            return 1
    if not np.isfinite(X).all():
        print("ERROR: X contains NaN or Inf values")
        return 1

    # y checks
    if not isinstance(y, np.ndarray):
        print("ERROR: y is not a numpy array")
        return 1
    if y.ndim != 1:
        print(f"ERROR: y must be 1D, got ndim={y.ndim}")
        return 1
    if y.shape[0] != n_cells:
        print(f"ERROR: y length ({y.shape[0]}) != n_cells ({n_cells})")
        return 1

    # A checks
    if not isinstance(A, np.ndarray):
        print("ERROR: A is not a numpy array")
        return 1
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print(f"ERROR: A must be square 2D matrix, got shape={A.shape}")
        return 1
    if A.shape[0] != n_genes:
        print(f"ERROR: A shape ({A.shape}) inconsistent with n_genes ({n_genes})")
        return 1
    # ensure boolean
    if A.dtype != bool:
        try:
            A = A.astype(bool)
            print("WARN: A was not boolean, casted to bool for checks")
        except Exception:
            print("ERROR: A is not boolean and cannot be cast to bool")
            return 1
    # diagonal true
    if not np.all(np.diag(A)):
        print("ERROR: not all diagonal entries of A are True")
        return 1
    # symmetric
    if not np.array_equal(A, A.T):
        print("ERROR: A is not symmetric")
        return 1

    # var_names
    if not isinstance(var_names, np.ndarray):
        print("ERROR: var_names is not a numpy array")
        return 1
    if var_names.shape[0] != n_genes:
        print(f"ERROR: var_names length ({var_names.shape[0]}) != n_genes ({n_genes})")
        return 1

    # basic label sanity
    if not np.issubdtype(y.dtype, np.integer):
        try:
            y = y.astype(int)
            print("WARN: y was not integer, casted to int for checks")
        except Exception:
            print("ERROR: y is not integer-like")
            return 1

    print("Data validation: PASS")
    print(f"  cells: {n_cells}, genes: {n_genes}, classes: {len(np.unique(y))}")
    return 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("npz", nargs='?', default='data/pbmc3k_hvg_knn.npz')
    args = p.parse_args()
    rc = run_checks(args.npz)
    sys.exit(rc)
