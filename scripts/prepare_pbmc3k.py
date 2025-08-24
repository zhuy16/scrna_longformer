import argparse, numpy as np
from src.scrna_longformer.data import load_pbmc3k_hvg

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--out", type=str, default="data/pbmc3k_hvg_knn.npz")
    args = ap.parse_args()

    X, y, A, var_names = load_pbmc3k_hvg(k=args.k, n_hvg=args.hvg)
    np.savez(args.out, X=X, y=y, A=A, var_names=var_names)
    np.save("data/pbmc3k_labels.npy", y)
    print(f"Saved {args.out} and labels to data/pbmc3k_labels.npy")
