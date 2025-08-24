import argparse, numpy as np
from src.scrna_longformer.data import load_pbmc3k_hvg, prepare_pbmc3k_fast

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--out", type=str, default="data/pbmc3k_hvg_knn.npz")
    ap.add_argument("--fast", action="store_true", help="Use fast small-HVG pipeline (n_hvg=256, k=64 by default)")
    args = ap.parse_args()

    if args.fast:
        X, y, A, var_names = prepare_pbmc3k_fast(k=max(args.k, 64), n_hvg=256)
    else:
        X, y, A, var_names = load_pbmc3k_hvg(k=args.k, n_hvg=args.hvg)

    np.savez(args.out, X=X, y=y, A=A, var_names=var_names)
    np.save("data/pbmc3k_labels.npy", y)
    print(f"Saved {args.out} and labels to data/pbmc3k_labels.npy")
