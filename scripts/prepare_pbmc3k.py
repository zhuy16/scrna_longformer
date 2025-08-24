import argparse, numpy as np
from scrna_longformer.data import load_pbmc3k_hvg, prepare_pbmc3k_fast

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--hvg", type=int, default=2000)
    ap.add_argument("--out", type=str, default="data/pbmc3k_hvg_knn.npz")
    ap.add_argument("--fast", action="store_true", help="Use fast small-HVG pipeline (n_hvg=256, k=64 by default)")
    args = ap.parse_args()

    if args.fast:
        X, y, A, var_names, gene_mean, gene_std = prepare_pbmc3k_fast(k=max(args.k, 64), n_hvg=256)
    else:
        X, y, A, var_names, gene_mean, gene_std = load_pbmc3k_hvg(k=args.k, n_hvg=args.hvg)

    np.savez(args.out, X=X, y=y, A=A, var_names=var_names, gene_mean=gene_mean, gene_std=gene_std)
    np.save("data/pbmc3k_labels.npy", y)
    print(f"Saved {args.out} and labels to data/pbmc3k_labels.npy")

    # run validation on the saved artifact so fast mode fails early if artifact is bad
    try:
        import subprocess, sys
        rc = subprocess.call([sys.executable, "scripts/validate_data.py", args.out])
        if rc != 0:
            print(f"Validation failed (exit {rc}) for {args.out}")
            sys.exit(rc)
    except Exception as e:
        print(f"Warning: failed to run validator: {e}")
