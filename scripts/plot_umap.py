import argparse, numpy as np, umap
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb",   type=str, required=True)
    ap.add_argument("--labels",type=str, required=True)
    ap.add_argument("--out",   type=str, default="data/umap.png")
    args = ap.parse_args()

    Z = np.load(args.emb)       # (N, D)
    y = np.load(args.labels)    # (N,)

    reducer = umap.UMAP(random_state=42)
    z2 = reducer.fit_transform(Z)

    plt.figure(figsize=(6,5))
    sc = plt.scatter(z2[:,0], z2[:,1], c=y, s=6, cmap="tab20")
    plt.title("UMAP of scrna-longformer embeddings")
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print("Saved", args.out)
