import argparse, yaml, numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from src.scrna_longformer.model import SCRNALongformer
from src.scrna_longformer.utils import seed_all, get_device

class CellsClsDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]

def split_indices(n, frac=0.8, seed=42):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    cut = int(frac*n)
    return idx[:cut], idx[cut:]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    ap.add_argument("--fast", action="store_true", help="Run a 1-epoch fast smoke run (small batch/hvg) for end-to-end checks")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed_all(cfg["seed"])
    dev = get_device(cfg["device_preference"])
    print("Device:", dev)

    dat = np.load(cfg["data"]["npz_path"], allow_pickle=True)
    X, y, A = dat["X"], dat["y"], dat["A"].astype(bool)
    # override for fast smoke run
    if args.fast:
        print("FAST MODE: overriding config for a 1-epoch smoke run")
        cfg["train"]["epochs"] = 1
        cfg["data"]["batch_size"] = min(8, X.shape[0])
        # optionally reduce model size for speed
        cfg["model"]["d_model"] = min(cfg["model"].get("d_model", 64), 64)
        cfg["model"]["depth"] = 1
    n_classes = (y.max() + 1) if cfg["model"]["n_classes"] == "auto" else cfg["model"]["n_classes"]

    train_idx, val_idx = split_indices(len(X), frac=cfg["data"]["train_frac"], seed=cfg["seed"])
    ds_tr = CellsClsDS(X[train_idx], y[train_idx])
    ds_va = CellsClsDS(X[val_idx],   y[val_idx])

    dl_tr = DataLoader(ds_tr, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=cfg["data"]["num_workers"]) 
    dl_va = DataLoader(ds_va, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"]["num_workers"]) 

    model = SCRNALongformer(
        n_genes=X.shape[1],
        n_classes=int(n_classes),
        d_model=cfg["model"]["d_model"],
        depth=cfg["model"]["depth"],
        n_heads=cfg["model"]["n_heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        pool=cfg["model"]["pool"],
    ).to(dev)

    mask = torch.tensor(A, dtype=torch.bool, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    ce = torch.nn.CrossEntropyLoss()

    best_f1, best_state = 0.0, None
    for epoch in range(1, cfg["train"]["epochs"]+1):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(dev), yb.to(dev)
            logits, emb = model(xb, mask)
            loss = ce(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
        # eval
        model.eval(); preds, trues, embs = [], [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(dev), yb.to(dev)
                logits, emb = model(xb, mask)
                preds += logits.argmax(1).cpu().tolist()
                trues += yb.cpu().tolist()
                embs.append(emb.cpu())
        f1 = f1_score(trues, preds, average="macro")
        acc = accuracy_score(trues, preds)
        print(f"epoch {epoch}  val_acc {acc:.3f}  val_f1 {f1:.3f}")
        if f1 > best_f1:
            best_f1, best_state = f1, model.state_dict().copy()

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), cfg["data"]["model_out"])
    print("Saved model to", cfg["data"]["model_out"])

    # save embeddings for all cells
    model.eval(); all_embs=[]
    with torch.no_grad():
        for xb, yb in DataLoader(CellsClsDS(X, y), batch_size=cfg["data"]["batch_size"]):
            xb = xb.to(dev)
            _, emb = model(xb, mask)
            all_embs.append(emb.cpu())
    embs = torch.cat(all_embs).numpy()
    np.save(cfg["data"]["emb_out"], embs)
    print("Saved embeddings to", cfg["data"]["emb_out"])
