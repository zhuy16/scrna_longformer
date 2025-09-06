import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from scrna_longformer.model import SCRNALongformer
from scrna_longformer.utils import seed_all, get_device


class CellsClsDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return self.X[i], self.y[i]


def train_one(model, dl_tr, opt, criterion, dev):
    model.train()
    for xb, yb in dl_tr:
        xb, yb = xb.to(dev), yb.to(dev)
        logits, emb = model(xb, mask)
        loss = criterion(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()


def eval_model(model, dl_va, dev):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in dl_va:
            xb, yb = xb.to(dev), yb.to(dev)
            out = model(xb, mask)
            logits, emb = out if not args.mlm else out[:2]
            preds += logits.argmax(1).cpu().tolist()
            trues += yb.cpu().tolist()
    f1 = f1_score(trues, preds, average='macro')
    acc = accuracy_score(trues, preds)
    return f1, acc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/exp_zscore.yaml')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--fast', action='store_true')
    ap.add_argument('--mlm', action='store_true')
    ap.add_argument('--outdir', type=str, default='results')
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    seed_all(cfg.get('seed', 42))
    dev = get_device(cfg.get('device_preference', 'auto'))
    print('Device:', dev)

    dat = np.load(cfg['data']['npz_path'], allow_pickle=True)
    X, y, A = dat['X'], dat['y'], dat['A'].astype(bool)
    gene_mean = dat['gene_mean'] if 'gene_mean' in dat else None
    gene_std = dat['gene_std'] if 'gene_std' in dat else None
    do_zscore = bool(cfg['data'].get('zscore', False))
    if do_zscore:
        if gene_mean is None or gene_std is None:
            raise RuntimeError('Z-scoring requested but gene_mean/gene_std missing in npz')
        X = (X - gene_mean[None, :]) / gene_std[None, :]

    # fast mode overrides
    if args.fast:
        print('FAST MODE: using 1 epoch smoke settings')
        epochs = 1
        batch_size = min(8, X.shape[0])
    else:
        epochs = cfg['train']['epochs']
        batch_size = cfg['data']['batch_size']

    n_splits = args.folds
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.get('seed', 42))

    import os
    os.makedirs(args.outdir, exist_ok=True)

    results = []
    global mask
    mask = torch.tensor(A, dtype=torch.bool, device=dev)

    fold = 0
    for tr_idx, va_idx in skf.split(X, y):
        fold += 1
        print(f'Running fold {fold}/{n_splits}')
        ds_tr = CellsClsDS(X[tr_idx], y[tr_idx])
        ds_va = CellsClsDS(X[va_idx], y[va_idx])
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=0)
        dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

        n_classes = int((y.max() + 1) if cfg['model']['n_classes'] == 'auto' else cfg['model']['n_classes'])
        model = SCRNALongformer(
            n_genes=X.shape[1],
            n_classes=n_classes,
            d_model=cfg['model']['d_model'],
            depth=cfg['model']['depth'],
            n_heads=cfg['model']['n_heads'],
            mlp_ratio=cfg['model']['mlp_ratio'],
            pool=cfg['model']['pool'],
            mlm=args.mlm,
        ).to(dev)

        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['train'].get('lr', 3e-4)), weight_decay=float(cfg['train'].get('weight_decay', 0.0)))
        ce = torch.nn.CrossEntropyLoss()

        for epoch in range(1, epochs+1):
            model.train()
            for xb, yb in dl_tr:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb, mask)
                logits, emb = out if not args.mlm else out[:2]
                loss = ce(logits, yb)
                opt.zero_grad(); loss.backward(); opt.step()

        f1, acc = eval_model(model, dl_va, dev)
        print(f'Fold {fold}  val_acc {acc:.3f}  val_f1 {f1:.3f}')
        results.append({'fold': fold, 'acc': float(acc), 'f1': float(f1)})
        # save model
        torch.save(model.state_dict(), os.path.join(args.outdir, f'model_fold{fold}.pt'))

    # aggregate
    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    summary = {
        'acc_mean': float(np.mean(accs)), 'acc_std': float(np.std(accs)),
        'f1_mean': float(np.mean(f1s)), 'f1_std': float(np.std(f1s)),
    }
    print('\nCV summary:')
    print(summary)
    import csv
    with open(os.path.join(args.outdir, 'cv_summary.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['fold','acc','f1'])
        w.writeheader()
        for r in results: w.writerow(r)
    with open(os.path.join(args.outdir, 'cv_agg.json'), 'w') as f:
        import json
        json.dump({'results': results, 'summary': summary}, f, indent=2)

    print('Saved CV results to', args.outdir)
