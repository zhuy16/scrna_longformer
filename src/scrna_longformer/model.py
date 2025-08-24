import torch, torch.nn as nn
from .layers import TransformerBlock

class SCRNALongformer(nn.Module):
    """
    Gene-token transformer with kNN mask. Minimal supervised variant with classifier head.
    """
    def __init__(self, n_genes, n_classes, d_model=128, depth=2, n_heads=4, mlp_ratio=4, pool="mean", mlm=False):
        super().__init__()
        self.n_genes = n_genes
        self.pool = pool
        self.gene_embed = nn.Embedding(n_genes, d_model)  # token id embedding
        self.val_proj   = nn.Linear(1, d_model)           # expression scalar -> d_model
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, mlp_ratio) for _ in range(depth)])
        self.cls_token = nn.Parameter(torch.zeros(1,1,d_model)) if pool == "cls" else None
        self.norm = nn.LayerNorm(d_model)
        self.clf_head = nn.Linear(d_model, n_classes)
        self.mlm = mlm
        # simple MLM head: project back to scalar expression per gene (regression)
        self.mlm_head = nn.Linear(d_model, 1) if mlm else None

    def forward(self, values, mask):
        """
        values: (B,G) float expression matrix (already normalized/log1p)
        mask:   (G,G) boolean kNN allow mask
        """
        B,G = values.shape
        gene_ids = torch.arange(G, device=values.device).long().unsqueeze(0).expand(B, -1)  # (B,G)
        x = self.gene_embed(gene_ids) + self.val_proj(values.unsqueeze(-1))  # (B,G,D)

        if self.pool == "cls":
            cls = self.cls_token.expand(B,-1,-1)  # (B,1,D)
            x = torch.cat([cls, x], dim=1)
            # extend mask to include CLS fully connected
            Gp1 = G+1
            m = values.new_ones((Gp1,Gp1), dtype=torch.bool)
            m[1:,1:] = mask
        else:
            m = mask

        for blk in self.blocks:
            x = blk(x, m)

        if self.pool == "cls":
            emb = x[:,0,:]
        else:
            emb = x.mean(dim=1)

        emb = self.norm(emb)
        logits = self.clf_head(emb)
        # compute per-token mlm predictions if requested
        mlm_pred = None
        if self.mlm:
            # x is (B, G, D) or (B, G+1, D) if cls token used; exclude cls if present
            x_tokens = x[:, 1:, :] if self.pool == "cls" else x
            mlm_pred = self.mlm_head(x_tokens).squeeze(-1)  # (B,G)
        return logits, emb, mlm_pred
