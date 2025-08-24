import os
import sys
import torch

# ensure `src/` is on sys.path so tests can import the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrna_longformer.model import SCRNALongformer

def test_scrna_longformer_forward_shapes_and_finite():
    torch.manual_seed(0)
    B = 2
    G = 128
    C = 5
    model = SCRNALongformer(n_genes=G, n_classes=C, d_model=32, depth=1, n_heads=4, pool='mean')
    xb = torch.randn(B, G)
    mask = torch.ones(G, G, dtype=torch.bool)
    logits, emb = model(xb, mask)
    assert logits.shape == (B, C)
    assert emb.shape[0] == B
    assert torch.isfinite(logits).all()
    assert torch.isfinite(emb).all()
