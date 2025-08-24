import os
import sys
import torch

# ensure `src/` is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrna_longformer.layers import LocalGraphAttention, TransformerBlock


def test_local_graph_attention_identity_mask_returns_v_projection():
    torch.manual_seed(0)
    B = 1
    G = 4
    D = 8
    H = 2

    x = torch.randn(B, G, D)
    attn = LocalGraphAttention(d_model=D, n_heads=H)

    # compute expected: with identity allow-mask, each query only attends to its own key
    # so the attention output per position equals the v vector for that position.
    qkv = attn.proj_qkv(x)  # (B,G,3*D)
    qkv = qkv.view(B, G, 3, H, D // H).permute(2, 0, 3, 1, 4)
    # q,k,v shapes: (B,H,G,d)
    q, k, v = qkv[0], qkv[1], qkv[2]

    # reconstructed concatenated v per gene: (B,G,D)
    y_expected = v.transpose(1, 2).contiguous().view(B, G, D)
    expected = attn.proj_out(y_expected)

    mask = torch.eye(G, dtype=torch.bool)
    out = attn(x, mask)

    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-5)


def test_transformer_block_forward_shape_and_finite():
    torch.manual_seed(1)
    B = 2
    G = 8
    D = 16
    block = TransformerBlock(d_model=D, n_heads=4, mlp_ratio=2)
    x = torch.randn(B, G, D)
    mask = torch.ones(G, G, dtype=torch.bool)
    out = block(x, mask)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
