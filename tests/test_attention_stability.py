import os
import sys
import torch
import numpy as np

# ensure `src/` is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from scrna_longformer.layers import LocalGraphAttention


def test_attention_extreme_values_and_gradients():
    torch.manual_seed(0)
    B = 1
    G = 6
    D = 8
    H = 2

    # extreme values (large and tiny) to exercise numerical stability
    x_large = torch.randn(B, G, D) * 1e6
    x_small = torch.randn(B, G, D) * 1e-6
    mask = torch.ones(G, G, dtype=torch.bool)

    attn = LocalGraphAttention(d_model=D, n_heads=H)

    for x in (x_large, x_small):
        x = x.clone().requires_grad_(True)
        out = attn(x, mask)
        # outputs must be finite
        assert torch.isfinite(out).all()
        # simple scalar objective and backward
        s = out.pow(2).sum()
        g = torch.autograd.grad(s, x)[0]
        assert torch.isfinite(g).all()


def test_attention_finite_difference_gradient_check():
    torch.manual_seed(1)
    B = 1
    G = 6
    D = 8
    H = 2

    # moderate inputs for FD checking
    x = torch.randn(B, G, D, dtype=torch.double, requires_grad=True)
    mask = torch.ones(G, G, dtype=torch.bool)
    attn = LocalGraphAttention(d_model=D, n_heads=H)

    # autograd gradient of scalar s = sum(attn(x)) in direction v
    out = attn(x.float(), mask)
    s = out.sum()
    g = torch.autograd.grad(s, x)[0].double()

    # directional finite difference
    v = torch.randn_like(x).double()
    v = v / (v.norm() + 1e-12)
    eps = 1e-3
    x_p = (x + eps * v).detach().clone()
    x_m = (x - eps * v).detach().clone()

    s_p = attn(x_p.float(), mask).sum().item()
    s_m = attn(x_m.float(), mask).sum().item()
    fd = (s_p - s_m) / (2 * eps)

    dir_grad = (g * v).sum().item()
    # relative error tolerance
    if abs(fd) > 1e-6:
        rel_err = abs(fd - dir_grad) / (abs(fd) + 1e-12)
    else:
        rel_err = abs(fd - dir_grad)

    assert rel_err < 1e-2
