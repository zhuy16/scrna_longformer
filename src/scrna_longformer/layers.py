import torch, torch.nn as nn, torch.nn.functional as F

class LocalGraphAttention(nn.Module):
    """
    Attention restricted by a boolean (G,G) mask A where True = allow.
    Uses SDPA with additive mask set to -inf for disallowed positions.
    """
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.d = d_model // n_heads
        self.proj_qkv = nn.Linear(d_model, 3*d_model)
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x, allow_mask_bool):
        """
        x: (B,G,D), allow_mask_bool: (G,G) boolean (device-agnostic)
        """
        B,G,D = x.shape
        qkv = self.proj_qkv(x).view(B,G,3,self.h,self.d).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]  # (B,H,G,d)

        # SDPA additive mask: float with -inf where attention is disallowed
        disallow = (~allow_mask_bool).to(x.device)
        add_mask = disallow.float().masked_fill(disallow, float("-inf"))  # (G,G)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask)
        y = y.transpose(1,2).contiguous().view(B,G,D)
        return self.proj_out(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = LocalGraphAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model*mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model*mlp_ratio, d_model),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x
