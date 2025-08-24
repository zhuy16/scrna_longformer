from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F

TensorOrArray = Union[torch.Tensor, object]

class LocalGraphAttention(nn.Module):
    """
    Attention restricted by a boolean (G,G) mask A where True = allow.
    Uses SDPA with additive mask set to -inf for disallowed positions.
    """
    def __init__(self, d_model: int = 128, n_heads: int = 4, attn_dropout: float = 0.0, resid_dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.d = d_model // n_heads
        self.proj_qkv = nn.Linear(d_model, 3*d_model)
        self.proj_out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout and attn_dropout > 0.0 else nn.Identity()
        self.resid_dropout = nn.Dropout(resid_dropout) if resid_dropout and resid_dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, allow_mask_bool: TensorOrArray) -> torch.Tensor:
        """Apply scaled-dot-product attention restricted by an allow-mask.

        Args:
            x: Tensor of shape (B, G, D) - batch of gene token embeddings.
            allow_mask_bool: boolean-like mask with one of shapes:
                (G, G) - global allow mask
                (B, G, G) - per-batch
                (H, G, G) - per-head
                (B, H, G, G) - per-batch-per-head

        Returns:
            Tensor of shape (B, G, D) after attention and output projection.
        """
        B, G, D = x.shape
        qkv = self.proj_qkv(x).view(B, G, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,H,G,d)

        # ensure mask is a torch boolean tensor on the correct device
        if not isinstance(allow_mask_bool, torch.Tensor):
            allow_mask = torch.as_tensor(allow_mask_bool, dtype=torch.bool, device=x.device)
        else:
            allow_mask = allow_mask_bool.to(device=x.device, dtype=torch.bool)

        # Normalize mask dims to (B,H,G,G) in boolean form where True means allow
        if allow_mask.dim() == 2:
            # (G,G) -> broadcast
            allow = allow_mask.unsqueeze(0).unsqueeze(0)  # (1,1,G,G)
            allow = allow.expand(B, self.h, G, G)
        elif allow_mask.dim() == 3:
            if allow_mask.shape[0] == B:
                # (B,G,G) -> expand heads
                allow = allow_mask.unsqueeze(1).expand(B, self.h, G, G)
            elif allow_mask.shape[0] == self.h:
                # (H,G,G) -> expand batch
                allow = allow_mask.unsqueeze(0).expand(B, self.h, G, G)
            else:
                raise ValueError(f"Unrecognized 3D mask first-dim {allow_mask.shape[0]} (expected B or H)")
        elif allow_mask.dim() == 4:
            # (B,H,G,G)
            allow = allow_mask
            if allow.shape[0] != B or allow.shape[1] != self.h:
                raise ValueError("4D mask must have shape (B,H,G,G)")
        else:
            raise ValueError("allow_mask_bool must be 2D, 3D, or 4D boolean-like array/tensor")

        # Build additive mask with -inf where disallowed (i.e., where allow==False)
        disallow = ~allow
        # Create float mask on same device
        add_mask = torch.zeros_like(disallow, dtype=torch.float32, device=x.device)
        add_mask = add_mask.masked_fill(disallow, float("-inf"))  # (B,H,G,G)

        # scaled_dot_product_attention accepts attn_mask broadcastable to (B,H,L,S)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=add_mask)
        y = self.attn_dropout(y)
        y = y.transpose(1, 2).contiguous().view(B, G, D)
        return self.resid_dropout(self.proj_out(y))

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, mlp_ratio: int = 4, attn_dropout: float = 0.1, resid_dropout: float = 0.0, mlp_dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = LocalGraphAttention(d_model, n_heads, attn_dropout=attn_dropout, resid_dropout=resid_dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(
            nn.Linear(d_model, d_model*mlp_ratio),
            nn.GELU(),
            nn.Dropout(mlp_dropout) if mlp_dropout and mlp_dropout > 0.0 else nn.Identity(),
            nn.Linear(d_model*mlp_ratio, d_model),
        )
        self.mlp_dropout = nn.Dropout(mlp_dropout) if mlp_dropout and mlp_dropout > 0.0 else nn.Identity()

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x
