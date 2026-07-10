"""
AdaLN Transformer Predictor (LeWM style).

Actions condition each Transformer layer via Adaptive Layer Normalization.
Zero initialization makes action influence grow gradually early in training.
Causal masking enables autoregressive prediction.
"""

import torch
import torch.nn as nn


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PredictorBlock(nn.Module):
    """AdaLN-zero block (le-wm ConditionalBlock style)."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, cond_dim=256,
                 dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, cond, mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaln(cond).chunk(6, dim=-1)
        )
        h_norm = _modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h_norm, h_norm, h_norm,
                       attn_mask=mask, need_weights=False)[0]
        x = x + gate_msa * h

        h = self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * h
        return x


class Predictor(nn.Module):

    def __init__(self, latent_dim=256, cond_dim=256,
                 depth=6, num_heads=4, mlp_ratio=4.0,
                 dropout=0.1, max_seq_len=64):
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, latent_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            PredictorBlock(latent_dim, num_heads, mlp_ratio, cond_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z, action_emb):
        """
        Args:
            z:          (B, W, D)  latent sequence from encoder
            action_emb: (B, W, D)  action embedding sequence
        Returns:
            z_hat: (B, W, D)  predicted next-step latent
        """
        W = z.shape[1]
        x = z + self.pos_embed[:, :W]

        mask = torch.triu(
            torch.full((W, W), float("-inf"), device=z.device), diagonal=1
        )

        for blk in self.blocks:
            x = blk(x, action_emb, mask)

        return self.norm(x)
