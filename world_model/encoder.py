"""
Encoder modules:
  - ViT          : HuggingFace ViT (SDPA/Flash Attention) → CLS + patches
  - StateEncoder : spacecraft proprioceptive state → state embedding
  - ActionEncoder: thruster actions → action embedding
  - Projector    : CLS + state → latent z (BatchNorm prevents LN collapse)
"""

import torch
import torch.nn as nn
from transformers import ViTConfig, ViTModel


# =========================== ViT ==================================

class ViT(nn.Module):
    """
    Visual encoder based on HuggingFace ViTModel.
    Automatically uses SDPA / Flash Attention (PyTorch 2.x).
    Returns CLS token and patch tokens with the same interface as before.
    """

    def __init__(self, img_size=224, patch_size=14, in_chans=3,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                 drop_rate=0.0):
        super().__init__()
        config = ViTConfig(
            image_size=img_size,
            patch_size=patch_size,
            num_channels=in_chans,
            hidden_size=embed_dim,
            num_hidden_layers=depth,
            num_attention_heads=num_heads,
            intermediate_size=int(embed_dim * mlp_ratio),
            hidden_dropout_prob=drop_rate,
            attention_probs_dropout_prob=drop_rate,
            add_pooling_layer=False,
        )
        try:
            self.model = ViTModel(config, attn_implementation="sdpa")
        except TypeError:
            self.model = ViTModel(config)

    def forward(self, x):
        """
        Args:  x: (B, 3, 224, 224)
        Returns:
            cls_token:    (B, D)
            patch_tokens: (B, N, D)   N = (img_size/patch_size)²
        """
        out = self.model(pixel_values=x, interpolate_pos_encoding=True)
        hidden = out.last_hidden_state       # (B, 1+N, D)
        return hidden[:, 0], hidden[:, 1:]   # CLS, patches


# ===================== State / Action Encoder ======================

class StateEncoder(nn.Module):
    def __init__(self, state_dim=16, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, s):
        """s: (*, state_dim) → (*, out_dim)"""
        return self.net(s)


class ActionEncoder(nn.Module):
    def __init__(self, action_dim=8, hidden_dim=128, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, a):
        """a: (*, action_dim) → (*, out_dim)"""
        return self.net(a)


# ========================= Projector ===============================

class Projector(nn.Module):
    """
    LeWM-style projector: Linear + BatchNorm.
    ViT's final LayerNorm is counteracted by BN so SIGReg can act on the latent distribution.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        """x: (B, W, in_dim) or (B, in_dim)"""
        shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        x = self.bn(self.linear(x))
        return x.reshape(*shape, -1)
