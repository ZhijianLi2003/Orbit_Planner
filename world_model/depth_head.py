"""
Depth auxiliary supervision head.

Predicts low-resolution depth maps from ViT patch tokens,
forcing the RGB encoder to encode spatial/obstacle information in the latent.
This module can be dropped at inference — sim-to-real needs RGB only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):

    def __init__(self, token_dim=192, patch_grid=16):
        """
        Args:
            token_dim:  ViT patch token dimension
            patch_grid: patch grid side length (224 / patch_size)
        """
        super().__init__()
        self.patch_grid = patch_grid
        self.head = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )

    def forward(self, patch_tokens):
        """
        Args:  patch_tokens: (B, N, D)   N = patch_grid²
        Returns: depth_pred:  (B, 1, G, G)  G = patch_grid
        """
        d = self.head(patch_tokens)                          # (B, N, 1)
        G = self.patch_grid
        d = d.reshape(-1, G, G, 1).permute(0, 3, 1, 2)      # (B, 1, G, G)
        return d

    @staticmethod
    def downsample_depth(depth, patch_grid):
        """Downsample ground-truth depth to patch grid resolution.
        Args:  depth: (B, 1, H, W) float [0, 1]
        Returns: (B, 1, G, G)
        """
        return F.adaptive_avg_pool2d(depth, (patch_grid, patch_grid))
