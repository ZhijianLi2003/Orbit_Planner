"""
Orbit Planner world model: composes all submodules.

Architecture:
  RGB  → ViT         → CLS token + patch tokens
  State → StateEncoder → state embedding
  [CLS ⊕ state_emb]  → Encoder Projector  → latent z  (prediction target)
  Action              → ActionEncoder      → action_emb
  z + action_emb      → Predictor          → raw output
  raw output          → Predictor Projector → z_hat    (prediction)
  patch tokens        → DepthHead          → depth_pred (auxiliary supervision)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ViT, StateEncoder, ActionEncoder, Projector
from .predictor import Predictor
from .depth_head import DepthHead


class OrbitPlanner(nn.Module):

    def __init__(self,
                 # ViT
                 img_size=224, patch_size=14, vit_dim=192,
                 vit_depth=12, vit_heads=3, vit_mlp_ratio=4.0,
                 # State / Action
                 state_dim=16, state_hidden=128, state_emb_dim=64,
                 action_dim=8, action_hidden=128,
                 # Latent
                 latent_dim=256,
                 # Predictor
                 pred_depth=6, pred_heads=4, pred_mlp_ratio=4.0,
                 pred_dropout=0.1, max_seq_len=64,
                 ):
        super().__init__()

        # ---------- encoders ----------
        self.vit = ViT(img_size, patch_size, 3,
                       vit_dim, vit_depth, vit_heads, vit_mlp_ratio)
        self.state_enc = StateEncoder(state_dim, state_hidden, state_emb_dim)
        self.action_enc = ActionEncoder(action_dim, action_hidden, latent_dim)

        # ---------- projectors ----------
        self.enc_proj = Projector(vit_dim + state_emb_dim, latent_dim)
        self.pred_proj = Projector(latent_dim, latent_dim)

        # ---------- predictor ----------
        self.predictor = Predictor(
            latent_dim, latent_dim,
            pred_depth, pred_heads, pred_mlp_ratio,
            pred_dropout, max_seq_len,
        )

        # ---------- depth head ----------
        patch_grid = img_size // patch_size
        self.depth_head = DepthHead(vit_dim, patch_grid)
        self.patch_grid = patch_grid

    # ----------------------------------------------------------------
    def encode(self, rgb, states):
        """
        Encode single or multiple frames.
        Args:
            rgb:    (B, 3, 224, 224) or (B, W, 3, 224, 224)
            states: (B, 16)          or (B, W, 16)
        Returns:
            z:             latent embeddings
            patch_tokens:  ViT patch tokens (for depth head)
        """
        multi = rgb.dim() == 5
        if multi:
            B, W = rgb.shape[:2]
            rgb_flat = rgb.reshape(B * W, *rgb.shape[2:])
            states_flat = states.reshape(B * W, -1)
        else:
            rgb_flat = rgb
            states_flat = states

        cls, patches = self.vit(rgb_flat)
        s_emb = self.state_enc(states_flat)
        fused = torch.cat([cls, s_emb], dim=-1)

        if multi:
            fused = fused.reshape(B, W, -1)
            patches = patches.reshape(B, W, *patches.shape[1:])

        z = self.enc_proj(fused)
        return z, patches

    # ----------------------------------------------------------------
    def forward(self, rgb, states, actions):
        """
        Full forward pass (training).
        Args:
            rgb:     (B, W, 3, 224, 224)
            states:  (B, W, 16)
            actions: (B, W, 8)
        Returns:
            z:          (B, W, D)
            z_hat:      (B, W, D)
            depth_pred: (B, W, 1, G, G)
        """
        B, W = rgb.shape[:2]

        # ---- encode all frames ----
        z, patches = self.encode(rgb, states)            # z: (B, W, D)

        # ---- action embedding ----
        act_flat = actions.reshape(B * W, -1)
        act_emb = self.action_enc(act_flat).reshape(B, W, -1)

        # ---- predict ----
        pred_raw = self.predictor(z, act_emb)            # (B, W, D)
        z_hat = self.pred_proj(pred_raw)                 # (B, W, D)

        # ---- depth supervision ----
        patches_flat = patches.reshape(B * W, *patches.shape[2:])
        depth_pred = self.depth_head(patches_flat)       # (B*W, 1, G, G)
        depth_pred = depth_pred.reshape(B, W, 1,
                                        self.patch_grid, self.patch_grid)

        return z, z_hat, depth_pred

    # ----------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, rgb_init, state_init, action_seq):
        """
        Autoregressive rollout at inference (for MPC / MPPI).
        Args:
            rgb_init:   (B, H, 3, 224, 224) history of H frames
            state_init: (B, H, 16)
            action_seq: (B, H+T, 8) full action sequence (history + future)
        Returns:
            z_rollout: (B, H+T, D) encoded history + predicted future latents
        """
        B, H = rgb_init.shape[:2]
        T_total = action_seq.shape[1]

        z_hist, _ = self.encode(rgb_init, state_init)    # (B, H, D)

        act_flat = action_seq.reshape(B * T_total, -1)
        act_emb_all = self.action_enc(act_flat).reshape(B, T_total, -1)

        z_seq = list(z_hist.unbind(dim=1))               # H tensors of shape (B, D)

        for t in range(H, T_total):
            ctx_start = max(0, t - H)
            z_ctx = torch.stack(z_seq[ctx_start:t], dim=1)     # (B, ?, D)
            a_ctx = act_emb_all[:, ctx_start:t]

            pred_raw = self.predictor(z_ctx, a_ctx)
            z_next = self.pred_proj(pred_raw[:, -1:])          # (B, 1, D)
            z_seq.append(z_next.squeeze(1))

        return torch.stack(z_seq, dim=1)
