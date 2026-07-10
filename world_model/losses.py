"""
Training losses:
  - SIGReg           : official implementation matching le-wm/module.py
  - OrbitPlannerLoss : L_pred + λ_sig * SIGReg + λ_depth * L_depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer (official le-wm implementation).

    Input proj: (T, B, D). Over T×B samples, random projections + Epps-Pulley
    characteristic function test encourage the latent distribution to be isotropic Gaussian.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        Args:
            proj: (T, B, D)  — time dimension first, matching le-wm
        Returns:
            scalar loss
        """
        proj = proj.float()
        t = self.t.to(device=proj.device)
        phi = self.phi.to(device=proj.device)
        weights = self.weights.to(device=proj.device)

        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * t
        err = (x_t.cos().mean(-3) - phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ weights) * proj.size(-2)
        return statistic.mean()


class OrbitPlannerLoss(nn.Module):

    def __init__(self, lambda_sigreg=0.1, lambda_depth=0.1,
                 sigreg_projections=1024, sigreg_knots=17):
        super().__init__()
        self.lambda_sig = lambda_sigreg
        self.lambda_depth = lambda_depth
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_projections)

    def forward(self, z, z_hat, depth_pred, depth_gt):
        """
        Args:
            z:          (B, W, D)  encoder latent (prediction target)
            z_hat:      (B, W, D)  predictor output (via predictor projector)
            depth_pred: (B, W, 1, G, G)
            depth_gt:   (B, W, 1, G, G)
        """
        # 1) Prediction loss — matches LeWM paper pseudocode:
        #    MSE(ẑ_{t}, z_{t+1}), teacher forcing, no stop-gradient
        loss_pred = F.mse_loss(z_hat[:, :-1], z[:, 1:])

        # 2) SIGReg — matches le-wm/train.py: emb.transpose(0, 1) → (W, B, D)
        loss_sig = self.sigreg(z.transpose(0, 1))

        # 3) Depth auxiliary supervision
        loss_depth = F.mse_loss(depth_pred, depth_gt)

        total = loss_pred + self.lambda_sig * loss_sig + self.lambda_depth * loss_depth

        metrics = {
            "loss_total": total.item(),
            "loss_pred": loss_pred.item(),
            "loss_sigreg": loss_sig.item(),
            "loss_depth": loss_depth.item(),
        }
        return total, metrics
