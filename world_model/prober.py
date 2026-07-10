"""
Physics-Inspired Prober.

Freeze Stage 1 encoder + predictor;
map frozen latent rollout to physically interpretable state quantities.

Spacecraft state: [p(3), v_body(3), rot6d(6), ω_body(3), φ(1)] = 16 dims

Prober structure:
  frozen ẑ_t  →  MLP  →  (Δp, Δv, Δω, Δφ)  state deltas
  x_{t+1} = integrate(x_t, Δp, Δv, Δω, Δφ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= SO(3) utilities ========================

def rot6d_to_matrix(rot6d):
    """6D rotation → 3×3 rotation matrix (Gram-Schmidt orthogonalization).

    rot6d storage format: R[:, :2].flatten() (row-major)
    i.e. [R00, R01, R10, R11, R20, R21]; reshape first, then extract by column.
    """
    mat = rot6d.reshape(*rot6d.shape[:-1], 3, 2)  # (..., 3, 2)
    a1 = mat[..., 0]  # first column: [R00, R10, R20]
    a2 = mat[..., 1]  # second column: [R01, R11, R21]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R):
    """3×3 rotation matrix → 6D rotation (first two columns)."""
    return R[..., :2].reshape(*R.shape[:-2], 6)


def exp_so3(omega, eps=1e-8):
    """so(3) → SO(3) exponential map (Rodrigues formula)."""
    theta = omega.norm(dim=-1, keepdim=True).clamp(min=eps)
    k = omega / theta

    K = torch.zeros(*omega.shape[:-1], 3, 3, device=omega.device, dtype=omega.dtype)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] = k[..., 0]

    theta = theta.unsqueeze(-1)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    R = I + theta.sin() * K + (1 - theta.cos()) * (K @ K)
    return R


# ========================= Prober MLP ============================

class ProberMLP(nn.Module):
    """Predict state deltas from frozen latent.

    Output: Δp(3) + Δv(3) + Δω(3) + Δφ(1) = 10 dims
    """

    def __init__(self, latent_dim=256, hidden_dim=256, num_layers=3):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 10))
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        """
        Returns:
            delta_p:     (..., 3) world-frame position delta
            delta_v:     (..., 3) body-frame velocity delta
            delta_omega: (..., 3) body-frame angular velocity delta
            delta_phi:   (..., 1) fuel delta
        """
        out = self.net(z)
        return out[..., :3], out[..., 3:6], out[..., 6:9], out[..., 9:10]


# ===================== Spacecraft kinematics integration ==========

class SpacecraftIntegrator(nn.Module):
    """Update state from MLP-predicted deltas while preserving SO(3) geometry.

    Δp, Δv, Δω, Δφ are predicted directly by the MLP (already scaled by dt).
    Rotation still needs explicit dt: R_{t+1} = R_t @ exp(ω_{t+1} × dt).
    """

    def __init__(self, dt=0.04):
        super().__init__()
        self.dt = dt

    def step(self, state, delta_p, delta_v, delta_omega, delta_phi):
        """
        Args:
            state:       (B, 16)
            delta_p:     (B, 3)  world-frame position delta
            delta_v:     (B, 3)  body-frame velocity delta
            delta_omega: (B, 3)  body-frame angular velocity delta
            delta_phi:   (B, 1)  fuel delta
        """
        p = state[..., 0:3]
        v = state[..., 3:6]
        rot6d = state[..., 6:12]
        omega = state[..., 12:15]
        phi = state[..., 15:16]

        R = rot6d_to_matrix(rot6d)

        # Position: add Δp directly (world frame)
        p_new = p + delta_p

        # Velocity: add Δv directly (body frame)
        v_new = v + delta_v

        # Rotation: ω × dt → SO(3) exponential map
        omega_new = omega + delta_omega
        dR = exp_so3(omega_new * self.dt)
        R_new = R @ dR
        rot6d_new = matrix_to_rot6d(R_new)

        # Fuel
        phi_new = (phi + delta_phi).clamp(0, 1)

        return torch.cat([p_new, v_new, rot6d_new, omega_new, phi_new], dim=-1)


# ======================== Physics Prober =========================

class PhysicsProber(nn.Module):
    """
    MLP predicts state deltas + integration.

    frozen ẑ_{1:T} + x_0 → autoregressive → x̃_{1:T}
    """

    def __init__(self, latent_dim=256, hidden_dim=256, num_layers=3,
                 dt=0.04, action_dim=8):
        super().__init__()
        self.mlp = ProberMLP(latent_dim, hidden_dim, num_layers)
        self.integrator = SpacecraftIntegrator(dt)

    def forward(self, z_seq, init_state, action_seq):
        """
        Args:
            z_seq:      (B, T, D) frozen latent sequence
            init_state: (B, 16)   integration start state
            action_seq: (B, T, A) corresponding actions (kept for API compatibility)
        """
        B, T, D = z_seq.shape
        states = []
        x = init_state

        for t in range(T):
            dp, dv, dw, dphi = self.mlp(z_seq[:, t])
            x = self.integrator.step(x, dp, dv, dw, dphi)
            states.append(x)

        return torch.stack(states, dim=1)


# ========================= Prober Loss ============================

class ProberLoss(nn.Module):
    """
    Prober training loss.
    Computes MSE per physical quantity with configurable weights.
    """

    def __init__(self, w_pos=1.0, w_vel=1.0, w_rot=1.0, w_omega=1.0, w_fuel=0.1):
        super().__init__()
        self.w = {
            "pos": w_pos,
            "vel": w_vel,
            "rot": w_rot,
            "omega": w_omega,
            "fuel": w_fuel,
        }

    def forward(self, pred, target):
        """
        Args:
            pred:   (B, T, 16) predicted states
            target: (B, T, 16) ground-truth states
        """
        losses = {
            "pos":   F.mse_loss(pred[..., 0:3],   target[..., 0:3]),
            "vel":   F.mse_loss(pred[..., 3:6],   target[..., 3:6]),
            "rot":   F.mse_loss(pred[..., 6:12],  target[..., 6:12]),
            "omega": F.mse_loss(pred[..., 12:15], target[..., 12:15]),
            "fuel":  F.mse_loss(pred[..., 15:16], target[..., 15:16]),
        }
        total = sum(self.w[k] * v for k, v in losses.items())
        metrics = {f"prober_{k}": v.item() for k, v in losses.items()}
        metrics["prober_total"] = total.item()
        return total, metrics
