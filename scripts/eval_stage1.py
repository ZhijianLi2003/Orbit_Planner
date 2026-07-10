"""
Orbit Planner Stage 1 evaluation script.

Metrics:
  1. Single-step prediction error (teacher forcing): z_hat_t vs z_{t+1}
  2. Multi-step autoregressive rollout error: latent prediction decay over horizon
  3. Depth prediction visualization: save GT vs Pred comparison plots
  4. Latent space visualization: PCA/t-SNE of z distribution

Usage:
    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/eval_stage1.py --ckpt checkpoints/stage1_worldmodel.pt --num_trajs 100 > logs/eval_stage1.log 2>&1 &

"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from dataset import list_trajectory_keys, load_episode
from world_model.depth_head import DepthHead
from world_model.world_model import OrbitPlanner


def load_model(ckpt_path, device):
    """Load model and config from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    mc = cfg["model"]

    model = OrbitPlanner(
        img_size=mc["img_size"], patch_size=mc["patch_size"],
        vit_dim=mc["vit_dim"], vit_depth=mc["vit_depth"],
        vit_heads=mc["vit_heads"], vit_mlp_ratio=mc["vit_mlp_ratio"],
        state_dim=mc["state_dim"], state_hidden=mc["state_hidden"],
        state_emb_dim=mc["state_emb_dim"],
        action_dim=mc["action_dim"], action_hidden=mc["action_hidden"],
        latent_dim=mc["latent_dim"],
        pred_depth=mc["pred_depth"], pred_heads=mc["pred_heads"],
        pred_mlp_ratio=mc["pred_mlp_ratio"],
        pred_dropout=mc["pred_dropout"], max_seq_len=mc["max_seq_len"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt["epoch"]
    print(f"Loaded checkpoint: epoch {epoch + 1}")
    return model, cfg


def load_trajectory(h5_path, traj_key):
    """Load one trajectory. traj_key is int (flat format) or str (legacy nested format)."""
    return load_episode(h5_path, traj_key)


# ==================== Eval 1: single-step prediction error ==================

@torch.no_grad()
def eval_single_step(model, rgb, depth, states, actions, device, patch_grid):
    """
    Single-step prediction error under teacher forcing.
    Returns loss_pred (latent MSE) and loss_depth (depth MSE).
    """
    T = rgb.shape[0]
    window = min(T, 50)

    rgb_w = rgb[:window].unsqueeze(0).to(device)
    states_w = states[:window].unsqueeze(0).to(device)
    actions_w = actions[:window].unsqueeze(0).to(device)
    depth_w = depth[:window].unsqueeze(0).to(device)

    z, z_hat, depth_pred = model(rgb_w, states_w, actions_w)

    loss_pred = F.mse_loss(z_hat[:, :-1], z[:, 1:]).item()

    B, W = depth_w.shape[:2]
    depth_flat = depth_w.reshape(B * W, *depth_w.shape[2:])
    depth_gt = DepthHead.downsample_depth(depth_flat, patch_grid)
    depth_gt = depth_gt.reshape(B, W, 1, patch_grid, patch_grid)
    loss_depth = F.mse_loss(depth_pred, depth_gt).item()

    return loss_pred, loss_depth


# ==================== Eval 2: multi-step rollout error ====================

@torch.no_grad()
def eval_rollout(model, rgb, states, actions, device, context_len=8):
    """
    Autoregressive rollout: initialize with the first context_len frames,
    predict subsequent frames, and compare against GT encoder outputs.

    Returns: per_step_mse (T-H,) latent prediction error at each step
    """
    T = min(rgb.shape[0], 60)
    H = context_len

    rgb_all = rgb[:T].unsqueeze(0).to(device)
    states_all = states[:T].unsqueeze(0).to(device)
    actions_all = actions[:T].unsqueeze(0).to(device)

    z_gt, _ = model.encode(rgb_all, states_all)  # (1, T, D)

    rgb_ctx = rgb_all[:, :H]
    state_ctx = states_all[:, :H]

    z_rollout = model.rollout(rgb_ctx, state_ctx, actions_all)  # (1, T, D)

    per_step_mse = (z_rollout[:, H:] - z_gt[:, H:]).pow(2).mean(dim=-1)  # (1, T-H)
    return per_step_mse.squeeze(0).cpu().numpy()


# ==================== Eval 3: depth prediction visualization =================

@torch.no_grad()
def save_depth_comparison(model, rgb, depth, states, actions, device,
                          patch_grid, save_dir, traj_name, num_frames=5):
    """Save GT vs Pred depth comparison (single PNG with multiple frame subplots)."""
    T = rgb.shape[0]
    indices = np.linspace(0, T - 1, num_frames, dtype=int)

    rgb_batch = rgb[indices].unsqueeze(0).to(device)
    states_batch = states[indices].unsqueeze(0).to(device)

    _, patches = model.encode(rgb_batch, states_batch)
    B, W = 1, len(indices)
    patches_flat = patches.reshape(B * W, *patches.shape[2:])
    depth_pred = model.depth_head(patches_flat)  # (N, 1, G, G)
    depth_pred = depth_pred.squeeze(1).cpu().numpy()

    depth_gt_full = depth[indices, 0].numpy()  # (N, 224, 224)
    depth_gt_down = DepthHead.downsample_depth(
        depth[indices].unsqueeze(0).reshape(W, 1, 224, 224),
        patch_grid
    ).squeeze(1).numpy()  # (N, G, G)

    rgb_np = rgb[indices].permute(0, 2, 3, 1).numpy()  # (N, 224, 224, 3)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(num_frames, 4, figsize=(12, 3 * num_frames))
        if num_frames == 1:
            axes = axes.reshape(1, -1)

        col_titles = ["RGB", "GT Depth", "GT Depth (down)", "Pred Depth"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title)

        for i in range(num_frames):
            t = indices[i]
            axes[i, 0].imshow(rgb_np[i])
            axes[i, 1].imshow(depth_gt_full[i], cmap="gray", vmin=0, vmax=1)
            axes[i, 2].imshow(depth_gt_down[i], cmap="gray", vmin=0, vmax=1)
            axes[i, 3].imshow(depth_pred[i], cmap="gray", vmin=0, vmax=1)
            for ax in axes[i]:
                ax.axis("off")
            axes[i, 0].set_ylabel(f"t={t}", rotation=0, labelpad=30, va="center")

        fig.suptitle(f"Depth Comparison - {traj_name}", fontsize=14)
        plt.tight_layout()
        out_path = save_dir / f"depth_comparison_{traj_name}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Depth comparison → {out_path}")
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping depth visualization")


# ==================== Eval 4: rollout error vs horizon ====================

def save_rollout_curve(all_rollout_errors, save_dir):
    """Plot rollout error as a function of prediction horizon."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        max_len = max(len(e) for e in all_rollout_errors)
        padded = np.zeros((len(all_rollout_errors), max_len))
        counts = np.zeros(max_len)
        for e in all_rollout_errors:
            padded[:len(e)] += 1
            for i, v in enumerate(e):
                padded[0, i] = 0  # placeholder
        mean_errors = np.zeros(max_len)
        for i in range(max_len):
            vals = [e[i] for e in all_rollout_errors if i < len(e)]
            mean_errors[i] = np.mean(vals) if vals else 0

        plt.figure(figsize=(10, 5))
        for e in all_rollout_errors:
            plt.plot(range(1, len(e) + 1), e, alpha=0.2, color="blue")
        plt.plot(range(1, len(mean_errors) + 1), mean_errors,
                 color="red", linewidth=3)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "rollout_error_curve.png", dpi=100)
        plt.close()
        print(f"  Rollout curve → {save_dir / 'rollout_error_curve.png'}")
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping rollout curve")


# ==================== Main ========================================

def resolve_h5_path(args, ckpt_cfg):
    """Resolve dataset path: CLI > configs/stage1.yaml > checkpoint config."""
    if args.h5_path:
        path = Path(args.h5_path)
    else:
        yaml_path = ROOT / "configs" / "stage1.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                yaml_cfg = yaml.safe_load(f)
            path = Path(yaml_cfg["data"]["h5_path"])
        else:
            path = Path(ckpt_cfg["data"]["h5_path"])
    if not path.is_absolute():
        path = ROOT / path
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Stage 1 checkpoint path")
    parser.add_argument("--h5_path", type=str, default=None,
                        help="HDF5 dataset path (default: read from config)")
    parser.add_argument("--num_trajs", type=int, default=5,
                        help="Number of trajectories to evaluate")
    parser.add_argument("--context_len", type=int, default=8,
                        help="Rollout context length")
    parser.add_argument("--output_dir", type=str, default="eval_output/stage1",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    model, cfg = load_model(ckpt_path, device)
    mc = cfg["model"]
    patch_grid = mc["img_size"] // mc["patch_size"]

    # Dataset path (prefer current yaml over stale path in checkpoint)
    h5_path = resolve_h5_path(args, cfg)

    # Output directory
    save_dir = Path(args.output_dir)
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # Select evaluation trajectories from validation set
    all_keys = list_trajectory_keys(h5_path)
    n_train = int(len(all_keys) * cfg["data"]["train_ratio"])
    val_keys = all_keys[n_train:]
    eval_keys = val_keys[:args.num_trajs]

    print(f"\n{'=' * 60}")
    print(f"Stage 1 Evaluation")
    print(f"  Checkpoint: {ckpt_path.name}")
    print(f"  Dataset:    {h5_path}")
    print(f"  Eval trajs: {len(eval_keys)} (from val set)")
    print(f"  Context H:  {args.context_len}")
    print(f"  Output:     {save_dir}")
    print(f"{'=' * 60}\n")

    # ==================== Per-trajectory evaluation ====================
    all_pred_losses = []
    all_depth_losses = []
    all_rollout_errors = []
    depth_saved = False

    for key in tqdm(eval_keys, desc="Evaluating"):
        traj_name = f"traj_{key:04d}" if isinstance(key, int) else str(key)
        rgb, depth, states, actions = load_trajectory(h5_path, key)

        if rgb.shape[0] < args.context_len + 5:
            print(f"  {traj_name}: too short ({rgb.shape[0]} steps), skipping")
            continue

        # Single-step prediction
        l_pred, l_depth = eval_single_step(
            model, rgb, depth, states, actions, device, patch_grid)
        all_pred_losses.append(l_pred)
        all_depth_losses.append(l_depth)

        # Multi-step rollout
        rollout_err = eval_rollout(
            model, rgb, states, actions, device, args.context_len)
        all_rollout_errors.append(rollout_err)

        # Depth visualization (first trajectory only)
        if not depth_saved:
            save_depth_comparison(
                model, rgb, depth, states, actions, device,
                patch_grid, save_dir, traj_name)
            depth_saved = True

    # ==================== Summary ====================
    print(f"\n{'=' * 60}")
    print("Summary:")
    print(f"  Single-step latent pred MSE:  {np.mean(all_pred_losses):.6f}")
    print(f"  Single-step depth MSE:        {np.mean(all_depth_losses):.6f}")

    if all_rollout_errors:
        horizon_5 = np.mean([e[4] for e in all_rollout_errors if len(e) > 4])
        horizon_10 = np.mean([e[9] for e in all_rollout_errors if len(e) > 9])
        horizon_20 = np.mean([e[19] for e in all_rollout_errors if len(e) > 19])
        print(f"  Rollout MSE @ horizon 5:      {horizon_5:.6f}")
        print(f"  Rollout MSE @ horizon 10:     {horizon_10:.6f}")
        print(f"  Rollout MSE @ horizon 20:     {horizon_20:.6f}")

        save_rollout_curve(all_rollout_errors, save_dir)

    print(f"{'=' * 60}")

    # Save numeric results
    results = {
        "pred_mse_mean": float(np.mean(all_pred_losses)),
        "pred_mse_std": float(np.std(all_pred_losses)),
        "depth_mse_mean": float(np.mean(all_depth_losses)),
        "depth_mse_std": float(np.std(all_depth_losses)),
        "num_trajs_evaluated": len(all_pred_losses),
    }
    if all_rollout_errors:
        results["rollout_mse_horizon_5"] = float(horizon_5)
        results["rollout_mse_horizon_10"] = float(horizon_10)
        results["rollout_mse_horizon_20"] = float(horizon_20)

    import json
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {save_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
