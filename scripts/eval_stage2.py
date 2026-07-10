"""
Orbit Planner Stage 2 evaluation script.

Metrics:
  1. Physics Prober per-component error: MSE & MAE for pos / vel / rot / omega / fuel
  2. Per-horizon error: total MAE at selected prediction steps

Usage:

    CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python scripts/eval_stage2.py --stage2_ckpt checkpoints/stage2_physicspober.pt --stage1_ckpt checkpoints/stage1_worldmodel.pt --pred_horizon 25 --num_trajs 100 > logs/eval_stage2.log 2>&1 &
"""

import argparse
import json
import sys
from collections import defaultdict
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
from world_model.prober import PhysicsProber
from world_model.world_model import OrbitPlanner

STATE_SLICES = {
    "pos":   (0, 3),
    "vel":   (3, 6),
    "rot":   (6, 12),
    "omega": (12, 15),
    "fuel":  (15, 16),
}


# ========================= Model loading ==========================

def load_frozen_world_model(cfg, ckpt_path, device):
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

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    del ckpt
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def load_stage2_models(cfg, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mc = cfg["model"]
    pc = cfg["prober"]

    prober = PhysicsProber(
        latent_dim=mc["latent_dim"],
        hidden_dim=pc["hidden_dim"],
        num_layers=pc["num_layers"],
        dt=pc["dt"],
    ).to(device)
    prober.load_state_dict(ckpt["prober"])
    prober.eval()

    epoch = ckpt["epoch"]
    print(f"Stage 2 checkpoint loaded: epoch {epoch + 1}")
    return prober


# ========================= Single-trajectory evaluation =============

@torch.no_grad()
def eval_trajectory(world_model, prober,
                    rgb, states, actions, device, H,
                    pred_horizon=12):
    """
    Run frozen rollout + prober prediction on one trajectory.

    Returns:
        pred_states:  (T, 16) predicted states
        gt_states:    (T, 16) ground-truth states
    """
    T_total = min(rgb.shape[0], H + pred_horizon)
    if T_total <= H + 1:
        return None

    rgb_t = rgb[:T_total].unsqueeze(0).to(device)
    states_t = states[:T_total].unsqueeze(0).to(device)
    actions_t = actions[:T_total].unsqueeze(0).to(device)

    rgb_ctx = rgb_t[:, :H]
    state_ctx = states_t[:, :H]

    z_all = world_model.rollout(rgb_ctx, state_ctx, actions_t)
    z_future = z_all[:, H:]  # (1, T, D)

    T = T_total - H
    init_state = states_t[:, H - 1]
    actions_prober = actions_t[:, H - 1:H - 1 + T]
    gt_s = states_t[:, H:]

    pred_s = prober(z_future, init_state, actions_prober)  # (1, T, 16)

    pred_states = pred_s.squeeze(0).cpu()
    gt_states = gt_s.squeeze(0).cpu()

    return pred_states, gt_states


# ========================= Metric computation =======================

def compute_per_step_errors(pred_states, gt_states):
    """Compute per-step MSE and MAE for each state component."""
    errors = {}
    for name, (s, e) in STATE_SLICES.items():
        diff = pred_states[:, s:e] - gt_states[:, s:e]
        errors[f"{name}_mse"] = diff.pow(2).mean(dim=-1).numpy()  # (T,)
        errors[f"{name}_mae"] = diff.abs().mean(dim=-1).numpy()   # (T,)
    all_diff = pred_states - gt_states
    errors["total_mse"] = all_diff.pow(2).mean(dim=-1).numpy()
    errors["total_mae"] = all_diff.abs().mean(dim=-1).numpy()
    return errors


# ========================= Main ====================================

def resolve_h5_path(args, cfg):
    if args.h5_path:
        path = Path(args.h5_path)
    else:
        yaml_path = ROOT / "configs" / "stage2.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                yaml_cfg = yaml.safe_load(f)
            path = Path(yaml_cfg["data"]["h5_path"])
        else:
            path = Path(cfg["data"]["h5_path"])
    if not path.is_absolute():
        path = ROOT / path
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_ckpt", type=str, required=True,
                        help="Stage 2 checkpoint path")
    parser.add_argument("--stage1_ckpt", type=str, default=None,
                        help="Stage 1 checkpoint (default: read from stage2 config)")
    parser.add_argument("--h5_path", type=str, default=None)
    parser.add_argument("--num_trajs", type=int, default=10)
    parser.add_argument("--context_len", type=int, default=8)
    parser.add_argument("--pred_horizon", type=int, default=12,
                        help="Prediction horizon in steps (default 12, matches training)")
    parser.add_argument("--output_dir", type=str, default="eval_output/stage2")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load config from Stage 2 checkpoint
    s2_ckpt = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    cfg = s2_ckpt["config"]
    del s2_ckpt

    H = args.context_len

    # Stage 1 world model
    stage1_path = args.stage1_ckpt or cfg["stage1"]["checkpoint"]
    stage1_path = Path(stage1_path)
    if not stage1_path.is_absolute():
        stage1_path = ROOT / stage1_path
    world_model = load_frozen_world_model(cfg, str(stage1_path), device)
    print(f"Stage 1 world model: {stage1_path.name}")

    # Stage 2 prober
    prober = load_stage2_models(cfg, args.stage2_ckpt, device)

    # Data
    h5_path = resolve_h5_path(args, cfg)
    all_keys = list_trajectory_keys(h5_path)
    n_train = int(len(all_keys) * cfg["data"]["train_ratio"])
    val_keys = all_keys[n_train:]
    eval_keys = val_keys[:args.num_trajs]

    # Output
    save_dir = Path(args.output_dir)
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("Stage 2 Evaluation")
    print(f"  Stage 1 ckpt:  {stage1_path.name}")
    print(f"  Stage 2 ckpt:  {Path(args.stage2_ckpt).name}")
    print(f"  Dataset:       {h5_path}")
    print(f"  Eval trajs:    {len(eval_keys)} (from val set)")
    print(f"  Context H:     {H}")
    print(f"  Pred horizon:  {args.pred_horizon}")
    print(f"  Output:        {save_dir}")
    print(f"{'=' * 60}\n")

    # ==================== Per-trajectory evaluation ====================
    all_step_errors = defaultdict(list)
    summary_prober = defaultdict(list)

    for key in tqdm(eval_keys, desc="Evaluating"):
        traj_name = f"traj_{key:04d}" if isinstance(key, int) else str(key)
        rgb, _, states, actions = load_episode(h5_path, key)

        if rgb.shape[0] < H + 5:
            print(f"  {traj_name}: too short ({rgb.shape[0]} steps), skipping")
            continue

        result = eval_trajectory(
            world_model, prober,
            rgb, states, actions, device, H,
            pred_horizon=args.pred_horizon,
        )
        if result is None:
            continue
        pred_states, gt_states = result

        # Per-step errors
        step_errors = compute_per_step_errors(pred_states, gt_states)
        for k, v in step_errors.items():
            all_step_errors[k].append(v)

        # Trajectory-level summary (mean over all steps)
        for name, (s, e) in STATE_SLICES.items():
            diff = pred_states[:, s:e] - gt_states[:, s:e]
            summary_prober[f"{name}_mse"].append(diff.pow(2).mean().item())
            summary_prober[f"{name}_mae"].append(diff.abs().mean().item())

    # ==================== Summary ====================
    print(f"\n{'=' * 60}")
    print("Summary (mean ± std across trajectories):")
    print("-" * 60)

    results = {}

    print("\n  [Physics Prober]")
    for name in ["pos", "vel", "rot", "omega", "fuel"]:
        for metric in ["mse", "mae"]:
            k = f"{name}_{metric}"
            vals = summary_prober[k]
            m, s = np.mean(vals), np.std(vals)
            print(f"    {k:>12s}:  {m:.6f} ± {s:.6f}")
            results[f"prober_{k}_mean"] = float(m)
            results[f"prober_{k}_std"] = float(s)

    # Per-horizon error report
    print("\n  [Per-Horizon Prober MAE]")
    for h in [1, 3, 5, 8, 12]:
        vals = [e[h - 1] for e in all_step_errors.get("total_mae", [])
                if len(e) >= h]
        if vals:
            m = np.mean(vals)
            print(f"    horizon {h:>2d}:  total_mae = {m:.6f}")
            results[f"prober_total_mae_h{h}"] = float(m)

    print(f"\n{'=' * 60}")

    results["num_trajs_evaluated"] = len(summary_prober["pos_mse"])

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {save_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
