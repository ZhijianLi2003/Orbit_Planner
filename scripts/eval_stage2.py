"""
Orbit-Planner Stage 2 evaluation script.

Evaluation:
  1. Physics prober prediction errors for each state component: MSE and MAE
     for position, velocity, angular velocity, and fuel. Orientation uses the
     SO(3) geodesic angular error (radians):
     θ = arccos(clip((tr(R_gt^T R_pred)-1)/2)).
  2. Visualization: 3D ground-truth and predicted trajectory comparison.

"""



import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from dataset import list_trajectory_keys, load_episode
from world_model.prober import PhysicsProber, rot6d_to_matrix
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


# ========================= Meta / window sampling ==================

def load_meta_records(meta_path):
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)["trajectories"]


def resolve_meta_path(args, h5_path):
    if getattr(args, "meta_path", None):
        path = Path(args.meta_path)
    else:
        candidates = [
            ROOT / "data" / "meta.json",
            h5_path.parent / "meta.json",
        ]
        path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.is_absolute():
        path = ROOT / path
    return path


def collision_step_for_key(records, key):
    """Return collision_step, or -1 if no collision / unknown."""
    if not records:
        return -1
    if isinstance(key, int):
        if 0 <= key < len(records):
            rec = records[key]
            if rec.get("has_collision", False):
                return int(rec["collision_step"])
        return -1
    for rec in records:
        if rec.get("traj_id") == key:
            if rec.get("has_collision", False):
                return int(rec["collision_step"])
            return -1
    return -1


def max_valid_start(T_ep, W, collision_step=-1):
    """
    窗口 [start, start+W) 不能包含碰撞帧。
    有碰撞时要求 start+W <= collision_step，即整窗落在首撞之前。
    """
    max_start = T_ep - W
    if collision_step is not None and collision_step >= 0:
        max_start = min(max_start, collision_step - W)
    return max_start


def sample_window_start(T_ep, H, pred_horizon, rng, collision_step=-1):
    """随机采样合法窗口起点；无合法起点时返回 None。"""
    W = H + pred_horizon
    max_start = max_valid_start(T_ep, W, collision_step)
    if max_start < 0:
        return None
    return rng.randint(0, max_start)


# ========================= Single-trajectory evaluation =============

@torch.no_grad()
def eval_trajectory(world_model, prober,
                    rgb, states, actions, device, H,
                    pred_horizon=12, start=0):
    """
    Run frozen rollout + prober on window [start, start+H+pred_horizon).

    Returns:
        pred_states:  (T, 16) predicted states
        gt_states:    (T, 16) ground-truth states
    """
    T_need = H + pred_horizon
    end = min(start + T_need, rgb.shape[0])
    T_total = end - start
    if T_total <= H + 1:
        return None

    rgb_t = rgb[start:end].unsqueeze(0).to(device)
    states_t = states[start:end].unsqueeze(0).to(device)
    actions_t = actions[start:end].unsqueeze(0).to(device)

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

def rotation_geodesic_error(pred_states, gt_states):
    """SO(3) geodesic angle between pred/gt rot6d.

    R_err = R_gt^T @ R_pred,  θ = arccos(clip((tr(R_err)-1)/2)).

    Args:
        pred_states: (..., 16) or rot handled via slices 6:12
        gt_states:   (..., 16)

    Returns:
        theta: (...) geodesic angle in radians
    """
    R_pred = rot6d_to_matrix(pred_states[..., 6:12])
    R_gt = rot6d_to_matrix(gt_states[..., 6:12])
    R_err = R_gt.transpose(-1, -2) @ R_pred
    tr = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos_theta)


def compute_per_step_errors(pred_states, gt_states):
    """Compute per-step MSE and MAE for each state component.

    Rotation uses SO(3) geodesic angle (rad): rot_mae = θ, rot_mse = θ².
    """
    errors = {}
    for name, (s, e) in STATE_SLICES.items():
        if name == "rot":
            theta = rotation_geodesic_error(pred_states, gt_states)
            errors["rot_mse"] = theta.pow(2).numpy()
            errors["rot_mae"] = theta.numpy()
        else:
            diff = pred_states[:, s:e] - gt_states[:, s:e]
            errors[f"{name}_mse"] = diff.pow(2).mean(dim=-1).numpy()  # (T,)
            errors[f"{name}_mae"] = diff.abs().mean(dim=-1).numpy()   # (T,)
    all_diff = pred_states - gt_states
    errors["total_mse"] = all_diff.pow(2).mean(dim=-1).numpy()
    errors["total_mae"] = all_diff.abs().mean(dim=-1).numpy()
    return errors


# ========================= Visualization ============================

def save_trajectory_3d(pred_states, gt_states, save_path, traj_name,
                       origin=None):
    """3D 轨迹对比: GT vs Pred。坐标相对窗口起点。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if origin is None:
            origin = gt_states[0, :3].numpy()
        else:
            origin = np.asarray(origin, dtype=np.float32)

        gt_pos = gt_states[:, :3].numpy() - origin
        pred_pos = pred_states[:, :3].numpy() - origin

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # 坐标映射: X→X, Y→Y, Z→竖直
        # matplotlib 3D 的竖直轴是第三个参数(Z)
        # 数据的 Z 放到竖直轴, X/Y 放水平面
        ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
                "b-o", markersize=3, alpha=0.8)
        ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2],
                "r-s", markersize=3, alpha=0.8)

        ax.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2],
                   c="green", s=80, marker="^")
        ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2],
                   c="black", s=80, marker="v")

        # 范围相对于窗口起点
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-2.0, 1.0)
        ax.set_zlim(-2.0, 1.0)

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

        # 调整视角：仰角30°，方位角-60°，让 Z 轴竖直更明显
        ax.view_init(elev=30, azim=-60)

        plt.tight_layout()
        plt.savefig(save_path, dpi=120)
        plt.close()
    except ImportError:
        print("  [WARN] matplotlib not installed, skipping 3D trajectory plot")


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
    parser.add_argument("--meta_path", type=str, default=None,
                        help="meta.json path (for collision_step; default data/meta.json)")
    parser.add_argument("--num_trajs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling trajs and window starts")
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
    meta_path = resolve_meta_path(args, h5_path)
    records = load_meta_records(meta_path) if meta_path.exists() else []
    if not records:
        print(f"[WARN] meta not found ({meta_path}); cannot exclude collision frames")

    all_keys = list_trajectory_keys(h5_path)
    n_train = int(len(all_keys) * cfg["data"]["train_ratio"])
    val_keys = all_keys[n_train:]
    rng = random.Random(args.seed)

    # Output
    save_dir = Path(args.output_dir)
    if not save_dir.is_absolute():
        save_dir = ROOT / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    W = H + args.pred_horizon
    print(f"\n{'=' * 60}")
    print("Stage 2 Evaluation")
    print(f"  Stage 1 ckpt:  {stage1_path.name}")
    print(f"  Stage 2 ckpt:  {Path(args.stage2_ckpt).name}")
    print(f"  Dataset:       {h5_path}")
    print(f"  Meta:          {meta_path if records else 'N/A'}")
    print(f"  Eval trajs:    {args.num_trajs} (random traj + random window, "
          f"no collision frames, seed={args.seed})")
    print(f"  Context H:     {H}")
    print(f"  Pred horizon:  {args.pred_horizon}")
    print(f"  Window W:      {W}")
    print(f"  Output:        {save_dir}")
    print(f"{'=' * 60}\n")

    # ==================== Per-trajectory evaluation ====================
    all_step_errors = defaultdict(list)
    summary_prober = defaultdict(list)

    # 随机抽轨迹；无合法无碰撞窗口则换下一条，直到凑够 num_trajs
    candidate_keys = list(val_keys)
    rng.shuffle(candidate_keys)
    n_done = 0

    for key in tqdm(candidate_keys, desc="Evaluating"):
        if n_done >= args.num_trajs:
            break

        traj_name = f"traj_{key:04d}" if isinstance(key, int) else str(key)
        rgb, _, states, actions = load_episode(h5_path, key)
        T_ep = rgb.shape[0]
        cs = collision_step_for_key(records, key)

        start = sample_window_start(
            T_ep, H, args.pred_horizon, rng, collision_step=cs)
        if start is None:
            continue

        result = eval_trajectory(
            world_model, prober,
            rgb, states, actions, device, H,
            pred_horizon=args.pred_horizon,
            start=start,
        )
        if result is None:
            continue
        pred_states, gt_states = result
        n_done += 1
        tag = f"{traj_name}_s{start}"

        # Per-step errors
        step_errors = compute_per_step_errors(pred_states, gt_states)
        for k, v in step_errors.items():
            all_step_errors[k].append(v)

        # Trajectory-level summary (mean over all steps)
        for name, (s, e) in STATE_SLICES.items():
            if name == "rot":
                theta = rotation_geodesic_error(pred_states, gt_states)
                summary_prober["rot_mse"].append(theta.pow(2).mean().item())
                summary_prober["rot_mae"].append(theta.mean().item())  # mean geodesic (rad)
            else:
                diff = pred_states[:, s:e] - gt_states[:, s:e]
                summary_prober[f"{name}_mse"].append(diff.pow(2).mean().item())
                summary_prober[f"{name}_mae"].append(diff.abs().mean().item())

        # Visualize the first 10 windows
        if n_done <= 10:
            origin = states[start, :3].numpy()
            save_trajectory_3d(
                pred_states, gt_states,
                save_dir / f"traj3d_{tag}.png", tag,
                origin=origin)

    # ==================== Summary ====================
    print(f"\n{'=' * 60}")
    print("Summary (mean ± std across trajectories):")
    print("-" * 60)

    results = {}

    print("\n  [Physics Prober]")
    print("  (rot_mae / rot_mse: SO(3) geodesic θ / θ² in radians)")
    for name in ["pos", "vel", "rot", "omega", "fuel"]:
        for metric in ["mse", "mae"]:
            k = f"{name}_{metric}"
            vals = summary_prober[k]
            m, s = np.mean(vals), np.std(vals)
            unit = " rad" if name == "rot" and metric == "mae" else ""
            print(f"    {k:>12s}:  {m:.6f} ± {s:.6f}{unit}")
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
    if results["num_trajs_evaluated"] < args.num_trajs:
        print(f"[WARN] only evaluated {results['num_trajs_evaluated']}/{args.num_trajs} "
              f"(others lack a full collision-free window of length {W})")

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved → {save_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()



