"""
Orbit Planner Stage 2 training script.

Freeze Stage 1 encoder + predictor; train Physics Prober.

Usage:

    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python scripts/train_stage2.py --config configs/stage2.yaml --stage1_ckpt checkpoints/stage1_worldmodel.pt > logs/train_stage2.log 2>&1 &
"""

import argparse
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    OrbitPlannerDataset, worker_init_fn, list_trajectory_keys, ChunkedRandomSampler,
)
from world_model.prober import PhysicsProber, ProberLoss
from world_model.world_model import OrbitPlanner


# ========================= Utils ==================================

def load_config(path, overrides=None):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for k, v in overrides.items():
            for section in cfg.values():
                if isinstance(section, dict) and k in section:
                    section[k] = type(section[k])(v)
    return cfg


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(optimizer, epoch, warmup, total):
    if epoch < warmup:
        lr_scale = (epoch + 1) / warmup
    else:
        progress = (epoch - warmup) / max(1, total - warmup)
        lr_scale = 0.5 * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = pg["_base_lr"] * lr_scale


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ========================= Stage 1 model loading ====================

def load_frozen_world_model(cfg, ckpt_path, device):
    """Load and freeze the Stage 1 world model."""
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

    print(f"Stage 1 world model loaded from {ckpt_path}")
    print(f"  (frozen, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params)")
    return model


# ========================= Training step ===========================

def train_step(world_model, prober, prober_criterion,
               batch, H, device, cfg):
    """
    Single training step.

    Data flow:
      1. First H frames as context, frozen rollout → z_all (B, W, D)
      2. z_future = z_all[:, H:] (B, T, D)
      3. Prober(z_future, x_{H-1}, a_{H-1:W-2}) → pred_states
    """
    W = batch["actions"].shape[1]
    T = W - H

    rgb = batch["rgb"].to(device)
    states = batch["states"].to(device)
    actions = batch["actions"].to(device)

    # --- Step 1: Frozen world model rollout ---
    rgb_ctx = rgb[:, :H]
    state_ctx = states[:, :H]

    with torch.no_grad():
        z_all = world_model.rollout(rgb_ctx, state_ctx, actions)

    z_future = z_all[:, H:]  # (B, T, D)

    # --- Step 2: Prober ---
    init_state = states[:, H - 1]
    actions_prober = actions[:, H - 1:H - 1 + T]
    gt_states = states[:, H:]

    pred_states = prober(z_future, init_state, actions_prober)
    prober_loss, prober_metrics = prober_criterion(pred_states, gt_states)

    # --- Total loss ---
    tc = cfg["training"]
    total_loss = tc["lambda_prober"] * prober_loss

    metrics = {**prober_metrics, "total_loss": total_loss.item()}
    return total_loss, metrics


# ========================= Validation ==============================

@torch.no_grad()
def validate(world_model, prober, prober_criterion,
             loader, H, device, cfg, use_amp, amp_dtype_torch):
    prober.eval()
    all_metrics = defaultdict(list)

    for batch in loader:
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
            _, metrics = train_step(
                world_model, prober, prober_criterion,
                batch, H, device, cfg,
            )
        for k, v in metrics.items():
            all_metrics[k].append(v)

    prober.train()
    return {k: sum(v) / len(v) for k, v in all_metrics.items()}


# ========================= Main ===================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=str(ROOT / "configs" / "stage2.yaml"))
    parser.add_argument("--stage1_ckpt", type=str, default=None,
                        help="Override stage1 checkpoint path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a stage2 checkpoint")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    overrides = {}
    if args.batch_size:
        overrides["batch_size"] = args.batch_size
    if args.epochs:
        overrides["epochs"] = args.epochs
    if args.device:
        overrides["device"] = args.device

    cfg = load_config(args.config, overrides)
    dc, mc, tc = cfg["data"], cfg["model"], cfg["training"]
    pc = cfg["prober"]

    h5_path = Path(dc["h5_path"])
    if not h5_path.is_absolute():
        h5_path = ROOT / h5_path
    out_dir = Path(tc["output_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    stage1_ckpt = args.stage1_ckpt or cfg["stage1"]["checkpoint"]
    stage1_ckpt = Path(stage1_ckpt)
    if not stage1_ckpt.is_absolute():
        stage1_ckpt = ROOT / stage1_ckpt
    if not stage1_ckpt.exists():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {stage1_ckpt}\n"
            "Train Stage 1 first, or pass --stage1_ckpt explicitly."
        )

    set_seed(tc["seed"])
    device = torch.device(tc["device"] if torch.cuda.is_available() else "cpu")
    H = dc["context_steps"]
    if H >= dc["window_size"]:
        raise ValueError(
            f"context_steps ({H}) must be < window_size ({dc['window_size']})"
        )
    print(f"Device: {device}")
    print(f"Context H={H}, Prediction T={dc['window_size'] - H}")

    # -------------------- W&B --------------------
    use_wandb = tc.get("wandb_entity") is not None
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=tc["wandb_project"], entity=tc["wandb_entity"],
                   config=cfg, tags=["stage2"])

    # -------------------- Data --------------------
    data_path = h5_path
    all_keys = list_trajectory_keys(data_path)
    n_train = int(len(all_keys) * dc["train_ratio"])
    train_ids = list(range(n_train))
    val_ids = list(range(n_train, len(all_keys)))

    train_set = OrbitPlannerDataset(
        data_path, dc["window_size"],
        window_stride=dc.get("window_stride", 1),
        traj_ids=train_ids, augment=dc.get("augment", False),
        load_events=False, load_depth=False, rgb_load_frames=H,
    )
    val_set = OrbitPlannerDataset(
        data_path, dc["window_size"],
        window_stride=dc.get("window_stride", 1),
        traj_ids=val_ids, augment=False,
        load_events=False, load_depth=False, rgb_load_frames=H,
    )

    nw = tc["num_workers"]
    train_sampler = ChunkedRandomSampler(train_set, chunk_size=5000, seed=tc["seed"])
    train_loader = DataLoader(
        train_set, batch_size=tc["batch_size"],
        sampler=train_sampler,
        num_workers=nw, pin_memory=True,
        worker_init_fn=worker_init_fn, drop_last=True,
        persistent_workers=nw > 0, prefetch_factor=3 if nw > 0 else None,
    )
    val_loader = DataLoader(
        val_set, batch_size=tc["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=nw > 0, prefetch_factor=3 if nw > 0 else None,
    )
    print(f"Train: {len(train_set)} windows  |  Val: {len(val_set)} windows")

    # -------------------- Models --------------------
    world_model = load_frozen_world_model(cfg, str(stage1_ckpt), device)

    prober = PhysicsProber(
        latent_dim=mc["latent_dim"],
        hidden_dim=pc["hidden_dim"],
        num_layers=pc["num_layers"],
        dt=pc["dt"],
    ).to(device)

    print(f"Prober params: {count_params(prober) / 1e3:.1f}K")

    # -------------------- Loss / Opt --------------------
    prober_criterion = ProberLoss(
        w_pos=tc["w_pos"], w_vel=tc["w_vel"],
        w_rot=tc["w_rot"], w_omega=tc["w_omega"],
        w_fuel=tc["w_fuel"],
    )

    all_params = list(prober.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=tc["lr"], weight_decay=tc["weight_decay"],
    )
    for pg in optimizer.param_groups:
        pg["_base_lr"] = tc["lr"]

    amp_dtype = tc.get("mixed_precision", "bf16")
    use_amp = amp_dtype in (True, "fp16", "bf16")
    amp_dtype_torch = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_scaler = amp_dtype not in ("bf16",)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler and use_amp)

    # -------------------- Output --------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    postfix_window = tc.get("postfix_window", 20)
    start_epoch = 0

    # -------------------- Resume --------------------
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = ROOT / resume_path
        resume_ckpt = torch.load(str(resume_path), map_location=device, weights_only=False)
        prober.load_state_dict(resume_ckpt["prober"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = resume_ckpt["epoch"] + 1
        print(f"Resumed from {resume_path.name}, starting at epoch {start_epoch + 1}")
        del resume_ckpt

    # -------------------- Train loop --------------------
    for epoch in range(start_epoch, tc["epochs"]):
        train_sampler.set_epoch(epoch)
        cosine_lr(optimizer, epoch, tc["warmup_epochs"], tc["epochs"])
        prober.train()

        epoch_metrics = defaultdict(list)
        running_store = defaultdict(list)
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{tc['epochs']}",
                    dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
                loss, metrics = train_step(
                    world_model, prober, prober_criterion,
                    batch, H, device, cfg,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(all_params, tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            for k, v in metrics.items():
                epoch_metrics[k].append(v)
                running_store[k].append(v)
                if len(running_store[k]) > postfix_window:
                    running_store[k] = running_store[k][-postfix_window:]

            display = {k: sum(v) / len(v) for k, v in running_store.items()}
            display_str = {k: f"{v:.4f}" for k, v in display.items()}
            display_str["lr"] = f"{lr_now:.2e}"
            pbar.set_postfix(**display_str, refresh=False)

            if (step + 1) % tc["log_interval"] == 0:
                n = tc["log_interval"]
                window_avg = {k: sum(v[-n:]) / n for k, v in epoch_metrics.items()}
                elapsed = time.time() - t0
                speed = (step + 1) / elapsed
                msg = (f"  [Step {step + 1}] "
                       + " | ".join(f"{k}={v:.4f}" for k, v in window_avg.items())
                       + f" | {speed:.2f} it/s")
                pbar.write(msg)

        # ---- epoch summary ----
        dt = time.time() - t0
        avg = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        epoch_metrics.clear()
        running_store.clear()
        print(f"[Epoch {epoch + 1}] "
              + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
              + f" | lr={lr_now:.2e} | time={dt:.0f}s")

        if use_wandb:
            wandb.log({"epoch": epoch + 1, "lr": lr_now,
                       **{f"train/{k}": v for k, v in avg.items()}})

        # ---- validation ----
        val_metrics = validate(
            world_model, prober, prober_criterion,
            val_loader, H, device, cfg, use_amp, amp_dtype_torch,
        )
        print(f"  [Val] "
              + " | ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

        if use_wandb:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()})

        # ---- save best ----
        val_loss = val_metrics.get("total_loss", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(out_dir / "best_stage2.pt",
                             epoch, prober, optimizer, cfg)
            print(f"  ★ Best val_loss={val_loss:.4f}, saved → best_stage2.pt")

        # ---- periodic save ----
        if (epoch + 1) % tc["save_interval"] == 0 or epoch == tc["epochs"] - 1:
            _save_checkpoint(out_dir / f"stage2_epoch{epoch + 1:03d}.pt",
                             epoch, prober, optimizer, cfg)
            print(f"  Saved → stage2_epoch{epoch + 1:03d}.pt")

    print("Stage 2 training complete.")


def _save_checkpoint(path, epoch, prober, optimizer, cfg):
    data = {
        "epoch": epoch,
        "prober": prober.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg,
    }
    torch.save(data, path)


if __name__ == "__main__":
    main()
