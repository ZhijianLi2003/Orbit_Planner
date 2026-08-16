"""
Orbit Planner Stage 1 training script.

"""

import argparse
import math
import random
import sys
import time
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
from world_model.depth_head import DepthHead
from world_model.losses import OrbitPlannerLoss
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


def update_running_metrics(metrics, window):
    """Keep the last `window` values per metric and return rolling averages."""
    store = update_running_metrics.store
    for k, v in metrics.items():
        store.setdefault(k, []).append(v)
        if len(store[k]) > window:
            store[k] = store[k][-window:]
    return {k: sum(vs) / len(vs) for k, vs in store.items()}


update_running_metrics.store = {}


def reset_running_metrics():
    update_running_metrics.store = {}


def format_postfix(metrics, lr=None):
    parts = {k: f"{v:.4f}" for k, v in metrics.items()}
    if lr is not None:
        parts["lr"] = f"{lr:.2e}"
    return parts


# ========================= Main ===================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                        default=str(ROOT / "configs" / "stage1.yaml"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path")
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

    # Resolve relative paths from project root
    h5_path = Path(dc["h5_path"])
    if not h5_path.is_absolute():
        h5_path = ROOT / h5_path
    out_dir = Path(tc["output_dir"])
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    set_seed(tc["seed"])
    device = torch.device(tc["device"] if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -------------------- W&B --------------------
    use_wandb = tc.get("wandb_entity") is not None
    wandb = None
    if use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=tc["wandb_project"], entity=tc["wandb_entity"],
                   config=cfg)

    # -------------------- Data --------------------
    data_path = h5_path
    all_keys = list_trajectory_keys(data_path)
    n_train = int(len(all_keys) * dc["train_ratio"])
    train_ids = list(range(n_train))
    val_ids = list(range(n_train, len(all_keys)))

    train_set = OrbitPlannerDataset(
        data_path, dc["window_size"],
        window_stride=dc.get("window_stride", 1),
        traj_ids=train_ids,
        augment=dc["augment"],
    )
    val_set = OrbitPlannerDataset(
        data_path, dc["window_size"],
        window_stride=dc.get("window_stride", 1),
        traj_ids=val_ids,
        augment=False,
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

    # -------------------- Model --------------------
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
    print(f"Model params: {count_params(model)/1e6:.2f} M")

    # -------------------- Loss / Opt --------------------
    criterion = OrbitPlannerLoss(
        lambda_sigreg=tc["lambda_sigreg"],
        lambda_depth=tc["lambda_depth"],
        sigreg_projections=tc["sigreg_projections"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"]
    )
    for pg in optimizer.param_groups:
        pg["_base_lr"] = tc["lr"]

    # bf16 does not need GradScaler (no overflow risk); fp16 does
    amp_dtype = tc.get("mixed_precision", "bf16")
    use_amp = amp_dtype in (True, "fp16", "bf16")
    amp_dtype_torch = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    use_scaler = amp_dtype not in ("bf16",)  # bf16 needs no scaler
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler and use_amp)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # -------------------- Output dir --------------------
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------- Train loop --------------------
    patch_grid = mc["img_size"] // mc["patch_size"]
    best_val_loss = float("inf")

    for epoch in range(start_epoch, tc["epochs"]):
        train_sampler.set_epoch(epoch)
        cosine_lr(optimizer, epoch, tc["warmup_epochs"], tc["epochs"])
        model.train()

        epoch_metrics = {}
        reset_running_metrics()
        postfix_window = tc.get("postfix_window", 20)
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{tc['epochs']}",
                    dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            rgb = batch["rgb"].to(device)          # (B, W, 3, 224, 224)
            depth = batch["depth"].to(device)      # (B, W, 1, 224, 224)
            states = batch["states"].to(device)    # (B, W, 16)
            actions = batch["actions"].to(device)  # (B, W, 8)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
                z, z_hat, depth_pred = model(rgb, states, actions)

                B, W = depth.shape[:2]
                depth_flat = depth.reshape(B * W, *depth.shape[2:])
                depth_gt = DepthHead.downsample_depth(depth_flat, patch_grid)
                depth_gt = depth_gt.reshape(B, W, 1, patch_grid, patch_grid)

                loss, metrics = criterion(z, z_hat, depth_pred, depth_gt)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            for k, v in metrics.items():
                epoch_metrics.setdefault(k, []).append(v)

            # Update tqdm every step: rolling average over last postfix_window steps
            display = update_running_metrics(metrics, postfix_window)
            pbar.set_postfix(**format_postfix(display, lr=lr_now), refresh=False)

            # Print detailed log every log_interval steps
            if (step + 1) % tc["log_interval"] == 0:
                window_avg = {k: sum(v[-tc["log_interval"]:]) / tc["log_interval"]
                              for k, v in epoch_metrics.items()}
                elapsed = time.time() - t0
                speed = (step + 1) / elapsed
                msg = (f"  [Step {step+1}] "
                       + " | ".join(f"{k}={v:.4f}" for k, v in window_avg.items())
                       + f" | {speed:.2f} it/s")
                pbar.write(msg)
                if use_wandb:
                    wandb.log({"train/step": epoch * len(train_loader) + step + 1,
                               **window_avg, "lr": lr_now})

        # ---- epoch summary ----
        dt = time.time() - t0
        avg = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[Epoch {epoch+1}] "
              + " | ".join(f"{k}={v:.4f}" for k, v in avg.items())
              + f" | lr={lr_now:.2e} | time={dt:.0f}s")

        if use_wandb:
            wandb.log({"epoch": epoch + 1, "lr": lr_now, **avg})

        # ---- validation ----
        val_interval = tc.get("val_interval", 1)
        if (epoch + 1) % val_interval == 0 or epoch == tc["epochs"] - 1:
            val_metrics = validate(model, val_loader, criterion,
                                   device, patch_grid,
                                   use_amp, amp_dtype_torch)
            print(f"  [Val] " + " | ".join(f"{k}={v:.4f}"
                                           for k, v in val_metrics.items()))
            if use_wandb:
                wandb.log({f"val_{k}": v for k, v in val_metrics.items()})

            val_loss = val_metrics.get("loss_total", float("inf"))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": cfg,
                }, out_dir / "best.pt")
                print(f"  ★ Best val_loss={val_loss:.4f}, saved → best.pt")

        # ---- save checkpoint every save_interval epochs ----
        if (epoch + 1) % tc["save_interval"] == 0 or epoch == tc["epochs"] - 1:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }, out_dir / f"stage1_epoch{epoch+1:03d}.pt")
            print(f"  Saved → {out_dir / f'stage1_epoch{epoch+1:03d}.pt'}")

    print("Training complete.")


# ========================= Validation =============================

@torch.no_grad()
def validate(model, loader, criterion, device, patch_grid,
             use_amp, amp_dtype_torch):
    model.eval()
    all_metrics = {}
    for batch in loader:
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype_torch):
            z, z_hat, depth_pred = model(rgb, states, actions)
            B, W = depth.shape[:2]
            depth_flat = depth.reshape(B * W, *depth.shape[2:])
            depth_gt = DepthHead.downsample_depth(depth_flat, patch_grid)
            depth_gt = depth_gt.reshape(B, W, 1, patch_grid, patch_grid)
            _, metrics = criterion(z, z_hat, depth_pred, depth_gt)

        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

    return {k: sum(v) / len(v) for k, v in all_metrics.items()}


if __name__ == "__main__":
    main()
