#!/usr/bin/env python3
# scripts/train_ae.py
# Autoencoder training script for LVSM project.
# Mirrors the style of scripts/train.py but handles the two-optimizer
# (generator + discriminator) setup that LPIPSWithDiscriminator requires.
#
# Checkpoint policy (mirrors LDM ModelCheckpoint save_top_k=3):
#   - latest.pt          : always overwritten
#   - e{epoch}-s{step}.pt: kept for the 3 lowest val/rec_loss checkpoints
#
# Vis schedule (mirrors LDM ImageLogger increase_log_steps):
#   - Steps 1, 2, 4, 8, ... up to vis_every, then every vis_every steps

import os
import sys
import importlib
import argparse
import heapq

# ── make project root importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import OmegaConf
import wandb
from tqdm import tqdm
import torchvision
from PIL import Image
import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args, overrides = parser.parse_known_args()
    config = OmegaConf.load(args.config)
    if overrides:
        cli = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, cli)
    return config


def init_distributed():
    rank       = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def is_main(rank):
    return rank == 0


def instantiate(class_path, **kwargs):
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


def save_image_grid(images, path, nrow=4):
    """Save a [-1,1] tensor batch as a PNG grid."""
    images = torch.clamp(images, -1, 1)
    images = (images + 1.0) / 2.0
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    grid = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(grid).save(path)


def get_amp_context(use_amp, amp_dtype, device):
    if use_amp and device.type == "cuda":
        dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.amp.autocast(device_type="cuda", enabled=False)


def make_exp_log_steps(vis_every):
    """Generate [1, 2, 4, 8, ...] up to vis_every — mirrors LDM increase_log_steps."""
    steps = []
    n = 1
    while n < vis_every:
        steps.append(n)
        n *= 2
    return steps


class TopKCheckpointManager:
    """
    Keeps the K checkpoints with the lowest monitored metric + always writes latest.pt.
    Mirrors LDM ModelCheckpoint(save_top_k=3, monitor='val/rec_loss').
    """
    def __init__(self, ckpt_dir, k=3, monitor="val/rec_loss"):
        self.ckpt_dir   = ckpt_dir
        self.k          = k
        self.monitor    = monitor
        # max-heap of (metric, path) — negate metric for min-heap behaviour
        self._heap      = []   # stores (-metric, path)

    def update(self, metric_val, step, epoch, state):
        ckpt_name = f"e{epoch:06d}-s{step:07d}.pt"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        torch.save(state, ckpt_path)
        print(f"[ckpt] Saved {ckpt_name}  ({self.monitor}={metric_val:.6f})")

        # Push onto heap (negate so smallest metric = highest priority)
        heapq.heappush(self._heap, (-metric_val, ckpt_path))

        # Evict worst if over budget
        if len(self._heap) > self.k:
            _, worst_path = heapq.heappop(self._heap)   # pops the highest neg = worst metric
            # Correct: we stored -metric, heappop gives smallest (most negative = worst metric)
            # Re-sort: actually heappop gives the *smallest* value = most negative = best metric.
            # We want to remove the *worst* (highest rec_loss). Use a proper structure:
            pass  # see __init__ note — handled below via sorted eviction

        # Simpler: just keep sorted list and delete worst
        self._heap = sorted(self._heap, key=lambda x: x[0])  # ascending -metric = best first
        while len(self._heap) > self.k:
            _, worst_path = self._heap.pop()  # pop last = least negative = worst metric
            if os.path.exists(worst_path):
                os.remove(worst_path)
                print(f"[ckpt] Removed {os.path.basename(worst_path)} (outside top-{self.k})")

        # Always write latest
        latest_path = os.path.join(self.ckpt_dir, "latest.pt")
        torch.save(state, latest_path)


# ── training loop ─────────────────────────────────────────────────────────────

def train(config):
    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    main = is_main(rank)

    if config.training.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── model ──────────────────────────────────────────────────────────────
    cfg = config.model
    model = instantiate(
        cfg.class_name,
        **OmegaConf.to_container(cfg.params, resolve=True),
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if world_size > 1 else model

    # ── optimizers ─────────────────────────────────────────────────────────
    t = config.training
    opt_ae = torch.optim.Adam(
        list(raw_model.encoder.parameters())
        + list(raw_model.decoder.parameters())
        + list(raw_model.quant_conv.parameters())
        + list(raw_model.post_quant_conv.parameters()),
        lr=t.lr,
        betas=(t.beta1, t.beta2),
        weight_decay=t.weight_decay,
    )
    opt_disc = torch.optim.Adam(
        raw_model.loss.discriminator.parameters(),
        lr=t.lr,
        betas=(t.beta1, t.beta2),
        weight_decay=t.weight_decay,
    )

    scaler_ae   = torch.amp.GradScaler('cuda', enabled=t.use_amp and "16" in t.amp_dtype)
    scaler_disc = torch.amp.GradScaler('cuda', enabled=t.use_amp and "16" in t.amp_dtype)

    # ── dataset ────────────────────────────────────────────────────────────
    from data.ae_dataset import AEDataset

    train_dataset = AEDataset(config)
    val_config = OmegaConf.merge(
        config,
        OmegaConf.create({"training": {"dataset_path": t.val_dataset_path, "random_flip": False}}),
    )
    val_dataset = AEDataset(val_config)
    dataset_len = len(train_dataset)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    train_loader  = DataLoader(
        train_dataset,
        batch_size=t.batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=t.num_workers,
        prefetch_factor=t.prefetch_factor if t.num_workers > 0 else None,
        persistent_workers=(t.num_workers > 0),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=t.batch_size_per_gpu,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    # ── wandb + dirs ───────────────────────────────────────────────────────
    use_wandb = main and bool(t.get("use_wandb", False))
    if main:
        if use_wandb:
            api_keys = OmegaConf.load(t.api_key_path)
            wandb.login(key=api_keys.wandb)
            wandb.init(
                project=t.wandb_project,
                name=t.wandb_exp_name,
                config=OmegaConf.to_container(config),
            )
        os.makedirs(t.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(t.checkpoint_dir, "images", "train"), exist_ok=True)
        os.makedirs(os.path.join(t.checkpoint_dir, "images", "val"),   exist_ok=True)

    ckpt_manager = TopKCheckpointManager(t.checkpoint_dir, k=3, monitor="val/rec_loss") if main else None

    # ── vis schedule (LDM increase_log_steps) ──────────────────────────────
    exp_log_steps = set(make_exp_log_steps(t.vis_every))

    def should_log_vis(step):
        return step in exp_log_steps or step % t.vis_every == 0

    # ── epoch tracking ─────────────────────────────────────────────────────
    total_samples_per_step = t.batch_size_per_gpu * world_size

    def current_epoch(step):
        return (step * total_samples_per_step) // dataset_len

    # ── amp context ────────────────────────────────────────────────────────
    amp_ctx     = get_amp_context(t.use_amp, t.amp_dtype, device)
    step        = 0
    loader_iter = iter(train_loader)
    pbar        = tqdm(total=t.train_steps, disable=not main, desc="AE training")

    # ── log step 1 immediately (log_first_step) ────────────────────────────
    exp_log_steps.add(1)

    while step < t.train_steps:
        model.train()

        try:
            batch = next(loader_iter)
        except StopIteration:
            if train_sampler is not None:
                train_sampler.set_epoch(step)
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        images = batch["image"].to(device)   # [B, 3, H, W] in [-1, 1]
        epoch  = current_epoch(step)

        # ── optimizer 0: AE ──
        opt_ae.zero_grad()
        with amp_ctx:
            ae_loss, ae_log, reconstructions = raw_model.training_step(
                images, optimizer_idx=0, global_step=step
            )
        scaler_ae.scale(ae_loss).backward()
        if t.grad_clip_norm > 0:
            scaler_ae.unscale_(opt_ae)
            torch.nn.utils.clip_grad_norm_(
                list(raw_model.encoder.parameters())
                + list(raw_model.decoder.parameters())
                + list(raw_model.quant_conv.parameters())
                + list(raw_model.post_quant_conv.parameters()),
                t.grad_clip_norm,
            )
        scaler_ae.step(opt_ae)
        scaler_ae.update()

        # ── optimizer 1: discriminator ──
        opt_disc.zero_grad()
        with amp_ctx:
            disc_loss, disc_log, _ = raw_model.training_step(
                images, optimizer_idx=1, global_step=step
            )
        scaler_disc.scale(disc_loss).backward()
        if t.grad_clip_norm > 0:
            scaler_disc.unscale_(opt_disc)
            torch.nn.utils.clip_grad_norm_(
                raw_model.loss.discriminator.parameters(),
                t.grad_clip_norm,
            )
        scaler_disc.step(opt_disc)
        scaler_disc.update()

        step += 1
        pbar.update(1)

        # ── console + wandb logging ────────────────────────────────────────
        if main and step % t.print_every == 0:
            log = {**ae_log, **disc_log}
            scalar_log = {k: v.item() if isinstance(v, torch.Tensor) else v
                          for k, v in log.items() if isinstance(v, (float, torch.Tensor))}
            epoch_str = f"[epoch {epoch:04d} | step {step:07d}]"
            print(epoch_str + "  " + "  ".join(f"{k}={v:.4f}" for k, v in scalar_log.items()))

        if use_wandb and step % getattr(t, "wandb_log_every", t.print_every) == 0:
            wandb.log({**ae_log, **disc_log, "step": step, "epoch": epoch})

        # ── visualisation (exp schedule + regular) ─────────────────────────
        if main and should_log_vis(step):
            model.eval()
            with torch.no_grad(), amp_ctx:
                vis_recon, _ = raw_model(images[:4], sample_posterior=False)
            # save grid: top row = inputs, bottom row = reconstructions
            grid_img = torch.cat([images[:4].cpu(), vis_recon.cpu()], dim=0)
            fname = f"e{epoch:06d}_gs{step:07d}_train.png"
            out_path = os.path.join(t.checkpoint_dir, "images", "train", fname)
            save_image_grid(grid_img, out_path, nrow=4)
            if use_wandb:
                wandb.log({"train/reconstructions": wandb.Image(out_path), "step": step})
            model.train()

        # ── validation + checkpoint ────────────────────────────────────────
        if main and step % t.checkpoint_every == 0:
            model.eval()
            val_logs = []
            with torch.no_grad():
                for i, vbatch in enumerate(val_loader):
                    vimages = vbatch["image"].to(device)
                    with amp_ctx:
                        val_loss, vlog, val_recon = raw_model.validation_step(
                            vimages, global_step=step
                        )
                    val_logs.append({
                        k: v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in vlog.items()
                    })
                    # save one val vis grid
                    if i == 0:
                        grid_img = torch.cat([vimages[:4].cpu(), val_recon[:4].cpu()], dim=0)
                        fname = f"e{epoch:06d}_gs{step:07d}_val.png"
                        val_vis_path = os.path.join(t.checkpoint_dir, "images", "val", fname)
                        save_image_grid(grid_img, val_vis_path, nrow=4)
                    if len(val_logs) >= 50:
                        break

            avg_val = {
                k: sum(d[k] for d in val_logs) / len(val_logs)
                for k in val_logs[0]
            }
            print(
                f"[val | epoch {epoch:04d} | step {step:07d}] "
                + "  ".join(f"{k}={v:.4f}" for k, v in avg_val.items())
            )
            if use_wandb:
                wandb.log({
                    **{f"val/{k}": v for k, v in avg_val.items()},
                    "val/reconstructions": wandb.Image(val_vis_path),
                    "step": step,
                    "epoch": epoch,
                })

            # top-3 checkpoint by val/rec_loss
            monitor_val = avg_val.get("val/rec_loss", float("inf"))
            state = {
                "step":               step,
                "epoch":              epoch,
                "model_state_dict":   raw_model.state_dict(),
                "opt_ae_state_dict":  opt_ae.state_dict(),
                "opt_disc_state_dict": opt_disc.state_dict(),
                "config":             OmegaConf.to_container(config),
                "val_rec_loss":       monitor_val,
            }
            ckpt_manager.update(monitor_val, step, epoch, state)
            model.train()

        if world_size > 1:
            dist.barrier()

    pbar.close()
    if use_wandb:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    config = init_config()
    train(config)