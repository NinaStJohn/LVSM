#!/usr/bin/env python3
# scripts/train_ae.py
# Autoencoder training script for LVSM project.
# Mirrors the style of scripts/train.py but handles the two-optimizer
# (generator + discriminator) setup that LPIPSWithDiscriminator requires.

import os
import sys
import importlib
import argparse

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
    images = (images + 1.0) / 2.0   # -> [0, 1]
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    grid = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    from PIL import Image
    Image.fromarray(grid).save(path)


def get_amp_context(use_amp, amp_dtype, device):
    if use_amp and device.type == "cuda":
        dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.amp.autocast(device_type="cuda", enabled=False)


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
        **cfg.params,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if world_size > 1 else model

    # ── optimizers ─────────────────────────────────────────────────────────
    # AE uses two optimizers: one for encoder/decoder, one for discriminator
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

    scaler_ae   = torch.cuda.amp.GradScaler(enabled=t.use_amp and "16" in t.amp_dtype)
    scaler_disc = torch.cuda.amp.GradScaler(enabled=t.use_amp and "16" in t.amp_dtype)

    # ── dataset ────────────────────────────────────────────────────────────
    # Import dataset class (defined relative to project root)
    from data.ae_dataset import AEDataset

    train_dataset = AEDataset(config)
    val_config = OmegaConf.merge(
        config,
        OmegaConf.create({
            "training": {
                "dataset_path": t.val_dataset_path,
                "random_flip": False,
            }
        }),
    )
    val_dataset = AEDataset(val_config)

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

    # ── wandb ──────────────────────────────────────────────────────────────
    if main:
        api_keys = OmegaConf.load(t.api_key_path)
        wandb.login(key=api_keys.wandb_api_key)
        wandb.init(project=t.wandb_project, name=t.wandb_exp_name, config=OmegaConf.to_container(config))
        os.makedirs(t.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(t.checkpoint_dir, "images"), exist_ok=True)

    # ── training ───────────────────────────────────────────────────────────
    amp_ctx  = get_amp_context(t.use_amp, t.amp_dtype, device)
    step     = 0
    loader_iter = iter(train_loader)

    pbar = tqdm(total=t.train_steps, disable=not main, desc="AE training")

    while step < t.train_steps:
        model.train()

        try:
            batch = next(loader_iter)
        except StopIteration:
            if train_sampler is not None:
                train_sampler.set_epoch(step)
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        images = batch["image"].to(device)   # [B, 3, H, W],  in [-1, 1]

        # ── optimizer 0: AE (encoder + decoder) ──
        opt_ae.zero_grad()
        with amp_ctx:
            ae_loss, ae_log, reconstructions = raw_model.training_step(images, optimizer_idx=0, global_step=step)
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
            disc_loss, disc_log, _ = raw_model.training_step(images, optimizer_idx=1, global_step=step)
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

        # ── logging ────────────────────────────────────────────────────────
        if main and step % t.print_every == 0:
            log = {**ae_log, **disc_log, "step": step}
            print(f"[step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in log.items() if isinstance(v, float)))

        if main and step % getattr(t, "wandb_log_every", t.print_every) == 0:
            wandb.log({**ae_log, **disc_log, "step": step})

        # ── visualise ──────────────────────────────────────────────────────
        if main and step % t.vis_every == 0:
            model.eval()
            with torch.no_grad(), amp_ctx:
                vis_recon, _ = raw_model(images[:4], sample_posterior=False)
            out_path = os.path.join(t.checkpoint_dir, "images", f"step_{step:07d}.png")
            save_image_grid(torch.cat([images[:4], vis_recon], dim=0), out_path, nrow=4)
            wandb.log({"reconstructions": wandb.Image(out_path), "step": step})
            model.train()

        # ── validation ─────────────────────────────────────────────────────
        if main and step % t.checkpoint_every == 0:
            model.eval()
            val_logs = []
            with torch.no_grad():
                for vbatch in val_loader:
                    vimages = vbatch["image"].to(device)
                    with amp_ctx:
                        val_loss, vlog, _ = raw_model.validation_step(vimages, global_step=step)
                    val_logs.append({k: v.item() if isinstance(v, torch.Tensor) else v for k, v in vlog.items()})
                    if len(val_logs) >= 50:   # cap to 50 batches
                        break
            avg_val = {k: sum(d[k] for d in val_logs) / len(val_logs) for k in val_logs[0]}
            print(f"[val step {step}] " + "  ".join(f"{k}={v:.4f}" for k, v in avg_val.items()))
            wandb.log({f"val/{k}": v for k, v in avg_val.items()} | {"step": step})
            model.train()

        # ── checkpoint ─────────────────────────────────────────────────────
        if main and step % t.checkpoint_every == 0:
            ckpt_path = os.path.join(t.checkpoint_dir, f"step_{step:07d}.pt")
            torch.save({
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "opt_ae_state_dict": opt_ae.state_dict(),
                "opt_disc_state_dict": opt_disc.state_dict(),
                "config": OmegaConf.to_container(config),
            }, ckpt_path)
            # also save latest
            torch.save({
                "step": step,
                "model_state_dict": raw_model.state_dict(),
                "opt_ae_state_dict": opt_ae.state_dict(),
                "opt_disc_state_dict": opt_disc.state_dict(),
                "config": OmegaConf.to_container(config),
            }, os.path.join(t.checkpoint_dir, "latest.pt"))
            print(f"[step {step}] Checkpoint saved -> {ckpt_path}")

        if world_size > 1:
            dist.barrier()

    pbar.close()
    if main:
        wandb.finish()
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    config = init_config()
    train(config)