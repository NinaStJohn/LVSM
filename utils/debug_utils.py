# Copyright (c) 2025 Nina St John. Debug utilities for LVSM project.

import os
import torch
from einops import rearrange
from PIL import Image
import numpy as np

try:
    from utils.dist_utils import print_rank0, is_rank0
except ImportError:
    # Fallback if dist_utils isn't available / different name in this repo
    def is_rank0():
        return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

    def print_rank0(*args, **kwargs):
        if is_rank0():
            print(*args, **kwargs)


def _to_uint8_image(tensor_01):
    """
    tensor_01: [c, h, w] or [h, w, c], values in [0,1], torch or numpy.
    Returns HWC uint8 numpy array ready for PIL.
    """
    if isinstance(tensor_01, torch.Tensor):
        arr = tensor_01.detach().float().cpu().numpy()
    else:
        arr = tensor_01

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)

    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    return arr


def log_tensor_stats(tensor, name, log_path=None, step=None):
    """
    Lightweight, text-only stats logger. No image I/O -- safe for high-frequency
    call sites (e.g. every step) where saving PNGs would be too slow.

    Args:
        tensor: any shape torch.Tensor
        name: str, identifies the call site (e.g. "post_vae_encode_input")
        log_path: optional path to append a text log line to, in addition to stdout
        step: optional training step, prepended to the log line
    """
    if not is_rank0():
        return

    t = tensor.detach()
    step_str = f"step={step} " if step is not None else ""
    line = (
        f"{step_str}[{name}] shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={t.min().item():.4f} max={t.max().item():.4f} "
        f"mean={t.mean().item():.4f} std={t.std().item():.4f}"
    )

    print_rank0(line)

    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")


def dump_tensor_state(
    tensor,
    name,
    out_dir,
    step=None,
    is_latent=False,
    vae=None,
    max_items=4,
    already_01=False,
):
    """
    Full debug dump: logs stats (via log_tensor_stats) AND saves a PNG visualization.
    Rank-0 only.

    Args:
        tensor: expected shape [b, v, c, h, w] or [n, c, h, w] (batch-of-views flattened is fine too).
            If is_latent=True, c should be the VAE latent channel count (e.g. 16).
            If is_latent=False, assumed already in [-1, 1] pixel range (LVSM convention)
            unless already_01=True is passed via a pre-rescaled tensor.
        name: str tag for filenames/log lines, e.g. "02_target_latent_encoded"
        out_dir: directory to write PNGs and the debug log into
        step: optional training step, used in filename and log line
        is_latent: if True, tensor is decoded through `vae` before visualization
        vae: required if is_latent=True; the frozen first_stage_model (AutoencoderKL)
        max_items: cap on how many images (flattened over batch*view) to save per call,
            to avoid flooding disk on large batches
    """
    if not is_rank0():
        return

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "debug_log.txt")

    log_tensor_stats(tensor, name, log_path=log_path, step=step)

    t = tensor.detach()

    # Flatten any leading batch/view dims down to [n, c, h, w]
    if t.dim() == 5:
        b, v, c, h, w = t.shape
        t = t.reshape(b * v, c, h, w)
    elif t.dim() != 4:
        print_rank0(f"[{name}] dump_tensor_state: unsupported ndim={t.dim()}, skipping image save")
        return

    n = min(t.shape[0], max_items)
    t = t[:n]

    if is_latent:
        if vae is None:
            print_rank0(f"[{name}] dump_tensor_state: is_latent=True but no vae provided, skipping image save")
            return
        with torch.no_grad():
            decoded = vae.decode(t)
        # log post-decode stats too, since this is where scale mismatches usually show up
        log_tensor_stats(decoded, f"{name}_decoded", log_path=log_path, step=step)
        pixel_01 = decoded * 0.5 + 0.5
    elif already_01:
        pixel_01 = t # No rescale, already [0,1]
    else:
        # assume already pixel-space in [-1, 1] (LVSM convention)
        pixel_01 = t * 0.5 + 0.5

    pixel_01 = pixel_01.clamp(0.0, 1.0)

    # tile the n items horizontally into one strip image
    strip = rearrange(pixel_01, "n c h w -> h (n w) c")
    img = _to_uint8_image(strip)

    step_str = f"step{step}_" if step is not None else ""
    filename = f"{step_str}{name}.png"
    Image.fromarray(img).save(os.path.join(out_dir, filename))


@torch.no_grad()
def vae_roundtrip_test(vae, image_batch, out_dir, step=None, max_items=4):
    """
    Standalone isolation test: encode -> decode through the frozen VAE only,
    no transformer involved. Use this to determine whether cast/warp artifacts
    originate in the VAE itself vs. downstream in the LVSM transformer pipeline.

    Args:
        vae: frozen first_stage_model (AutoencoderKL)
        image_batch: [b, v, c, h, w] or [n, c, h, w], values in [-1, 1]
            (pass input.image_pixel or target.image, pre-VAE-encode)
        out_dir: directory to write results into
        step: optional training step, for filename/log tagging
        max_items: cap on images saved per call
    """
    if not is_rank0():
        return

    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "debug_log.txt")

    t = image_batch.detach()
    if t.dim() == 5:
        b, v, c, h, w = t.shape
        t = t.reshape(b * v, c, h, w)

    n = min(t.shape[0], max_items)
    t = t[:n]

    log_tensor_stats(t, "vae_roundtrip_input_raw", log_path=log_path, step=step)

    z = vae.encode(t).sample()
    log_tensor_stats(z, "vae_roundtrip_latent", log_path=log_path, step=step)

    recon = vae.decode(z)
    log_tensor_stats(recon, "vae_roundtrip_decoded_raw", log_path=log_path, step=step)

    input_01 = (t * 0.5 + 0.5).clamp(0.0, 1.0)
    recon_01 = (recon * 0.5 + 0.5).clamp(0.0, 1.0)

    # side-by-side: input strip on top, reconstruction strip on bottom
    input_strip = rearrange(input_01, "n c h w -> h (n w) c")
    recon_strip = rearrange(recon_01, "n c h w -> h (n w) c")
    comparison = torch.cat(
        (torch.from_numpy(np.array(_to_uint8_image(input_strip))),
         torch.from_numpy(np.array(_to_uint8_image(recon_strip)))),
        dim=0,
    ).numpy()

    step_str = f"step{step}_" if step is not None else ""
    Image.fromarray(comparison).save(
        os.path.join(out_dir, f"{step_str}vae_roundtrip_input_vs_recon.png")
    )

    # quick numeric divergence check
    mse = torch.nn.functional.mse_loss(recon_01, input_01).item()
    print_rank0(f"[vae_roundtrip_test] step={step} MSE(input_01, recon_01)={mse:.6f}")
    with open(log_path, "a") as f:
        f.write(f"step={step} [vae_roundtrip_test] MSE(input_01, recon_01)={mse:.6f}\n")