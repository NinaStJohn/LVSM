# This is a self reference readme so I can unserstand what is going on

```
Input RGB [b,v,3,H,W]
    → AE encode → latent [b,v,16,H/4,W/4]        # frozen VAE encoder
    → cat(latent, plucker rays) → [b,v,22,H/4,W/4] # get_posed_input
    → patch embed → [b, v*n_patches, 768]           # image_tokenizer
    → cat(learnable latents [3072,768], img_tokens)
    → transformer encoder (12 layers)
    → split → latent_tokens [b,3072,768]            # scene representation

Target pose (rays only) [b,v,6,H/4,W/4]
    → patch embed → [b*v, n_patches, 768]           # target_pose_tokenizer
    → cat(pose_tokens, latent_tokens)
    → transformer decoder (12 layers)
    → split → image_tokens [b*v, n_patches, 768]
    → image_token_decoder → [b*v, n_patches, patch²×16]
    → rearrange → [b,v,16,H/4,W/4]
    → AE decode → RGB [b,v,3,H,W]                  # frozen VAE decoder

    # LVSM Scene Encoder-Decoder — My Architecture Notes
```

This is my working reference for `model/LVSM_scene_encoder_decoder.py`.  
The class is `Images2LatentScene`. It takes multi-view RGB images + camera poses as input and renders novel views.

---

## Big Picture

```
Input RGB images + camera poses
        ↓
   VAE encode (frozen)          # RGB → latent space
        ↓
   Plucker ray concat           # add pose conditioning
        ↓
   Patch embed (tokenize)       # latent patches → transformer tokens
        ↓
   Transformer ENCODER          # compress scene into latent vectors
        ↓
   [scene latent vectors]       # the scene representation
        ↓
   Transformer DECODER          # query from target camera pose
        ↓
   image_token_decoder          # tokens → latent patches
        ↓
   VAE decode (frozen)          # latent → RGB output
        ↓
   Rendered novel view
```

---

## Step 1 — VAE Encode Input Images

**Where:** `forward()`, lines after `input.image_pixel = input.image`

```python
with torch.no_grad():
    b, v, c, h, w = input.image.shape
    input.image = self.first_stage_model.encode(
        input.image.reshape(b*v, c, h, w)
    ).sample().reshape(b, v, 16, h//4, w//4)
```

**What's happening:**
- `input.image` starts as `[b, v, 3, H, W]` (RGB)
- The frozen VAE encoder compresses it to `[b, v, 16, H/4, W/4]`
- `16` = number of latent channels (`z_channels` in the yaml)
- `H/4` = 4x spatial downsampling from the 3-level encoder (`ch_mult: [1,2,4]`)
- `.sample()` = draw from the KL posterior (reparameterization trick — you'll learn this with VAEs)
- `torch.no_grad()` = don't backprop through the frozen AE

**Why latent space and not pixels?**  
The transformer operates on patches. At pixel resolution (256×256), patches are huge. At latent resolution (64×64), it's manageable and the AE has already learned a good compressed representation.

---

## Step 2 — Recompute Rays at Latent Resolution

```python
lh, lw = h//4, w//4
input.ray_o, input.ray_d = self.process_data.compute_rays(
    fxfycxcy=input.fxfycxcy, c2w=input.c2w,
    h=lh, w=lw, device=input.image.device
)
```

**What's happening:**  
The ray maps (one ray per pixel) need to match the spatial size of the latents. Originally rays were at `256×256`, now we recompute them at `64×64` so we can concatenate them with the latents channel-wise.

---

## Step 3 — Plucker Ray Encoding (Pose Conditioning)

**Where:** `get_posed_input()`

```python
o_cross_d = torch.cross(ray_o, ray_d, dim=2)
pose_cond = torch.cat([o_cross_d, ray_d], dim=2)  # [b, v, 6, h, w]
```

Then:
```python
return torch.cat([images, pose_cond], dim=2)  # [b, v, 22, h, w]
```

**What's happening:**  
- Each pixel/latent cell gets 6 extra channels encoding its camera ray: `[ray_o × ray_d, ray_d]`
- This is **Plucker coordinates** — a way to represent a 3D ray with 6 numbers
- Concatenated with the 16 latent channels → 22 channels total (matches `in_channels: 22` in the yaml)
- Target pose uses rays only (no image) → 6 channels (matches `in_channels: 6`)

---

## Step 4 — Patch Embedding (Tokenization)

**Where:** `_create_tokenizer()` and `_init_tokenizers()`

```python
tokenizer = nn.Sequential(
    Rearrange(
        "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
        ph=patch_size, pw=patch_size,
    ),
    nn.Linear(in_channels * (patch_size**2), d_model, bias=False),
)
```

**What's happening:**
- The spatial grid `[H, W]` is split into non-overlapping patches of size `patch_size × patch_size`
- Each patch is flattened: `patch_size² × n_channels` values → one vector
- That vector is linearly projected to `d_model=768` (transformer hidden dim)
- Output: `[b*v, n_patches, 768]`

**patch_size=8 vs patch_size=1:**
- `patch_size=8` at latent res 64×64 → `(64/8)² = 64` patches per view
- `patch_size=1` → `64×64 = 4096` patches per view — sequence length explodes, and each "patch" is a single latent cell (very little spatial context per token)

---

## Step 5 — Transformer Encoder

**Where:** `forward()`, encoder section

```python
latent_vector_tokens = self.n_light_field_latent.expand(b, -1, -1)  # [b, 3072, 768]

encoder_input_tokens = torch.cat((latent_vector_tokens, input_img_tokens), dim=1)
# shape: [b, 3072 + v*n_patches, 768]

intermediate_tokens = self.pass_layers(self.transformer_encoder, encoder_input_tokens, ...)

latent_tokens, input_img_tokens = intermediate_tokens.split(
    [n_latent_vectors, v_input * n_patches], dim=1
)
```

**What's happening:**
- `n_light_field_latent` is a learned parameter — `3072` vectors of dim `768`
- These are **bottleneck tokens** that learn to aggregate scene information from all input views
- The encoder attends over `[latent tokens | image tokens]` jointly for 12 layers
- After encoding, we split and keep only the `3072` latent tokens — they now encode the full scene
- The image tokens are discarded — their info has been compressed into the latents

**Why 3072?**  
`3072 = 48×64`. Roughly enough to represent a 48×64 latent scene representation. Could be tuned.

---

## Step 6 — Transformer Decoder

**Where:** `forward()`, decoder section

```python
repeated_latent_tokens = repeat(latent_tokens, 'b nl d -> (b v_target) nl d', v_target=v_target)

decoder_input_tokens = torch.cat((target_pose_tokens, repeated_latent_tokens), dim=1)
# shape: [b*v_target, n_patches + 3072, 768]

transformer_output_tokens = self.pass_layers(self.transformer_decoder, decoder_input_tokens, ...)

target_image_tokens, _ = transformer_output_tokens.split([n_patches, n_latent_vectors], dim=1)
```

**What's happening:**
- For each target view, we give the decoder: `[target pose tokens | scene latent tokens]`
- The pose tokens (from Plucker rays) tell the decoder *where* we want to look from
- The scene latent tokens carry the scene representation from the encoder
- After 12 decoder layers, the pose tokens have been filled in with scene content
- The latent tokens are discarded — we only keep the `n_patches` output tokens

---

## Step 7 — image_token_decoder

**Where:** `_init_tokenizers()`

```python
self.image_token_decoder = nn.Sequential(
    nn.LayerNorm(self.config.model.transformer.d, bias=False),
    nn.Linear(
        self.config.model.transformer.d,
        (patch_size**2) * z_channels,   # patch_size² × 16
        bias=False,
    ),
    # nn.Sigmoid()  <-- REMOVED, was squashing latents to [0,1] — wrong range for VAE
)
```

**What's happening:**
- Each output token `[768]` is projected to `patch_size² × 16` values
- This reconstructs the patch of latent channels that the VAE decoder expects
- **Critical:** No Sigmoid here. VAE latents are approximately Gaussian `~N(0,1)`, not `[0,1]`

---

## Step 8 — Rearrange Patches → Latent Grid

```python
rendered_images = rearrange(
    rendered_images, "(b v) (h w) (p1 p2 c) -> b v c (h p1) (w p2)",
    v=v_target,
    h=lh // patch_size,
    w=lw // patch_size,
    p1=patch_size, p2=patch_size,
    c=16
)
# output: [b, v, 16, lh, lw]  e.g. [b, v, 16, 64, 64]
```

**What's happening:**  
Inverse of Step 4. Reassemble the flat list of patch vectors back into a spatial latent grid.

---

## Step 9 — VAE Decode → RGB

```python
bv = rendered_images.shape[0] * rendered_images.shape[1]
rendered_images = self.first_stage_model.decode(
    rendered_images.reshape(bv, 16, lh, lw)
).reshape(b, v_target, 3, pixel_height, pixel_width)
```

**What's happening:**
- The frozen VAE decoder maps `[b*v, 16, 64, 64]` → `[b*v, 3, 256, 256]`
- Gradients *do* flow back through this into the transformer (no `torch.no_grad()` here)
- This is intentional — the transformer must learn to produce latents the VAE can decode well

---

## Known Issues / Things Being Debugged

### 1. Sigmoid in `image_token_decoder` (FIXED)
The original code had `nn.Sigmoid()` at the end of `image_token_decoder`.  
This squashed output to `[0,1]` but VAE decoder expects ~`[-3, 3]` (Gaussian latents).  
Result: decoded images were black. **Removed.**

### 2. `scale_factor` (UNDER INVESTIGATION)
Standard LDM-style VAEs apply a learned `scale_factor` (~0.18) to normalize latents before/after the transformer:
```python
z = ae.encode(x).sample() * ae.scale_factor   # into transformer
ae.decode(z / ae.scale_factor)                  # back to pixel
```
Need to check if `AutoencoderKL` here has `scale_factor` and whether it's being applied.

### 3. Sanity check to run before training

```python
with torch.no_grad():
    x = torch.randn(1, 3, 256, 256).cuda()
    z = model.first_stage_model.encode(x).sample()
    print(f"latent: mean={z.mean():.3f} std={z.std():.3f}")
    # expect: mean~0, std~1

    x_rec = model.first_stage_model.decode(z)
    print(f"decoded: min={x_rec.min():.3f} max={x_rec.max():.3f}")
    # expect: reasonable RGB range, not all zeros

    fake_token = torch.randn(1, 1, 768).cuda()
    out = model.image_token_decoder(fake_token)
    print(f"token_decoder out: min={out.min():.3f} max={out.max():.3f}")
    # after Sigmoid removal: should NOT be clamped to [0,1]
```

---

## Shape Reference Card

| Tensor | Shape | Notes |
|--------|-------|-------|
| Input RGB | `[b, v, 3, 256, 256]` | raw images |
| AE latent | `[b, v, 16, 64, 64]` | after encode |
| Posed input | `[b, v, 22, 64, 64]` | 16 latent + 6 plucker |
| Image tokens | `[b*v, n_patches, 768]` | after patch embed |
| Latent vectors | `[b, 3072, 768]` | scene bottleneck |
| Encoder input | `[b, 3072+v*n_patches, 768]` | full encoder sequence |
| Target pose | `[b, v, 6, 64, 64]` | rays only |
| Decoder input | `[b*v, n_patches+3072, 768]` | pose + scene |
| Token decoder out | `[b*v, n_patches, patch²×16]` | pre-rearrange |
| Rendered latent | `[b, v, 16, 64, 64]` | post-rearrange |
| Rendered RGB | `[b, v, 3, 256, 256]` | after AE decode |

*(with patch_size=8, n_patches = (64/8)² = 64)*