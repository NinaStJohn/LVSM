import torch
import torch.nn as nn
import torch.nn.functional as F


def nonlinearity(x):
    return F.silu(x)


def Normalize(in_channels, num_groups=32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        B, C, H, W = q.shape
        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        w = torch.bmm(q, k) * (C ** -0.5)
        w = F.softmax(w, dim=2)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        h = torch.bmm(w, v).permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj_out(h)


class LinearAttnBlock(nn.Module):
    """Memory-efficient linear attention — matches attn_type='linear' in ddconfig."""
    def __init__(self, in_channels):
        super().__init__()
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        B, C, H, W = q.shape
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)
        kv = torch.bmm(k, v)  # (B, C, C)
        q = q.reshape(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
        h_out = torch.bmm(q, kv)  # (B, HW, C)
        h_out = h_out.permute(0, 2, 1).reshape(B, C, H, W)
        return x + self.proj_out(h_out)


def make_attn(in_channels, attn_type="vanilla"):
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "linear":
        return LinearAttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown attn_type: {attn_type}")


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        ch,
        ch_mult,
        num_res_blocks,
        z_channels,
        double_z=True,
        dropout=0.0,
        attn_resolutions=None,
        attn_type="linear",
        resolution=256,
        out_ch=None,  # unused, accepted for yaml compat
        **kwargs,
    ):
        super().__init__()
        attn_resolutions = attn_resolutions or []
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution

        # Downsampling
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        curr_res = resolution
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res //= 2
            self.down.append(down)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        # Output
        self.norm_out = Normalize(block_in)
        out_channels = 2 * z_channels if double_z else z_channels
        self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        for i_level, down in enumerate(self.down):
            for i_block in range(self.num_res_blocks):
                h = down.block[i_block](h)
                if len(down.attn) > 0:
                    h = down.attn[i_block](h)
            if hasattr(down, "downsample"):
                h = down.downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch,
        out_ch,
        ch_mult,
        num_res_blocks,
        z_channels,
        dropout=0.0,
        attn_resolutions=None,
        attn_type="linear",
        resolution=256,
        in_channels=None,   # unused, accepted for yaml compat
        double_z=None,      # unused, accepted for yaml compat
        **kwargs,
    ):
        super().__init__()
        attn_resolutions = attn_resolutions or []
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[-1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            up = self.up[i_level]
            for i_block in range(self.num_res_blocks + 1):
                h = up.block[i_block](h)
                if len(up.attn) > 0:
                    h = up.attn[i_block](h)
            if hasattr(up, "upsample"):
                h = up.upsample(h)
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h