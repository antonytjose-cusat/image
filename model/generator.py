import torch
import torch.nn as nn
import torch.nn.functional as F
from .cbam import CBAM


# ─────────────────────────────────────────────
# AdaIN: style path modulates structure path
# ─────────────────────────────────────────────

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization.
    Uses style features to modulate structure features.
    AdaIN(x, y) = σ(y) * ((x - μ(x)) / σ(x)) + μ(y)
    """
    def __init__(self, channels):
        super().__init__()
        # Learn scale and shift from style features
        self.style_scale = nn.Linear(channels, channels)
        self.style_shift = nn.Linear(channels, channels)

    def forward(self, struct_feat, style_feat):
        # Compute style statistics from global average pooling
        style_gap = style_feat.mean(dim=[2, 3])          # (B, C)
        scale = self.style_scale(style_gap).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
        shift = self.style_shift(style_gap).unsqueeze(-1).unsqueeze(-1)

        # Normalize structure features
        mean = struct_feat.mean(dim=[2, 3], keepdim=True)
        std  = struct_feat.std(dim=[2, 3], keepdim=True) + 1e-5
        normalized = (struct_feat - mean) / std

        return scale * normalized + shift


# ─────────────────────────────────────────────
# Self-Attention (for bottleneck global context)
# ─────────────────────────────────────────────

class SelfAttention(nn.Module):
    """
    Non-local self-attention block.
    Lets the model relate distant spatial regions — critical for
    consistent cartoon style across the whole image.
    """
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key   = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)  # (B, HW, C//8)
        k = self.key(x).view(B, -1, H * W)                      # (B, C//8, HW)
        attn = F.softmax(torch.bmm(q, k), dim=-1)               # (B, HW, HW)
        v = self.value(x).view(B, -1, H * W)                    # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.GELU(),   # smoother gradients than LeakyReLU
    )


class ResBlock(nn.Module):
    def __init__(self, channels, use_cbam=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )
        self.cbam = CBAM(channels) if use_cbam else nn.Identity()

    def forward(self, x):
        return self.cbam(self.block(x)) + x


# ─────────────────────────────────────────────
# Dual-Path Generator with AdaIN + Self-Attention
# ─────────────────────────────────────────────

class DualPathGenerator(nn.Module):
    """
    Improvements over previous version:
      - AdaIN: style path dynamically modulates structure path features
      - Self-attention in bottleneck for global context
      - GELU activations for smoother gradients
      - CBAM in both shared block and style path
    """

    def __init__(self, in_ch=3, base_ch=64, n_res=4):
        super().__init__()

        # Encoder
        self.e1 = conv_block(in_ch,       base_ch,     stride=1)   # 256
        self.e2 = conv_block(base_ch,     base_ch * 2, stride=2)   # 128
        self.e3 = conv_block(base_ch * 2, base_ch * 4, stride=2)   # 64

        # Shared bottleneck with self-attention for global context
        self.shared_res  = ResBlock(base_ch * 4, use_cbam=True)
        self.self_attn   = SelfAttention(base_ch * 4)

        # Structure path — shape, boundaries (no CBAM, pure structural)
        self.struct_path = nn.Sequential(
            *[ResBlock(base_ch * 4, use_cbam=False) for _ in range(n_res)]
        )

        # Style path — texture, artistic style with CBAM
        self.style_path = nn.Sequential(
            *[ResBlock(base_ch * 4, use_cbam=True) for _ in range(n_res)]
        )

        # AdaIN: style modulates structure
        self.adain = AdaIN(base_ch * 4)

        # Learnable fusion weight (after AdaIN)
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # Decoder with skip connections
        self.d1 = self._up_block(base_ch * 4,     base_ch * 2)
        self.d2 = self._up_block(base_ch * 2 * 2, base_ch)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.GELU(),
            nn.Conv2d(base_ch, in_ch, 7, padding=3),
            nn.Tanh(),
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        # Encode
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        # Shared bottleneck + self-attention
        shared = self.self_attn(self.shared_res(e3))

        # Dual paths
        struct_feat = self.struct_path(shared)
        style_feat  = self.style_path(shared)

        # AdaIN: style dynamically modulates structure
        adain_feat = self.adain(struct_feat, style_feat)

        # Weighted fusion of AdaIN output and style features
        w = torch.sigmoid(self.fusion_weight)
        fused = w * adain_feat + (1 - w) * style_feat

        # Decode with skip connections
        d1 = self.d1(fused)
        d2 = self.d2(torch.cat([d1, e2], dim=1))
        return self.out_conv(torch.cat([d2, e1], dim=1))
