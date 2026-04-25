import torch
import torch.nn as nn
from .cbam import CBAM


def conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )


class ResBlock(nn.Module):
    def __init__(self, channels, use_cbam=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )
        self.cbam = CBAM(channels) if use_cbam else nn.Identity()

    def forward(self, x):
        return self.cbam(self.block(x)) + x


class DualPathGenerator(nn.Module):
    """
    Deeper encoder-decoder with dual-path bottleneck and CBAM.
    base_ch=64 -> 128 -> 256 -> 512 for stronger feature capacity.
    """

    def __init__(self, in_ch=3, base_ch=64, n_res=4):
        super().__init__()

        # Encoder: 3 downsampling levels (memory-efficient)
        self.e1 = conv_block(in_ch,       base_ch,     stride=1)   # 256
        self.e2 = conv_block(base_ch,     base_ch * 2, stride=2)   # 128
        self.e3 = conv_block(base_ch * 2, base_ch * 4, stride=2)   # 64

        # Shared bottleneck block
        self.shared_res = nn.Sequential(
            ResBlock(base_ch * 4, use_cbam=True),
        )

        # Structure path — preserves edges/boundaries
        self.struct_path = nn.Sequential(
            *[ResBlock(base_ch * 4, use_cbam=False) for _ in range(n_res)]
        )

        # Style path — captures texture/artistic style with CBAM
        self.style_path = nn.Sequential(
            *[ResBlock(base_ch * 4, use_cbam=True) for _ in range(n_res)]
        )

        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        # Decoder with skip connections
        self.d1 = self._up_block(base_ch * 4,     base_ch * 2)  # 64->128
        self.d2 = self._up_block(base_ch * 2 * 2, base_ch)      # 128->256 (skip e2)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 7, padding=3),
            nn.Tanh(),
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)

        shared = self.shared_res(e3)

        struct_feat = self.struct_path(shared)
        style_feat  = self.style_path(shared)

        w = torch.sigmoid(self.fusion_weight)
        fused = w * struct_feat + (1 - w) * style_feat

        d1 = self.d1(fused)
        d2 = self.d2(torch.cat([d1, e2], dim=1))
        return self.out_conv(torch.cat([d2, e1], dim=1))
