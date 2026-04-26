"""
Single-Path Baseline Generator (CBA-GAN style).
U-Net encoder-decoder with CBAM in encoder, 4 ResNet blocks, Tanh output.
No dual-path split — single bottleneck path.
"""
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
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return self.block(x) + x


class SinglePathGenerator(nn.Module):
    """
    U-Net with CBAM in encoder + 4 ResBlocks bottleneck.
    Matches CBA-GAN paper architecture (single path).
    """

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()

        # Encoder with CBAM
        self.e1 = nn.Sequential(conv_block(in_ch, base_ch), CBAM(base_ch))
        self.e2 = nn.Sequential(conv_block(base_ch, base_ch * 2, stride=2), CBAM(base_ch * 2))
        self.e3 = nn.Sequential(conv_block(base_ch * 2, base_ch * 4, stride=2), CBAM(base_ch * 4))

        # Bottleneck: 4 ResBlocks (single path)
        self.bottleneck = nn.Sequential(
            *[ResBlock(base_ch * 4) for _ in range(4)]
        )

        # Decoder with skip connections
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2 * 2, base_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, in_ch, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        b = self.bottleneck(e3)
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e2], dim=1))
        return self.out_conv(torch.cat([d2, e1], dim=1))
