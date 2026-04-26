import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def sn_block(in_ch, out_ch, stride=2, norm=True):
    layers = [spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=True))]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    """PatchGAN with spectral normalization."""

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.model = nn.Sequential(
            sn_block(in_ch,       base_ch,     norm=False),
            sn_block(base_ch,     base_ch * 2),
            sn_block(base_ch * 2, base_ch * 4),
            sn_block(base_ch * 4, base_ch * 8, stride=1),
            spectral_norm(nn.Conv2d(base_ch * 8, 1, 4, padding=1)),
        )

    def forward(self, x):
        return self.model(x)


class EdgeDiscriminator(PatchDiscriminator):
    """Focuses on sharp cartoon edges via Sobel map."""

    def __init__(self, base_ch=64):
        super().__init__(in_ch=4, base_ch=base_ch)

    def forward(self, x):
        gray = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],
                           dtype=x.dtype, device=x.device).view(1,1,3,3)
        ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]],
                           dtype=x.dtype, device=x.device).view(1,1,3,3)
        edges = torch.sqrt(F.conv2d(gray,kx,padding=1)**2 +
                           F.conv2d(gray,ky,padding=1)**2 + 1e-6)
        return self.model(torch.cat([x, edges], dim=1))


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator (from pix2pixHD).
    3 discriminators operating at different scales:
      D1 - full resolution  (256x256) — fine texture details
      D2 - half resolution  (128x128) — mid-level structure
      D3 - quarter res      (64x64)   — global composition

    Plus an edge discriminator for sharp cartoon outlines.
    """

    def __init__(self):
        super().__init__()
        self.D1   = PatchDiscriminator(in_ch=3)   # full res
        self.D2   = PatchDiscriminator(in_ch=3)   # half res
        self.D3   = PatchDiscriminator(in_ch=3)   # quarter res
        self.D_edge = EdgeDiscriminator()

    def forward(self, x):
        x_half    = F.avg_pool2d(x, kernel_size=2)
        x_quarter = F.avg_pool2d(x, kernel_size=4)
        return (
            self.D1(x),
            self.D2(x_half),
            self.D3(x_quarter),
            self.D_edge(x),
        )


# Keep alias for backward compatibility
MultiDiscriminator = MultiScaleDiscriminator
