import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


def sn_block(in_ch, out_ch, stride=2, norm=True):
    """Conv block with spectral normalization (stabilizes GAN training)."""
    layers = [spectral_norm(nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=True))]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization."""

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.model = nn.Sequential(
            sn_block(in_ch, base_ch,     norm=False),
            sn_block(base_ch,   base_ch * 2),
            sn_block(base_ch*2, base_ch * 4),
            sn_block(base_ch*4, base_ch * 8, stride=1),
            spectral_norm(nn.Conv2d(base_ch * 8, 1, 4, padding=1)),
        )

    def forward(self, x):
        return self.model(x)


class SurfaceDiscriminator(PatchDiscriminator):
    """
    Ds from White-Box: distinguishes surface representations
    (guided-filtered smooth regions) of real vs generated cartoons.
    """
    pass


class TextureDiscriminator(PatchDiscriminator):
    """
    Dt from White-Box: distinguishes texture representations
    (random color shift maps) of real vs generated cartoons.
    Input is single-channel texture map.
    """
    def __init__(self, base_ch=64):
        super().__init__(in_ch=1, base_ch=base_ch)


class EdgeDiscriminator(PatchDiscriminator):
    """
    Edge sharpness discriminator from original paper.
    Appends Sobel edge map to input (4 channels total).
    """
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


class MultiDiscriminator(nn.Module):
    """
    Combined discriminator with 3 specialized heads:
      D_surface  - smooth surface regions (White-Box Ds)
      D_texture  - high-freq texture maps (White-Box Dt)
      D_edge     - sharp cartoon edges (original paper)
    """
    def __init__(self):
        super().__init__()
        self.D_surface = SurfaceDiscriminator(in_ch=3)
        self.D_texture = TextureDiscriminator()
        self.D_edge    = EdgeDiscriminator()

    def forward(self, x, surface=None, texture=None):
        surf  = surface if surface is not None else x
        # Use grayscale as texture if not provided
        gray  = 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]
        tex   = texture if texture is not None else gray
        out_s = self.D_surface(surf)
        out_t = self.D_texture(tex)
        out_e = self.D_edge(x)
        return out_s, out_t, out_e
