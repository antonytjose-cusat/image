"""
Compute FID, SSIM, and LPIPS on the test set.
Auto-detects old vs new architecture.

Usage:
  python evaluate.py --checkpoint checkpoints/ckpt_epoch100.pt \
                     --data_dir data/test
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from model import DualPathGenerator
from model.cbam import CBAM
from dataset import CartoonDataset


# ── Legacy architecture for old checkpoints ──
def _conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, inplace=True))

class LegacyResBlock(nn.Module):
    def __init__(self, channels, use_cbam=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels))
        self.cbam = CBAM(channels) if use_cbam else nn.Identity()
    def forward(self, x):
        return self.cbam(self.block(x)) + x

class LegacyGenerator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_res=4):
        super().__init__()
        self.e1 = _conv_block(in_ch, base_ch, stride=1)
        self.e2 = _conv_block(base_ch, base_ch*2, stride=2)
        self.e3 = _conv_block(base_ch*2, base_ch*4, stride=2)
        self.shared_res  = nn.Sequential(LegacyResBlock(base_ch*4, use_cbam=True))
        self.struct_path = nn.Sequential(*[LegacyResBlock(base_ch*4) for _ in range(n_res)])
        self.style_path  = nn.Sequential(*[LegacyResBlock(base_ch*4, use_cbam=True) for _ in range(n_res)])
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.d1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch*2), nn.ReLU(inplace=True))
        self.d2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*2*2, base_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 7, padding=3), nn.Tanh())
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(e1); e3 = self.e3(e2)
        shared = self.shared_res(e3)
        w = torch.sigmoid(self.fusion_weight)
        fused = w * self.struct_path(shared) + (1-w) * self.style_path(shared)
        d1 = self.d1(fused)
        d2 = self.d2(torch.cat([d1, e2], dim=1))
        return self.out_conv(torch.cat([d2, e1], dim=1))


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    G = DualPathGenerator().to(device)
    try:
        G.load_state_dict(ckpt["G"])
        print("  Loaded: new architecture (AdaIN + Self-Attention)")
    except RuntimeError:
        G = LegacyGenerator().to(device)
        G.load_state_dict(ckpt["G"])
        print("  Loaded: legacy architecture")
    G.eval()
    return G


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating: {args.checkpoint}")
    G = load_model(args.checkpoint, device)

    dataset = CartoonDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=8, num_workers=0)

    fid   = FrechetInceptionDistance(normalize=True).to(device)
    ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    ssim_scores, lpips_scores = [], []

    with torch.no_grad():
        for photos, cartoons in loader:
            photos, cartoons = photos.to(device), cartoons.to(device)
            fake    = G(photos)
            fake_01 = (fake * 0.5 + 0.5).clamp(0, 1)
            real_01 = (cartoons * 0.5 + 0.5).clamp(0, 1)

            fid.update((real_01 * 255).byte(), real=True)
            fid.update((fake_01 * 255).byte(), real=False)
            ssim_scores.append(ssim(fake_01, real_01).item())
            lpips_scores.append(lpips(fake_01, real_01).item())

    print(f"FID:   {fid.compute().item():.2f}")
    print(f"SSIM:  {sum(ssim_scores)/len(ssim_scores):.4f}")
    print(f"LPIPS: {sum(lpips_scores)/len(lpips_scores):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir",   required=True)
    main(parser.parse_args())
