"""
Train all model variants on a new dataset and compare results.

Models:
  1. Single-Path Baseline (CBA-GAN style, no dual-path)
  2. Dual-Path + CBAM (your paper's original architecture)
  3. Dual-Path + AdaIN + Self-Attention + Multi-Scale D (V4)

Usage:
  python train_and_compare_all.py --epochs 50 --batch_size 16
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from model import DualPathGenerator, SinglePathGenerator, MultiDiscriminator, total_variation_loss
from model.cbam import CBAM
from dataset import CartoonDataset
from losses import (VGGFeatures, ContentLoss, StyleLoss, EdgeLoss,
                    pixel_loss, adversarial_loss_g, adversarial_loss_d)


# ── Legacy Dual-Path (V1 architecture, no AdaIN) ──
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

class LegacyDualPathGenerator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_res=4):
        super().__init__()
        self.e1 = _conv_block(in_ch, base_ch)
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


def train_model(G, D, loader, device, args, ckpt_dir, model_name):
    """Generic training loop for any generator."""
    vgg        = VGGFeatures().to(device)
    content_fn = ContentLoss(vgg)
    style_fn   = StyleLoss(vgg)
    edge_fn    = EdgeLoss().to(device)
    scaler     = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    opt_G   = Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D   = Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    sched_G = StepLR(opt_G, step_size=25, gamma=0.5)
    sched_D = StepLR(opt_D, step_size=25, gamma=0.5)

    os.makedirs(ckpt_dir, exist_ok=True)

    # Pretrain
    print(f"  Pretraining {model_name}...")
    G.train()
    step = 0
    while step < args.pretrain_steps:
        for photos, cartoons in loader:
            if step >= args.pretrain_steps:
                break
            photos, cartoons = photos.to(device), cartoons.to(device)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                fake = G(photos)
                loss = 100 * pixel_loss(fake, cartoons) + 10 * content_fn(fake, cartoons)
            opt_G.zero_grad()
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt_G); scaler.update()
            else:
                loss.backward(); opt_G.step()
            step += 1
    print(f"  Pretraining done.")

    # GAN training
    print(f"  Training {model_name} for {args.epochs} epochs...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        for i, (photos, cartoons) in enumerate(loader):
            photos, cartoons = photos.to(device), cartoons.to(device)

            # D step
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    fake = G(photos)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                loss_D = adversarial_loss_d(D(cartoons), D(fake.detach()))
            opt_D.zero_grad()
            if scaler:
                scaler.scale(loss_D).backward(); scaler.step(opt_D); scaler.update()
            else:
                loss_D.backward(); opt_D.step()

            # G step
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                fake = G(photos)
                loss_G = (adversarial_loss_g(D(fake))
                          + 100 * pixel_loss(fake, cartoons)
                          + 10  * content_fn(fake, cartoons)
                          + 100 * style_fn(fake, cartoons)
                          + 10  * edge_fn(fake, cartoons)
                          + 1   * total_variation_loss(fake))
            opt_G.zero_grad()
            if scaler:
                scaler.scale(loss_G).backward(); scaler.step(opt_G); scaler.update()
            else:
                loss_G.backward(); opt_G.step()

        sched_G.step(); sched_D.step()
        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"    Epoch [{epoch}/{args.epochs}] D:{loss_D.item():.4f} G:{loss_G.item():.4f}")
            torch.save({"G": G.state_dict(), "D": D.state_dict()},
                       os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}.pt"))

    elapsed = time.time() - t0
    print(f"  {model_name} done in {elapsed/60:.1f} min")
    return G


def evaluate_model(G, loader, device, label):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    fid   = FrechetInceptionDistance(normalize=True).to(device)
    ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    ss, ls = [], []
    G.eval()
    with torch.no_grad():
        for photos, cartoons in loader:
            photos, cartoons = photos.to(device), cartoons.to(device)
            fake = G(photos)
            f01 = (fake * 0.5 + 0.5).clamp(0, 1)
            r01 = (cartoons * 0.5 + 0.5).clamp(0, 1)
            fid.update((r01 * 255).byte(), real=True)
            fid.update((f01 * 255).byte(), real=False)
            ss.append(ssim(f01, r01).item())
            ls.append(lpips(f01, r01).item())
    fid_val   = fid.compute().item()
    ssim_val  = sum(ss) / len(ss)
    lpips_val = sum(ls) / len(ls)
    print(f"  {label}: FID={fid_val:.2f}  SSIM={ssim_val:.4f}  LPIPS={lpips_val:.4f}")
    return {"label": label, "FID": fid_val, "SSIM": ssim_val, "LPIPS": lpips_val}


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = CartoonDataset(os.path.join(args.data_dir, "train"), size=256)
    test_ds  = CartoonDataset(os.path.join(args.data_dir, "test"),  size=256)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, num_workers=0)

    results = []

    # ── Model 1: Single-Path Baseline ──
    print("\n" + "="*60)
    print("MODEL 1: Single-Path Baseline (CBA-GAN style)")
    print("="*60)
    G1 = SinglePathGenerator().to(device)
    D1 = MultiDiscriminator().to(device)
    G1 = train_model(G1, D1, train_loader, device, args, "ckpt_single", "Single-Path")
    results.append(evaluate_model(G1, test_loader, device, "Single-Path"))
    del D1; torch.cuda.empty_cache()

    # ── Model 2: Dual-Path + CBAM (original paper) ──
    print("\n" + "="*60)
    print("MODEL 2: Dual-Path + CBAM (Paper Architecture)")
    print("="*60)
    G2 = LegacyDualPathGenerator().to(device)
    D2 = MultiDiscriminator().to(device)
    G2 = train_model(G2, D2, train_loader, device, args, "ckpt_dualpath", "Dual-Path+CBAM")
    results.append(evaluate_model(G2, test_loader, device, "Dual-Path+CBAM"))
    del D2; torch.cuda.empty_cache()

    # ── Model 3: Dual-Path + AdaIN + Self-Attention (V4) ──
    print("\n" + "="*60)
    print("MODEL 3: Dual-Path + AdaIN + Self-Attention (V4)")
    print("="*60)
    G3 = DualPathGenerator().to(device)
    D3 = MultiDiscriminator().to(device)
    G3 = train_model(G3, D3, train_loader, device, args, "ckpt_adain", "DualPath+AdaIN")
    results.append(evaluate_model(G3, test_loader, device, "DualPath+AdaIN"))
    del D3; torch.cuda.empty_cache()

    # ── Print comparison table ──
    print("\n" + "="*70)
    print(f"  {'Model':<25} {'FID ↓':>10} {'SSIM ↑':>10} {'LPIPS ↓':>10}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['label']:<25} {r['FID']:>10.2f} {r['SSIM']:>10.4f} {r['LPIPS']:>10.4f}")
    print("="*70)

    # Save results
    with open("results_comparison.txt", "w") as f:
        f.write(f"{'Model':<25} {'FID':>10} {'SSIM':>10} {'LPIPS':>10}\n")
        for r in results:
            f.write(f"{r['label']:<25} {r['FID']:>10.2f} {r['SSIM']:>10.4f} {r['LPIPS']:>10.4f}\n")
    print("\nResults saved to results_comparison.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str, required=True)
    parser.add_argument("--epochs",         type=int, default=50)
    parser.add_argument("--batch_size",     type=int, default=16)
    parser.add_argument("--pretrain_steps", type=int, default=2000)
    args = parser.parse_args()
    main(args)
