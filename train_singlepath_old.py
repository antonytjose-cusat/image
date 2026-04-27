#!/usr/bin/env python3
"""
Train Single-Path Baseline on the OLD dataset (dqymaggie/cartoonizer-dataset).
50 epochs, same losses as V1 for fair comparison.
"""
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from model import SinglePathGenerator, MultiDiscriminator, total_variation_loss
from dataset import CartoonDataset
from losses import (VGGFeatures, ContentLoss, StyleLoss, EdgeLoss,
                    pixel_loss, adversarial_loss_g, adversarial_loss_d)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data"
EPOCHS = 50
BATCH_SIZE = 8
PRETRAIN_STEPS = 2000
LR = 2e-4

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    train_ds = CartoonDataset(f"{DATA_DIR}/train", size=256)
    test_ds  = CartoonDataset(f"{DATA_DIR}/test",  size=256)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, num_workers=0)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    G = SinglePathGenerator().to(DEVICE)
    D = MultiDiscriminator().to(DEVICE)

    vgg = VGGFeatures().to(DEVICE)
    content_fn = ContentLoss(vgg)
    style_fn   = StyleLoss(vgg)
    edge_fn    = EdgeLoss().to(DEVICE)
    scaler     = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    opt_G = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    sch_G = StepLR(opt_G, step_size=25, gamma=0.5)
    sch_D = StepLR(opt_D, step_size=25, gamma=0.5)

    ckpt_dir = "ckpt_singlepath_old"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Pretrain
    print(f"\nPretraining ({PRETRAIN_STEPS} steps)...")
    G.train(); step = 0
    while step < PRETRAIN_STEPS:
        for photos, cartoons in train_loader:
            if step >= PRETRAIN_STEPS: break
            photos, cartoons = photos.to(DEVICE), cartoons.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                fake = G(photos)
                loss = 100*pixel_loss(fake,cartoons) + 10*content_fn(fake,cartoons)
            opt_G.zero_grad()
            if scaler: scaler.scale(loss).backward(); scaler.step(opt_G); scaler.update()
            else: loss.backward(); opt_G.step()
            step += 1
            if step % 500 == 0: print(f"  step {step}/{PRETRAIN_STEPS} loss={loss.item():.4f}")

    # GAN Training
    print(f"\nGAN training ({EPOCHS} epochs)...")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        for photos, cartoons in train_loader:
            photos, cartoons = photos.to(DEVICE), cartoons.to(DEVICE)

            # D
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    fake = G(photos)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                ld = adversarial_loss_d(D(cartoons), D(fake.detach()))
            opt_D.zero_grad()
            if scaler: scaler.scale(ld).backward(); scaler.step(opt_D); scaler.update()
            else: ld.backward(); opt_D.step()

            # G
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                fake = G(photos)
                lg = (adversarial_loss_g(D(fake))
                      + 100*pixel_loss(fake,cartoons)
                      + 10*content_fn(fake,cartoons)
                      + 50*style_fn(fake,cartoons))
            opt_G.zero_grad()
            if scaler: scaler.scale(lg).backward(); scaler.step(opt_G); scaler.update()
            else: lg.backward(); opt_G.step()

        sch_G.step(); sch_D.step()
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  Epoch [{epoch}/{EPOCHS}] D={ld.item():.4f} G={lg.item():.4f}")
            torch.save({"G": G.state_dict(), "D": D.state_dict()},
                       os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}.pt"))

    print(f"\nTraining done in {(time.time()-t0)/60:.1f} min")

    # Evaluate
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    fid   = FrechetInceptionDistance(normalize=True).to(DEVICE)
    ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)
    ss, ls = [], []
    G.eval()
    with torch.no_grad():
        for photos, cartoons in test_loader:
            photos, cartoons = photos.to(DEVICE), cartoons.to(DEVICE)
            fake = G(photos)
            f01 = (fake*0.5+0.5).clamp(0,1)
            r01 = (cartoons*0.5+0.5).clamp(0,1)
            fid.update((r01*255).byte(), real=True)
            fid.update((f01*255).byte(), real=False)
            ss.append(ssim(f01,r01).item())
            ls.append(lpips(f01,r01).item())

    print(f"\n{'='*50}")
    print(f"Single-Path Baseline on OLD dataset:")
    print(f"  FID:   {fid.compute().item():.2f}")
    print(f"  SSIM:  {sum(ss)/len(ss):.4f}")
    print(f"  LPIPS: {sum(ls)/len(ls):.4f}")
    print(f"{'='*50}")
