#!/usr/bin/env python3
"""
ONE SCRIPT TO RULE THEM ALL.
Downloads new dataset, trains all 3 models, evaluates, prints results.
No user intervention needed.

Run:
  python run_everything.py
"""
import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

# ═══════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════
DATA_DIR       = "data_new"
EPOCHS         = 50
BATCH_SIZE     = 16
PRETRAIN_STEPS = 2000
LR             = 2e-4
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════
# STEP 1: DOWNLOAD DATASET
# ═══════════════════════════════════════════════

def download_dataset():
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING DATASET")
    print("="*60)

    if os.path.exists(f"{DATA_DIR}/train/photos") and len(os.listdir(f"{DATA_DIR}/train/photos")) > 100:
        print("Dataset already exists, skipping download.")
        return

    from datasets import load_dataset

    for split in ("train", "test"):
        for sub in ("photos", "cartoons"):
            Path(f"{DATA_DIR}/{split}/{sub}").mkdir(parents=True, exist_ok=True)

    print("Downloading instruction-tuning-sd/cartoonization ...")
    ds = load_dataset("instruction-tuning-sd/cartoonization", split="train")
    print(f"Total: {len(ds)} samples, Columns: {ds.column_names}")

    def to_pil(img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        import numpy as np
        return Image.fromarray(img).convert("RGB")

    def extract_pair(sample):
        for pk, ck in [("input_image","edit_image"),("original_image","cartoonized_image"),
                       ("image","cartoon"),("input","output")]:
            if pk in sample and ck in sample:
                return sample[pk], sample[ck]
        img_keys = [k for k, v in sample.items() if isinstance(v, Image.Image)]
        if len(img_keys) >= 2:
            return sample[img_keys[0]], sample[img_keys[1]]
        raise ValueError(f"Cannot find columns: {list(sample.keys())}")

    split_idx = int(len(ds) * 0.8)

    print(f"Saving train ({split_idx})...")
    for idx in tqdm(range(split_idx)):
        photo, cartoon = extract_pair(ds[idx])
        to_pil(photo).resize((256,256)).save(f"{DATA_DIR}/train/photos/{idx:05d}.jpg")
        to_pil(cartoon).resize((256,256)).save(f"{DATA_DIR}/train/cartoons/{idx:05d}.jpg")

    print(f"Saving test ({len(ds)-split_idx})...")
    for idx in tqdm(range(split_idx, len(ds))):
        photo, cartoon = extract_pair(ds[idx])
        tidx = idx - split_idx
        to_pil(photo).resize((256,256)).save(f"{DATA_DIR}/test/photos/{tidx:05d}.jpg")
        to_pil(cartoon).resize((256,256)).save(f"{DATA_DIR}/test/cartoons/{tidx:05d}.jpg")

    print("Dataset ready.\n")


# ═══════════════════════════════════════════════
# IMPORTS (after dataset download)
# ═══════════════════════════════════════════════

from model import DualPathGenerator, SinglePathGenerator, MultiDiscriminator, total_variation_loss
from model.cbam import CBAM
from dataset import CartoonDataset
from losses import (VGGFeatures, ContentLoss, StyleLoss, EdgeLoss,
                    pixel_loss, adversarial_loss_g, adversarial_loss_d)


# ═══════════════════════════════════════════════
# LEGACY DUAL-PATH (paper's original, no AdaIN)
# ═══════════════════════════════════════════════

def _cb(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, True))

class _RB(nn.Module):
    def __init__(self, ch, cbam=False):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.InstanceNorm2d(ch))
        self.c = CBAM(ch) if cbam else nn.Identity()
    def forward(self, x): return self.c(self.b(x)) + x

class LegacyDualPath(nn.Module):
    def __init__(self, in_ch=3, bc=64, nr=4):
        super().__init__()
        self.e1=_cb(in_ch,bc); self.e2=_cb(bc,bc*2,2); self.e3=_cb(bc*2,bc*4,2)
        self.shared_res=nn.Sequential(_RB(bc*4,True))
        self.struct_path=nn.Sequential(*[_RB(bc*4) for _ in range(nr)])
        self.style_path=nn.Sequential(*[_RB(bc*4,True) for _ in range(nr)])
        self.fusion_weight=nn.Parameter(torch.tensor(0.5))
        self.d1=nn.Sequential(nn.ConvTranspose2d(bc*4,bc*2,4,2,1,bias=False),nn.InstanceNorm2d(bc*2),nn.ReLU(True))
        self.d2=nn.Sequential(nn.ConvTranspose2d(bc*4,bc,4,2,1,bias=False),nn.InstanceNorm2d(bc),nn.ReLU(True))
        self.out_conv=nn.Sequential(nn.Conv2d(bc*2,bc,3,padding=1,bias=False),nn.InstanceNorm2d(bc),nn.ReLU(True),nn.Conv2d(bc,in_ch,7,padding=3),nn.Tanh())
    def forward(self, x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2)
        s=self.shared_res(e3); w=torch.sigmoid(self.fusion_weight)
        f=w*self.struct_path(s)+(1-w)*self.style_path(s)
        return self.out_conv(torch.cat([self.d2(torch.cat([self.d1(f),e2],1)),e1],1))


# ═══════════════════════════════════════════════
# STEP 2: TRAINING FUNCTION
# ═══════════════════════════════════════════════

def train_model(G, D, loader, ckpt_dir, name):
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"{'='*60}")

    vgg = VGGFeatures().to(DEVICE)
    content_fn = ContentLoss(vgg)
    style_fn   = StyleLoss(vgg)
    edge_fn    = EdgeLoss().to(DEVICE)
    scaler     = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    opt_G = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    sch_G = StepLR(opt_G, step_size=25, gamma=0.5)
    sch_D = StepLR(opt_D, step_size=25, gamma=0.5)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Pretrain ──
    print(f"  Pretraining ({PRETRAIN_STEPS} steps)...")
    G.train(); step = 0
    while step < PRETRAIN_STEPS:
        for photos, cartoons in loader:
            if step >= PRETRAIN_STEPS: break
            photos, cartoons = photos.to(DEVICE), cartoons.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                fake = G(photos)
                loss = 100*pixel_loss(fake,cartoons) + 10*content_fn(fake,cartoons)
            opt_G.zero_grad()
            if scaler: scaler.scale(loss).backward(); scaler.step(opt_G); scaler.update()
            else: loss.backward(); opt_G.step()
            step += 1
            if step % 500 == 0: print(f"    pretrain step {step}/{PRETRAIN_STEPS} loss={loss.item():.4f}")

    # ── GAN Training ──
    print(f"  GAN training ({EPOCHS} epochs)...")
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        G.train(); D.train()
        for photos, cartoons in loader:
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
                      + 100*style_fn(fake,cartoons)
                      + 10*edge_fn(fake,cartoons)
                      + 1*total_variation_loss(fake))
            opt_G.zero_grad()
            if scaler: scaler.scale(lg).backward(); scaler.step(opt_G); scaler.update()
            else: lg.backward(); opt_G.step()

        sch_G.step(); sch_D.step()
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"    Epoch [{epoch}/{EPOCHS}] D={ld.item():.4f} G={lg.item():.4f}")
            torch.save({"G": G.state_dict(), "D": D.state_dict()},
                       os.path.join(ckpt_dir, f"ckpt_epoch{epoch:03d}.pt"))

    mins = (time.time() - t0) / 60
    print(f"  {name} complete in {mins:.1f} min\n")
    return G


# ═══════════════════════════════════════════════
# STEP 3: EVALUATION FUNCTION
# ═══════════════════════════════════════════════

def evaluate(G, loader, label):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    fid   = FrechetInceptionDistance(normalize=True).to(DEVICE)
    ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)
    ss, ls = [], []
    G.eval()
    with torch.no_grad():
        for photos, cartoons in loader:
            photos, cartoons = photos.to(DEVICE), cartoons.to(DEVICE)
            fake = G(photos)
            f01 = (fake*0.5+0.5).clamp(0,1)
            r01 = (cartoons*0.5+0.5).clamp(0,1)
            fid.update((r01*255).byte(), real=True)
            fid.update((f01*255).byte(), real=False)
            ss.append(ssim(f01,r01).item())
            ls.append(lpips(f01,r01).item())
    r = {"label": label, "FID": fid.compute().item(),
         "SSIM": sum(ss)/len(ss), "LPIPS": sum(ls)/len(ls)}
    print(f"  {label}: FID={r['FID']:.2f}  SSIM={r['SSIM']:.4f}  LPIPS={r['LPIPS']:.4f}")
    return r


# ═══════════════════════════════════════════════
# MAIN: RUN EVERYTHING
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    total_start = time.time()
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Step 1: Download
    download_dataset()

    # Step 2: Load data
    train_ds = CartoonDataset(f"{DATA_DIR}/train", size=256)
    test_ds  = CartoonDataset(f"{DATA_DIR}/test",  size=256)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=8, num_workers=0)
    print(f"Train: {len(train_ds)} images, Test: {len(test_ds)} images")

    results = []

    # Step 3: Train + Evaluate Model 1
    G1 = SinglePathGenerator().to(DEVICE)
    D1 = MultiDiscriminator().to(DEVICE)
    G1 = train_model(G1, D1, train_loader, "ckpt_single", "Single-Path Baseline")
    results.append(evaluate(G1, test_loader, "Single-Path Baseline"))
    del G1, D1; torch.cuda.empty_cache()

    # Step 4: Train + Evaluate Model 2
    G2 = LegacyDualPath().to(DEVICE)
    D2 = MultiDiscriminator().to(DEVICE)
    G2 = train_model(G2, D2, train_loader, "ckpt_dualpath", "Dual-Path + CBAM")
    results.append(evaluate(G2, test_loader, "Dual-Path + CBAM"))
    del G2, D2; torch.cuda.empty_cache()

    # Step 5: Train + Evaluate Model 3
    G3 = DualPathGenerator().to(DEVICE)
    D3 = MultiDiscriminator().to(DEVICE)
    G3 = train_model(G3, D3, train_loader, "ckpt_adain", "DualPath+AdaIN+SelfAttn")
    results.append(evaluate(G3, test_loader, "DualPath+AdaIN+SelfAttn"))
    del G3, D3; torch.cuda.empty_cache()

    # Step 6: Print final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"  {'Model':<30} {'FID ↓':>10} {'SSIM ↑':>10} {'LPIPS ↓':>10}")
    print(f"  {'-'*62}")
    for r in results:
        fid_mark  = " ✓" if r["FID"]  == min(x["FID"]  for x in results) else ""
        ssim_mark = " ✓" if r["SSIM"] == max(x["SSIM"] for x in results) else ""
        lpips_mark= " ✓" if r["LPIPS"]== min(x["LPIPS"]for x in results) else ""
        print(f"  {r['label']:<30} {r['FID']:>8.2f}{fid_mark:<2} {r['SSIM']:>8.4f}{ssim_mark:<2} {r['LPIPS']:>8.4f}{lpips_mark:<2}")
    print("="*70)

    # Save to file
    with open("results_comparison.txt", "w") as f:
        f.write("PHOTO-TO-CARTOON GAN — MODEL COMPARISON\n")
        f.write(f"Dataset: instruction-tuning-sd/cartoonization\n")
        f.write(f"Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, Pretrain: {PRETRAIN_STEPS}\n\n")
        f.write(f"{'Model':<30} {'FID':>10} {'SSIM':>10} {'LPIPS':>10}\n")
        f.write("-"*62 + "\n")
        for r in results:
            f.write(f"{r['label']:<30} {r['FID']:>10.2f} {r['SSIM']:>10.4f} {r['LPIPS']:>10.4f}\n")

    total_mins = (time.time() - total_start) / 60
    print(f"\nTotal time: {total_mins:.1f} minutes")
    print("Results saved to results_comparison.txt")
    print("Checkpoints: ckpt_single/ ckpt_dualpath/ ckpt_adain/")
