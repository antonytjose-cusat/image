#!/usr/bin/env python3
"""
Run everything for diffusion experiments:
1. Install deps
2. Train LoRA-only baseline
3. Train dual-path conditioned diffusion
4. Evaluate both + compare with GAN results

Usage:
  python diffusion/run_diffusion_experiments.py
"""
import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data_new"
GAN_CKPT = "ckpt_dualpath/ckpt_epoch050.pt"


class EvalDataset(Dataset):
    def __init__(self, root, size=512):
        self.photo_dir = os.path.join(root, "photos")
        self.cartoon_dir = os.path.join(root, "cartoons")
        self.files = sorted(os.listdir(self.photo_dir))
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        photo = Image.open(os.path.join(self.photo_dir, name)).convert("RGB")
        cartoon = Image.open(os.path.join(self.cartoon_dir, name)).convert("RGB")
        return self.tf(photo), self.tf(cartoon), name


def generate_with_lora(lora_dir, test_dir, output_dir, num_images=100):
    """Generate cartoon images using LoRA-only diffusion."""
    from diffusers import StableDiffusionImg2ImgPipeline
    from peft import PeftModel
    import torch

    print(f"\nGenerating with LoRA from {lora_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)

    # Load LoRA weights via PEFT
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_dir).to(DEVICE)

    photo_dir = os.path.join(test_dir, "photos")
    files = sorted(os.listdir(photo_dir))[:num_images]

    for i, fname in enumerate(files):
        img = Image.open(os.path.join(photo_dir, fname)).convert("RGB").resize((512, 512))
        with torch.autocast("cuda"):
            result = pipe("a cartoon style image", image=img, strength=0.6,
                          num_inference_steps=30, guidance_scale=7.5).images[0]
        result.save(os.path.join(output_dir, fname))
        if (i + 1) % 20 == 0:
            print(f"    Generated {i+1}/{len(files)}")

    print(f"  Generated {len(files)} images to {output_dir}")
    pipe.to("cpu")
    del pipe
    torch.cuda.empty_cache()


def evaluate_generated(gen_dir, test_dir, label, size=256):
    """Compute FID, SSIM, LPIPS on generated images."""
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    fid = FrechetInceptionDistance(normalize=True).to(DEVICE)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(DEVICE)

    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

    cartoon_dir = os.path.join(test_dir, "cartoons")
    gen_files = sorted(os.listdir(gen_dir))
    ss, ls = [], []

    for fname in gen_files:
        gen_path = os.path.join(gen_dir, fname)
        real_path = os.path.join(cartoon_dir, fname)
        if not os.path.exists(real_path):
            continue

        gen_img = tf(Image.open(gen_path).convert("RGB")).unsqueeze(0).to(DEVICE)
        real_img = tf(Image.open(real_path).convert("RGB")).unsqueeze(0).to(DEVICE)

        fid.update((real_img * 255).byte(), real=True)
        fid.update((gen_img * 255).byte(), real=False)
        ss.append(ssim(gen_img, real_img).item())
        ls.append(lpips(gen_img, real_img).item())

    r = {
        "label": label,
        "FID": fid.compute().item(),
        "SSIM": sum(ss) / len(ss) if ss else 0,
        "LPIPS": sum(ls) / len(ls) if ls else 0,
    }
    print(f"  {label}: FID={r['FID']:.2f}  SSIM={r['SSIM']:.4f}  LPIPS={r['LPIPS']:.4f}")
    return r


def main():
    t0 = time.time()
    results = []
    test_dir = f"{DATA_DIR}/test"

    # ── Step 1: Train LoRA-only baseline ──
    print("="*60)
    print("STEP 1: Training Diffusion + LoRA (baseline)")
    print("="*60)
    if not os.path.exists("diffusion_lora_ckpt/lora_epoch05"):
        os.system(f"{sys.executable} diffusion/train_lora.py "
                  f"--data_dir {DATA_DIR}/train --epochs 5 --batch_size 1")
    else:
        print("LoRA checkpoint exists, skipping training.")

    # ── Step 2: Train Dual-Path Conditioned Diffusion ──
    print("\n" + "="*60)
    print("STEP 2: Training Diffusion + Dual-Path Conditioning")
    print("="*60)
    dualpath_ready = False
    dualpath_lora = "diffusion_dualpath_ckpt/epoch05/lora"
    if os.path.exists(os.path.join(dualpath_lora, "adapter_config.json")):
        print("Dual-path diffusion checkpoint exists, skipping training.")
        dualpath_ready = True
    else:
        try:
            ret = os.system(f"{sys.executable} diffusion/train_dualpath_diffusion.py "
                      f"--data_dir {DATA_DIR}/train "
                      f"--gan_checkpoint {GAN_CKPT} --epochs 5 --batch_size 1")
            if ret == 0 and os.path.exists(os.path.join(dualpath_lora, "adapter_config.json")):
                dualpath_ready = True
            else:
                print("WARNING: Dual-path diffusion training failed. Skipping.")
        except Exception as e:
            print(f"WARNING: Dual-path diffusion training error: {e}. Skipping.")

    # ── Step 3: Generate images ──
    print("\n" + "="*60)
    print("STEP 3: Generating test images")
    print("="*60)
    generate_with_lora("diffusion_lora_ckpt/lora_epoch05",
                       test_dir, "gen_lora_only", num_images=100)
    if dualpath_ready:
        generate_with_lora(dualpath_lora,
                           test_dir, "gen_dualpath_diff", num_images=100)

    # ── Step 4: Evaluate ──
    print("\n" + "="*60)
    print("STEP 4: Evaluating")
    print("="*60)
    results.append(evaluate_generated("gen_lora_only", test_dir, "Diffusion+LoRA (alone)"))
    if dualpath_ready and os.path.exists("gen_dualpath_diff"):
        results.append(evaluate_generated("gen_dualpath_diff", test_dir, "Diffusion+DualPath"))

    # ── Step 5: Print comparison ──
    # Add GAN results for reference
    gan_results = [
        {"label": "Single-Path GAN", "FID": 39.80, "SSIM": 0.7444, "LPIPS": 0.1225},
        {"label": "Dual-Path+CBAM GAN", "FID": 38.46, "SSIM": 0.7463, "LPIPS": 0.1137},
    ]
    all_results = gan_results + results

    print("\n" + "="*70)
    print("FINAL COMPARISON: GAN vs Diffusion")
    print("="*70)
    print(f"  {'Model':<30} {'FID ↓':>10} {'SSIM ↑':>10} {'LPIPS ↓':>10}")
    print(f"  {'-'*62}")
    for r in all_results:
        print(f"  {r['label']:<30} {r['FID']:>10.2f} {r['SSIM']:>10.4f} {r['LPIPS']:>10.4f}")
    print("="*70)

    # Save
    with open("results_diffusion_comparison.txt", "w") as f:
        f.write("GAN vs DIFFUSION COMPARISON\n\n")
        f.write(f"{'Model':<30} {'FID':>10} {'SSIM':>10} {'LPIPS':>10}\n")
        f.write("-"*62 + "\n")
        for r in all_results:
            f.write(f"{r['label']:<30} {r['FID']:>10.2f} {r['SSIM']:>10.4f} {r['LPIPS']:>10.4f}\n")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("Results saved to results_diffusion_comparison.txt")


if __name__ == "__main__":
    main()
