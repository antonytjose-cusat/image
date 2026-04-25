"""
Compute FID, SSIM, and LPIPS on the test set.

Usage:
  python evaluate.py --checkpoint checkpoints/ckpt_epoch100.pt \
                     --data_dir data/test
"""
import argparse
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from model import DualPathGenerator
from dataset import CartoonDataset


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = DualPathGenerator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    dataset = CartoonDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=8, num_workers=4)

    fid = FrechetInceptionDistance(normalize=True).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    ssim_scores, lpips_scores = [], []

    with torch.no_grad():
        for photos, cartoons in loader:
            photos, cartoons = photos.to(device), cartoons.to(device)
            fake = G(photos)

            # Denorm to [0,1]
            fake_01 = (fake * 0.5 + 0.5).clamp(0, 1)
            real_01 = (cartoons * 0.5 + 0.5).clamp(0, 1)

            # FID expects uint8 [0,255]
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
    parser.add_argument("--data_dir", required=True)
    main(parser.parse_args())
