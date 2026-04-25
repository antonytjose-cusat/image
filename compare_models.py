"""
Compare two model checkpoints quantitatively (FID/SSIM/LPIPS)
and generate a side-by-side visual comparison grid.

Usage:
  python compare_models.py \
      --checkpoint_a checkpoints_old/ckpt_epoch050.pt \
      --checkpoint_b checkpoints_new/ckpt_epoch100.pt \
      --data_dir data/test \
      --output_dir comparison/
"""
import argparse
import os
import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from PIL import Image, ImageDraw, ImageFont
from dataset import CartoonDataset
from model import DualPathGenerator


def load_model(checkpoint_path, device):
    G = DualPathGenerator().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()
    return G


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def evaluate_model(G, loader, device, label):
    fid  = FrechetInceptionDistance(normalize=True).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    ssim_scores, lpips_scores = [], []

    with torch.no_grad():
        for photos, cartoons in loader:
            photos, cartoons = photos.to(device), cartoons.to(device)
            fake = G(photos)
            fake_01 = denorm(fake)
            real_01 = denorm(cartoons)

            fid.update((real_01 * 255).byte(), real=True)
            fid.update((fake_01 * 255).byte(), real=False)
            ssim_scores.append(ssim(fake_01, real_01).item())
            lpips_scores.append(lpips(fake_01, real_01).item())

    fid_score   = fid.compute().item()
    ssim_score  = sum(ssim_scores)  / len(ssim_scores)
    lpips_score = sum(lpips_scores) / len(lpips_scores)

    print(f"\n{'='*40}")
    print(f"  {label}")
    print(f"{'='*40}")
    print(f"  FID   ↓ : {fid_score:.2f}")
    print(f"  SSIM  ↑ : {ssim_score:.4f}")
    print(f"  LPIPS ↓ : {lpips_score:.4f}")

    return {"FID": fid_score, "SSIM": ssim_score, "LPIPS": lpips_score}


def save_comparison_grid(G_a, G_b, dataset, device, output_dir, n_samples=8):
    """
    Save a visual grid:
    Row layout per sample: [Input Photo | Model A | Model B | Ground Truth]
    """
    os.makedirs(output_dir, exist_ok=True)
    indices = list(range(min(n_samples, len(dataset))))

    rows = []
    for idx in indices:
        photo, cartoon = dataset[idx]
        photo_t   = photo.unsqueeze(0).to(device)
        cartoon_t = cartoon.unsqueeze(0).to(device)

        with torch.no_grad():
            out_a = G_a(photo_t)
            out_b = G_b(photo_t)

        row = torch.cat([
            denorm(photo_t),
            denorm(out_a),
            denorm(out_b),
            denorm(cartoon_t),
        ], dim=0)  # 4 images per row
        rows.append(row)

    # Stack all rows: (n_samples*4, C, H, W)
    grid_tensor = torch.cat(rows, dim=0)
    grid = vutils.make_grid(grid_tensor, nrow=4, padding=4, pad_value=1.0)

    # Save grid
    grid_path = os.path.join(output_dir, "comparison_grid.png")
    vutils.save_image(grid, grid_path)
    print(f"\nVisual comparison saved to: {grid_path}")
    print("Column order: [Input Photo | Model A (old) | Model B (new) | Ground Truth]")


def print_summary_table(metrics_a, metrics_b, label_a, label_b):
    print(f"\n{'='*55}")
    print(f"  {'Metric':<10} {'Model A':>15} {'Model B':>15}  {'Better':>8}")
    print(f"  {'-'*50}")

    for metric, arrow in [("FID", "↓"), ("SSIM", "↑"), ("LPIPS", "↓")]:
        a = metrics_a[metric]
        b = metrics_b[metric]
        if arrow == "↓":
            better = label_b if b < a else label_a
        else:
            better = label_b if b > a else label_a
        print(f"  {metric+' '+arrow:<10} {a:>15.4f} {b:>15.4f}  {better:>8}")

    print(f"{'='*55}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CartoonDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=8, num_workers=0)

    print(f"Loading Model A: {args.checkpoint_a}")
    G_a = load_model(args.checkpoint_a, device)

    print(f"Loading Model B: {args.checkpoint_b}")
    G_b = load_model(args.checkpoint_b, device)

    label_a = os.path.basename(os.path.dirname(args.checkpoint_a)) or "Model A"
    label_b = os.path.basename(os.path.dirname(args.checkpoint_b)) or "Model B"

    metrics_a = evaluate_model(G_a, loader, device, label_a)
    metrics_b = evaluate_model(G_b, loader, device, label_b)

    print_summary_table(metrics_a, metrics_b, label_a, label_b)

    save_comparison_grid(G_a, G_b, dataset, device, args.output_dir, n_samples=args.n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", required=True, help="Old model checkpoint")
    parser.add_argument("--checkpoint_b", required=True, help="New model checkpoint")
    parser.add_argument("--data_dir",     required=True, help="Test data directory")
    parser.add_argument("--output_dir",   default="comparison")
    parser.add_argument("--n_samples",    type=int, default=8,
                        help="Number of images in visual grid")
    main(parser.parse_args())
