"""
Compare multiple model checkpoints quantitatively (FID/SSIM/LPIPS)
and generate a side-by-side visual comparison grid.

Usage:
  python compare_models.py \
      --checkpoints checkpoints_old/ckpt_epoch050.pt \
                    checkpoints_v2/ckpt_epoch050.pt \
                    checkpoints/ckpt_epoch050.pt \
      --labels "Model-V1 (Baseline)" "Model-V2 (WhiteBox)" "Model-V3 (Edge+Cycle)" \
      --data_dir data/test \
      --output_dir comparison/
"""
import argparse
import os
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from dataset import CartoonDataset
from model import DualPathGenerator
from model.cbam import CBAM


# ── Legacy architecture (V1/V2/V3 checkpoints) ──
def conv_lrelu(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
        nn.LeakyReLU(0.2, inplace=True),
    )

class LegacyResBlock(nn.Module):
    def __init__(self, channels, use_cbam=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels),
        )
        self.cbam = CBAM(channels) if use_cbam else nn.Identity()
    def forward(self, x):
        return self.cbam(self.block(x)) + x

class LegacyGenerator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, n_res=4):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_lrelu(in_ch, base_ch, stride=1),
            conv_lrelu(base_ch, base_ch*2, stride=2),
            conv_lrelu(base_ch*2, base_ch*4, stride=2),
        )
        self.shared_res  = nn.Sequential(LegacyResBlock(base_ch*4, use_cbam=True))
        self.struct_path = nn.Sequential(*[LegacyResBlock(base_ch*4) for _ in range(n_res)])
        self.style_path  = nn.Sequential(*[LegacyResBlock(base_ch*4, use_cbam=True) for _ in range(n_res)])
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch*2), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch*4, base_ch, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(
            nn.Conv2d(base_ch*2, in_ch, 7, padding=3), nn.Tanh())
    def forward(self, x):
        e1 = self.encoder[0](x)
        e2 = self.encoder[1](e1)
        e3 = self.encoder[2](e2)
        shared = self.shared_res(e3)
        struct_feat = self.struct_path(shared)
        style_feat  = self.style_path(shared)
        w = torch.sigmoid(self.fusion_weight)
        fused = w * struct_feat + (1-w) * style_feat
        d1 = self.up1(fused)
        d2 = self.up2(torch.cat([d1, e2], dim=1))
        return self.out_conv(torch.cat([d2, e1], dim=1))


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    G = DualPathGenerator().to(device)
    try:
        G.load_state_dict(ckpt["G"])
    except RuntimeError:
        # Old architecture — rebuild with old generator
        G = LegacyGenerator().to(device)
        G.load_state_dict(ckpt["G"])
    G.eval()
    return G


def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)


def evaluate_model(G, loader, device, label):
    fid   = FrechetInceptionDistance(normalize=True).to(device)
    ssim  = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    ssim_scores, lpips_scores = [], []

    with torch.no_grad():
        for photos, cartoons in loader:
            photos, cartoons = photos.to(device), cartoons.to(device)
            fake    = G(photos)
            fake_01 = denorm(fake)
            real_01 = denorm(cartoons)

            fid.update((real_01 * 255).byte(), real=True)
            fid.update((fake_01 * 255).byte(), real=False)
            ssim_scores.append(ssim(fake_01, real_01).item())
            lpips_scores.append(lpips(fake_01, real_01).item())

    return {
        "label": label,
        "FID":   fid.compute().item(),
        "SSIM":  sum(ssim_scores)  / len(ssim_scores),
        "LPIPS": sum(lpips_scores) / len(lpips_scores),
    }


def print_table(results):
    print(f"\n{'='*70}")
    header = f"  {'Metric':<10}"
    for r in results:
        header += f"  {r['label'][:18]:>18}"
    print(header)
    print(f"  {'-'*65}")

    for metric, arrow in [("FID", "↓"), ("SSIM", "↑"), ("LPIPS", "↓")]:
        row = f"  {metric+' '+arrow:<10}"
        values = [r[metric] for r in results]
        best   = min(values) if arrow == "↓" else max(values)
        for v in values:
            marker = " ✓" if v == best else "  "
            row += f"  {v:>16.4f}{marker}"
        print(row)

    print(f"{'='*70}\n")


def save_comparison_grid(models, labels, dataset, device, output_dir, n_samples=8):
    """
    Grid layout per sample row:
    [Input] | [Model1] | [Model2] | [Model3] | [Ground Truth]
    """
    os.makedirs(output_dir, exist_ok=True)
    indices = list(range(min(n_samples, len(dataset))))
    n_cols  = len(models) + 2  # input + all models + ground truth

    rows = []
    for idx in indices:
        photo, cartoon = dataset[idx]
        photo_t   = photo.unsqueeze(0).to(device)
        cartoon_t = cartoon.unsqueeze(0).to(device)

        cols = [denorm(photo_t)]
        for G in models:
            with torch.no_grad():
                cols.append(denorm(G(photo_t)))
        cols.append(denorm(cartoon_t))

        rows.append(torch.cat(cols, dim=0))

    grid_tensor = torch.cat(rows, dim=0)
    grid = vutils.make_grid(grid_tensor, nrow=n_cols, padding=4, pad_value=1.0)
    grid_path = os.path.join(output_dir, "comparison_grid.png")
    vutils.save_image(grid, grid_path)

    print(f"Visual grid saved: {grid_path}")
    col_names = ["Input"] + labels + ["Ground Truth"]
    print(f"Columns: {' | '.join(col_names)}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert len(args.checkpoints) == len(args.labels), \
        "Number of --checkpoints and --labels must match"

    dataset = CartoonDataset(args.data_dir)
    loader  = DataLoader(dataset, batch_size=8, num_workers=0)

    models  = []
    results = []

    for ckpt_path, label in zip(args.checkpoints, args.labels):
        print(f"Loading {label} from {ckpt_path} ...")
        G = load_model(ckpt_path, device)
        models.append(G)
        metrics = evaluate_model(G, loader, device, label)
        results.append(metrics)
        print(f"  FID:{metrics['FID']:.2f}  SSIM:{metrics['SSIM']:.4f}  LPIPS:{metrics['LPIPS']:.4f}")

    print_table(results)
    save_comparison_grid(models, args.labels, dataset, device,
                         args.output_dir, args.n_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="List of checkpoint paths")
    parser.add_argument("--labels",      nargs="+", required=True,
                        help="Label for each checkpoint")
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--output_dir",  default="comparison")
    parser.add_argument("--n_samples",   type=int, default=8)
    main(parser.parse_args())
