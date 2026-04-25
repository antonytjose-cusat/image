"""
White-Box cartoon representations (from Wang et al. CVPR 2020).
Three explicit decompositions used as additional loss signals:
  1. Surface  - differentiable guided filter (smooth surfaces)
  2. Structure - superpixel segmentation (sparse color blocks)
  3. Texture  - random color shift (high-freq, color-independent)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import segmentation


# ─────────────────────────────────────────────
# 1. Surface Representation: Differentiable Guided Filter
# ─────────────────────────────────────────────

def box_filter(x, r):
    ch = x.shape[1]
    k  = 2 * r + 1
    w  = torch.ones((ch, 1, k, k), dtype=x.dtype, device=x.device) / (k * k)
    return F.conv2d(x, w, stride=1, padding=r, groups=ch)


def guided_filter(x, y, r=5, eps=1e-2):
    _, _, H, W = x.shape
    N      = box_filter(torch.ones(1, 1, H, W, dtype=x.dtype, device=x.device), r)
    mean_x = box_filter(x, r) / N
    mean_y = box_filter(y, r) / N
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x
    A      = cov_xy / (var_x + eps)
    b      = mean_y - A * mean_x
    return box_filter(A, r) / N * x + box_filter(b, r) / N


def surface_representation(img, r=5):
    """Extract smooth surface by guided filtering (self-guided)."""
    return guided_filter(img, img, r=r)


# ─────────────────────────────────────────────
# 2. Structure Representation: Superpixel + Adaptive Coloring
# ─────────────────────────────────────────────

def adaptive_coloring(image_np, seg_labels):
    """
    Adaptive coloring algorithm from White-Box paper.
    Uses mean/median depending on std of each segment.
    """
    out = np.zeros_like(image_np, dtype=np.float32)
    for label in np.unique(seg_labels):
        mask = seg_labels == label
        region = image_np[mask].astype(np.float32)
        std = region.std()
        if std < 20:
            color = region.mean(axis=0)
        elif std < 40:
            color = 0.5 * region.mean(axis=0) + 0.5 * np.median(region, axis=0)
        else:
            color = np.median(region, axis=0)
        out[mask] = color
    # Contrast enhancement (μ=1.2 from paper)
    out = np.clip(out, 0, 255)
    out = (out / 255.0) ** (1 / 1.2) * 255.0
    return out.astype(np.uint8)


def structure_representation(img_tensor):
    """
    Extract structure representation for a batch.
    img_tensor: (B, 3, H, W) in [-1, 1]
    Returns:    (B, 3, H, W) in [-1, 1]
    Runs superpixel on CPU in parallel threads for speed.
    """
    from concurrent.futures import ThreadPoolExecutor

    B, C, H, W = img_tensor.shape
    img_np = ((img_tensor.detach().cpu() * 0.5 + 0.5) * 255).byte().numpy()

    def process_one(b):
        img = img_np[b].transpose(1, 2, 0)
        seg = segmentation.felzenszwalb(img, scale=10, sigma=0.8, min_size=100)
        colored = adaptive_coloring(img, seg)
        t = torch.from_numpy(colored.transpose(2, 0, 1)).float() / 255.0
        return t * 2 - 1

    with ThreadPoolExecutor(max_workers=B) as ex:
        results = list(ex.map(process_one, range(B)))

    return torch.stack(results).to(img_tensor.device)


# ─────────────────────────────────────────────
# 3. Texture Representation: Random Color Shift
# ─────────────────────────────────────────────

class RandomColorShift(nn.Module):
    """
    Extracts single-channel texture map by random weighted RGB combination.
    α=0.8, β ~ U(-1,1) as in White-Box paper.
    """
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, img):
        # img: (B, 3, H, W) in [-1,1]
        beta = torch.FloatTensor(3).uniform_(-1, 1).to(img.device)
        beta = beta / (beta.abs().sum() + 1e-6)
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        texture = (1 - self.alpha) * (beta[0]*r + beta[1]*g + beta[2]*b) + self.alpha * gray
        return texture  # (B, 1, H, W)


# ─────────────────────────────────────────────
# 4. Total Variation Loss
# ─────────────────────────────────────────────

def total_variation_loss(img):
    """Spatial smoothness loss to reduce high-frequency noise."""
    B, C, H, W = img.shape
    tv_h = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean()
    tv_w = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean()
    return tv_h + tv_w
