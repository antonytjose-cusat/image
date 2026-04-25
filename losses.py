import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────
# VGG Feature Extractor (shared)
# ─────────────────────────────────────────────

class VGGFeatures(nn.Module):
    """VGG16 feature slices used for content, structure and perceptual losses."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        f = list(vgg.features)
        self.slice1 = nn.Sequential(*f[:4])   # relu1_2
        self.slice2 = nn.Sequential(*f[4:9])  # relu2_2
        self.slice3 = nn.Sequential(*f[9:16]) # relu3_3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return f1, f2, f3


# ─────────────────────────────────────────────
# Content Loss (VGG perceptual, photo <-> generated)
# ─────────────────────────────────────────────

class ContentLoss(nn.Module):
    def __init__(self, vgg: VGGFeatures):
        super().__init__()
        self.vgg = vgg

    def forward(self, generated, target):
        gf1, gf2, gf3 = self.vgg(generated)
        tf1, tf2, tf3 = self.vgg(target)
        return (nn.functional.l1_loss(gf1, tf1) +
                nn.functional.l1_loss(gf2, tf2) +
                nn.functional.l1_loss(gf3, tf3))


# ─────────────────────────────────────────────
# Structure Loss (White-Box): VGG(generated) vs VGG(superpixel)
# ─────────────────────────────────────────────

class StructureLoss(nn.Module):
    """
    Enforces spatial consistency between generated image and its
    superpixel structure representation using VGG feature space.
    """
    def __init__(self, vgg: VGGFeatures):
        super().__init__()
        self.vgg = vgg

    def forward(self, generated, structure):
        _, _, gf3 = self.vgg(generated)
        _, _, sf3 = self.vgg(structure)
        return nn.functional.l1_loss(gf3, sf3)


# ─────────────────────────────────────────────
# Style Loss (Gram matrix)
# ─────────────────────────────────────────────

def gram_matrix(feat):
    b, c, h, w = feat.size()
    f = feat.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, vgg: VGGFeatures):
        super().__init__()
        self.vgg = vgg

    def forward(self, generated, target):
        gf1, gf2, _ = self.vgg(generated)
        tf1, tf2, _ = self.vgg(target)
        return (nn.functional.l1_loss(gram_matrix(gf1), gram_matrix(tf1)) +
                nn.functional.l1_loss(gram_matrix(gf2), gram_matrix(tf2)))


# ─────────────────────────────────────────────
# Pixel Loss
# ─────────────────────────────────────────────

def pixel_loss(generated, target):
    return nn.functional.l1_loss(generated, target)


# ─────────────────────────────────────────────
# Adversarial Losses (LSGAN)
# ─────────────────────────────────────────────

def adversarial_loss_g(disc_outputs):
    """Generator: fool all discriminators."""
    return sum(torch.mean((d - 1) ** 2) for d in disc_outputs) / len(disc_outputs)


def adversarial_loss_d(real_outputs, fake_outputs):
    """Discriminator: real=1, fake=0."""
    loss = 0
    for real, fake in zip(real_outputs, fake_outputs):
        loss += 0.5 * (torch.mean((real - 1) ** 2) + torch.mean(fake ** 2))
    return loss / len(real_outputs)
