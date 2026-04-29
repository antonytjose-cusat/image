#!/usr/bin/env python3
"""
Dual-Path Conditioned Diffusion for Cartoon Generation.

Key idea: Use the pre-trained dual-path GAN's style-path features as
additional conditioning for the diffusion model's denoising process.
Structure path features are injected via a lightweight adapter.

Usage:
  python diffusion/train_dualpath_diffusion.py \
      --data_dir data_new/train \
      --gan_checkpoint ckpt_dualpath/ckpt_epoch050.pt \
      --epochs 5
"""
import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.cbam import CBAM


# ── Dual-Path Feature Extractor (frozen, from trained GAN) ──
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

class DualPathEncoder(nn.Module):
    """Extract structure and style features from the trained dual-path GAN."""
    def __init__(self, bc=64, nr=4):
        super().__init__()
        self.e1=_cb(3,bc); self.e2=_cb(bc,bc*2,2); self.e3=_cb(bc*2,bc*4,2)
        self.shared_res=nn.Sequential(_RB(bc*4,True))
        self.struct_path=nn.Sequential(*[_RB(bc*4) for _ in range(nr)])
        self.style_path=nn.Sequential(*[_RB(bc*4,True) for _ in range(nr)])

    def forward(self, x):
        e3 = self.e3(self.e2(self.e1(x)))
        shared = self.shared_res(e3)
        return self.struct_path(shared), self.style_path(shared)


# ── Style Adapter: projects dual-path features to UNet cross-attention dim ──
class StyleAdapter(nn.Module):
    """Projects style features from dual-path GAN to UNet cross-attention space."""
    def __init__(self, in_channels=256, cross_attn_dim=768, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attn_dim = cross_attn_dim
        self.pool = nn.AdaptiveAvgPool2d(1)  # global average pool -> (B, C, 1, 1)
        self.proj = nn.Sequential(
            nn.Linear(in_channels, cross_attn_dim * num_tokens),
            nn.GELU(),
        )

    def forward(self, style_feat):
        b = style_feat.shape[0]
        pooled = self.pool(style_feat).view(b, -1)  # (B, 256)
        projected = self.proj(pooled)  # (B, 768 * num_tokens)
        return projected.view(b, self.num_tokens, self.cross_attn_dim)  # (B, 8, 768)


# ── Structure Adapter: adds structure info to noisy latents ──
class StructureAdapter(nn.Module):
    """Injects structure features into the latent space as additional channels."""
    def __init__(self, struct_channels=256, latent_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(struct_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, latent_channels, 1),
        )

    def forward(self, struct_feat, latent_size):
        # Resize struct features to match latent spatial dims
        h, w = latent_size
        resized = nn.functional.interpolate(struct_feat, size=(h, w), mode='bilinear', align_corners=False)
        return self.conv(resized)  # (B, 4, h, w)


class CartoonPairDataset(Dataset):
    def __init__(self, root, size=512):
        self.photo_dir = os.path.join(root, "photos")
        self.cartoon_dir = os.path.join(root, "cartoons")
        self.files = sorted(os.listdir(self.photo_dir))
        self.tf_sd = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.tf_gan = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        photo = Image.open(os.path.join(self.photo_dir, name)).convert("RGB")
        cartoon = Image.open(os.path.join(self.cartoon_dir, name)).convert("RGB")
        return self.tf_sd(photo), self.tf_sd(cartoon), self.tf_gan(photo)


def load_gan_encoder(checkpoint_path, device):
    """Load the encoder part of the trained dual-path GAN."""
    encoder = DualPathEncoder().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    gan_state = ckpt["G"]
    # Load only encoder + path weights
    enc_state = {}
    for k, v in gan_state.items():
        if any(k.startswith(p) for p in ["e1", "e2", "e3", "shared_res", "struct_path", "style_path"]):
            enc_state[k] = v
    encoder.load_state_dict(enc_state)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"Loaded dual-path encoder from {checkpoint_path}")
    return encoder


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = "runwayml/stable-diffusion-v1-5"

    # Load SD components
    print("Loading Stable Diffusion...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add LoRA to UNet
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)

    # Load frozen dual-path GAN encoder
    gan_encoder = load_gan_encoder(args.gan_checkpoint, device)

    # Trainable adapters
    style_adapter = StyleAdapter(in_channels=256, cross_attn_dim=768).to(device)
    struct_adapter = StructureAdapter(struct_channels=256, latent_channels=4).to(device)

    # Dataset
    dataset = CartoonPairDataset(args.data_dir, size=512)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Optimizer: LoRA params + adapter params
    trainable_params = (list(unet.parameters()) +
                        list(style_adapter.parameters()) +
                        list(struct_adapter.parameters()))
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    # Text embeddings
    text_input = tokenizer(
        "a cartoon style image", padding="max_length",
        max_length=tokenizer.model_max_length, truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(text_input)[0]

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training dual-path conditioned diffusion for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        unet.train(); style_adapter.train(); struct_adapter.train()
        total_loss = 0

        for step, (photos_sd, cartoons_sd, photos_gan) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            cartoons_sd = cartoons_sd.to(device)
            photos_gan = photos_gan.to(device)

            # Extract dual-path features (frozen)
            with torch.no_grad():
                struct_feat, style_feat = gan_encoder(photos_gan)

            # Style conditioning: project to cross-attention space
            style_tokens = style_adapter(style_feat)  # (B, 8, 768)
            batch_text = text_embeddings.expand(cartoons_sd.shape[0], -1, -1)
            # Concatenate style tokens with text embeddings
            combined_cond = torch.cat([batch_text, style_tokens], dim=1)

            # Encode target to latents
            with torch.no_grad():
                latents = vae.encode(cartoons_sd).latent_dist.sample() * 0.18215

            # Structure conditioning: add to noisy latents
            struct_cond = struct_adapter(struct_feat, (latents.shape[2], latents.shape[3]))

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Add structure conditioning to noisy latents
            conditioned_latents = noisy_latents + 0.1 * struct_cond

            # Predict noise with combined conditioning
            noise_pred = unet(conditioned_latents, timesteps,
                              encoder_hidden_states=combined_cond).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.4f}")

        # Save everything
        save_dir = os.path.join(args.output_dir, f"epoch{epoch:02d}")
        os.makedirs(save_dir, exist_ok=True)
        unet.save_pretrained(os.path.join(save_dir, "lora"))
        torch.save({
            "style_adapter": style_adapter.state_dict(),
            "struct_adapter": struct_adapter.state_dict(),
        }, os.path.join(save_dir, "adapters.pt"))

    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gan_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="diffusion_dualpath_ckpt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    args = parser.parse_args()
    train(args)
