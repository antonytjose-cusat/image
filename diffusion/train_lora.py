#!/usr/bin/env python3
"""
Fine-tune Stable Diffusion with LoRA for cartoon style transfer.
This is the "diffusion alone" baseline — no dual-path conditioning.

Usage:
  python diffusion/train_lora.py --data_dir data_new/train --epochs 5
"""
import argparse
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionImg2ImgPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers import DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


class CartoonPairDataset(Dataset):
    def __init__(self, root, size=512):
        self.photo_dir = os.path.join(root, "photos")
        self.cartoon_dir = os.path.join(root, "cartoons")
        self.files = sorted(os.listdir(self.photo_dir))
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.prompt = "a cartoon style image"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        photo = Image.open(os.path.join(self.photo_dir, name)).convert("RGB")
        cartoon = Image.open(os.path.join(self.cartoon_dir, name)).convert("RGB")
        return self.transform(photo), self.transform(cartoon)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_id = "runwayml/stable-diffusion-v1-5"

    # Load components
    print("Loading Stable Diffusion...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)

    # Freeze everything except LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Add LoRA to UNet
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Dataset
    dataset = CartoonPairDataset(args.data_dir, size=512)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)

    # Encode prompt once
    text_input = tokenizer(
        "a cartoon style image", padding="max_length",
        max_length=tokenizer.model_max_length, truncation=True,
        return_tensors="pt"
    ).input_ids.to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(text_input)[0]

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    print(f"Training LoRA for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        unet.train()
        total_loss = 0
        for step, (photos, cartoons) in enumerate(tqdm(loader, desc=f"Epoch {epoch}")):
            cartoons = cartoons.to(device)

            # Encode target cartoon to latent space
            with torch.no_grad():
                latents = vae.encode(cartoons).latent_dist.sample() * 0.18215

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Expand text embeddings for batch
            batch_text = text_embeddings.expand(latents.shape[0], -1, -1)

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=batch_text).sample

            # Loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"  Epoch {epoch}/{args.epochs} — Loss: {avg_loss:.4f}")

        # Save LoRA weights
        unet.save_pretrained(os.path.join(args.output_dir, f"lora_epoch{epoch:02d}"))

    print(f"LoRA weights saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="diffusion_lora_ckpt")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=8)
    args = parser.parse_args()
    train(args)
