"""
Run cartoonization on a single image or a directory of images.

Usage:
  python inference.py --checkpoint checkpoints/ckpt_epoch100.pt \
                      --input photo.jpg --output cartoon.jpg
"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from model import DualPathGenerator


def load_image(path, size=256):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


def tensor_to_pil(t):
    """Convert a (1, 3, H, W) tensor in [-1,1] to a PIL image."""
    t = (t.squeeze(0) * 0.5 + 0.5).clamp(0, 1)   # -> [0,1]
    t = (t * 255).byte().cpu().numpy()              # -> uint8
    t = t.transpose(1, 2, 0)                        # CHW -> HWC
    return Image.fromarray(t, mode="RGB")


def cartoonize(G, img_tensor, device):
    G.eval()
    with torch.no_grad():
        out = G(img_tensor.to(device))
        print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
        return tensor_to_pil(out)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = DualPathGenerator().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["G"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = [f for f in os.listdir(args.input)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        for fname in files:
            img = load_image(os.path.join(args.input, fname))
            out = cartoonize(G, img, device)
            out_path = os.path.join(args.output, fname)
            out.save(out_path)
            print(f"  Saved: {out_path}")
    else:
        img = load_image(args.input)
        out = cartoonize(G, img, device)
        # Always save as PNG to avoid JPEG compression issues
        out_path = args.output if args.output.lower().endswith(".png") \
                   else os.path.splitext(args.output)[0] + ".png"
        out.save(out_path)
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input",      required=True)
    parser.add_argument("--output",     required=True)
    main(parser.parse_args())
