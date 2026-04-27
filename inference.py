"""
Run cartoonization on a single image or a directory of images.
Auto-detects architecture from checkpoint.

Usage:
  python inference.py --checkpoint checkpoints/ckpt_epoch050.pt \
                      --input photo.jpg --output cartoon.png
"""
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

from model import DualPathGenerator
from model.cbam import CBAM


# ── Legacy architecture (checkpoints_old, checkpoints_v2, checkpoints_v3) ──
def _conv_block(in_ch, out_ch, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch), nn.LeakyReLU(0.2, True))

class LegacyResBlock(nn.Module):
    def __init__(self, ch, use_cbam=False):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.InstanceNorm2d(ch))
        self.cbam = CBAM(ch) if use_cbam else nn.Identity()
    def forward(self, x): return self.cbam(self.block(x)) + x

class LegacyGenerator(nn.Module):
    def __init__(self, in_ch=3, bc=64, nr=4):
        super().__init__()
        self.e1=_conv_block(in_ch,bc); self.e2=_conv_block(bc,bc*2,2); self.e3=_conv_block(bc*2,bc*4,2)
        self.shared_res=nn.Sequential(LegacyResBlock(bc*4,True))
        self.struct_path=nn.Sequential(*[LegacyResBlock(bc*4) for _ in range(nr)])
        self.style_path=nn.Sequential(*[LegacyResBlock(bc*4,True) for _ in range(nr)])
        self.fusion_weight=nn.Parameter(torch.tensor(0.5))
        self.d1=nn.Sequential(nn.ConvTranspose2d(bc*4,bc*2,4,2,1,bias=False),nn.InstanceNorm2d(bc*2),nn.ReLU(True))
        self.d2=nn.Sequential(nn.ConvTranspose2d(bc*4,bc,4,2,1,bias=False),nn.InstanceNorm2d(bc),nn.ReLU(True))
        self.out_conv=nn.Sequential(nn.Conv2d(bc*2,bc,3,padding=1,bias=False),nn.InstanceNorm2d(bc),nn.ReLU(True),nn.Conv2d(bc,in_ch,7,padding=3),nn.Tanh())
    def forward(self, x):
        e1=self.e1(x); e2=self.e2(e1); e3=self.e3(e2)
        s=self.shared_res(e3); w=torch.sigmoid(self.fusion_weight)
        f=w*self.struct_path(s)+(1-w)*self.style_path(s)
        return self.out_conv(torch.cat([self.d2(torch.cat([self.d1(f),e2],1)),e1],1))


# ── WhiteBox architecture (short var names: b, c) ──
class _RB(nn.Module):
    def __init__(self, ch, cbam=False):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.InstanceNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,padding=1,bias=False), nn.InstanceNorm2d(ch))
        self.c = CBAM(ch) if cbam else nn.Identity()
    def forward(self, x): return self.c(self.b(x)) + x

class WhiteBoxGenerator(nn.Module):
    def __init__(self, in_ch=3, bc=64, nr=4):
        super().__init__()
        self.e1=_conv_block(in_ch,bc); self.e2=_conv_block(bc,bc*2,2); self.e3=_conv_block(bc*2,bc*4,2)
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


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Try each architecture until one works
    for GenClass, name in [
        (DualPathGenerator, "V4 AdaIN+SelfAttn"),
        (LegacyGenerator,  "Legacy Dual-Path"),
        (WhiteBoxGenerator, "WhiteBox Dual-Path"),
    ]:
        try:
            G = GenClass().to(device)
            G.load_state_dict(ckpt["G"])
            print(f"  Architecture: {name}")
            G.eval()
            return G
        except RuntimeError:
            continue
    raise RuntimeError("Could not load checkpoint — unknown architecture")


def load_image(path, size=256):
    tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0)


def tensor_to_pil(t):
    t = (t.squeeze(0)*0.5+0.5).clamp(0,1)
    t = (t*255).byte().cpu().numpy().transpose(1,2,0)
    return Image.fromarray(t, mode="RGB")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading: {args.checkpoint}")
    G = load_model(args.checkpoint, device)

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        files = [f for f in os.listdir(args.input) if f.lower().endswith((".jpg",".jpeg",".png"))]
        for fname in files:
            img = load_image(os.path.join(args.input, fname))
            with torch.no_grad():
                out = G(img.to(device))
                print(f"  {fname} range: [{out.min():.3f}, {out.max():.3f}]")
            tensor_to_pil(out).save(os.path.join(args.output, fname.rsplit('.',1)[0]+".png"))
            print(f"  Saved: {fname}")
    else:
        img = load_image(args.input)
        with torch.no_grad():
            out = G(img.to(device))
            print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
        out_path = os.path.splitext(args.output)[0] + ".png"
        tensor_to_pil(out).save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
