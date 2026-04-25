# Photo-to-Cartoon Translation — Dual-Path Generator GAN

Implementation of the paper *"Photo-to-Cartoon Translation Using Dual-Path Generator GAN"*.

## Architecture

- **Dual-Path Generator**: encoder → shared ResNet block → [structure path | style path (+ CBAM)] → weighted fusion → decoder with skip connections
- **Multi-Discriminator**: smoothness D, edge D, texture D
- **Losses**: LSGAN adversarial + VGG perceptual content loss

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Expects this layout:
```
data/
  train/
    photos/      # real photos
    cartoons/    # paired cartoon images (same filenames)
  test/
    photos/
    cartoons/
```

## Train

```bash
python train.py --data_dir data/train --epochs 100 --batch_size 4 --lambda_content 10
```

## Inference

```bash
# Single image
python inference.py --checkpoint checkpoints/ckpt_epoch100.pt \
                    --input photo.jpg --output cartoon.jpg

# Directory
python inference.py --checkpoint checkpoints/ckpt_epoch100.pt \
                    --input data/test/photos/ --output results/
```

## Evaluate (FID / SSIM / LPIPS)

```bash
python evaluate.py --checkpoint checkpoints/ckpt_epoch100.pt --data_dir data/test
```
