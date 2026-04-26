# Experiment Log — Photo-to-Cartoon GAN

## Architecture
- Dual-Path Generator (structure path + style path with CBAM)
- Multi-Discriminator (surface D, texture D, edge D) with spectral norm
- Based on paper: "Photo-to-Cartoon Translation Using Dual-Path Generator GAN"
- Improved with ideas from White-Box Cartoonization (CVPR 2020)

---

## Model V1 — Baseline Dual-Path GAN
**Checkpoint:** `checkpoints_old/`
**Dataset:** dqymaggie/cartoonizer-dataset (paired, 4000 train / 1000 test)
**Epochs:** 50
**Batch size:** 4
**Losses:** Adversarial + Pixel (λ=100) + VGG Content (λ=10) + Style/Gram (λ=50)
**Notes:**
- First working model
- Output was slightly blurry, weak cartoon stylization
- Dataset cartoon style was too subtle (AI-generated cartoons, not real anime)

---

## Model V2 — White-Box Improvements
**Checkpoint:** `checkpoints_v2/`
**Dataset:** dqymaggie/cartoonizer-dataset (same as V1)
**Epochs:** 100
**Batch size:** 8
**Losses:** Adversarial + Pixel (λ=100) + Content (λ=10) + Style (λ=50) + TV (λ=1)
**New additions:**
- Generator pretraining (2000 steps, content+pixel only)
- TV loss for spatial smoothness
- Mixed precision (FP16) training
- Spectral normalization on discriminators
- LR decay scheduler (StepLR, γ=0.5 every 50 epochs)
**Notes:**
- Faster training due to FP16
- TV loss weight of 1e4 caused black output (fixed to 1.0)
- Still limited by dataset quality

---

## Model V3 — Edge + Cycle Consistency
**Checkpoint:** `checkpoints_v3/`
**Dataset:** dqymaggie/cartoonizer-dataset (same)
**Epochs:** 50
**Batch size:** 8
**Losses:** All V2 losses + Edge (λ=10) + Cycle (λ=10) + stronger Style (λ=100)
**New additions:**
- Edge sharpening loss (Sobel map matching)
- Cycle consistency (G_inv reconstructs photo from fake cartoon)
- Stronger style weight (100 vs 50)
**Notes:**
- Cycle consistency prevents content loss during stylization
- Edge loss enforces sharp cartoon outlines

---

## Model V4 — Anime Dataset
**Checkpoint:** `checkpoints_v4/`
**Dataset:** huggan/anime-faces (21,551 real anime faces) + FFHQ real photos (unpaired)
**Epochs:** 100
**Batch size:** 8
**Losses:** Same as V3
**New additions:**
- Real anime dataset (not AI-generated cartoons)
- Unpaired training setup
**Notes:**
- Expected to produce strongest cartoon stylization
- Real anime frames provide authentic cartoon style signal

---

## Key Lessons Learned

1. **Dataset quality > model complexity** — real anime frames produce far better
   cartoon stylization than AI-generated cartoon datasets
2. **TV loss weight matters** — λ=1e4 collapses the generator to black output; λ=1.0 is safe
3. **Generator pretraining is critical** — 2000+ steps of content-only training
   before GAN prevents mode collapse
4. **Spectral norm stabilizes discriminators** — prevents discriminator from
   overpowering the generator early in training
5. **Superpixel structure loss is slow** — Felzenszwalb on CPU per batch is a
   bottleneck; removed from training loop for speed
6. **Mixed precision (FP16)** — ~2x speedup on RTX 5070 with no quality loss

---

## Metrics Comparison (to be filled after evaluation)

| Model | FID ↓ | SSIM ↑ | LPIPS ↓ |
|-------|-------|--------|---------|
| V1 Baseline | - | - | - |
| V2 WhiteBox | - | - | - |
| V3 Edge+Cycle | - | - | - |
| V4 Anime Dataset | - | - | - |

---

## Model V4 — AdaIN + Self-Attention + Multi-Scale Discriminator
**Checkpoint:** `checkpoints_v4/`
**Dataset:** dqymaggie/cartoonizer-dataset (same paired dataset)
**Epochs:** 100
**Batch size:** 8
**Architecture changes:**
- AdaIN: style path dynamically modulates structure path (key novelty)
- Self-attention in bottleneck for global spatial context
- GELU activations replacing LeakyReLU in generator
- Multi-scale discriminator (3 scales: full/half/quarter + edge D)
**Losses:** Adversarial + Pixel (λ=100) + Content (λ=10) + Style (λ=100) + Edge (λ=10) + TV (λ=1)
**Expected improvements:**
- AdaIN enables proper style transfer between paths
- Self-attention ensures globally consistent cartoon style
- Multi-scale D catches both fine texture and global structure
