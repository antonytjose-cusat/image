import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from model import (DualPathGenerator, MultiDiscriminator,
                   surface_representation, RandomColorShift, total_variation_loss)
from dataset import CartoonDataset
from losses import (VGGFeatures, ContentLoss, StyleLoss,
                    pixel_loss, adversarial_loss_g, adversarial_loss_d)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = CartoonDataset(args.data_dir, size=256)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=0, pin_memory=True)

    G = DualPathGenerator().to(device)
    D = MultiDiscriminator().to(device)

    vgg        = VGGFeatures().to(device)
    content_fn = ContentLoss(vgg)
    style_fn   = StyleLoss(vgg)
    scaler     = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    opt_G = Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    sched_G = StepLR(opt_G, step_size=50, gamma=0.5)
    sched_D = StepLR(opt_D, step_size=50, gamma=0.5)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Phase 1: Pretrain generator with pixel + content loss only ──
    if args.pretrain_steps > 0:
        print(f"\n=== Pretraining generator for {args.pretrain_steps} steps ===")
        G.train()
        step = 0
        while step < args.pretrain_steps:
            for photos, cartoons in loader:
                if step >= args.pretrain_steps:
                    break
                photos, cartoons = photos.to(device), cartoons.to(device)
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    fake = G(photos)
                    loss = (args.lambda_pixel * pixel_loss(fake, cartoons)
                            + args.lambda_content * content_fn(fake, cartoons))
                opt_G.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(opt_G)
                    scaler.update()
                else:
                    loss.backward(); opt_G.step()
                step += 1
                if step % 200 == 0:
                    print(f"  Pretrain [{step}/{args.pretrain_steps}] loss: {loss.item():.4f}")
        print("Pretraining done.\n")

    # ── Phase 2: GAN training ──
    print("=== GAN Training ===")
    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()

        for i, (photos, cartoons) in enumerate(loader):
            photos, cartoons = photos.to(device), cartoons.to(device)

            # Surface representations (White-Box guided filter)
            surf_real = surface_representation(cartoons)

            # ── Discriminator step ──
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=scaler is not None):
                    fake = G(photos)
            surf_fake = surface_representation(fake.float())

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                real_outs = D(cartoons, surface=surf_real)
                fake_outs = D(fake,     surface=surf_fake)
                loss_D    = adversarial_loss_d(real_outs, fake_outs)

            opt_D.zero_grad()
            if scaler:
                scaler.scale(loss_D).backward(); scaler.step(opt_D); scaler.update()
            else:
                loss_D.backward(); opt_D.step()

            # ── Generator step ──
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                fake      = G(photos)
                surf_fake = surface_representation(fake.float())
                fake_outs = D(fake, surface=surf_fake)

                loss_adv     = adversarial_loss_g(fake_outs)
                loss_pix     = pixel_loss(fake, cartoons)
                loss_content = content_fn(fake, cartoons)
                loss_style   = style_fn(fake, cartoons)
                loss_tv      = total_variation_loss(fake)

                loss_G = (loss_adv
                          + args.lambda_pixel   * loss_pix
                          + args.lambda_content * loss_content
                          + args.lambda_style   * loss_style
                          + args.lambda_tv      * loss_tv)

            opt_G.zero_grad()
            if scaler:
                scaler.scale(loss_G).backward(); scaler.step(opt_G); scaler.update()
            else:
                loss_G.backward(); opt_G.step()

            if i % 10 == 0:
                print(f"Epoch [{epoch}/{args.epochs}] Step [{i}/{len(loader)}] "
                      f"D:{loss_D.item():.4f} G:{loss_G.item():.4f} "
                      f"adv:{loss_adv.item():.4f} pix:{loss_pix.item():.4f} "
                      f"cont:{loss_content.item():.4f} style:{loss_style.item():.5f} "
                      f"tv:{loss_tv.item():.5f}")

        sched_G.step(); sched_D.step()
        torch.save({"epoch": epoch, "G": G.state_dict(), "D": D.state_dict()},
                   os.path.join(args.checkpoint_dir, f"ckpt_epoch{epoch:03d}.pt"))

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str,   required=True)
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--pretrain_steps", type=int,   default=2000)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--lambda_pixel",   type=float, default=100.0)
    parser.add_argument("--lambda_content", type=float, default=10.0)
    parser.add_argument("--lambda_style",   type=float, default=50.0)
    parser.add_argument("--lambda_tv",      type=float, default=1.0)
    args = parser.parse_args()
    train(args)
