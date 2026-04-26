"""
Train all 3 model variants sequentially and compare results.

Usage:
  python train_all.py --data_dir data_anime
"""
import argparse
import os
import subprocess
import sys


def run(cmd):
    print(f"\n>>> {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        sys.exit(1)


def main(args):
    env = "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    py  = sys.executable

    # ── V1: Baseline (pixel + content + style) ──
    print("\n" + "="*60)
    print("TRAINING V1 — Baseline Dual-Path GAN")
    print("="*60)
    run(f"{env} {py} train.py "
        f"--data_dir {args.data_dir}/train "
        f"--checkpoint_dir checkpoints_v1 "
        f"--pretrain_steps 2000 "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--lambda_pixel 100 --lambda_content 10 "
        f"--lambda_style 50 --lambda_tv 1.0 "
        f"--lambda_edge 0 --lambda_cycle 0")

    # ── V2: + stronger style + TV ──
    print("\n" + "="*60)
    print("TRAINING V2 — Stronger Style + TV Loss")
    print("="*60)
    run(f"{env} {py} train.py "
        f"--data_dir {args.data_dir}/train "
        f"--checkpoint_dir checkpoints_v2 "
        f"--pretrain_steps 2000 "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--lambda_pixel 100 --lambda_content 10 "
        f"--lambda_style 100 --lambda_tv 1.0 "
        f"--lambda_edge 0 --lambda_cycle 0")

    # ── V3: + edge loss + cycle consistency ──
    print("\n" + "="*60)
    print("TRAINING V3 — Edge Loss + Cycle Consistency")
    print("="*60)
    run(f"{env} {py} train.py "
        f"--data_dir {args.data_dir}/train "
        f"--checkpoint_dir checkpoints_v3 "
        f"--pretrain_steps 2000 "
        f"--epochs {args.epochs} "
        f"--batch_size {args.batch_size} "
        f"--lambda_pixel 100 --lambda_content 10 "
        f"--lambda_style 100 --lambda_tv 1.0 "
        f"--lambda_edge 10 --lambda_cycle 10")

    # ── Compare all 3 ──
    print("\n" + "="*60)
    print("COMPARING ALL MODELS")
    print("="*60)

    ckpt_v1 = f"checkpoints_v1/ckpt_epoch{args.epochs:03d}.pt"
    ckpt_v2 = f"checkpoints_v2/ckpt_epoch{args.epochs:03d}.pt"
    ckpt_v3 = f"checkpoints_v3/ckpt_epoch{args.epochs:03d}.pt"

    run(f"{py} compare_models.py "
        f"--checkpoints {ckpt_v1} {ckpt_v2} {ckpt_v3} "
        f"--labels V1-Baseline V2-StrongerStyle V3-Edge+Cycle "
        f"--data_dir {args.data_dir}/test "
        f"--output_dir comparison_anime "
        f"--n_samples 8")

    print("\n" + "="*60)
    print("ALL DONE!")
    print(f"  Checkpoints: checkpoints_v1/ v2/ v3/")
    print(f"  Results:     comparison_anime/comparison_grid.png")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
