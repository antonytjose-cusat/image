"""
Download and prepare dataset for training.

Two modes:
  1. paired   -- dqymaggie/cartoonizer-dataset (paired photo+cartoon, default)
  2. unpaired -- FFHQ photos + anime cartoon frames (like White-Box paper)

Usage:
  python prepare_dataset.py --mode paired   --output_dir data
  python prepare_dataset.py --mode unpaired --output_dir data
"""
import argparse
import os
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def prepare_folders(output_dir, mode):
    if mode == "paired":
        for split in ("train", "test"):
            for sub in ("photos", "cartoons"):
                Path(f"{output_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)
    else:
        for sub in ("photos", "cartoons"):
            Path(f"{output_dir}/train/{sub}").mkdir(parents=True, exist_ok=True)
            Path(f"{output_dir}/test/{sub}").mkdir(parents=True, exist_ok=True)


def to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    import numpy as np
    return Image.fromarray(img).convert("RGB")


def extract_pair(sample):
    key_map = [
        ("original_image", "cartoonized_image"),
        ("image", "cartoon"),
        ("photo", "cartoon"),
        ("input", "output"),
    ]
    for pk, ck in key_map:
        if pk in sample and ck in sample:
            return sample[pk], sample[ck]
    img_keys = [k for k, v in sample.items() if isinstance(v, Image.Image)]
    if len(img_keys) >= 2:
        return sample[img_keys[0]], sample[img_keys[1]]
    raise ValueError(f"Cannot find photo/cartoon columns. Keys: {list(sample.keys())}")


# ─────────────────────────────────────────────
# Mode 1: Paired dataset (dqymaggie)
# ─────────────────────────────────────────────

def prepare_paired(output_dir):
    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install datasets")
        from datasets import load_dataset

    print("Downloading dqymaggie/cartoonizer-dataset ...")
    dataset = load_dataset("dqymaggie/cartoonizer-dataset")

    splits = list(dataset.keys())
    print(f"Available splits: {splits}")
    print(f"Columns: {list(dataset[splits[0]][0].keys())}")

    all_data = dataset[splits[0]]
    split_idx = int(len(all_data) * 0.8)

    def save_split(data, split_dir, label):
        print(f"\nSaving {label} ({len(data)} samples)...")
        for idx, sample in enumerate(tqdm(data, desc=label)):
            photo, cartoon = extract_pair(sample)
            fname = f"{idx:05d}.jpg"
            to_pil(photo).save(f"{split_dir}/photos/{fname}")
            to_pil(cartoon).save(f"{split_dir}/cartoons/{fname}")

    save_split(all_data.select(range(split_idx)),             f"{output_dir}/train", "train")
    save_split(all_data.select(range(split_idx, len(all_data))), f"{output_dir}/test",  "test")


# ─────────────────────────────────────────────
# Mode 2: Unpaired dataset (like White-Box paper)
# Photos: FFHQ thumbnails (publicly available, 70k faces)
# Cartoons: Anime face dataset (publicly available)
# ─────────────────────────────────────────────

def download_file(url, dest):
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest, reporthook=lambda b, bs, t:
        print(f"\r  {min(b*bs, t)}/{t} bytes", end="", flush=True))
    print()


def prepare_unpaired(output_dir):
    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install datasets")
        from datasets import load_dataset

    # ── Real photos: FFHQ 256x256 thumbnails from HuggingFace ──
    print("\nDownloading FFHQ photos (real faces)...")
    ffhq = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True,
                        trust_remote_code=True)

    # Fallback: use a smaller publicly available face dataset
    print("Loading real photos from datasets...")
    try:
        photos_ds = load_dataset("nielsr/ffhq-1024", split="train", streaming=True)
        photo_source = "ffhq"
    except Exception:
        # Use CelebA-HQ as fallback
        photos_ds = load_dataset("mattmdjaga/human_parsing_dataset", split="train", streaming=True)
        photo_source = "celeba"

    print(f"Using photo source: {photo_source}")
    photo_dir = f"{output_dir}/train/photos"
    count = 0
    for sample in tqdm(photos_ds, desc="Saving photos", total=4000):
        if count >= 4000:
            break
        try:
            img = sample.get("image") or sample.get("img") or list(sample.values())[0]
            to_pil(img).resize((256, 256)).save(f"{photo_dir}/{count:05d}.jpg")
            count += 1
        except Exception:
            continue

    # ── Cartoon images: anime face dataset ──
    print("\nDownloading anime cartoon images...")
    try:
        anime_ds = load_dataset("animelover1984/anime-images", split="train", streaming=True)
    except Exception:
        anime_ds = load_dataset("jlbaker361/anime_faces_dim_64", split="train", streaming=True)

    cartoon_dir = f"{output_dir}/train/cartoons"
    count = 0
    for sample in tqdm(anime_ds, desc="Saving cartoons", total=4000):
        if count >= 4000:
            break
        try:
            img = sample.get("image") or sample.get("img") or list(sample.values())[0]
            to_pil(img).resize((256, 256)).save(f"{cartoon_dir}/{count:05d}.jpg")
            count += 1
        except Exception:
            continue

    # Copy a subset to test
    print("\nCreating test split from last 20% of train...")
    for sub in ("photos", "cartoons"):
        src = f"{output_dir}/train/{sub}"
        dst = f"{output_dir}/test/{sub}"
        files = sorted(os.listdir(src))
        split = int(len(files) * 0.8)
        for f in tqdm(files[split:], desc=f"test/{sub}"):
            img = Image.open(f"{src}/{f}")
            img.save(f"{dst}/{f}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print(f"Dataset Preparation — mode: {args.mode}")
    print("=" * 60)

    prepare_folders(args.output_dir, args.mode)

    if args.mode == "paired":
        prepare_paired(args.output_dir)
    else:
        prepare_unpaired(args.output_dir)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Train: {args.output_dir}/train/")
    print(f"  Test:  {args.output_dir}/test/")
    print("=" * 60)
    print(f"\nNext: python train.py --data_dir {args.output_dir}/train --epochs 100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--mode", type=str, default="paired",
                        choices=["paired", "unpaired"],
                        help="paired: dqymaggie dataset | unpaired: FFHQ + anime (like White-Box)")
    args = parser.parse_args()
    main(args)
