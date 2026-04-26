"""
Download instruction-tuning-sd/cartoonization dataset (5K paired images).
This has stronger cartoon stylization than dqymaggie.

Usage:
  python download_new_dataset.py --output_dir data_new
"""
import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    import numpy as np
    return Image.fromarray(img).convert("RGB")


def extract_pair(sample):
    key_map = [
        ("input_image", "edit_image"),
        ("original_image", "cartoonized_image"),
        ("image", "cartoon"),
        ("input", "output"),
    ]
    for pk, ck in key_map:
        if pk in sample and ck in sample:
            return sample[pk], sample[ck]
    img_keys = [k for k, v in sample.items() if isinstance(v, Image.Image)]
    if len(img_keys) >= 2:
        return sample[img_keys[0]], sample[img_keys[1]]
    raise ValueError(f"Cannot find columns. Keys: {list(sample.keys())}")


def main(args):
    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install datasets")
        from datasets import load_dataset

    for split in ("train", "test"):
        for sub in ("photos", "cartoons"):
            Path(f"{args.output_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)

    print("Downloading instruction-tuning-sd/cartoonization ...")
    ds = load_dataset("instruction-tuning-sd/cartoonization", split="train")
    print(f"Total: {len(ds)} samples")
    print(f"Columns: {ds.column_names}")

    split_idx = int(len(ds) * 0.8)

    print(f"\nSaving train ({split_idx} samples)...")
    for idx in tqdm(range(split_idx)):
        sample = ds[idx]
        photo, cartoon = extract_pair(sample)
        to_pil(photo).resize((256, 256)).save(f"{args.output_dir}/train/photos/{idx:05d}.jpg")
        to_pil(cartoon).resize((256, 256)).save(f"{args.output_dir}/train/cartoons/{idx:05d}.jpg")

    print(f"\nSaving test ({len(ds) - split_idx} samples)...")
    for idx in tqdm(range(split_idx, len(ds))):
        sample = ds[idx]
        photo, cartoon = extract_pair(sample)
        tidx = idx - split_idx
        to_pil(photo).resize((256, 256)).save(f"{args.output_dir}/test/photos/{tidx:05d}.jpg")
        to_pil(cartoon).resize((256, 256)).save(f"{args.output_dir}/test/cartoons/{tidx:05d}.jpg")

    print(f"\nDone! Dataset at {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data_new")
    args = parser.parse_args()
    main(args)
