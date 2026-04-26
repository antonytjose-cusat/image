"""
Download anime faces (huggan/anime-faces) as cartoon images
and use existing real photos as the photo domain.

Usage:
  python prepare_anime_dataset.py --output_dir data_anime
"""
import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def prepare_folders(output_dir):
    for split in ("train", "test"):
        for sub in ("photos", "cartoons"):
            Path(f"{output_dir}/{split}/{sub}").mkdir(parents=True, exist_ok=True)


def to_pil(img):
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    import numpy as np
    return Image.fromarray(img).convert("RGB")


def main(args):
    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install datasets")
        from datasets import load_dataset

    prepare_folders(args.output_dir)

    # ── Download anime faces as cartoon domain ──
    print("Downloading huggan/anime-faces (21,551 anime images)...")
    anime_ds = load_dataset("huggan/anime-faces", split="train")
    print(f"Total anime images: {len(anime_ds)}")

    # Use 8000 for train, 2000 for test
    train_cartoon = anime_ds.select(range(8000))
    test_cartoon  = anime_ds.select(range(8000, 10000))

    print("\nSaving cartoon (anime) images...")
    for idx, sample in enumerate(tqdm(train_cartoon, desc="train/cartoons")):
        img = to_pil(sample["image"]).resize((256, 256))
        img.save(f"{args.output_dir}/train/cartoons/{idx:05d}.jpg")

    for idx, sample in enumerate(tqdm(test_cartoon, desc="test/cartoons")):
        img = to_pil(sample["image"]).resize((256, 256))
        img.save(f"{args.output_dir}/test/cartoons/{idx:05d}.jpg")

    # ── Use existing real photos or download FFHQ ──
    if args.photo_dir and os.path.exists(args.photo_dir):
        print(f"\nCopying real photos from {args.photo_dir}...")
        import shutil
        photos = sorted(os.listdir(args.photo_dir))
        train_photos = photos[:8000]
        test_photos  = photos[8000:10000] if len(photos) > 8000 else photos[:2000]

        for idx, fname in enumerate(tqdm(train_photos, desc="train/photos")):
            src = os.path.join(args.photo_dir, fname)
            dst = f"{args.output_dir}/train/photos/{idx:05d}.jpg"
            img = Image.open(src).convert("RGB").resize((256, 256))
            img.save(dst)

        for idx, fname in enumerate(tqdm(test_photos, desc="test/photos")):
            src = os.path.join(args.photo_dir, fname)
            dst = f"{args.output_dir}/test/photos/{idx:05d}.jpg"
            img = Image.open(src).convert("RGB").resize((256, 256))
            img.save(dst)
    else:
        # Download CelebA-HQ faces as real photo domain
        print("\nDownloading real face photos (FFHQ subset)...")
        try:
            photo_ds = load_dataset("nielsr/ffhq-1024", split="train", streaming=True)
        except Exception:
            photo_ds = load_dataset("Andyrasika/celeba-hq-1024-HF",
                                    split="train", streaming=True)

        train_count, test_count = 0, 0
        for sample in tqdm(photo_ds, desc="Downloading photos", total=10000):
            try:
                img_key = "image" if "image" in sample else list(sample.keys())[0]
                img = to_pil(sample[img_key]).resize((256, 256))
                if train_count < 8000:
                    img.save(f"{args.output_dir}/train/photos/{train_count:05d}.jpg")
                    train_count += 1
                elif test_count < 2000:
                    img.save(f"{args.output_dir}/test/photos/{test_count:05d}.jpg")
                    test_count += 1
                else:
                    break
            except Exception:
                continue

    print("\n" + "="*60)
    print("Anime dataset ready!")
    print(f"  Train: {args.output_dir}/train/  (photos + cartoons)")
    print(f"  Test:  {args.output_dir}/test/")
    print("="*60)
    print(f"\nNext: python train_all.py --data_dir {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",  type=str, default="data_anime")
    parser.add_argument("--photo_dir",   type=str, default=None,
                        help="Optional: path to existing real photos folder")
    args = parser.parse_args()
    main(args)
