import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_transform(size=256):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class CartoonDataset(Dataset):
    """
    Paired dataset:
      root/photos/   - real photos
      root/cartoons/ - paired cartoon images (same filenames)
    """
    def __init__(self, root, size=256):
        self.photo_dir   = os.path.join(root, "photos")
        self.cartoon_dir = os.path.join(root, "cartoons")
        self.files     = sorted(os.listdir(self.photo_dir))
        self.transform = get_transform(size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name    = self.files[idx]
        photo   = Image.open(os.path.join(self.photo_dir,   name)).convert("RGB")
        cartoon = Image.open(os.path.join(self.cartoon_dir, name)).convert("RGB")
        return self.transform(photo), self.transform(cartoon)


class UnpairedDataset(Dataset):
    """
    Unpaired dataset (like White-Box paper):
      root/photos/   - real photos  (any filenames)
      root/cartoons/ - cartoon images (different filenames, different count ok)
    Photos and cartoons are sampled independently.
    """
    def __init__(self, root, size=256):
        self.photo_dir   = os.path.join(root, "photos")
        self.cartoon_dir = os.path.join(root, "cartoons")
        self.photos   = sorted(os.listdir(self.photo_dir))
        self.cartoons = sorted(os.listdir(self.cartoon_dir))
        self.transform = get_transform(size)

    def __len__(self):
        # Length is the larger of the two sets
        return max(len(self.photos), len(self.cartoons))

    def __getitem__(self, idx):
        photo   = Image.open(os.path.join(self.photo_dir,
                    self.photos[idx % len(self.photos)])).convert("RGB")
        cartoon = Image.open(os.path.join(self.cartoon_dir,
                    self.cartoons[idx % len(self.cartoons)])).convert("RGB")
        return self.transform(photo), self.transform(cartoon)
