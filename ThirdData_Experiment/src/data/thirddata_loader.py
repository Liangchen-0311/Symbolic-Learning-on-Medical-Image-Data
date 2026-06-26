"""
ThirdData / BUSI Breast Ultrasound Data Loader (grayscale).

Dataset characteristics:
  - Breast Ultrasound Images (BUSI-style): 3 classes benign / malignant / normal.
  - On-disk layout already split: <root>/{train,val,test}/<class>/*.png
  - Class-IMBALANCED (e.g. train benign 310 / malignant 148 / normal 87).
  - Images stored as RGB but content is grayscale B-mode ultrasound; we load as
    grayscale 'L' (consistent with the grayscale symbolic terminals/operators).
  - Variable resolution -> resized to a fixed square.

Mirrors the Brain loader's interface so the shared pipeline works unchanged,
except there is NO val carving here (val split already exists on disk); the
`val_fraction` / `seed` args are accepted but ignored.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import Counter


# Class order is fixed (indices stable). benign=0, malignant=1, normal=2.
THIRD_NAMES = ['benign', 'malignant', 'normal']

THIRD_FULL_NAMES = {
    'benign':    'Benign tumor',
    'malignant': 'Malignant tumor',
    'normal':    'Normal tissue',
}

# Hierarchical grouping: lesion (benign+malignant) vs. normal, for the
# hierarchical RL evaluation in early training.
THIRD_SUPERCLASS = {
    'lesion': [0, 1],   # benign, malignant
    'normal': [2],      # normal
}

SUPERCLASS_NAMES = ['lesion', 'normal']

_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


class ThirdDataDataset(Dataset):
    """BUSI breast-ultrasound dataset (grayscale), 3-class.

    split in {'train','val','test'} maps directly to the on-disk folder.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        resolution: int = 224,
        augment: bool = False,
        val_fraction: float = 0.15,   # accepted for interface parity; ignored
        seed: int = 42,               # accepted for interface parity; ignored
        class_subset=None,
    ):
        assert split in ('train', 'val', 'test'), f"bad split: {split}"
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        self.augment = augment

        self.active_names = list(class_subset) if class_subset else list(THIRD_NAMES)
        self.num_classes = len(self.active_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.active_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.active_names)}

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Split dir not found: {split_dir}")

        self.samples = []
        self.labels = []
        for class_name in self.active_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = self.class_to_idx[class_name]
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.startswith('._'):
                    continue
                if img_name.lower().endswith(_IMG_EXTS) and 'mask' not in img_name.lower():
                    self.samples.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

        self.labels = np.array(self.labels, dtype=np.int64)
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split={split} under {split_dir}")

        dist = Counter(self.labels.tolist())
        print(f"  ThirdData {split}: {len(self.samples)} images")
        for i in range(self.num_classes):
            name = self.active_names[i]
            print(f"    {name:11s} ({THIRD_FULL_NAMES[name]:18s}): {dist.get(i, 0):5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('L')   # grayscale ultrasound
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0      # [H, W] in [0, 1]
        img = torch.from_numpy(img).unsqueeze(0)           # [1, H, W]

        if self.augment and self.split == 'train':
            img = self._apply_augmentation(img)

        return img, label

    def _apply_augmentation(self, img):
        """Light augmentation for ultrasound. Horizontal flip is fine; avoid
        vertical flips/rotations that create implausible scan geometry."""
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [2])
        if torch.rand(1).item() > 0.7:
            img = torch.clamp(img + torch.randn_like(img) * 0.02, 0, 1)
        return img


class ThirdDataDataModule:
    """train/val/test data module for ThirdData/BUSI."""

    def __init__(self, data_dir, resolution=224, batch_size=64, num_workers=4,
                 augment=False, val_fraction=0.15, seed=42):
        self.train_dataset = ThirdDataDataset(data_dir, 'train', resolution, augment)
        self.val_dataset = ThirdDataDataset(data_dir, 'val', resolution, False)
        self.test_dataset = ThirdDataDataset(data_dir, 'test', resolution, False)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


def build_third_data_batch(data_dir, resolution=128, batch_size=32, num_workers=4,
                           augment=False, split='train', val_fraction=0.15, seed=42):
    """Build one data batch (used by Step 0 dataset validation)."""
    dataset = ThirdDataDataset(data_dir, split=split, resolution=resolution, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    images, labels = next(iter(loader))
    return {'images': images, 'labels': labels,
            'class_names': THIRD_NAMES, 'num_classes': len(THIRD_NAMES)}


def build_third_superclass_mapping():
    """Return (superclass_mapping_dict, superclass_names) for hierarchical eval."""
    return THIRD_SUPERCLASS, SUPERCLASS_NAMES
