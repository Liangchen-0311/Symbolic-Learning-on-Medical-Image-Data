"""
Brain Tumor MRI Data Loader (grayscale).

Dataset characteristics:
  - Brain Tumor MRI Dataset (Kaggle, Msoud Nickparvar version)
  - 4 classes: glioma, meningioma, pituitary, notumor
  - On-disk layout: <root>/Training/<class>/*.jpg and <root>/Testing/<class>/*.jpg
  - Class-balanced: Training ~1400/class (5600 total), Testing 400/class (1600).
  - Images: single-channel grayscale (PIL mode 'L'), typically 512x512.

Key design decisions (mirrors ham10000_loader, adapted for grayscale):
  - Images are loaded as GRAYSCALE [1, H, W] in [0, 1]. No mean/std normalization
    (symbolic formulas operate on raw intensity).
  - The dataset ships only Training/ and Testing/. We carve a validation split
    out of Training/ with a deterministic, per-class (stratified) hold-out so
    train/val never leak; Testing/ is used untouched as the test set.
  - Single-label (multiclass) classification, 4 classes.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import Counter


# Class order is fixed here (NOT alphabetical) so indices are stable and readable.
BRAIN_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']

BRAIN_FULL_NAMES = {
    'glioma':     'Glioma tumor',
    'meningioma': 'Meningioma tumor',
    'pituitary':  'Pituitary tumor',
    'notumor':    'No tumor',
}

# Hierarchical grouping: tumor vs. no-tumor (used by hierarchical RL evaluation).
BRAIN_SUPERCLASS = {
    'tumor':   [0, 1, 2],   # glioma, meningioma, pituitary
    'notumor': [3],         # notumor
}

SUPERCLASS_NAMES = ['tumor', 'notumor']

_IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


class BrainTumorDataset(Dataset):
    """Brain Tumor MRI dataset (grayscale) for 4-class classification.

    split:
      'train' / 'val' -> sampled from <root>/Training with a deterministic,
                         per-class stratified hold-out (val_fraction).
      'test'          -> all images under <root>/Testing.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        resolution: int = 224,
        augment: bool = False,
        val_fraction: float = 0.15,
        seed: int = 42,
        class_subset=None,
    ):
        assert split in ('train', 'val', 'test'), f"bad split: {split}"
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        self.augment = augment

        # class_subset (list of class names) restricts to those classes and
        # remaps labels to contiguous 0..k-1 in subset order — used for focused
        # binary/sub-problem RL searches (e.g. glioma vs meningioma).
        self.active_names = list(class_subset) if class_subset else list(BRAIN_NAMES)
        self.num_classes = len(self.active_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.active_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.active_names)}

        disk_split = 'Testing' if split == 'test' else 'Training'
        split_dir = os.path.join(root_dir, disk_split)

        self.samples = []
        self.labels = []

        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Split dir not found: {split_dir}")

        rng = np.random.RandomState(seed)
        for class_name in self.active_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = self.class_to_idx[class_name]
            files = sorted(
                f for f in os.listdir(class_dir)
                if f.lower().endswith(_IMG_EXTS)
            )

            if split in ('train', 'val'):
                # Deterministic per-class stratified split of Training/.
                idx = np.arange(len(files))
                perm = rng.permutation(idx)          # seed depends only on `seed`
                n_val = int(round(len(files) * val_fraction))
                val_idx = set(perm[:n_val].tolist())
                keep = (
                    [i for i in range(len(files)) if i in val_idx]
                    if split == 'val'
                    else [i for i in range(len(files)) if i not in val_idx]
                )
                files = [files[i] for i in keep]

            for img_name in files:
                self.samples.append(os.path.join(class_dir, img_name))
                self.labels.append(class_idx)

        self.labels = np.array(self.labels, dtype=np.int64)
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split={split} under {split_dir}")

        dist = Counter(self.labels.tolist())
        print(f"  BrainTumor {split}: {len(self.samples)} images")
        for i in range(self.num_classes):
            name = self.active_names[i]
            print(f"    {name:11s} ({BRAIN_FULL_NAMES[name]:18s}): {dist.get(i, 0):5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('L')   # grayscale
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0      # [H, W] in [0, 1]
        img = torch.from_numpy(img).unsqueeze(0)           # [1, H, W]

        if self.augment and self.split == 'train':
            img = self._apply_augmentation(img)

        return img, label

    def _apply_augmentation(self, img):
        """Light augmentation. MRI orientation is meaningful, so we avoid
        vertical flips / 90-degree rotations (those create anatomically
        implausible brains and would also corrupt left-right symmetry cues)."""
        # Horizontal flip is fine — brains are roughly bilaterally symmetric.
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [2])
        # Small additive noise.
        if torch.rand(1).item() > 0.7:
            img = torch.clamp(img + torch.randn_like(img) * 0.02, 0, 1)
        return img


class BrainTumorDataModule:
    """train/val/test data module for the Brain Tumor MRI dataset."""

    def __init__(
        self,
        data_dir: str,
        resolution: int = 224,
        batch_size: int = 64,
        num_workers: int = 4,
        augment: bool = False,
        val_fraction: float = 0.15,
        seed: int = 42,
    ):
        self.data_dir = data_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = BrainTumorDataset(
            data_dir, split='train', resolution=resolution, augment=augment,
            val_fraction=val_fraction, seed=seed,
        )
        self.val_dataset = BrainTumorDataset(
            data_dir, split='val', resolution=resolution, augment=False,
            val_fraction=val_fraction, seed=seed,
        )
        self.test_dataset = BrainTumorDataset(
            data_dir, split='test', resolution=resolution, augment=False,
            val_fraction=val_fraction, seed=seed,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers, pin_memory=True,
        )


def build_brain_data_batch(
    data_dir: str,
    resolution: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = False,
    split: str = 'train',
    val_fraction: float = 0.15,
    seed: int = 42,
):
    """Build a single data batch (used by Step 0 dataset validation)."""
    dataset = BrainTumorDataset(
        data_dir, split=split, resolution=resolution, augment=augment,
        val_fraction=val_fraction, seed=seed,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    images, labels = next(iter(loader))
    return {
        'images': images,
        'labels': labels,
        'class_names': BRAIN_NAMES,
        'num_classes': len(BRAIN_NAMES),
    }


def build_brain_superclass_mapping():
    """Return (superclass_mapping_dict, superclass_names) for hierarchical eval."""
    return BRAIN_SUPERCLASS, SUPERCLASS_NAMES
