"""
HAM10000 (Skin Cancer MNIST) Data Loader.

Dataset characteristics:
  - 10,015 dermoscopic lesion images from ISIC 2018 Task 3
  - 7 classes: akiec, bcc, bkl, df, mel, nv, vasc
  - Highly imbalanced: nv (6705) dominates, df (115) rarest
  - Images: variable resolution (typically 600x450), RGB
  - Pre-split: 70% train / 15% val / 15% test (stratified)
  - Multiple images per lesion_id (same lesion, different angles)

Key design decisions:
  - No mean/std normalization: symbolic formulas operate on raw pixel values [0,1]
  - Single-label (multiclass) classification
  - Stratified splits ensure all 7 classes present in each set
  - Supports configurable resolution for Phase 1 (128x128) vs Phase 2/3 (450x450)
  - Dermoscopic-specific: color channels (RGB/HSV), pigment network, border analysis
"""

import os
import csv
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image
from typing import Optional, List, Dict, Tuple
from collections import Counter


HAM10000_NAMES = [
    'akiec',  # 0: Actinic Keratosis / Bowen's Disease (pre-cancerous)
    'bcc',    # 1: Basal Cell Carcinoma (common skin cancer)
    'bkl',    # 2: Benign Keratosis (seborrheic keratosis, lichen planus)
    'df',     # 3: Dermatofibroma (benign skin nodule)
    'mel',    # 4: Melanoma (most dangerous skin cancer)
    'nv',     # 5: Melanocytic Nevus (common mole, benign)
    'vasc',   # 6: Vascular Lesion (angiomas, angiokeratomas, etc.)
]

HAM10000_FULL_NAMES = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus',
    'vasc': 'Vascular Lesion',
}

# Clinical grouping for hierarchical classification
HAM10000_SUPERCLASS = {
    'malignant': [0, 1, 4],       # akiec, bcc, mel
    'benign': [2, 3, 5, 6],       # bkl, df, nv, vasc
}

SUPERCLASS_NAMES = ['malignant', 'benign']

# ABCD rule mapping for interpretability
ABCD_FEATURES = {
    'A_asymmetry': ['flip_diff', 'rotational_symmetry'],
    'B_border': ['gradient_magnitude', 'edge_density', 'border_sharpness'],
    'C_color': ['color_variety', 'hue_range', 'saturation_stats'],
    'D_diameter': ['area_ratio', 'major_axis', 'compactness'],
}


class HAM10000Dataset(Dataset):
    """HAM10000 skin lesion dataset for multiclass classification."""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        resolution: int = 450,
        augment: bool = False,
    ):
        """
        Args:
            root_dir: Path to datasets directory containing train/val/test subdirs
            split: 'train', 'val', or 'test'
            resolution: Target image resolution
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        self.augment = augment
        self.num_classes = 7

        self.class_to_idx = {name: i for i, name in enumerate(HAM10000_NAMES)}
        self.idx_to_class = {i: name for i, name in enumerate(HAM10000_NAMES)}

        # Load images from split directory
        split_dir = os.path.join(root_dir, split)
        self.samples = []
        self.labels = []

        if os.path.isdir(split_dir):
            for class_name in HAM10000_NAMES:
                class_dir = os.path.join(split_dir, class_name)
                if not os.path.isdir(class_dir):
                    continue
                class_idx = self.class_to_idx[class_name]
                for img_name in sorted(os.listdir(class_dir)):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append(img_path)
                        self.labels.append(class_idx)

        self.labels = np.array(self.labels, dtype=np.int64)

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {split_dir}")

        # Print distribution
        dist = Counter(self.labels.tolist())
        print(f"  HAM10000 {split}: {len(self.samples)} images")
        for i in range(self.num_classes):
            name = HAM10000_NAMES[i]
            count = dist.get(i, 0)
            print(f"    {name:8s} ({HAM10000_FULL_NAMES[name]:30s}): {count:5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)    # [3, H, W]

        if self.augment and self.split == 'train':
            img = self._apply_augmentation(img)

        return img, label

    def _apply_augmentation(self, img):
        """Apply dermoscopy-specific augmentation."""
        # Random horizontal flip
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [2])
        # Random vertical flip
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [1])
        # Random rotation (90 degree increments)
        k = torch.randint(0, 4, (1,)).item()
        img = torch.rot90(img, k, [1, 2])
        # Color jitter (subtle, dermoscopic images are color-critical)
        if torch.rand(1).item() > 0.7:
            noise = torch.randn_like(img) * 0.02
            img = torch.clamp(img + noise, 0, 1)
        return img


class HAM10000DataModule:
    """Data module for HAM10000 with train/val/test splits."""

    def __init__(
        self,
        data_dir: str,
        resolution: int = 450,
        batch_size: int = 64,
        num_workers: int = 4,
        augment: bool = False,
    ):
        self.data_dir = data_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

        self.train_dataset = HAM10000Dataset(
            data_dir, split='train', resolution=resolution, augment=augment,
        )
        self.val_dataset = HAM10000Dataset(
            data_dir, split='val', resolution=resolution, augment=False,
        )
        self.test_dataset = HAM10000Dataset(
            data_dir, split='test', resolution=resolution, augment=False,
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


def build_ham10000_data_batch(
    data_dir: str,
    resolution: int = 128,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = False,
    split: str = 'train',
):
    """Build a data batch for symbolic feature discovery.

    Returns:
        dict with keys:
            'images': [B, 3, H, W] tensor
            'labels': [B] int tensor
            'class_names': list of str
            'num_classes': int
    """
    dataset = HAM10000Dataset(
        data_dir, split=split, resolution=resolution, augment=augment,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )

    images, labels = next(iter(loader))

    return {
        'images': images,
        'labels': labels,
        'class_names': HAM10000_NAMES,
        'num_classes': 7,
    }


def build_ham10000_superclass_mapping():
    """Return superclass mapping for hierarchical classification."""
    return HAM10000_SUPERCLASS, SUPERCLASS_NAMES
