"""
COVIDx CT-3A Data Loader (grayscale chest CT slices).

3 classes: normal / pneumonia / covid. Built from a class-balanced symlink
subset of the official COVIDx CT-3A manifests:
    COVIDx_CT_subset/{train,val,test}/<class>/*.png
Images are grayscale 512x512 CT slices -> loaded as 'L' (consistent with the
grayscale symbolic terminals/operators). Variable-content but uniform size.

Mirrors the ThirdData/Brain loader interface; no val carving (train/val/test
already materialized as folders). val_fraction/seed accepted but ignored.
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from collections import Counter


COVIDX_NAMES = ['normal', 'pneumonia', 'covid']

COVIDX_FULL_NAMES = {
    'normal':    'Normal',
    'pneumonia': 'Pneumonia (non-COVID)',
    'covid':     'COVID-19',
}

# Hierarchical grouping: abnormal (pneumonia+covid) vs. normal.
COVIDX_SUPERCLASS = {'abnormal': [1, 2], 'normal': [0]}
SUPERCLASS_NAMES = ['abnormal', 'normal']

_IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


class COVIDXDataset(Dataset):
    """COVIDx CT-3A grayscale dataset (3-class). split in {train,val,test}."""

    def __init__(self, root_dir, split='train', resolution=224, augment=False,
                 val_fraction=0.15, seed=42, class_subset=None):
        assert split in ('train', 'val', 'test'), f"bad split: {split}"
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        self.augment = augment

        self.active_names = list(class_subset) if class_subset else list(COVIDX_NAMES)
        self.num_classes = len(self.active_names)
        self.class_to_idx = {name: i for i, name in enumerate(self.active_names)}
        self.idx_to_class = {i: name for i, name in enumerate(self.active_names)}

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise RuntimeError(f"Split dir not found: {split_dir}")

        self.samples, self.labels = [], []
        for class_name in self.active_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            class_idx = self.class_to_idx[class_name]
            for img_name in sorted(os.listdir(class_dir)):
                if img_name.startswith('._'):
                    continue
                if img_name.lower().endswith(_IMG_EXTS):
                    self.samples.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)

        self.labels = np.array(self.labels, dtype=np.int64)
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split={split} under {split_dir}")

        dist = Counter(self.labels.tolist())
        print(f"  COVIDx {split}: {len(self.samples)} images")
        for i in range(self.num_classes):
            name = self.active_names[i]
            print(f"    {name:11s} ({COVIDX_FULL_NAMES[name]:22s}): {dist.get(i, 0):5d}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('L')   # grayscale CT
        img = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)            # [1, H, W]
        if self.augment and self.split == 'train':
            img = self._apply_augmentation(img)
        return img, self.labels[idx]

    def _apply_augmentation(self, img):
        """Light aug for axial CT: horizontal flip OK (left/right lung); avoid
        vertical flips that invert anatomy."""
        if torch.rand(1).item() > 0.5:
            img = torch.flip(img, [2])
        if torch.rand(1).item() > 0.7:
            img = torch.clamp(img + torch.randn_like(img) * 0.02, 0, 1)
        return img


class COVIDXDataModule:
    def __init__(self, data_dir, resolution=224, batch_size=64, num_workers=4,
                 augment=False, val_fraction=0.15, seed=42):
        self.train_dataset = COVIDXDataset(data_dir, 'train', resolution, augment)
        self.val_dataset = COVIDXDataset(data_dir, 'val', resolution, False)
        self.test_dataset = COVIDXDataset(data_dir, 'test', resolution, False)
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


def build_covidx_data_batch(data_dir, resolution=128, batch_size=32, num_workers=4,
                            augment=False, split='train', val_fraction=0.15, seed=42):
    dataset = COVIDXDataset(data_dir, split=split, resolution=resolution, augment=augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    images, labels = next(iter(loader))
    return {'images': images, 'labels': labels,
            'class_names': COVIDX_NAMES, 'num_classes': len(COVIDX_NAMES)}


def build_covidx_superclass_mapping():
    return COVIDX_SUPERCLASS, SUPERCLASS_NAMES
