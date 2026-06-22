"""
HBFMID (Human Bone Fractures Multi-modal Image Dataset) Data Loader.

Dataset characteristics:
  - YOLO-format bounding box annotations (class x_center y_center width height)
  - 10 fracture types: Comminuted, Greenstick, Healthy, Linear, Oblique Displaced,
                       Oblique, Segmental, Spiral, Transverse Displaced, Transverse
  - 640x640 RGB X-ray images
  - Imbalanced classes (Class 8 Transverse: 630 vs Class 6 Segmental: 18)
  - Multi-label: 189/1347 images have multiple fracture annotations

Key design decisions:
  - No mean/std normalization: symbolic formulas operate on raw pixel values [0,1]
  - Supports configurable resolution for Phase 1 (128x128) vs Phase 2/3 (640x640)
  - Image-level labels derived from YOLO boxes (multi-hot for multi-label)
  - Stratified sampling for balanced class representation
  - ROI cropping: extract fracture regions as additional terminals
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from PIL import Image
from typing import Optional, Dict, List, Tuple
from collections import Counter


FRACTURE_NAMES = [
    'Comminuted',        # 0: 粉碎性骨折
    'Greenstick',        # 1: 青枝骨折
    'Healthy',           # 2: 健康
    'Linear',            # 3: 线性骨折
    'Oblique Displaced', # 4: 斜形移位骨折
    'Oblique',           # 5: 斜形骨折
    'Segmental',         # 6: 节段性骨折
    'Spiral',            # 7: 螺旋骨折
    'Transverse Displaced', # 8: 横形移位骨折
    'Transverse',        # 9: 横形骨折
]

FRACTURE_SUPERCLASS = {
    'simple': [2, 3, 5, 9],
    'displaced': [0, 4, 8],
    'complex': [1, 6, 7],
}

SUPERCLASS_NAMES = ['simple', 'displaced', 'complex']


class HBFMIDDataset(Dataset):
    """HBFMID fracture detection dataset with YOLO-format annotations."""

    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        resolution: int = 640,
        augment: bool = False,
        task: str = 'classification',
    ):
        """
        Args:
            root_dir: Path to 'Bone Fractures Detection' directory
            split: 'train', 'valid', or 'test'
            resolution: Target image resolution
            augment: Whether to apply data augmentation
            task: 'classification' (image-level) or 'detection' (with boxes)
        """
        self.root_dir = root_dir
        self.split = split
        self.resolution = resolution
        self.augment = augment
        self.task = task
        self.num_classes = 10

        # Handle 'val' vs 'valid' directory naming
        actual_split = split
        if split == 'val' and not os.path.isdir(os.path.join(root_dir, split)):
            if os.path.isdir(os.path.join(root_dir, 'valid')):
                actual_split = 'valid'

        self.image_dir = os.path.join(root_dir, actual_split, 'images')
        self.label_dir = os.path.join(root_dir, actual_split, 'labels')

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

        if augment and split == 'train':
            tf_list = [
                transforms.RandomResizedCrop(resolution, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
            ]
        else:
            tf_list = [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        self.transform = transforms.Compose(tf_list)

        self._labels_cache = {}
        self._build_label_cache()

    def _build_label_cache(self):
        for img_file in self.image_files:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_file)
            boxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f.read().strip().split('\n'):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            x_c, y_c, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            boxes.append((cls_id, x_c, y_c, w, h))
            self._labels_cache[img_file] = boxes

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (self.resolution, self.resolution), (0, 0, 0))

        orig_w, orig_h = image.size
        image = self.transform(image)

        boxes = self._labels_cache[img_file]

        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for cls_id, _, _, _, _ in boxes:
            label[cls_id] = 1.0

        if self.task == 'classification':
            primary_class = label.argmax().item() if label.sum() > 0 else 2
            return image, primary_class
        else:
            return image, label, boxes

    def get_class_distribution(self):
        counter = Counter()
        for img_file in self.image_files:
            for cls_id, _, _, _, _ in self._labels_cache[img_file]:
                counter[cls_id] += 1
        return counter


class HBFMIDDataModule:
    """Data module for HBFMID with one-time stratified split (v2).

    v2 核心策略：
    - 合并所有原始数据，做一次分层划分（70/15/15），保存到 split_indices.npz
    - 后续所有步骤（RL搜索、特征提取、分类器）都使用同一份划分
    - RL 搜索只看 train split → 无数据泄露
    - 测试集类别丰富（每类至少 2-3 张）→ 评估更可靠
    """

    def __init__(
        self,
        data_dir: str,
        resolution: int = 640,
        batch_size: int = 32,
        num_workers: int = 4,
        augment: bool = False,
        samples_per_class: Optional[int] = None,
        task: str = 'classification',
        split_file: Optional[str] = None,
    ):
        self.data_dir = data_dir
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment
        self.samples_per_class = samples_per_class
        self.task = task
        self.split_file = split_file

        self.num_classes = 10
        self.input_channels = 3
        self.image_size = resolution

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        train_ds = HBFMIDDataset(
            self.data_dir, split='train', resolution=self.resolution,
            augment=self.augment, task=self.task,
        )
        val_ds = HBFMIDDataset(
            self.data_dir, split='val', resolution=self.resolution,
            augment=False, task=self.task,
        )
        test_ds = HBFMIDDataset(
            self.data_dir, split='test', resolution=self.resolution,
            augment=False, task=self.task,
        )

        all_images = []
        all_labels = []
        for ds in [train_ds, val_ds, test_ds]:
            for i in range(len(ds)):
                img, lbl = ds[i]
                all_images.append(img)
                all_labels.append(lbl)
        all_labels = np.array(all_labels)
        n_total = len(all_labels)
        print(f"[HBFMID] Total images across all splits: {n_total}")

        present_classes = sorted(set(all_labels.tolist()))
        active_classes = [c for c in present_classes if (all_labels == c).sum() >= 3]
        inactive_classes = [c for c in present_classes if c not in active_classes]
        if inactive_classes:
            print(f"[HBFMID] Dropping classes with <3 samples: {[FRACTURE_NAMES[c] for c in inactive_classes]}")

        keep_mask = np.isin(all_labels, active_classes)
        keep_indices = np.where(keep_mask)[0]
        all_labels = all_labels[keep_indices]
        all_images = [all_images[i] for i in keep_indices]
        self.num_classes = len(active_classes)
        self._active_classes = active_classes
        self._class_map = {c: i for i, c in enumerate(active_classes)}
        remapped = np.array([self._class_map[l] for l in all_labels])
        print(f"[HBFMID] Active classes ({self.num_classes}): {[FRACTURE_NAMES[c] for c in active_classes]}")

        if self.split_file and os.path.exists(self.split_file):
            split_data = np.load(self.split_file, allow_pickle=True)
            train_idx = split_data['train_idx'].tolist()
            val_idx = split_data['val_idx'].tolist()
            test_idx = split_data['test_idx'].tolist()
            print(f"[HBFMID-v2] Loaded existing split from {self.split_file}")
        else:
            rng = np.random.RandomState(42)
            train_idx, val_idx, test_idx = [], [], []
            for cls_id in active_classes:
                mapped_id = self._class_map[cls_id]
                cls_mask = remapped == mapped_id
                cls_indices = np.where(cls_mask)[0]
                n = len(cls_indices)
                rng.shuffle(cls_indices)
                n_test = max(2, n // 7)
                n_val = max(2, n // 7)
                n_train = n - n_test - n_val
                test_idx.extend(cls_indices[:n_test].tolist())
                val_idx.extend(cls_indices[n_test:n_test + n_val].tolist())
                train_idx.extend(cls_indices[n_test + n_val:].tolist())
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            rng.shuffle(test_idx)

            if self.split_file:
                np.savez(
                    self.split_file,
                    train_idx=np.array(train_idx),
                    val_idx=np.array(val_idx),
                    test_idx=np.array(test_idx),
                    active_classes=np.array(active_classes),
                )
                print(f"[HBFMID-v2] Saved split to {self.split_file}")

        class _SplitDataset(Dataset):
            def __init__(self, images, indices, labels, num_classes):
                self.images = [images[i] for i in indices]
                self.labels = [labels[i] for i in indices]
                self.num_classes = num_classes
            def __len__(self):
                return len(self.images)
            def __getitem__(self, idx):
                return self.images[idx], self.labels[idx]

        self.train_dataset = _SplitDataset(all_images, train_idx, remapped, self.num_classes)
        self.val_dataset = _SplitDataset(all_images, val_idx, remapped, self.num_classes)
        self.test_dataset = _SplitDataset(all_images, test_idx, remapped, self.num_classes)

        if self.samples_per_class is not None:
            self.train_dataset = self._stratified_subset(self.train_dataset, self.samples_per_class)

        print(f"[HBFMID-v2] One-time stratified split (70/15/15), shared across all steps")
        print(f"[HBFMID] Train: {len(self.train_dataset)} images")
        print(f"[HBFMID] Val: {len(self.val_dataset)} images")
        print(f"[HBFMID] Test: {len(self.test_dataset)} images")
        print(f"[HBFMID] Resolution: {self.resolution}x{self.resolution}")

        for split_name, ds in [('Train', self.train_dataset), ('Val', self.val_dataset), ('Test', self.test_dataset)]:
            cnt = Counter()
            for i in range(len(ds)):
                _, lbl = ds[i]
                cnt[int(lbl)] += 1
            print(f"[HBFMID] {split_name} class distribution: {dict(sorted(cnt.items()))}")

    def _stratified_subset(self, dataset, samples_per_class):
        targets = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            targets.append(label)
        targets = np.array(targets)

        indices = []
        rng = np.random.RandomState(42)
        for cls_id in range(self.num_classes):
            cls_indices = np.where(targets == cls_id)[0]
            if len(cls_indices) > samples_per_class:
                chosen = rng.choice(cls_indices, size=samples_per_class, replace=False)
            else:
                chosen = cls_indices
            indices.extend(chosen.tolist())
        rng.shuffle(indices)
        return Subset(dataset, indices)

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=True, drop_last=False,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
        )

    def get_test_loader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=True,
        )


def build_fracture_data_batch(images, device):
    """Build terminal dict from a batch of X-ray images [B, 3, H, W].

    X-ray specific channels:
      - I_R, I_G, I_B: raw RGB channels
      - I_GRAY: standard grayscale
      - I_BONE: bone-enhanced channel (high contrast for bone edges)
      - I_SOFT: soft tissue suppressed channel
      - I_EDGE_PRIOR: edge-prior channel (gradient magnitude of grayscale)
      - I_NEG: inverted X-ray (bright bone on dark background)
      - I_RG: red-green difference (subtle tissue variation)
      - I_BY: blue-yellow opponent channel
      - I_H, I_S: HSV hue and saturation
    """
    images = images.to(device, dtype=torch.float32)
    I_R = images[:, 0]
    I_G = images[:, 1]
    I_B = images[:, 2]

    I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B

    I_NEG = 1.0 - I_GRAY

    I_BONE = torch.clamp((I_GRAY - 0.3) / 0.7, 0.0, 1.0)

    I_SOFT = torch.clamp(I_GRAY / 0.5, 0.0, 1.0) * (1.0 - I_BONE)

    import torch.nn.functional as F
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                           device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                           device=device).view(1, 1, 3, 3)
    gray_4d = I_GRAY.unsqueeze(1)
    gx = F.conv2d(gray_4d, sobel_x, padding=1).squeeze(1)
    gy = F.conv2d(gray_4d, sobel_y, padding=1).squeeze(1)
    I_EDGE_PRIOR = torch.sqrt(gx * gx + gy * gy + 1e-8)

    Cmax, _ = images.max(dim=1)
    Cmin, _ = images.min(dim=1)
    delta = Cmax - Cmin + 1e-8
    H = torch.zeros_like(I_R)
    mr = (Cmax == I_R)
    mg = (Cmax == I_G) & ~mr
    mb = ~mr & ~mg
    H[mr] = (((I_G[mr] - I_B[mr]) / delta[mr]) % 6)
    H[mg] = ((I_B[mg] - I_R[mg]) / delta[mg]) + 2
    H[mb] = ((I_R[mb] - I_G[mb]) / delta[mb]) + 4
    H = H / 6.0
    S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))

    total = I_R + I_G + I_B + 1e-8

    return {
        'I_R': I_R, 'I_G': I_G, 'I_B': I_B,
        'I_GRAY': I_GRAY,
        'I_NEG': I_NEG,
        'I_BONE': I_BONE,
        'I_SOFT': I_SOFT,
        'I_EDGE_PRIOR': I_EDGE_PRIOR,
        'I_H': H, 'I_S': S,
        'I_r': I_R / total, 'I_g': I_G / total,
        'I_RG': I_R - I_G, 'I_BY': I_B - (I_R + I_G) / 2,
    }


def build_fracture_superclass_mapping():
    """Map 10 fracture classes to 3 superclasses for hierarchical evaluation."""
    mapping = {}
    for sup_id, (_, cls_list) in enumerate(FRACTURE_SUPERCLASS.items()):
        for cls_id in cls_list:
            mapping[cls_id] = sup_id
    return mapping
