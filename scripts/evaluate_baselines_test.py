#!/usr/bin/env python3
"""
统一测试集评估脚本
加载 baseline_comparison 中所有已训练模型的权重，在 dataset_split/test 上评估，输出 JSON 报告
"""

import os
import sys
import json
import argparse
import importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

FRACTURE_NAMES = ['Comminuted', 'Greenstick', 'Healthy', 'Linear',
                  'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
                  'Transverse Displaced', 'Transverse']

FRACTURE_CONCEPTS = ['cortical_break', 'fracture_line_horizontal',
                     'fracture_line_oblique_45', 'fracture_line_oblique_135',
                     'fracture_line_vertical', 'displacement',
                     'bone_fragment', 'soft_tissue_swelling']

CONCEPT_CLASS_MAP = {
    0: [1, 0, 0, 0, 0, 1, 1, 0],
    1: [0, 1, 0, 0, 0, 0, 0, 1],
    2: [0, 0, 0, 0, 0, 0, 0, 0],
    3: [0, 1, 0, 0, 0, 0, 0, 0],
    4: [1, 0, 1, 0, 0, 1, 1, 1],
    5: [0, 0, 1, 0, 0, 0, 0, 0],
    6: [1, 0, 0, 0, 1, 1, 1, 1],
    7: [0, 0, 0, 1, 0, 0, 0, 0],
    8: [1, 1, 0, 0, 0, 1, 1, 1],
    9: [0, 1, 0, 0, 0, 0, 0, 0],
}


class FractureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def get_transforms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                               transforms.ToTensor(), norm])


def load_test_data(split_dir, split_file):
    sd = np.load(split_file, allow_pickle=True)
    active_classes = sd['active_classes']
    num_classes = len(active_classes)
    active_names = [FRACTURE_NAMES[int(c)] for c in active_classes]
    class_map = {int(c): i for i, c in enumerate(active_classes)}

    img_dir = os.path.join(split_dir, 'test', 'images')
    lbl_dir = os.path.join(split_dir, 'test', 'labels')
    paths, labels = [], []
    for img_file in sorted(os.listdir(img_dir)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        label_file = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + '.txt')
        cls_id = 0
        if os.path.exists(label_file):
            with open(label_file) as f:
                lines = f.read().strip().split('\n')
            if lines and lines[0].strip():
                cls_id = int(lines[0].strip().split()[0])
        paths.append(os.path.join(img_dir, img_file))
        labels.append(class_map.get(cls_id, -1))

    valid = [l >= 0 for l in labels]
    paths = [p for p, v in zip(paths, valid) if v]
    labels = np.array([l for l, v in zip(labels, valid) if v])

    print(f"[Test Data] {len(paths)} images, {num_classes} classes")
    return paths, labels, num_classes, active_names, active_classes


def evaluate_standard_model(model, loader, device, num_classes):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(device))
            if isinstance(out, tuple):
                out = out[0]
            probs = F.softmax(out, dim=1)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    return _compute_metrics(np.array(all_preds), np.array(all_labels),
                            np.array(all_probs), num_classes)


def _compute_metrics(preds, labels, probs, num_classes):
    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average='macro', zero_division=0)
    wf1 = f1_score(labels, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    per_class_acc = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            per_class_acc[str(c)] = float((preds[mask] == c).mean())

    return dict(accuracy=float(acc), balanced_accuracy=float(bacc),
                macro_f1=float(mf1), weighted_f1=float(wf1), auc=float(auc),
                per_class_accuracy=per_class_acc)


def eval_resnet50(model_dir, test_loader, device, num_classes):
    from torchvision import models as tv_models
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_dir / 'best_weights.pth', map_location='cpu'))
    model = model.to(device)
    return evaluate_standard_model(model, test_loader, device, num_classes)


def eval_densenet121(model_dir, test_loader, device, num_classes):
    from torchvision import models as tv_models
    model = tv_models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.load_state_dict(torch.load(model_dir / 'best_weights.pth', map_location='cpu'))
    model = model.to(device)
    return evaluate_standard_model(model, test_loader, device, num_classes)


def eval_efficientnet_b0(model_dir, test_loader, device, num_classes):
    from torchvision import models as tv_models
    model = tv_models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(torch.load(model_dir / 'best_weights.pth', map_location='cpu'))
    model = model.to(device)
    return evaluate_standard_model(model, test_loader, device, num_classes)


def eval_swin_tiny(model_dir, test_loader, device, num_classes):
    import timm
    report = json.load(open(model_dir / 'report.json'))
    variant = report.get('model_variant', 'swin_tiny_patch4_window7_224')
    model = timm.create_model(variant, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_dir / 'best_weights.pth', map_location='cpu'))
    model = model.to(device)
    return evaluate_standard_model(model, test_loader, device, num_classes)


def eval_gradcam(model_dir, test_loader, device, num_classes):
    from torchvision import models as tv_models
    model = tv_models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_dir / 'best_weights.pth', map_location='cpu'))
    model = model.to(device)
    return evaluate_standard_model(model, test_loader, device, num_classes)


def eval_cbm(model_dir, test_loader, device, num_classes):
    ncon = len(FRACTURE_CONCEPTS)

    class CBMModel(nn.Module):
        def __init__(self):
            super().__init__()
            from torchvision import models as tv_models
            backbone = tv_models.resnet34(weights=None)
            self.backbone = backbone
            self.concept_head = nn.Linear(backbone.fc.in_features, ncon)
            self.classifier = nn.Sequential(nn.Linear(ncon, 64), nn.ReLU(),
                                            nn.Linear(64, num_classes))
            backbone.fc = nn.Identity()

        def forward(self, x):
            features = self.backbone(x)
            concepts = torch.sigmoid(self.concept_head(features))
            logits = self.classifier(concepts)
            return logits, concepts

    model = CBMModel()
    model.load_state_dict(torch.load(model_dir / 'best_weights.pth', map_location='cpu'))
    model = model.to(device)
    return evaluate_standard_model(model, test_loader, device, num_classes)


def eval_crl(model_dir, test_loader, device, num_classes):
    ncon = len(FRACTURE_CONCEPTS)

    class CRLModel(nn.Module):
        def __init__(self):
            super().__init__()
            from torchvision import models as tv_models
            backbone = tv_models.resnet34(weights=None)
            self.backbone = backbone
            self.concept_head = nn.Linear(backbone.fc.in_features, ncon)
            self.bool_weight = nn.Parameter(torch.randn(num_classes, ncon) * 0.1)
            self.bool_bias = nn.Parameter(torch.zeros(num_classes))
            backbone.fc = nn.Identity()

        def forward(self, x):
            features = self.backbone(x)
            concepts = torch.sigmoid(self.concept_head(features))
            logits = F.linear(concepts, self.bool_weight, self.bool_bias)
            return logits, concepts

    model = CRLModel()
    model.load_state_dict(torch.load(model_dir / 'best_weights.pth', map_location='cpu'))
    model = model.to(device)
    return evaluate_standard_model(model, test_loader, device, num_classes)


def eval_dinov2_linear(model_dir, test_paths, test_labels, device, num_classes):
    import pickle
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    dino_t = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])])
    test_ds = FractureDataset(test_paths, test_labels, dino_t)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    feats = []
    with torch.no_grad():
        for images, _ in loader:
            feats.append(backbone(images.to(device)).cpu())
    te_f = torch.cat(feats).numpy()

    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / 'classifier.pkl', 'rb') as f:
        clf = pickle.load(f)

    te_s = scaler.transform(te_f)
    preds = clf.predict(te_s)
    probs = clf.predict_proba(te_s)
    return _compute_metrics(preds, test_labels, probs, num_classes)


def eval_rulefit(model_dir, test_paths, test_labels, device, num_classes):
    import pickle
    from torchvision import models as tv_models

    backbone = tv_models.mobilenet_v2(weights=None)
    backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, num_classes)
    backbone.load_state_dict(torch.load(model_dir / 'backbone_weights.pth', map_location='cpu'))
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    norm_t = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])])
    test_ds = FractureDataset(test_paths, test_labels, norm_t)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    feats = []
    with torch.no_grad():
        for images, _ in loader:
            out = backbone(images.to(device))
            feats.append(out.cpu())
    te_f = torch.cat(feats).numpy()

    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(model_dir / 'selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    with open(model_dir / 'rule_model.pkl', 'rb') as f:
        rule_model = pickle.load(f)

    te_s = scaler.transform(te_f)
    te_sel = selector.transform(te_s)
    preds = rule_model.predict(te_sel)
    try:
        probs = rule_model.predict_proba(te_sel)
    except Exception:
        probs = np.zeros((len(preds), num_classes))
    return _compute_metrics(preds, test_labels, probs, num_classes)


MODEL_EVALUATORS = {
    'resnet50': ('CNN baseline', 'deep'),
    'densenet121': ('CNN baseline', 'deep'),
    'efficientnet_b0': ('CNN baseline', 'deep'),
    'swin_tiny': ('Transformer baseline', 'deep'),
    'dinov2_linear': ('Self-supervised baseline', 'sklearn'),
    'gradcam': ('XAI baseline', 'deep'),
    'rulefit': ('Rule baseline', 'rulefit'),
    'cbm': ('Neuro-symbolic baseline', 'deep'),
    'crl': ('Neuro-symbolic baseline', 'deep'),
}


def main():
    parser = argparse.ArgumentParser(description='Unified test set evaluation for all baseline models')
    parser.add_argument('--models_dir', type=str,
                        default='outputs/fracture_v3_expanded/baseline_comparison',
                        help='Directory containing all model subdirectories')
    parser.add_argument('--split_dir', type=str, default='dataset_split',
                        help='dataset_split directory with test images')
    parser.add_argument('--split_file', type=str,
                        default='outputs/fracture_v3_expanded/split_indices.npz',
                        help='split_indices.npz file')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / args.models_dir
    split_dir = base_dir / args.split_dir
    split_file = base_dir / args.split_file
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print("=" * 70)
    print("  UNIFIED TEST SET EVALUATION")
    print("=" * 70)
    print(f"  Models dir:  {models_dir}")
    print(f"  Test data:   {split_dir}/test")
    print(f"  Split file:  {split_file}")
    print(f"  Device:      {device}")
    print("=" * 70)

    test_paths, test_labels, num_classes, active_names, active_classes = \
        load_test_data(str(split_dir), str(split_file))

    test_t = get_transforms()
    test_ds = FractureDataset(test_paths, test_labels, test_t)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=4, pin_memory=True)

    all_results = {}
    summary_rows = []

    for model_name, (group, eval_type) in MODEL_EVALUATORS.items():
        model_dir = models_dir / model_name
        if not model_dir.is_dir():
            print(f"\n  [{model_name}] SKIP - directory not found")
            continue

        print(f"\n  Evaluating: {model_name} ({group})...", flush=True)

        try:
            if eval_type == 'deep':
                eval_fn = globals()[f'eval_{model_name}']
                results = eval_fn(model_dir, test_loader, device, num_classes)
            elif eval_type == 'sklearn':
                results = eval_dinov2_linear(model_dir, test_paths, test_labels,
                                             device, num_classes)
            elif eval_type == 'rulefit':
                results = eval_rulefit(model_dir, test_paths, test_labels,
                                       device, num_classes)
            else:
                print(f"    Unknown eval type: {eval_type}")
                continue

            results['group'] = group
            results['model_name'] = model_name
            all_results[model_name] = results

            print(f"    Acc={results['accuracy']:.4f}  BAcc={results['balanced_accuracy']:.4f}  "
                  f"F1={results['macro_f1']:.4f}  AUC={results['auc']:.4f}")

            summary_rows.append({
                'Method': model_name,
                'Group': group,
                'Acc': f"{results['accuracy']:.4f}",
                'BAcc': f"{results['balanced_accuracy']:.4f}",
                'F1': f"{results['macro_f1']:.4f}",
                'AUC': f"{results['auc']:.4f}",
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            all_results[model_name] = {'error': str(e), 'group': group}

    print("\n" + "=" * 70)
    print("   TEST SET EVALUATION SUMMARY")
    print("=" * 70)
    header = f"{'Method':<22} {'Group':<28} {'Acc':>6} {'BAcc':>6} {'F1':>6} {'AUC':>6}"
    print(header)
    print("-" * 70)
    for row in summary_rows:
        print(f"{row['Method']:<22} {row['Group']:<28} {row['Acc']:>6} {row['BAcc']:>6} "
              f"{row['F1']:>6} {row['AUC']:>6}")
    print("=" * 70)

    out_file = models_dir / 'test_set_evaluation.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results saved to: {out_file}")

    per_class_file = models_dir / 'test_set_per_class.json'
    per_class = {}
    for name, res in all_results.items():
        if 'per_class_accuracy' in res:
            per_class[name] = {
                active_names[int(k)]: v for k, v in res['per_class_accuracy'].items()
            }
    with open(per_class_file, 'w') as f:
        json.dump(per_class, f, indent=2, ensure_ascii=False)
    print(f"  Per-class results saved to: {per_class_file}")


if __name__ == '__main__':
    main()
