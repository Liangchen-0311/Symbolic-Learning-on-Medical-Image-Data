#!/usr/bin/env python3
"""
Baseline Comparison for Fracture Symbolic Classification

9 baselines across 4 groups, all sharing the same data split and evaluation metrics.
Each model saves to: outputs/fracture_v3_expanded/baseline_comparison/<model_name>/
  - model.py          : standalone model definition
  - best_weights.pth  : best model weights
  - report.json       : evaluation report
  - predict.py        : prediction script using saved weights

Usage:
    python experiments/baseline_comparison.py --config configs/fracture_v3_expanded.yaml --gpu 0
    python experiments/baseline_comparison.py --config configs/fracture_v3_expanded.yaml --gpu 0 --methods resnet50 densenet121
    python experiments/baseline_comparison.py --config configs/fracture_v3_expanded.yaml --gpu 0 --methods gradcam
"""

import argparse, json, os, sys, time, pickle
from pathlib import Path
from collections import Counter

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from src.data.fracture_loader import FRACTURE_NAMES

ALL_METHODS = [
    'resnet50', 'densenet121', 'efficientnet_b0',
    'swin_tiny', 'dinov2_linear',
    'gradcam', 'rulefit',
    'cbm', 'crl',
]

GROUP_LABELS = {
    'resnet50': 'CNN baseline', 'densenet121': 'CNN baseline', 'efficientnet_b0': 'CNN baseline',
    'swin_tiny': 'Transformer baseline', 'dinov2_linear': 'Self-supervised baseline',
    'gradcam': 'XAI baseline', 'rulefit': 'Rule baseline',
    'cbm': 'Neuro-symbolic baseline', 'crl': 'Neuro-symbolic baseline',
}

FRACTURE_CONCEPTS = [
    'cortical_break', 'fracture_line_horizontal', 'fracture_line_oblique_45',
    'fracture_line_oblique_135', 'fracture_line_vertical', 'displacement',
    'bone_fragment', 'soft_tissue_swelling',
]

CONCEPT_CLASS_MAP = {
    0: [1,1,0,0,0,1,1,0], 1: [1,0,0,0,0,0,0,1], 2: [0,0,0,0,0,0,0,0],
    3: [1,1,0,0,0,0,0,0], 4: [1,0,1,1,0,1,1,0], 5: [1,0,1,0,0,0,0,0],
    6: [1,1,0,0,1,1,1,0], 7: [1,0,1,1,0,0,0,0], 8: [1,1,0,0,0,1,0,0],
    9: [1,1,0,0,0,0,0,0],
}


class FractureDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths, self.labels, self.transform = image_paths, labels, transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


def load_split_data(config):
    split_dir = str(Path(__file__).parent.parent / 'dataset_split')
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"dataset_split directory not found: {split_dir}\n"
            "Please run scripts/resplit_dataset.py first to generate it."
        )

    split_file = str(Path(config['output_dir']) / 'split_indices.npz')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}. Run pipeline first.")
    sd = np.load(split_file, allow_pickle=True)
    active_classes = sd['active_classes']
    num_classes = len(active_classes)
    active_names = [FRACTURE_NAMES[int(c)] for c in active_classes]
    class_map = {int(c): i for i, c in enumerate(active_classes)}

    def _load_split(split_name):
        img_dir = os.path.join(split_dir, split_name, 'images')
        lbl_dir = os.path.join(split_dir, split_name, 'labels')
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
        return paths, labels

    tp, tl = _load_split('train')
    vp, vl = _load_split('val')
    tep, tel = _load_split('test')

    print(f"[Data] Train={len(tp)}, Val={len(vp)}, Test={len(tep)}, Classes={num_classes}")
    print(f"[Data] Source: {split_dir}")
    print(f"[Data] Active: {active_names}")
    return dict(train_paths=tp, val_paths=vp, test_paths=tep,
                train_labels=tl, val_labels=vl, test_labels=tel,
                num_classes=num_classes, active_names=active_names,
                active_classes=active_classes, class_map=class_map)


def get_transforms(augment=False):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(), norm])
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm])


def make_loaders(data, batch_size=32, augment_train=True):
    train_ds = FractureDataset(data['train_paths'], data['train_labels'], get_transforms(augment=augment_train))
    val_ds = FractureDataset(data['val_paths'], data['val_labels'], get_transforms())
    test_ds = FractureDataset(data['test_paths'], data['test_labels'], get_transforms())
    kw = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, shuffle=False, **kw),
            DataLoader(test_ds, shuffle=False, **kw))


def evaluate_model(model, loader, device, num_classes):
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
    all_preds, all_labels, all_probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    acc = accuracy_score(all_labels, all_preds)
    bacc = balanced_accuracy_score(all_labels, all_preds)
    mf1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    wf1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    per_class, sens, spec = {}, {}, {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[c] = float((all_preds[mask] == c).mean())
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        tn = ((all_preds != c) & (all_labels != c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        sens[c] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec[c] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return dict(accuracy=float(acc), balanced_accuracy=float(bacc), macro_f1=float(mf1),
                weighted_f1=float(wf1), auc=float(auc),
                per_class_accuracy=per_class, sensitivity=sens, specificity=spec)


def train_deep_model(model, train_loader, val_loader, device, num_epochs=100, lr=1e-3,
                     weight_decay=1e-4, class_weights=None, patience=15):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    cw = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights else None
    criterion = nn.CrossEntropyLoss(weight=cw) if cw is not None else nn.CrossEntropyLoss()

    history = {'loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_val_acc, best_state, no_improve = 0, None, 0
    for epoch in range(num_epochs):
        model.train()
        tl, c, t = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            if isinstance(out, tuple):
                out = out[0]
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            tl += loss.item() * images.size(0)
            c += (out.argmax(1) == labels).sum().item()
            t += images.size(0)
        scheduler.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                out = model(images.to(device))
                if isinstance(out, tuple):
                    out = out[0]
                vc += (out.argmax(1) == labels.to(device)).sum().item()
                vt += images.size(0)
        va = vc / vt

        history['loss'].append(tl / t)
        history['train_acc'].append(c / t)
        history['val_acc'].append(va)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if va > best_val_acc:
            best_val_acc = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}: loss={tl/t:.4f}, train_acc={c/t:.4f}, val_acc={va:.4f}")
        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val_acc, history


def plot_training_curves(history, save_dir, model_name):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(history['loss']) + 1)

        axes[0].plot(epochs, history['loss'], 'b-', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Training Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history['train_acc'], 'r-', linewidth=1.5, label='Train Acc')
        axes[1].plot(epochs, history['val_acc'], 'g-', linewidth=1.5, label='Val Acc')
        best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
        axes[1].axvline(x=best_epoch, color='k', linestyle='--', alpha=0.5, label=f'Best Epoch={best_epoch}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{model_name} - Accuracy (Best Val={max(history["val_acc"]):.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, history['lr'], 'm-', linewidth=1.5)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title(f'{model_name} - Learning Rate')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Training curves saved to {save_dir}/training_curves.png")
    except ImportError:
        print(f"  [WARN] matplotlib not available, skipping training curves plot")

    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)


def class_weights(labels, nc):
    counts = Counter(labels.tolist())
    total = len(labels)
    return [total / (nc * counts.get(c, 1)) for c in range(nc)]


def _save_report(path, results):
    s = {}
    for k, v in results.items():
        if isinstance(v, (np.floating, np.integer)):
            v = float(v)
        elif isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, dict):
            v = {str(kk): float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()}
        s[k] = v
    with open(path, 'w') as f:
        json.dump(s, f, indent=2, ensure_ascii=False)


def _standard_predict_code(model_name, create_fn_str, data):
    nc, names = data['num_classes'], data['active_names']
    return f'''#!/usr/bin/env python3
"""Predict with {model_name}"""
import os, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms
from PIL import Image

FRACTURE_NAMES = {names}

{create_fn_str}

def predict(image_dir, weights_path, gpu=0):
    device = torch.device(f"cuda:{{gpu}}" if torch.cuda.is_available() else "cpu")
    model = create_model()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    results = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, tuple):
                out = out[0]
            prob = F.softmax(out, dim=1)
            pred = out.argmax(1).item()
        results.append({{"file": f, "pred_class": pred, "pred_name": FRACTURE_NAMES[pred],
                         "confidence": prob[0, pred].item()}})
    for r in results:
        print(f"{{r['file']}}: {{r['pred_name']}} ({{r['confidence']:.4f}})")
    out_path = os.path.join(os.path.dirname(weights_path), "predict_results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {{out_path}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_weights.pth"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.weights, args.gpu)
'''


def _save_standard_model(model_dir, model_name, model, results, create_fn_str, data):
    model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_dir / 'best_weights.pth')
    _save_report(model_dir / 'report.json', results)
    with open(model_dir / 'predict.py', 'w') as f:
        f.write(_standard_predict_code(model_name, create_fn_str, data))
    with open(model_dir / 'model.py', 'w') as f:
        f.write(f'"""{model_name} for Fracture Classification"""\nimport torch, torch.nn as nn\nfrom torchvision import models\n\nFRACTURE_NAMES = {data["active_names"]}\n\n{create_fn_str}\n')
    print(f"  Saved model to {model_dir}/  (model.py, best_weights.pth, report.json, predict.py)")


# ======================================================================
# Group 1: CNN Baselines
# ======================================================================

def run_resnet50(data, device, save_dir):
    print(f"\n  --- ResNet50 ---", flush=True)
    nc = data['num_classes']
    cw = class_weights(data['train_labels'], nc)
    train_l, val_l, test_l = make_loaders(data)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, nc)
    model, bv, history = train_deep_model(model, train_l, val_l, device, num_epochs=100, class_weights=cw)
    results = evaluate_model(model, test_l, device, nc)
    results['best_val_accuracy'] = float(bv)
    print(f"  ResNet50: acc={results['accuracy']:.4f}, bacc={results['balanced_accuracy']:.4f}, macro_f1={results['macro_f1']:.4f}, auc={results['auc']:.4f}")
    fn = f"def create_model(num_classes={nc}):\n    model = models.resnet50(weights=None)\n    model.fc = nn.Linear(model.fc.in_features, num_classes)\n    return model"
    _save_standard_model(save_dir / 'resnet50', 'ResNet50', model, results, fn, data)
    plot_training_curves(history, save_dir / 'resnet50', 'ResNet50')
    return results


def run_densenet121(data, device, save_dir):
    print(f"\n  --- DenseNet121 ---", flush=True)
    nc = data['num_classes']
    cw = class_weights(data['train_labels'], nc)
    train_l, val_l, test_l = make_loaders(data)
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, nc)
    model, bv, history = train_deep_model(model, train_l, val_l, device, num_epochs=100, class_weights=cw)
    results = evaluate_model(model, test_l, device, nc)
    results['best_val_accuracy'] = float(bv)
    print(f"  DenseNet121: acc={results['accuracy']:.4f}, bacc={results['balanced_accuracy']:.4f}, macro_f1={results['macro_f1']:.4f}, auc={results['auc']:.4f}")
    fn = f"def create_model(num_classes={nc}):\n    model = models.densenet121(weights=None)\n    model.classifier = nn.Linear(model.classifier.in_features, num_classes)\n    return model"
    _save_standard_model(save_dir / 'densenet121', 'DenseNet121', model, results, fn, data)
    plot_training_curves(history, save_dir / 'densenet121', 'DenseNet121')
    return results


def run_efficientnet_b0(data, device, save_dir):
    print(f"\n  --- EfficientNet-B0 ---", flush=True)
    nc = data['num_classes']
    cw = class_weights(data['train_labels'], nc)
    train_l, val_l, test_l = make_loaders(data)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, nc)
    model, bv, history = train_deep_model(model, train_l, val_l, device, num_epochs=100, class_weights=cw)
    results = evaluate_model(model, test_l, device, nc)
    results['best_val_accuracy'] = float(bv)
    print(f"  EfficientNet-B0: acc={results['accuracy']:.4f}, bacc={results['balanced_accuracy']:.4f}, macro_f1={results['macro_f1']:.4f}, auc={results['auc']:.4f}")
    fn = f"def create_model(num_classes={nc}):\n    model = models.efficientnet_b0(weights=None)\n    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)\n    return model"
    _save_standard_model(save_dir / 'efficientnet_b0', 'EfficientNet-B0', model, results, fn, data)
    plot_training_curves(history, save_dir / 'efficientnet_b0', 'EfficientNet-B0')
    return results


# ======================================================================
# Group 2: Transformer / Foundation Model Baselines
# ======================================================================

def run_swin_tiny(data, device, save_dir):
    print(f"\n  --- Swin-Tiny ---", flush=True)
    import timm
    nc = data['num_classes']
    cw = class_weights(data['train_labels'], nc)
    train_l, val_l, test_l = make_loaders(data)

    model = None
    model_name_str = 'swin_tiny_patch4_window7_224'
    for name in ['swin_tiny_patch4_window7_224', 'swinv2_tiny_window16_256', 'deit_small_patch16_224']:
        try:
            print(f"  Trying to create model: {name}...", flush=True)
            model = timm.create_model(name, pretrained=True, num_classes=nc)
            model_name_str = name
            print(f"  Success: {name}", flush=True)
            break
        except Exception as e:
            print(f"  Failed: {name} ({e})", flush=True)
            model = None
    if model is None:
        print("  All transformer models failed, using DeiT-Small as fallback")
        try:
            model = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=nc)
            model_name_str = 'deit_small_patch16_224'
        except Exception:
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            model.heads.head = nn.Linear(model.heads.head.in_features, nc)
            model_name_str = 'vit_b_16'

    model, bv, history = train_deep_model(model, train_l, val_l, device, num_epochs=200, lr=2e-4, class_weights=cw, patience=30)
    results = evaluate_model(model, test_l, device, nc)
    results['best_val_accuracy'] = float(bv)
    results['model_variant'] = model_name_str
    print(f"  Swin-Tiny ({model_name_str}): acc={results['accuracy']:.4f}, bacc={results['balanced_accuracy']:.4f}, macro_f1={results['macro_f1']:.4f}, auc={results['auc']:.4f}")
    fn = f"def create_model(num_classes={nc}):\n    import timm\n    return timm.create_model('{model_name_str}', pretrained=False, num_classes=num_classes)"
    _save_standard_model(save_dir / 'swin_tiny', f'Swin-Tiny ({model_name_str})', model, results, fn, data)
    plot_training_curves(history, save_dir / 'swin_tiny', f'Swin-Tiny ({model_name_str})')
    return results


def run_dinov2_linear(data, device, save_dir):
    print(f"\n  --- DINOv2 + Linear Probe ---", flush=True)
    nc = data['num_classes']
    print("  Loading DINOv2 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad = False

    dino_t = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_ds = FractureDataset(data['train_paths'], data['train_labels'], dino_t)
    val_ds = FractureDataset(data['val_paths'], data['val_labels'], dino_t)
    test_ds = FractureDataset(data['test_paths'], data['test_labels'], dino_t)

    def extract_feats(ds):
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        feats, labs = [], []
        with torch.no_grad():
            for images, labels in loader:
                feats.append(backbone(images.to(device)).cpu())
                labs.append(labels)
        return torch.cat(feats).numpy(), torch.cat(labs).numpy()

    print("  Extracting DINOv2 features...")
    tr_f, tr_l = extract_feats(train_ds)
    va_f, va_l = extract_feats(val_ds)
    te_f, te_l = extract_feats(test_ds)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    scaler = StandardScaler()
    tr_s = scaler.fit_transform(tr_f)
    va_s = scaler.transform(va_f)
    te_s = scaler.transform(te_f)
    comb_f = np.concatenate([tr_s, va_s])
    comb_l = np.concatenate([tr_l, va_l])

    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', solver='lbfgs')
    clf.fit(comb_f, comb_l)
    preds = clf.predict(te_s)
    probs = clf.predict_proba(te_s)

    acc = accuracy_score(te_l, preds)
    bacc = balanced_accuracy_score(te_l, preds)
    mf1 = f1_score(te_l, preds, average='macro', zero_division=0)
    wf1 = f1_score(te_l, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(te_l, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    results = dict(accuracy=float(acc), balanced_accuracy=float(bacc), macro_f1=float(mf1),
                   weighted_f1=float(wf1), auc=float(auc))
    print(f"  DINOv2+Linear: acc={acc:.4f}, bacc={bacc:.4f}, macro_f1={mf1:.4f}, auc={auc:.4f}")

    model_dir = save_dir / 'dinov2_linear'
    model_dir.mkdir(exist_ok=True, parents=True)
    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(model_dir / 'classifier.pkl', 'wb') as f:
        pickle.dump(clf, f)
    _save_report(model_dir / 'report.json', results)

    nc2, names = nc, data['active_names']
    pred_code = f'''#!/usr/bin/env python3
"""Predict with DINOv2 + Linear Probe"""
import os, json, argparse, pickle
import torch, torch.nn.functional as F
from torchvision import transforms
from PIL import Image

FRACTURE_NAMES = {names}

def predict(image_dir, model_dir, gpu=0):
    device = torch.device(f"cuda:{{gpu}}" if torch.cuda.is_available() else "cpu")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    backbone.eval().to(device)
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    clf = pickle.load(open(os.path.join(model_dir, "classifier.pkl"), "rb"))
    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    results = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(tensor).cpu().numpy()
        feat_s = scaler.transform(feat)
        pred = clf.predict(feat_s)[0]
        prob = clf.predict_proba(feat_s)[0]
        results.append({{"file": f, "pred_class": int(pred), "pred_name": FRACTURE_NAMES[pred],
                         "confidence": float(prob[pred])}})
    for r in results:
        print(f"{{r['file']}}: {{r['pred_name']}} ({{r['confidence']:.4f}})")
    out_path = os.path.join(model_dir, "predict_results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {{out_path}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.model_dir, args.gpu)
'''
    with open(model_dir / 'predict.py', 'w') as f:
        f.write(pred_code)
    with open(model_dir / 'model.py', 'w') as f:
        f.write(f'"""DINOv2 + Linear Probe for Fracture Classification"""\nimport torch, pickle\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LogisticRegression\n\nFRACTURE_NAMES = {names}\n')
    print(f"  Saved model to {model_dir}/  (scaler.pkl, classifier.pkl, report.json, predict.py)")
    return results


# ======================================================================
# Group 3: XAI / Rule Baselines
# ======================================================================

def run_gradcam(data, device, save_dir):
    print(f"\n  --- Grad-CAM + ResNet50 ---", flush=True)
    nc = data['num_classes']
    cw = class_weights(data['train_labels'], nc)
    train_l, val_l, test_l = make_loaders(data)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, nc)
    model, bv, history = train_deep_model(model, train_l, val_l, device, num_epochs=100, class_weights=cw)
    results = evaluate_model(model, test_l, device, nc)
    results['best_val_accuracy'] = float(bv)
    print(f"  Grad-CAM+ResNet50: acc={results['accuracy']:.4f}, bacc={results['balanced_accuracy']:.4f}")

    try:
        from captum.attr import LayerGradCam
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        target_layer = model.layer4[-1]
        gc = LayerGradCam(model, target_layer)
        cam_dir = save_dir / 'gradcam' / 'heatmaps'
        cam_dir.mkdir(exist_ok=True, parents=True)
        model.eval()
        n_saved = 0
        for images, labels in test_l:
            images_dev = images.to(device)
            for i in range(min(images.size(0), 10)):
                inp = images_dev[i:i+1]
                inp.requires_grad = True
                pred = model(inp).argmax(dim=1).item()
                attr = gc.attribute(inp, target=pred)
                attr = F.interpolate(attr, size=(224, 224), mode='bilinear', align_corners=False)
                hm = attr.squeeze().cpu().detach().numpy()
                hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                img_np = images[i].numpy().transpose(1, 2, 0)
                mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
                img_np = np.clip((img_np * std + mean), 0, 1)
                fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                ax.imshow(img_np)
                ax.imshow(hm, cmap='jet', alpha=0.4)
                ax.set_title(f"Pred: {data['active_names'][pred]}")
                ax.axis('off')
                fig.savefig(cam_dir / f'gradcam_{n_saved}.png', dpi=100, bbox_inches='tight')
                plt.close(fig)
                n_saved += 1
            if n_saved >= 10:
                break
        results['gradcam_saved'] = n_saved
        print(f"  Saved {n_saved} Grad-CAM heatmaps to {cam_dir}")
    except Exception as e:
        print(f"  Grad-CAM visualization skipped: {e}")

    fn = f"def create_model(num_classes={nc}):\n    model = models.resnet50(weights=None)\n    model.fc = nn.Linear(model.fc.in_features, num_classes)\n    return model"
    _save_standard_model(save_dir / 'gradcam', 'Grad-CAM+ResNet50', model, results, fn, data)
    plot_training_curves(history, save_dir / 'gradcam', 'Grad-CAM+ResNet50')
    return results


def run_rulefit(data, device, save_dir):
    print(f"\n  --- MobileNetV2 + RuleFit ---", flush=True)
    nc = data['num_classes']
    train_l, val_l, test_l = make_loaders(data, augment_train=False)

    backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, nc)
    backbone = backbone.to(device)
    cw = class_weights(data['train_labels'], nc)
    backbone, _, rulefit_history = train_deep_model(backbone, train_l, val_l, device, num_epochs=80, class_weights=cw)

    backbone.eval()
    feat_ext = nn.Sequential(*list(backbone.features.children()))
    pool = nn.AdaptiveAvgPool2d(1)

    def extract_feats(loader):
        afs, als = [], []
        with torch.no_grad():
            for images, labels in loader:
                afs.append(pool(feat_ext(images.to(device))).flatten(1).cpu().numpy())
                als.append(labels.numpy())
        return np.concatenate(afs), np.concatenate(als)

    print("  Extracting MobileNetV2 features...")
    tr_f, tr_l = extract_feats(train_l)
    va_f, va_l = extract_feats(val_l)
    te_f, te_l = extract_feats(test_l)
    comb_f = np.concatenate([tr_f, va_f])
    comb_l = np.concatenate([tr_l, va_l])

    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

    scaler = StandardScaler()
    comb_s = scaler.fit_transform(comb_f)
    te_s = scaler.transform(te_f)
    selector = SelectKBest(f_classif, k=min(200, comb_s.shape[1]))
    comb_sel = selector.fit_transform(comb_s, comb_l)
    te_sel = selector.transform(te_s)

    rule_model = None
    rulefit_used = False
    try:
        from imodels import RuleFitClassifier
        print("  Training RuleFit...")
        rule_model = RuleFitClassifier(random_state=42, max_rules=100, n_estimators=200)
        rule_model.fit(comb_sel, comb_l)
        preds = rule_model.predict(te_sel)
        try:
            probs = rule_model.predict_proba(te_sel)
            rulefit_used = True
        except Exception:
            probs = None
        try:
            rules = rule_model.get_rules()
            rules = rules[rules['type'] == 'rule'].sort_values('importance', ascending=False).head(20)
            rf_path = save_dir / 'rulefit' / 'rulefit_rules.txt'
            rf_path.parent.mkdir(exist_ok=True, parents=True)
            with open(rf_path, 'w') as f:
                f.write("Top-20 RuleFit Rules:\n")
                for _, row in rules.iterrows():
                    f.write(f"  {row['rule']}: importance={row['importance']:.4f}\n")
        except Exception:
            pass
    except Exception as e:
        print(f"  RuleFit failed ({e}), falling back to GradientBoosting with shallow trees")

    if rule_model is None or not rulefit_used:
        from sklearn.ensemble import GradientBoostingClassifier
        print("  Training GradientBoosting (max_depth=3, n_estimators=200)...")
        rule_model = GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.1,
                                                 random_state=42, subsample=0.8)
        rule_model.fit(comb_sel, comb_l)
        preds = rule_model.predict(te_sel)
        probs = rule_model.predict_proba(te_sel)
        rf_path = save_dir / 'rulefit' / 'rulefit_rules.txt'
        rf_path.parent.mkdir(exist_ok=True, parents=True)
        with open(rf_path, 'w') as f:
            f.write("Top-20 Feature Importances (GradientBoosting):\n")
            importances = rule_model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]
            for rank, idx in enumerate(top_idx):
                f.write(f"  Feature_{idx}: importance={importances[idx]:.4f}\n")

    acc = accuracy_score(te_l, preds)
    bacc = balanced_accuracy_score(te_l, preds)
    mf1 = f1_score(te_l, preds, average='macro', zero_division=0)
    wf1 = f1_score(te_l, preds, average='weighted', zero_division=0)
    try:
        auc = roc_auc_score(te_l, probs, multi_class='ovr', average='macro') if probs is not None else 0.0
    except Exception:
        auc = 0.0
    results = dict(accuracy=float(acc), balanced_accuracy=float(bacc), macro_f1=float(mf1),
                   weighted_f1=float(wf1), auc=float(auc))
    print(f"  RuleFit: acc={acc:.4f}, bacc={bacc:.4f}, macro_f1={mf1:.4f}")

    model_dir = save_dir / 'rulefit'
    model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(backbone.state_dict(), model_dir / 'backbone_weights.pth')
    with open(model_dir / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(model_dir / 'selector.pkl', 'wb') as f:
        pickle.dump(selector, f)
    with open(model_dir / 'rule_model.pkl', 'wb') as f:
        pickle.dump(rule_model, f)
    _save_report(model_dir / 'report.json', results)

    nc2, names = nc, data['active_names']
    pred_code = f'''#!/usr/bin/env python3
"""Predict with MobileNetV2 + RuleFit"""
import os, json, argparse, pickle
import torch, torch.nn as nn
from torchvision import transforms, models
from PIL import Image

FRACTURE_NAMES = {names}

def predict(image_dir, model_dir, gpu=0):
    device = torch.device(f"cuda:{{gpu}}" if torch.cuda.is_available() else "cpu")
    backbone = models.mobilenet_v2(weights=None)
    backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, {nc2})
    backbone.load_state_dict(torch.load(os.path.join(model_dir, "backbone_weights.pth"), map_location="cpu"))
    backbone = backbone.to(device).eval()
    scaler = pickle.load(open(os.path.join(model_dir, "scaler.pkl"), "rb"))
    selector = pickle.load(open(os.path.join(model_dir, "selector.pkl"), "rb"))
    rule_model = pickle.load(open(os.path.join(model_dir, "rule_model.pkl"), "rb"))
    feat_ext = nn.Sequential(*list(backbone.features.children()))
    pool = nn.AdaptiveAvgPool2d(1)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    results = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = pool(feat_ext(tensor)).flatten(1).cpu().numpy()
        feat_s = scaler.transform(feat)
        feat_sel = selector.transform(feat_s)
        pred = rule_model.predict(feat_sel)[0]
        results.append({{"file": f, "pred_class": int(pred), "pred_name": FRACTURE_NAMES[pred]}})
    for r in results:
        print(f"{{r['file']}}: {{r['pred_name']}}")
    out_path = os.path.join(model_dir, "predict_results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {{out_path}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.model_dir, args.gpu)
'''
    with open(model_dir / 'predict.py', 'w') as f:
        f.write(pred_code)
    with open(model_dir / 'model.py', 'w') as f:
        f.write(f'"""MobileNetV2 + RuleFit for Fracture Classification"""\nimport torch, pickle\nfrom torchvision import models\nimport torch.nn as nn\n\nFRACTURE_NAMES = {names}\n')
    plot_training_curves(rulefit_history, model_dir, 'MobileNetV2+RuleFit')
    print(f"  Saved model to {model_dir}/")
    return results


# ======================================================================
# Group 4: Neuro-symbolic Baselines
# ======================================================================

# ======================================================================
# CBM (Concept Bottleneck Model) — following original CBM design
#   - Independent FC heads per concept (nn.ModuleList)
#   - Per-concept BCEWithLogitsLoss with class weighting
#   - End2EndModel: X→C (concept_heads) + C→Y (MLP classifier)
#   - use_sigmoid=True between X→C and C→Y
# ======================================================================

class CBMModel(nn.Module):
    """Concept Bottleneck Model following the original CBM paper design.

    Key design choices aligned with original:
    - Each concept has its own independent FC head (nn.ModuleList)
    - Concepts are predicted with BCEWithLogitsLoss (per-concept)
    - C→Y classifier takes sigmoid-activated concept probabilities
    - Supports class weighting for imbalanced concept labels
    """

    def __init__(self, backbone, num_concepts, num_classes, expand_dim=64):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.backbone = backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()

        # Independent FC head for each concept (original CBM design)
        self.concept_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, expand_dim), nn.ReLU(), nn.Linear(expand_dim, 1))
            for _ in range(num_concepts)
        ])

        # C→Y classifier: MLP taking sigmoid(concept_logits) as input
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, expand_dim),
            nn.ReLU(),
            nn.Linear(expand_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        # Each concept head outputs a 1-dim logit
        concept_logits = torch.cat([head(features) for head in self.concept_heads], dim=1)
        concept_probs = torch.sigmoid(concept_logits)
        logits = self.classifier(concept_probs)
        return logits, concept_probs, concept_logits


# ======================================================================
# CRL (Concept Rule Learner) — following original CRL design
#   - BinarizeLayer: hard binarization + Gradient Grafting
#   - UnionLayer: ConjunctionLayer(AND) + DisjunctionLayer(OR)
#   - LRLayer: final linear classification
#   - Trainable softmax temperature
#   - ClipWeights after each batch
#   - Dual learning rate (concept lr/100, rule layers lr)
# ======================================================================

class _GradGraft(torch.autograd.Function):
    """Gradient Grafting: forward uses binary result, backward uses continuous gradient."""
    @staticmethod
    def forward(ctx, X, Y):
        return X
    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class _Binarizer(torch.autograd.Function):
    """Hard binarization with straight-through estimator."""
    @staticmethod
    def forward(_, concepts):
        return (concepts.detach() > 0.0).float()
    @staticmethod
    def backward(_, grad_output):
        return grad_output.clone()


class _BinarizeLayer(nn.Module):
    """Binarize concept predictions; optionally append NOT (1-x) of each concept."""
    def __init__(self, n_concepts, use_not=True):
        super().__init__()
        self.n_concepts = n_concepts
        self.use_not = use_not
        self.input_dim = n_concepts
        self.output_dim = 2 * n_concepts if use_not else n_concepts
        self.layer_type = "binarization"
        self.dim2id = {i: i for i in range(self.output_dim)}
        self.rule_name = None

    def forward(self, x):
        x = _Binarizer.apply(x)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        pass

    def get_rule_name(self, concept_names):
        self.rule_name = []
        for i in range(self.n_concepts):
            self.rule_name.append(concept_names[i])
        if self.use_not:
            for i in range(self.n_concepts):
                self.rule_name.append("~" + concept_names[i])


class _Product(torch.autograd.Function):
    """Tensor product for continuous AND/OR approximation."""
    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X + 1e-10), dim=1))
        ctx.save_for_backward(X, y)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        X, y = ctx.saved_tensors
        grad_input = grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + 1e-10))
        return grad_input


class _ConjunctionLayer(nn.Module):
    """Conjunction (AND) layer with Gradient Grafting."""
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))

    def forward(self, x):
        res_tilde = self._continuous_forward(x)
        res_bar = self._binarized_forward(x)
        return _GradGraft.apply(res_bar, res_tilde)

    def _continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return _Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)

    @torch.no_grad()
    def _binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = _Binarizer.apply(self.W - 0.5)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class _DisjunctionLayer(nn.Module):
    """Disjunction (OR) layer with Gradient Grafting."""
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))

    def forward(self, x):
        res_tilde = self._continuous_forward(x)
        res_bar = self._binarized_forward(x)
        return _GradGraft.apply(res_bar, res_tilde)

    def _continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - _Product.apply(1 - x.unsqueeze(-1) * self.W)

    @torch.no_grad()
    def _binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = _Binarizer.apply(self.W - 0.5)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)

    def clip(self):
        self.W.data.clamp_(0.0, 1.0)


class _UnionLayer(nn.Module):
    """UnionLayer = ConjunctionLayer(AND) + DisjunctionLayer(OR), following original CRL."""
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2
        self.con_layer = _ConjunctionLayer(input_dim, output_dim, use_not=use_not)
        self.dis_layer = _DisjunctionLayer(input_dim, output_dim, use_not=use_not)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

    @torch.no_grad()
    def binarized_forward(self, x):
        return torch.cat(
            [self.con_layer._binarized_forward(x), self.dis_layer._binarized_forward(x)], dim=1
        )

    def clip(self):
        self.con_layer.clip()
        self.dis_layer.clip()

    def l2_norm(self):
        return torch.sum(self.con_layer.W ** 2) + torch.sum(self.dis_layer.W ** 2)

    def edge_count(self):
        con_Wb = _Binarizer.apply(self.con_layer.W - 0.5)
        dis_Wb = _Binarizer.apply(self.dis_layer.W - 0.5)
        return torch.sum(con_Wb) + torch.sum(dis_Wb)


class _LRLayer(nn.Module):
    """Linear classification layer at the end of CRL."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc1(x)

    @torch.no_grad()
    def binarized_forward(self, x):
        return self.forward(x)

    def clip(self):
        for param in self.fc1.parameters():
            param.data.clamp_(-1.0, 1.0)

    def l2_norm(self):
        return torch.sum(self.fc1.weight ** 2)


class CRLModel(nn.Module):
    """Concept Rule Learner following the original CRL paper design.

    Key design choices aligned with original:
    - BinarizeLayer: hard binarization with Gradient Grafting
    - UnionLayer(s): ConjunctionLayer(AND) + DisjunctionLayer(OR)
    - LRLayer: final linear classification
    - Trainable softmax temperature
    - ClipWeights after each training batch
    - use_not: append NOT (1-x) of each concept for negation
    - use_skip: skip connections between layers
    """

    def __init__(self, backbone, num_concepts, num_classes,
                 l1=256, l2=256, use_not=True, use_skip=True, temperature=1.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.use_not = use_not
        self.use_skip = use_skip

        # Backbone + concept predictor
        self.backbone = backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.concept_predictor = nn.Linear(in_features, num_concepts)

        # Trainable temperature
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

        # Build layer list: BinarizeLayer → UnionLayer(s) → LRLayer
        self.layer_list = nn.ModuleList()
        dim_list = [num_concepts, l1, l2, num_classes]
        prev_layer_dim = None

        for idx, dim in enumerate(dim_list):
            # Compute effective input dimension (including skip connection)
            effective_input_dim = prev_layer_dim
            if use_skip and idx >= 3 and len(self.layer_list) >= 2:
                skip_dim = self.layer_list[-2].output_dim
                effective_input_dim = prev_layer_dim + skip_dim

            if idx == 0:
                layer = _BinarizeLayer(dim, use_not)
            elif idx == len(dim_list) - 1:
                # LRLayer: input includes skip connection
                layer = _LRLayer(effective_input_dim, dim)
            else:
                # First UnionLayer does NOT use_not (BinarizeLayer already appended NOT)
                layer_use_not = True if idx != 1 else False
                layer = _UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)

            prev_layer_dim = layer.output_dim
            self.layer_list.append(layer)

        # Store skip connection info
        self._skip_indices = {}
        for idx in range(3, len(dim_list)):
            self._skip_indices[idx] = idx - 2

    def forward(self, x):
        features = self.backbone(x)
        concept_logits = self.concept_predictor(features)
        concept_probs = torch.sigmoid(concept_logits)

        h = concept_logits  # Feed raw logits into BinarizeLayer
        skip_cache = {}

        for idx, layer in enumerate(self.layer_list):
            # Handle skip connection
            if idx in self._skip_indices:
                skip_idx = self._skip_indices[idx]
                h = torch.cat((h, skip_cache[skip_idx]), dim=1)
            h = layer(h)
            # Cache for skip connections
            if idx in self._skip_indices.values():
                skip_cache[idx] = h

        # Apply temperature scaling to logits
        logits = h / torch.exp(self.t)
        return logits, concept_probs, concept_logits

    @torch.no_grad()
    def bi_forward(self, x, count=False):
        """Binarized forward pass for rule extraction."""
        features = self.backbone(x)
        concept_logits = self.concept_predictor(features)
        concept_probs = torch.sigmoid(concept_logits)

        h = concept_logits
        skip_cache = {}

        for idx, layer in enumerate(self.layer_list):
            if idx in self._skip_indices:
                skip_idx = self._skip_indices[idx]
                h = torch.cat((h, skip_cache[skip_idx]), dim=1)
            if hasattr(layer, 'binarized_forward'):
                h = layer.binarized_forward(h)
            else:
                h = layer(h)
            if idx in self._skip_indices.values():
                skip_cache[idx] = h
            if count and hasattr(layer, 'node_activation_cnt'):
                layer.node_activation_cnt += torch.sum(h, dim=0)
                layer.forward_tot += h.shape[0]

        logits = h / torch.exp(self.t)
        return logits, concept_probs

    def clip_weights(self):
        """Clip weights after each training batch (original CRL ClipWeights callback)."""
        for layer in self.layer_list[:-1]:
            layer.clip()

    def l2_penalty(self):
        """L2 penalty on UnionLayer weights (original CRL regularization)."""
        penalty = 0.0
        for layer in self.layer_list[1:]:
            if hasattr(layer, 'l2_norm'):
                penalty += layer.l2_norm()
        return penalty

    def get_rules(self, threshold=0.5, active_names=None, concept_names=None):
        """Extract rules from the CRL model by analyzing UnionLayer weights."""
        rules = {}
        if concept_names is None:
            concept_names = [f"c{i}" for i in range(self.num_concepts)]

        # Get UnionLayer weights for rule extraction
        union_layers = [l for l in self.layer_list if isinstance(l, _UnionLayer)]
        lr_layer = self.layer_list[-1]

        # Simple rule extraction based on UnionLayer edge analysis
        for ci in range(self.num_classes):
            cn = active_names[ci] if active_names else f"class_{ci}"
            conds = []
            for ul_idx, ul in enumerate(union_layers):
                # Analyze conjunction weights
                W_con = ul.con_layer.W.detach().cpu().numpy()
                W_dis = ul.dis_layer.W.detach().cpu().numpy()
                for j in range(W_con.shape[1]):
                    active_con = np.where(W_con[:, j] > threshold)[0]
                    if len(active_con) > 0:
                        terms = []
                        for a in active_con:
                            if a < len(concept_names):
                                terms.append(concept_names[a])
                            else:
                                terms.append("~" + concept_names[a - len(concept_names)])
                        conds.append(f"AND({', '.join(terms)})")
                for j in range(W_dis.shape[1]):
                    active_dis = np.where(W_dis[:, j] > threshold)[0]
                    if len(active_dis) > 0:
                        terms = []
                        for a in active_dis:
                            if a < len(concept_names):
                                terms.append(concept_names[a])
                            else:
                                terms.append("~" + concept_names[a - len(concept_names)])
                        conds.append(f"OR({', '.join(terms)})")
            rules[cn] = conds if conds else ["(no active rule)"]
        return rules


def _get_concept_labels(labels_np, active_classes, num_concepts):
    cl = []
    for l in labels_np:
        orig_cls = active_classes[l]
        cl.append(CONCEPT_CLASS_MAP.get(int(orig_cls), [0]*num_concepts))
    return np.array(cl, dtype=np.float32)


def run_cbm(data, device, save_dir):
    """CBM training following original CBM paper design.

    Key changes from previous simplified implementation:
    - Independent FC heads per concept with per-concept BCEWithLogitsLoss
    - Class-weighted concept loss (upweight rare concept values)
    - SGD optimizer with StepLR scheduler (original CBM default)
    - Two-stage option: Independent (X→C then C→Y) or Joint (end-to-end)
    """
    print(f"\n  --- CBM (Concept Bottleneck Model) ---", flush=True)
    nc = data['num_classes']
    ncon = len(FRACTURE_CONCEPTS)
    cw = class_weights(data['train_labels'], nc)
    train_l, val_l, test_l = make_loaders(data)

    backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model = CBMModel(backbone, ncon, nc, expand_dim=64).to(device)

    # Original CBM uses SGD + StepLR; we keep AdamW + CosineAnnealingLR for stability
    # but add per-concept class weighting
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))

    # Per-concept BCEWithLogitsLoss with class weighting (original CBM design)
    # Compute concept class weights: upweight rare concept values
    concept_labels_train = _get_concept_labels(data['train_labels'], data['active_classes'], ncon)
    concept_weights_list = []
    for ci in range(ncon):
        pos_count = concept_labels_train[:, ci].sum()
        neg_count = len(concept_labels_train) - pos_count
        if pos_count > 0 and neg_count > 0:
            w_pos = neg_count / (pos_count + neg_count)
            w_neg = pos_count / (pos_count + neg_count)
        else:
            w_pos, w_neg = 0.5, 0.5
        concept_weights_list.append(torch.tensor([w_neg, w_pos], dtype=torch.float32).to(device))

    concept_criterions = [
        nn.BCEWithLogitsLoss(pos_weight=cw_[1].reshape(1))
        for cw_ in concept_weights_list
    ]

    num_epochs = 200
    patience = 30
    cbm_history = {'loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_va, best_st, no_improve = 0, None, 0

    for epoch in range(num_epochs):
        model.train()
        tl, c, t = 0, 0, 0
        for images, labels in train_l:
            images, labels = images.to(device), labels.to(device)
            ct = torch.tensor(
                _get_concept_labels(labels.cpu().numpy(), data['active_classes'], ncon),
                dtype=torch.float32
            ).to(device)

            logits, concept_probs, concept_logits = model(images)

            # Classification loss
            loss_y = ce_loss(logits, labels)

            # Per-concept loss (original CBM: independent BCEWithLogitsLoss per concept)
            loss_c = 0.0
            for ci in range(ncon):
                loss_c += concept_criterions[ci](concept_logits[:, ci], ct[:, ci])
            loss_c = loss_c / ncon

            # Total loss: CE + weighted concept loss (original uses additional_loss_weighting)
            loss = loss_y + 0.5 * loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tl += loss.item() * images.size(0)
            c += (logits.argmax(1) == labels).sum().item()
            t += images.size(0)

        scheduler.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for images, labels in val_l:
                logits, _, _ = model(images.to(device))
                vc += (logits.argmax(1) == labels.to(device)).sum().item()
                vt += images.size(0)
        va = vc / vt

        cbm_history['loss'].append(tl / t)
        cbm_history['train_acc'].append(c / t)
        cbm_history['val_acc'].append(va)
        cbm_history['lr'].append(optimizer.param_groups[0]['lr'])

        if va > best_va:
            best_va = va
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}: loss={tl/t:.4f}, train_acc={c/t:.4f}, val_acc={va:.4f}")
        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_st:
        model.load_state_dict(best_st)
    model = model.to(device)

    # Evaluate — need wrapper for evaluate_model which expects (logits, concepts) = model(x)
    class _CBMEvalWrapper(nn.Module):
        def __init__(self, cbm_model):
            super().__init__()
            self.cbm = cbm_model
        def forward(self, x):
            logits, _, _ = self.cbm(x)
            return logits, _
    eval_model = _CBMEvalWrapper(model)
    results = evaluate_model(eval_model, test_l, device, nc)
    results['best_val_accuracy'] = float(best_va)

    concept_accs = {}
    model.eval()
    acp, act = [], []
    with torch.no_grad():
        for images, labels in test_l:
            _, concept_probs, _ = model(images.to(device))
            ct = torch.tensor(_get_concept_labels(labels.numpy(), data['active_classes'], ncon), dtype=torch.float32)
            acp.append(concept_probs.cpu().numpy())
            act.append(ct.numpy())
    acp, act = np.concatenate(acp), np.concatenate(act)
    for i, name in enumerate(FRACTURE_CONCEPTS):
        pb = (acp[:, i] > 0.5).astype(float)
        concept_accs[name] = float((pb == act[:, i]).mean())
    results['concept_accuracy'] = concept_accs
    print(f"  CBM: acc={results['accuracy']:.4f}, bacc={results['balanced_accuracy']:.4f}")

    model_dir = save_dir / 'cbm'
    model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_dir / 'best_weights.pth')
    _save_report(model_dir / 'report.json', results)
    plot_training_curves(cbm_history, model_dir, 'CBM')

    names = data['active_names']
    pred_code = f'''#!/usr/bin/env python3
"""Predict with CBM (original design: independent concept heads)"""
import os, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

FRACTURE_NAMES = {names}
FRACTURE_CONCEPTS = {FRACTURE_CONCEPTS}

class CBMModel(nn.Module):
    def __init__(self, num_concepts={ncon}, num_classes={nc}, expand_dim=64):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        backbone = models.resnet34(weights=None)
        self.backbone = backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.concept_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, expand_dim), nn.ReLU(), nn.Linear(expand_dim, 1))
            for _ in range(num_concepts)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, expand_dim), nn.ReLU(), nn.Linear(expand_dim, num_classes)
        )
    def forward(self, x):
        features = self.backbone(x)
        concept_logits = torch.cat([head(features) for head in self.concept_heads], dim=1)
        concept_probs = torch.sigmoid(concept_logits)
        logits = self.classifier(concept_probs)
        return logits, concept_probs, concept_logits

def predict(image_dir, weights_path, gpu=0):
    device = torch.device(f"cuda:{{gpu}}" if torch.cuda.is_available() else "cpu")
    model = CBMModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    results = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, concepts, _ = model(tensor)
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
            cv = {{FRACTURE_CONCEPTS[i]: float(concepts[0, i].item()) for i in range(len(FRACTURE_CONCEPTS))}}
        results.append({{"file": f, "pred_class": pred, "pred_name": FRACTURE_NAMES[pred],
                         "confidence": prob[0, pred].item(), "concepts": cv}})
    for r in results:
        print(f"{{r['file']}}: {{r['pred_name']}} ({{r['confidence']:.4f}})")
        tc = sorted(r['concepts'].items(), key=lambda x: -x[1])[:3]
        print(f"  Top concepts: {{tc}}")
    out_path = os.path.join(os.path.dirname(weights_path), "predict_results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {{out_path}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_weights.pth"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.weights, args.gpu)
'''
    with open(model_dir / 'predict.py', 'w') as f:
        f.write(pred_code)
    print(f"  Saved model to {model_dir}/")
    return results


def run_crl(data, device, save_dir):
    """CRL training following original CRL paper design.

    Key changes from previous simplified implementation:
    - Full CRL architecture: BinarizeLayer → UnionLayer(AND/OR) → LRLayer
    - Gradient Grafting: forward uses binary, backward uses continuous
    - Trainable softmax temperature
    - ClipWeights after each batch (W.clamp_ for Union/LR layers)
    - Dual learning rate: concept predictor lr/100, rule layers lr
    - L2 regularization on UnionLayer weights (l2_weight=5e-6)
    - Loss = concept_loss(BCE) + l2_weight * l2_penalty + rrl_loss(CE)
    """
    print(f"\n  --- CRL (Concept Rule Learner) ---", flush=True)
    nc = data['num_classes']
    ncon = len(FRACTURE_CONCEPTS)
    cw = class_weights(data['train_labels'], nc)
    train_l, val_l, test_l = make_loaders(data)

    backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model = CRLModel(backbone, ncon, nc, l1=256, l2=256,
                     use_not=True, use_skip=True, temperature=1.0).to(device)

    # Dual learning rate (original CRL design):
    # concept predictor: lr/100, weight_decay=1e-2
    # rule layers: lr, weight_decay=0
    base_lr = 5e-4
    concept_params = [p for n, p in model.named_parameters() if 'concept' in n]
    rule_params = [p for n, p in model.named_parameters() if 'concept' not in n]

    optimizer = torch.optim.AdamW([
        {'params': concept_params, 'lr': base_lr / 100, 'weight_decay': 1e-2},
        {'params': rule_params, 'lr': base_lr, 'weight_decay': 0},
    ], lr=base_lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

    # Loss functions (original CRL design)
    ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    bce_loss = nn.BCELoss()
    l2_weight = 5e-6  # Original CRL default

    num_epochs = 200
    patience = 30
    crl_history = {'loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    best_va, best_st, no_improve = 0, None, 0

    # Initialize node activation counters for rule extraction
    for layer in model.layer_list:
        if hasattr(layer, 'output_dim'):
            layer.node_activation_cnt = torch.zeros(layer.output_dim, dtype=torch.double)
            layer.forward_tot = 0

    for epoch in range(num_epochs):
        model.train()
        tl, c, t = 0, 0, 0
        for images, labels in train_l:
            images, labels = images.to(device), labels.to(device)
            ct = torch.tensor(
                _get_concept_labels(labels.cpu().numpy(), data['active_classes'], ncon),
                dtype=torch.float32
            ).to(device)

            logits, concept_probs, concept_logits = model(images)

            # Concept loss: BCE on sigmoid(concept_logits) vs concept labels
            concept_loss = bce_loss(concept_probs, ct)

            # L2 penalty on UnionLayer weights
            l2_loss = l2_weight * model.l2_penalty()

            # Classification loss
            rrl_loss = ce_loss(logits, labels)

            # Total loss (original CRL: concept_loss + l2_loss + rrl_loss)
            loss = concept_loss + l2_loss + rrl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ClipWeights after each batch (original CRL ClipWeights callback)
            model.clip_weights()

            tl += loss.item() * images.size(0)
            c += (logits.argmax(1) == labels).sum().item()
            t += images.size(0)

        scheduler.step()

        model.eval()
        vc, vt = 0, 0
        with torch.no_grad():
            for images, labels in val_l:
                logits, _, _ = model(images.to(device))
                vc += (logits.argmax(1) == labels.to(device)).sum().item()
                vt += images.size(0)
        va = vc / vt

        crl_history['loss'].append(tl / t)
        crl_history['train_acc'].append(c / t)
        crl_history['val_acc'].append(va)
        crl_history['lr'].append(optimizer.param_groups[0]['lr'])

        if va > best_va:
            best_va = va
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{num_epochs}: loss={tl/t:.4f}, train_acc={c/t:.4f}, val_acc={va:.4f}")
        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_st:
        model.load_state_dict(best_st)
    model = model.to(device)

    # Evaluate — wrapper for evaluate_model
    class _CRLEvalWrapper(nn.Module):
        def __init__(self, crl_model):
            super().__init__()
            self.crl = crl_model
        def forward(self, x):
            logits, _, _ = self.crl(x)
            return logits, _
    eval_model = _CRLEvalWrapper(model)
    results = evaluate_model(eval_model, test_l, device, nc)
    results['best_val_accuracy'] = float(best_va)

    # Rule extraction
    rules = model.get_rules(threshold=0.5, active_names=data['active_names'], concept_names=FRACTURE_CONCEPTS)
    results['learned_rules'] = rules
    print(f"  CRL: acc={results['accuracy']:.4f}, bacc={results['balanced_accuracy']:.4f}")

    model_dir = save_dir / 'crl'
    model_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), model_dir / 'best_weights.pth')
    _save_report(model_dir / 'report.json', results)
    plot_training_curves(crl_history, model_dir, 'CRL')

    rule_file = model_dir / 'crl_rules.txt'
    with open(rule_file, 'w') as f:
        f.write("CRL Learned Rules (original CRL design: AND/OR logic rules):\n")
        for cls_name, conditions in rules.items():
            f.write(f"\n  {cls_name}:\n")
            for cond in conditions:
                f.write(f"    {cond}\n")

    names = data['active_names']
    pred_code = f'''#!/usr/bin/env python3
"""Predict with CRL (original design: BinarizeLayer + UnionLayer + LRLayer)"""
import os, json, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

FRACTURE_NAMES = {names}
FRACTURE_CONCEPTS = {FRACTURE_CONCEPTS}

class _GradGraft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y): return X
    @staticmethod
    def backward(ctx, grad_output): return None, grad_output.clone()

class _Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(_, concepts): return (concepts.detach() > 0.0).float()
    @staticmethod
    def backward(_, grad_output): return grad_output.clone()

class _BinarizeLayer(nn.Module):
    def __init__(self, n_concepts, use_not=True):
        super().__init__()
        self.n_concepts = n_concepts
        self.use_not = use_not
        self.output_dim = 2 * n_concepts if use_not else n_concepts
    def forward(self, x):
        x = _Binarizer.apply(x)
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        return x

class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X + 1e-10), dim=1))
        ctx.save_for_backward(X, y)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        X, y = ctx.saved_tensors
        return grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + 1e-10))

class _ConjunctionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))
    def forward(self, x):
        res_tilde = self._continuous_forward(x)
        res_bar = self._binarized_forward(x)
        return _GradGraft.apply(res_bar, res_tilde)
    def _continuous_forward(self, x):
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        return _Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)
    @torch.no_grad()
    def _binarized_forward(self, x):
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        Wb = _Binarizer.apply(self.W - 0.5)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)

class _DisjunctionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))
    def forward(self, x):
        res_tilde = self._continuous_forward(x)
        res_bar = self._binarized_forward(x)
        return _GradGraft.apply(res_bar, res_tilde)
    def _continuous_forward(self, x):
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        return 1 - _Product.apply(1 - x.unsqueeze(-1) * self.W)
    @torch.no_grad()
    def _binarized_forward(self, x):
        if self.use_not: x = torch.cat((x, 1 - x), dim=1)
        Wb = _Binarizer.apply(self.W - 0.5)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)

class _UnionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2
        self.con_layer = _ConjunctionLayer(input_dim, output_dim, use_not=use_not)
        self.dis_layer = _DisjunctionLayer(input_dim, output_dim, use_not=use_not)
    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)

class _LRLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc1(x)

class CRLModel(nn.Module):
    def __init__(self, num_concepts={ncon}, num_classes={nc}, l1=256, l2=256, use_not=True, use_skip=True, temperature=1.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.use_not = use_not
        self.use_skip = use_skip
        backbone = models.resnet34(weights=None)
        self.backbone = backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.concept_predictor = nn.Linear(in_features, num_concepts)
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))
        self.layer_list = nn.ModuleList()
        dim_list = [num_concepts, l1, l2, num_classes]
        prev_layer_dim = None
        for idx, dim in enumerate(dim_list):
            if idx == 0:
                layer = _BinarizeLayer(dim, use_not)
            elif idx == len(dim_list) - 1:
                layer = _LRLayer(prev_layer_dim, dim)
            else:
                layer_use_not = True if idx != 1 else False
                layer = _UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)
            prev_layer_dim = layer.output_dim
            if use_skip and idx >= 3:
                skip_dim = self.layer_list[-2].output_dim
                prev_layer_dim += skip_dim
            self.layer_list.append(layer)
        self._skip_indices = {{}}
        for idx in range(3, len(dim_list)):
            self._skip_indices[idx] = idx - 2
    def forward(self, x):
        features = self.backbone(x)
        concept_logits = self.concept_predictor(features)
        concept_probs = torch.sigmoid(concept_logits)
        h = concept_logits
        skip_cache = {{}}
        for idx, layer in enumerate(self.layer_list):
            if idx in self._skip_indices:
                skip_idx = self._skip_indices[idx]
                h = torch.cat((h, skip_cache[skip_idx]), dim=1)
            h = layer(h)
            if idx in self._skip_indices.values():
                skip_cache[idx] = h
        logits = h / torch.exp(self.t)
        return logits, concept_probs, concept_logits

def predict(image_dir, weights_path, gpu=0):
    device = torch.device(f"cuda:{{gpu}}" if torch.cuda.is_available() else "cpu")
    model = CRLModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model = model.to(device).eval()
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    results = []
    for f in sorted(os.listdir(image_dir)):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        img = Image.open(os.path.join(image_dir, f)).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, concepts, _ = model(tensor)
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
            cv = {{FRACTURE_CONCEPTS[i]: float(concepts[0, i].item()) for i in range(len(FRACTURE_CONCEPTS))}}
        results.append({{"file": f, "pred_class": pred, "pred_name": FRACTURE_NAMES[pred],
                         "confidence": prob[0, pred].item(), "concepts": cv}})
    for r in results:
        print(f"{{r['file']}}: {{r['pred_name']}} ({{r['confidence']:.4f}})")
        tc = sorted(r['concepts'].items(), key=lambda x: -x[1])[:3]
        print(f"  Top concepts: {{tc}}")
    out_path = os.path.join(os.path.dirname(weights_path), "predict_results.json")
    with open(out_path, "w") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"Saved to {{out_path}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_weights.pth"))
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    predict(args.image_dir, args.weights, args.gpu)
'''
    with open(model_dir / 'predict.py', 'w') as f:
        f.write(pred_code)
    print(f"  Saved model to {model_dir}/")
    return results


# ======================================================================
# Main
# ======================================================================

METHOD_MAP = {
    'resnet50': run_resnet50, 'densenet121': run_densenet121, 'efficientnet_b0': run_efficientnet_b0,
    'swin_tiny': run_swin_tiny, 'dinov2_linear': run_dinov2_linear,
    'gradcam': run_gradcam, 'rulefit': run_rulefit,
    'cbm': run_cbm, 'crl': run_crl,
}


def main():
    parser = argparse.ArgumentParser(description='Baseline Comparison for Fracture Classification')
    parser.add_argument('--config', type=str, default='configs/fracture_v3_expanded.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--methods', nargs='+', default=None,
                        help=f'Methods to run: {ALL_METHODS}. Default: all')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    data = load_split_data(config)
    output_dir = Path(config['output_dir']) / 'baseline_comparison'
    output_dir.mkdir(exist_ok=True, parents=True)

    methods = args.methods if args.methods else ALL_METHODS
    methods = [m for m in methods if m in METHOD_MAP]

    all_results = {}
    for method_name in methods:
        print(f"\n{'='*70}")
        print(f"  Running: {method_name} ({GROUP_LABELS.get(method_name, '')})")
        print(f"{'='*70}")
        t0 = time.time()
        try:
            result = METHOD_MAP[method_name](data, device, output_dir)
            result['group'] = GROUP_LABELS.get(method_name, '')
            result['method'] = method_name
            result['elapsed_seconds'] = int(time.time() - t0)
            all_results[method_name] = result
        except Exception as e:
            print(f"  [ERROR] {method_name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[method_name] = {'error': str(e)}

    print(f"\n\n{'='*70}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Method':20s} {'Group':25s} {'Acc':>6s} {'BAcc':>6s} {'F1':>6s} {'AUC':>6s}")
    print(f"  {'-'*70}")
    for m in methods:
        r = all_results.get(m, {})
        if 'error' in r:
            print(f"  {m:20s} {GROUP_LABELS.get(m, ''):25s} ERROR: {r['error'][:30]}")
            continue
        print(f"  {m:20s} {r.get('group', ''):25s} "
              f"{r.get('accuracy', 0):.4f} "
              f"{r.get('balanced_accuracy', 0):.4f} "
              f"{r.get('macro_f1', 0):.4f} "
              f"{r.get('auc', 0):.4f}")

    results_path = output_dir / 'baseline_comparison_results.json'
    serializable = {}
    for k, v in all_results.items():
        sv = {}
        for kk, vv in v.items():
            if isinstance(vv, (np.floating, np.integer)):
                vv = float(vv)
            elif isinstance(vv, np.ndarray):
                vv = vv.tolist()
            elif isinstance(vv, dict):
                vv = {str(kkk): float(vvv) if isinstance(vvv, (np.floating, np.integer)) else vvv
                      for kkk, vvv in vv.items()}
            sv[kk] = vv
        serializable[k] = sv
    with open(results_path, 'w') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved summary to {results_path}")


if __name__ == '__main__':
    main()
