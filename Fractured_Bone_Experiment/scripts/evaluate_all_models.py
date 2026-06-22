#!/usr/bin/env python3
"""
Unified evaluation: load all trained models (baseline + symbolic),
predict on the same test set, and save prediction probabilities for comparison.

Usage:
    python scripts/evaluate_all_models.py --config configs/fracture_v3_expanded.yaml --gpu 0
"""

import argparse, json, os, sys, time, pickle
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from src.data.fracture_loader import FRACTURE_NAMES


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
    split_file = str(Path(config['output_dir']) / 'split_indices.npz')
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
    return dict(train_paths=tp, val_paths=vp, test_paths=tep,
                train_labels=tl, val_labels=vl, test_labels=tel,
                num_classes=num_classes, active_names=active_names,
                active_classes=active_classes, class_map=class_map)


# ============================================================
# Model definitions (must match training code exactly)
# ============================================================

FRACTURE_CONCEPTS = [
    'cortical_break', 'fracture_line_horizontal', 'fracture_line_oblique_45',
    'fracture_line_oblique_135', 'fracture_line_vertical', 'displacement',
    'bone_fragment', 'soft_tissue_swelling',
]


class CBMModel(nn.Module):
    def __init__(self, backbone, num_concepts, num_classes, expand_dim=64):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.backbone = backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.concept_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_features, expand_dim), nn.ReLU(), nn.Linear(expand_dim, 1))
            for _ in range(num_concepts)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, expand_dim),
            nn.ReLU(),
            nn.Linear(expand_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        concept_logits = torch.cat([head(features) for head in self.concept_heads], dim=1)
        concept_probs = torch.sigmoid(concept_logits)
        logits = self.classifier(concept_probs)
        return logits, concept_probs, concept_logits


class _GradGraft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y):
        return X
    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class _Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(_, concepts):
        return (concepts.detach() > 0.0).float()
    @staticmethod
    def backward(_, grad_output):
        return grad_output.clone()


class _BinarizeLayer(nn.Module):
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


class _Product(torch.autograd.Function):
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


class _LRLayer(nn.Module):
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
    def __init__(self, backbone, num_concepts, num_classes,
                 l1=256, l2=256, use_not=True, use_skip=True, temperature=1.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.use_not = use_not
        self.use_skip = use_skip
        self.backbone = backbone
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.concept_predictor = nn.Linear(in_features, num_concepts)
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

        self.layer_list = nn.ModuleList()
        dim_list = [num_concepts, l1, l2, num_classes]
        prev_layer_dim = None

        for idx, dim in enumerate(dim_list):
            effective_input_dim = prev_layer_dim
            if use_skip and idx >= 3 and len(self.layer_list) >= 2:
                skip_dim = self.layer_list[-2].output_dim
                effective_input_dim = prev_layer_dim + skip_dim

            if idx == 0:
                layer = _BinarizeLayer(dim, use_not)
            elif idx == len(dim_list) - 1:
                layer = _LRLayer(effective_input_dim, dim)
            else:
                layer_use_not = True if idx != 1 else False
                layer = _UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)

            prev_layer_dim = layer.output_dim
            self.layer_list.append(layer)

        self._skip_indices = {}
        for idx in range(3, len(dim_list)):
            self._skip_indices[idx] = idx - 2

    def forward(self, x):
        features = self.backbone(x)
        concept_logits = self.concept_predictor(features)
        concept_probs = torch.sigmoid(concept_logits)
        h = concept_logits
        skip_cache = {}
        for idx, layer in enumerate(self.layer_list):
            if idx in self._skip_indices:
                skip_idx = self._skip_indices[idx]
                h = torch.cat((h, skip_cache[skip_idx]), dim=1)
            h = layer(h)
            if idx in self._skip_indices.values():
                skip_cache[idx] = h
        logits = h / torch.exp(self.t)
        return logits, concept_probs, concept_logits


# ============================================================
# Evaluation functions
# ============================================================

def get_test_loader(data, batch_size=32):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                    transforms.ToTensor(), norm])
    ds = FractureDataset(data['test_paths'], data['test_labels'], transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


def predict_deep_model(model, loader, device):
    """Standard deep model prediction returning (preds, probs, labels)."""
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
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def evaluate_standard_model(name, model, loader, device):
    """Evaluate a standard deep model and return results dict."""
    preds, probs, labels = predict_deep_model(model, loader, device)
    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    print(f"  {name}: Acc={acc:.4f}, BAcc={bacc:.4f}, F1={mf1:.4f}, AUC={auc:.4f}")
    return dict(preds=preds, probs=probs, labels=labels,
                accuracy=float(acc), balanced_accuracy=float(bacc),
                macro_f1=float(mf1), auc=float(auc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/fracture_v3_expanded.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(config['output_dir'])
    bl_dir = output_dir / 'baseline_comparison'
    data = load_split_data(config)
    nc = data['num_classes']
    loader = get_test_loader(data)
    test_labels = data['test_labels']

    all_results = {}

    # ------------------------------------------------------------------
    # 1. CNN baselines: ResNet50, DenseNet121, EfficientNet-B0
    # ------------------------------------------------------------------
    cnn_configs = {
        'resnet50': lambda nc: (models.resnet50(weights=None), lambda m, nc: setattr(m, 'fc', nn.Linear(m.fc.in_features, nc))),
        'densenet121': lambda nc: (models.densenet121(weights=None), lambda m, nc: setattr(m, 'classifier', nn.Linear(m.classifier.in_features, nc))),
        'efficientnet_b0': lambda nc: (models.efficientnet_b0(weights=None), lambda m, nc: setattr(m, 'classifier', nn.Sequential(nn.Dropout(p=0.2), nn.Linear(m.classifier[1].in_features, nc)))),
    }
    for name, builder in cnn_configs.items():
        wpath = bl_dir / name / 'best_weights.pth'
        if not wpath.exists():
            print(f"  [SKIP] {name}: weights not found")
            continue
        model, set_head = builder(nc)
        set_head(model, nc)
        state = torch.load(wpath, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        all_results[name] = evaluate_standard_model(name, model, loader, device)

    # ------------------------------------------------------------------
    # 2. Swin-Tiny (DeiT-Small fallback)
    # ------------------------------------------------------------------
    swin_path = bl_dir / 'swin_tiny' / 'best_weights.pth'
    if swin_path.exists():
        swin_report = json.load(open(bl_dir / 'swin_tiny' / 'report.json'))
        variant = swin_report.get('model_variant', 'deit_small_patch16_224')
        loaded = False

        # Try timm models first
        try:
            import timm
            variants_to_try = [variant, 'deit_small_patch16_224', 'swin_tiny_patch4_window7_224']
            for v in variants_to_try:
                try:
                    model = timm.create_model(v, pretrained=False, num_classes=nc)
                    state = torch.load(swin_path, map_location=device, weights_only=True)
                    model.load_state_dict(state)
                    model.to(device)
                    all_results['swin_tiny'] = evaluate_standard_model(f'swin_tiny ({v})', model, loader, device)
                    loaded = True
                    break
                except Exception:
                    continue
        except ImportError:
            pass

        # Try torchvision ViT-B-16
        if not loaded and variant == 'vit_b_16':
            try:
                from torchvision.models import vit_b_16
                model = vit_b_16(weights=None)
                model.heads.head = nn.Linear(model.heads.head.in_features, nc)
                state = torch.load(swin_path, map_location=device, weights_only=True)
                model.load_state_dict(state)
                model.to(device)
                all_results['swin_tiny'] = evaluate_standard_model(f'swin_tiny (vit_b_16)', model, loader, device)
                loaded = True
            except Exception as e:
                print(f"  [ERROR] swin_tiny vit_b_16: {e}")

        if not loaded:
            # Fallback to saved report
            all_results['swin_tiny'] = dict(
                preds=None, probs=None, labels=test_labels,
                accuracy=swin_report.get('accuracy', 0),
                balanced_accuracy=swin_report.get('balanced_accuracy', 0),
                macro_f1=swin_report.get('macro_f1', 0),
                auc=swin_report.get('auc', 0))
            print(f"  swin_tiny (from report): Acc={swin_report.get('accuracy', 0):.4f}")

    # ------------------------------------------------------------------
    # 3. GradCAM (ResNet50 backbone)
    # ------------------------------------------------------------------
    gradcam_path = bl_dir / 'gradcam' / 'best_weights.pth'
    if gradcam_path.exists():
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, nc)
        state = torch.load(gradcam_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        all_results['gradcam'] = evaluate_standard_model('gradcam', model, loader, device)

    # ------------------------------------------------------------------
    # 4. DINOv2 + Linear Probe
    # ------------------------------------------------------------------
    dinov2_dir = bl_dir / 'dinov2_linear'
    if (dinov2_dir / 'scaler.pkl').exists() and (dinov2_dir / 'classifier.pkl').exists():
        print("  [DINOv2] Loading backbone and extracting features...")
        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
        backbone.eval().to(device)
        for p in backbone.parameters():
            p.requires_grad = False

        dino_t = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
        dino_ds = FractureDataset(data['test_paths'], data['test_labels'], dino_t)
        dino_loader = DataLoader(dino_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

        feats = []
        with torch.no_grad():
            for images, _ in dino_loader:
                feats.append(backbone(images.to(device)).cpu().numpy())
        feats = np.concatenate(feats)

        scaler = pickle.load(open(dinov2_dir / 'scaler.pkl', 'rb'))
        clf = pickle.load(open(dinov2_dir / 'classifier.pkl', 'rb'))
        feats_s = scaler.transform(feats)
        preds = clf.predict(feats_s)
        try:
            probs = clf.predict_proba(feats_s)
        except AttributeError:
            # sklearn version mismatch - retrain a quick LR
            print("  [WARN] DINOv2 classifier version mismatch, retraining LR...")
            from sklearn.linear_model import LogisticRegression
            # Re-extract train features
            train_ds = FractureDataset(data['train_paths'], data['train_labels'], dino_t)
            val_ds = FractureDataset(data['val_paths'], data['val_labels'], dino_t)
            tr_feats, tr_labels = [], []
            with torch.no_grad():
                for images, labels in DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True):
                    tr_feats.append(backbone(images.to(device)).cpu().numpy())
                    tr_labels.append(labels.numpy())
                for images, labels in DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True):
                    tr_feats.append(backbone(images.to(device)).cpu().numpy())
                    tr_labels.append(labels.numpy())
            tr_feats = scaler.transform(np.concatenate(tr_feats))
            tr_labels = np.concatenate(tr_labels)
            clf2 = LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced', solver='lbfgs')
            clf2.fit(tr_feats, tr_labels)
            preds = clf2.predict(feats_s)
            probs = clf2.predict_proba(feats_s)

        acc = accuracy_score(test_labels, preds)
        bacc = balanced_accuracy_score(test_labels, preds)
        mf1 = f1_score(test_labels, preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(test_labels, probs, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
        print(f"  dinov2_linear: Acc={acc:.4f}, BAcc={bacc:.4f}, F1={mf1:.4f}, AUC={auc:.4f}")
        all_results['dinov2_linear'] = dict(preds=preds, probs=probs, labels=test_labels,
                                             accuracy=float(acc), balanced_accuracy=float(bacc),
                                             macro_f1=float(mf1), auc=float(auc))

    # ------------------------------------------------------------------
    # 5. RuleFit (MobileNetV2 backbone + GradientBoosting)
    # ------------------------------------------------------------------
    rulefit_dir = bl_dir / 'rulefit'
    rulefit_backbone_path = rulefit_dir / 'backbone_weights.pth'
    if rulefit_backbone_path.exists():
        print("  [RuleFit] Loading MobileNetV2 backbone...")
        backbone = models.mobilenet_v2(weights=None)
        backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, nc)
        state = torch.load(rulefit_backbone_path, map_location=device, weights_only=True)
        backbone.load_state_dict(state)
        backbone.eval().to(device)

        feat_ext = nn.Sequential(*list(backbone.features.children()))
        pool = nn.AdaptiveAvgPool2d(1)

        feats = []
        with torch.no_grad():
            for images, _ in loader:
                feats.append(pool(feat_ext(images.to(device))).flatten(1).cpu().numpy())
        feats = np.concatenate(feats)

        # Load RuleFit components
        scaler_path = rulefit_dir / 'scaler.pkl'
        selector_path = rulefit_dir / 'selector.pkl'
        rule_model_path = rulefit_dir / 'rule_model.pkl'

        if scaler_path.exists() and rule_model_path.exists():
            scaler = pickle.load(open(scaler_path, 'rb'))
            feats_s = scaler.transform(feats)
            if selector_path.exists():
                selector = pickle.load(open(selector_path, 'rb'))
                feats_sel = selector.transform(feats_s)
            else:
                feats_sel = feats_s
            rule_model = pickle.load(open(rule_model_path, 'rb'))
            preds = rule_model.predict(feats_sel)
            try:
                probs = rule_model.predict_proba(feats_sel)
            except Exception:
                probs = np.zeros((len(preds), nc))
                for i, p in enumerate(preds):
                    probs[i, p] = 1.0

            acc = accuracy_score(test_labels, preds)
            bacc = balanced_accuracy_score(test_labels, preds)
            mf1 = f1_score(test_labels, preds, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(test_labels, probs, multi_class='ovr', average='macro')
            except Exception:
                auc = 0.0
            print(f"  rulefit: Acc={acc:.4f}, BAcc={bacc:.4f}, F1={mf1:.4f}, AUC={auc:.4f}")
            all_results['rulefit'] = dict(preds=preds, probs=probs, labels=test_labels,
                                           accuracy=float(acc), balanced_accuracy=float(bacc),
                                           macro_f1=float(mf1), auc=float(auc))
        else:
            print("  [SKIP] rulefit: scaler/rule_model not found, re-extracting features...")
            # Fallback: re-extract and use saved report
            report = json.load(open(rulefit_dir / 'report.json'))
            all_results['rulefit'] = dict(
                preds=None, probs=None, labels=test_labels,
                accuracy=report.get('accuracy', 0),
                balanced_accuracy=report.get('balanced_accuracy', 0),
                macro_f1=report.get('macro_f1', 0),
                auc=report.get('auc', 0))

    # ------------------------------------------------------------------
    # 6. CBM
    # ------------------------------------------------------------------
    cbm_path = bl_dir / 'cbm' / 'best_weights.pth'
    if cbm_path.exists():
        backbone = models.resnet34(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, nc)  # temp, will be replaced
        model = CBMModel(backbone, len(FRACTURE_CONCEPTS), nc)
        state = torch.load(cbm_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        all_results['cbm'] = evaluate_standard_model('cbm', model, loader, device)

    # ------------------------------------------------------------------
    # 7. CRL
    # ------------------------------------------------------------------
    crl_path = bl_dir / 'crl' / 'best_weights.pth'
    if crl_path.exists():
        backbone = models.resnet34(weights=None)
        backbone.fc = nn.Linear(backbone.fc.in_features, nc)  # temp, will be replaced
        model = CRLModel(backbone, len(FRACTURE_CONCEPTS), nc)
        state = torch.load(crl_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        all_results['crl'] = evaluate_standard_model('crl', model, loader, device)

    # ------------------------------------------------------------------
    # 8. Symbolic Model (v6 - HistGB with MI feature selection + sample_weight)
    # ------------------------------------------------------------------
    print("\n  [Symbolic v6] Loading classifier...")
    v6_dir = output_dir / 'v6'
    v6_path = v6_dir / 'best_classifier_v6.pkl'
    if not v6_path.exists():
        v6_path = output_dir / 'best_classifier_v6.pkl'
    if v6_path.exists():
        import joblib
        try:
            clf_data = joblib.load(v6_path)
            pipe = clf_data['pipe']
            non_const_mask = clf_data.get('non_const', None)
            scaler_sym = clf_data.get('scaler', None)
            anova_sel = clf_data.get('anova_selector', None)
            mi_pool_sel = clf_data.get('mi_pool_selector', None)
            mi_sel = clf_data.get('mi_selector', None)

            features_path = output_dir / 'features.npz'
            fdata = np.load(features_path, allow_pickle=True)
            X_test = fdata['test_features']
            y_test = fdata['test_labels']

            # Apply pipeline
            if non_const_mask is not None:
                X_test = X_test[:, non_const_mask]
            if scaler_sym is not None:
                X_test = scaler_sym.transform(X_test)
            if anova_sel is not None:
                X_test = anova_sel.transform(X_test)
            if mi_pool_sel is not None:
                X_test = mi_pool_sel.transform(X_test)
            if mi_sel is not None:
                X_test = mi_sel.transform(X_test)

            preds = pipe.predict(X_test)
            try:
                probs = pipe.predict_proba(X_test)
            except Exception:
                probs = np.zeros((len(preds), nc))
                for i, p in enumerate(preds):
                    probs[i, p] = 1.0

            acc = accuracy_score(y_test, preds)
            bacc = balanced_accuracy_score(y_test, preds)
            mf1 = f1_score(y_test, preds, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(y_test, probs, multi_class='ovr', average='macro')
            except Exception:
                auc = 0.0
            print(f"  symbolic_v6: Acc={acc:.4f}, BAcc={bacc:.4f}, F1={mf1:.4f}, AUC={auc:.4f}")
            all_results['symbolic_v6'] = dict(preds=preds, probs=probs, labels=y_test,
                                               accuracy=float(acc), balanced_accuracy=float(bacc),
                                               macro_f1=float(mf1), auc=float(auc))
        except Exception as e:
            print(f"  [WARN] symbolic_v6 sklearn version issue ({e}), using saved report...")
            # Fallback: use saved test_predictions.json
            tp_path = output_dir / 'test_predictions.json'
            if tp_path.exists():
                tp = json.load(open(tp_path))
                all_results['symbolic_v6'] = dict(
                    preds=None, probs=None, labels=test_labels,
                    accuracy=tp.get('overall_accuracy', 0),
                    balanced_accuracy=tp.get('balanced_accuracy', 0),
                    macro_f1=0, auc=0)
                print(f"  symbolic_v6 (from report): Acc={tp.get('overall_accuracy', 0):.4f}")
            else:
                print("  [SKIP] symbolic_v6: no saved predictions found")
    else:
        print("  [SKIP] symbolic_v6: classifier not found")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  UNIFIED COMPARISON ON TEST SET")
    print("=" * 90)
    print(f"  {'Method':<22s} {'Acc':>8s} {'BAcc':>8s} {'F1':>8s} {'AUC':>8s} {'Group'}")
    print("-" * 90)

    group_map = {
        'resnet50': 'CNN', 'densenet121': 'CNN', 'efficientnet_b0': 'CNN',
        'swin_tiny': 'Transformer', 'dinov2_linear': 'Self-supervised',
        'gradcam': 'XAI', 'rulefit': 'Rule',
        'cbm': 'Neuro-symbolic', 'crl': 'Neuro-symbolic',
        'symbolic_v6': 'Symbolic (Ours)',
    }

    sorted_methods = sorted(all_results.keys(),
                            key=lambda k: all_results[k].get('accuracy', 0), reverse=True)
    for name in sorted_methods:
        r = all_results[name]
        group = group_map.get(name, '')
        print(f"  {name:<22s} {r['accuracy']:>8.4f} {r['balanced_accuracy']:>8.4f} "
              f"{r['macro_f1']:>8.4f} {r['auc']:>8.4f} {group}")

    # ------------------------------------------------------------------
    # Save all predictions
    # ------------------------------------------------------------------
    save_dir = output_dir / 'unified_evaluation'
    save_dir.mkdir(exist_ok=True, parents=True)

    # Save summary
    summary = {}
    for name, r in all_results.items():
        summary[name] = {
            'accuracy': r['accuracy'],
            'balanced_accuracy': r['balanced_accuracy'],
            'macro_f1': r['macro_f1'],
            'auc': r['auc'],
            'group': group_map.get(name, ''),
        }
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save prediction probabilities
    pred_data = {}
    for name, r in all_results.items():
        pred_data[name] = {
            'preds': r['preds'].tolist() if r['preds'] is not None else None,
            'probs': r['probs'].tolist() if r['probs'] is not None else None,
            'labels': r['labels'].tolist() if r['labels'] is not None else None,
        }
    with open(save_dir / 'all_predictions.json', 'w') as f:
        json.dump(pred_data, f, indent=2)

    # Save as npz for easier loading
    np.savez_compressed(save_dir / 'all_predictions.npz',
                        **{f'{name}_preds': r['preds'] for name, r in all_results.items() if r['preds'] is not None},
                        **{f'{name}_probs': r['probs'] for name, r in all_results.items() if r['probs'] is not None},
                        test_labels=test_labels,
                        active_names=data['active_names'])

    print(f"\n  Saved to {save_dir}/")
    print(f"    summary.json          - metrics comparison table")
    print(f"    all_predictions.json  - all prediction probabilities")
    print(f"    all_predictions.npz   - numpy format predictions")


if __name__ == '__main__':
    main()
