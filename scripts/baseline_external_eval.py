#!/usr/bin/env python3
"""Baseline models external generalization evaluation.

Evaluates all baseline models from fracture_symbolic_v3 on the external test dataset,
comparing their generalization performance with the symbolic model.
"""
import os, sys, json, argparse, time, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]

# External class -> v3 class mapping (exact + loose + superclass)
EXTERNAL_CLASS_MAP = {
    'Comminuted fracture': {
        'exact': ['Comminuted'],
        'loose': ['Comminuted'],
        'superclass': 'complex',
    },
    'Greenstick fracture': {
        'exact': ['Greenstick'],
        'loose': ['Greenstick'],
        'superclass': 'complex',
    },
    'Oblique fracture': {
        'exact': ['Oblique'],
        'loose': ['Oblique', 'Oblique Displaced'],
        'superclass': 'simple',
    },
    'Segmental fracture': {
        'exact': ['Segmental'],
        'loose': ['Segmental'],
        'superclass': 'complex',
    },
    'Spiral Fracture': {
        'exact': ['Spiral'],
        'loose': ['Spiral'],
        'superclass': 'complex',
    },
    'Transverse fracture': {
        'exact': ['Transverse'],
        'loose': ['Transverse', 'Transverse Displaced'],
        'superclass': 'simple',
    },
}

SUPERCLASS_MAP = {
    'simple': ['Healthy', 'Linear', 'Oblique', 'Transverse'],
    'displaced': ['Comminuted', 'Oblique Displaced', 'Transverse Displaced'],
    'complex': ['Greenstick', 'Segmental', 'Spiral'],
}


def name_to_idx(name):
    return FRACTURE_NAMES.index(name) if name in FRACTURE_NAMES else -1


def collect_external_images(external_dir):
    """Collect external images with ground truth labels."""
    ext_dir = Path(external_dir)
    samples = []
    for class_dir in sorted(ext_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in EXTERNAL_CLASS_MAP:
            print(f"  [Skip] Unknown class: {class_name}")
            continue
        mapping = EXTERNAL_CLASS_MAP[class_name]
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp'):
                continue
            samples.append({
                'path': str(img_file),
                'external_class': class_name,
                'exact_targets': [name_to_idx(n) for n in mapping['exact']],
                'loose_targets': [name_to_idx(n) for n in mapping['loose']],
                'superclass': mapping['superclass'],
            })
    print(f"  Collected {len(samples)} external images from {len(set(s['external_class'] for s in samples))} classes")
    return samples


def evaluate_predictions(predictions, samples):
    """Evaluate predictions against exact/loose/superclass matching."""
    exact_correct = 0
    loose_correct = 0
    superclass_correct = 0
    total = len(predictions)

    per_class = defaultdict(lambda: {'exact': 0, 'loose': 0, 'superclass': 0, 'total': 0})

    for pred, sample in zip(predictions, samples):
        pred_name = pred['pred_name']
        pred_idx = pred.get('pred_class', name_to_idx(pred_name))
        ext_class = sample['external_class']

        per_class[ext_class]['total'] += 1

        # Exact match
        if pred_idx in sample['exact_targets']:
            exact_correct += 1
            per_class[ext_class]['exact'] += 1

        # Loose match
        if pred_idx in sample['loose_targets']:
            loose_correct += 1
            per_class[ext_class]['loose'] += 1

        # Superclass match
        pred_superclass = None
        for sc, members in SUPERCLASS_MAP.items():
            if pred_name in members:
                pred_superclass = sc
                break
        if pred_superclass == sample['superclass']:
            superclass_correct += 1
            per_class[ext_class]['superclass'] += 1

    results = {
        'exact_acc': exact_correct / total if total > 0 else 0,
        'loose_acc': loose_correct / total if total > 0 else 0,
        'superclass_acc': superclass_correct / total if total > 0 else 0,
        'total': total,
        'per_class': {},
    }
    for cls, counts in sorted(per_class.items()):
        t = counts['total']
        results['per_class'][cls] = {
            'exact': counts['exact'] / t if t > 0 else 0,
            'loose': counts['loose'] / t if t > 0 else 0,
            'superclass': counts['superclass'] / t if t > 0 else 0,
            'total': t,
        }
    return results


# ============================================================
# Model prediction functions
# ============================================================

def predict_cnn_model(model_name, create_model_fn, samples, baseline_dir, gpu=0):
    """Generic CNN prediction (ResNet50, DenseNet121, EfficientNet, Swin, GradCAM)."""
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(baseline_dir, "best_weights.pth")
    if not os.path.exists(weights_path):
        print(f"  [Skip] {model_name}: weights not found at {weights_path}")
        return None

    model = create_model_fn()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []
    for sample in samples:
        img = Image.open(sample['path']).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(tensor)
            if isinstance(out, tuple):
                out = out[0]
            prob = F.softmax(out, dim=1)
            pred = out.argmax(1).item()
        predictions.append({
            'pred_class': pred,
            'pred_name': FRACTURE_NAMES[pred],
            'confidence': prob[0, pred].item(),
        })
    return predictions


def predict_dinov2(samples, baseline_dir, gpu=0):
    """DINOv2 + Linear probe prediction."""
    import torch
    import pickle
    from torchvision import transforms
    from PIL import Image

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    scaler_path = os.path.join(baseline_dir, "scaler.pkl")
    clf_path = os.path.join(baseline_dir, "classifier.pkl")
    if not os.path.exists(scaler_path) or not os.path.exists(clf_path):
        print(f"  [Skip] DINOv2: model files not found")
        return None

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', verbose=False)
    backbone.eval().to(device)
    scaler = pickle.load(open(scaler_path, "rb"))
    clf = pickle.load(open(clf_path, "rb"))

    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []
    for sample in samples:
        img = Image.open(sample['path']).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = backbone(tensor).cpu().numpy()
        feat_s = scaler.transform(feat)
        pred = int(clf.predict(feat_s)[0])
        try:
            prob = clf.predict_proba(feat_s)[0]
            conf = float(prob[pred])
        except (AttributeError, Exception):
            conf = 0.0
        predictions.append({
            'pred_class': pred,
            'pred_name': FRACTURE_NAMES[pred],
            'confidence': conf,
        })
    return predictions


def predict_rulefit(samples, baseline_dir, gpu=0):
    """MobileNetV2 + RuleFit prediction."""
    import torch
    import torch.nn as nn
    import pickle
    from torchvision import transforms, models
    from PIL import Image

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    backbone_path = os.path.join(baseline_dir, "backbone_weights.pth")
    if not os.path.exists(backbone_path):
        print(f"  [Skip] RuleFit: backbone not found")
        return None

    backbone = models.mobilenet_v2(weights=None)
    backbone.classifier[1] = nn.Linear(backbone.classifier[1].in_features, 10)
    backbone.load_state_dict(torch.load(backbone_path, map_location="cpu", weights_only=True))
    backbone = backbone.to(device).eval()

    scaler = pickle.load(open(os.path.join(baseline_dir, "scaler.pkl"), "rb"))
    selector = pickle.load(open(os.path.join(baseline_dir, "selector.pkl"), "rb"))
    rule_model = pickle.load(open(os.path.join(baseline_dir, "rule_model.pkl"), "rb"))

    feat_ext = nn.Sequential(*list(backbone.features.children()))
    pool = nn.AdaptiveAvgPool2d(1)

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []
    for sample in samples:
        img = Image.open(sample['path']).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = pool(feat_ext(tensor)).flatten(1).cpu().numpy()
        feat_s = scaler.transform(feat)
        feat_sel = selector.transform(feat_s)
        pred = rule_model.predict(feat_sel)[0]
        predictions.append({
            'pred_class': int(pred),
            'pred_name': FRACTURE_NAMES[int(pred)],
            'confidence': 0.0,
        })
    return predictions


def predict_cbm(samples, baseline_dir, gpu=0):
    """Concept Bottleneck Model prediction."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    from PIL import Image

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(baseline_dir, "best_weights.pth")
    if not os.path.exists(weights_path):
        print(f"  [Skip] CBM: weights not found")
        return None

    FRACTURE_CONCEPTS = ['cortical_break', 'fracture_line_horizontal', 'fracture_line_oblique_45',
                         'fracture_line_oblique_135', 'fracture_line_vertical', 'displacement',
                         'bone_fragment', 'soft_tissue_swelling']

    class CBMModel(nn.Module):
        def __init__(self, num_concepts=8, num_classes=10, expand_dim=64):
            super().__init__()
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

    model = CBMModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []
    for sample in samples:
        img = Image.open(sample['path']).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, concepts, _ = model(tensor)
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
        predictions.append({
            'pred_class': pred,
            'pred_name': FRACTURE_NAMES[pred],
            'confidence': prob[0, pred].item(),
        })
    return predictions


def predict_crl(samples, baseline_dir, gpu=0):
    """Concept Reasoning Layer prediction (complex CRL with Binarize/Union/LR layers)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms, models
    from PIL import Image

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    weights_path = os.path.join(baseline_dir, "best_weights.pth")
    if not os.path.exists(weights_path):
        print(f"  [Skip] CRL: weights not found")
        return None

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
            self.output_dim = output_dim  # Fix: add output_dim attribute
            self.fc1 = nn.Linear(input_dim, output_dim)
        def forward(self, x):
            return self.fc1(x)

    class CRLModel(nn.Module):
        def __init__(self, num_concepts=8, num_classes=10, l1=256, l2=256, use_not=True, use_skip=True, temperature=1.0):
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

    model = CRLModel(l1=256, l2=256)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    predictions = []
    for sample in samples:
        img = Image.open(sample['path']).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, concepts, _ = model(tensor)
            prob = F.softmax(logits, dim=1)
            pred = logits.argmax(1).item()
        predictions.append({
            'pred_class': pred,
            'pred_name': FRACTURE_NAMES[pred],
            'confidence': prob[0, pred].item(),
        })
    return predictions


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline models external generalization evaluation")
    parser.add_argument("--external_dir", required=True, help="Path to external test directory")
    parser.add_argument("--baseline_dir", required=True, help="Path to baseline_comparison directory")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    external_dir = args.external_dir
    gpu = args.gpu

    print("=" * 70)
    print("  Baseline Models External Generalization Evaluation")
    print("=" * 70)

    # Collect external images
    print("\n[1] Collecting external images...")
    samples = collect_external_images(external_dir)

    # Internal test accuracy (from report.json)
    internal_results = {}
    for model_name in ['resnet50', 'densenet121', 'efficientnet_b0', 'swin_tiny',
                       'dinov2_linear', 'rulefit', 'cbm', 'crl', 'gradcam']:
        report_path = baseline_dir / model_name / "report.json"
        if report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            internal_results[model_name] = {
                'test_acc': report.get('accuracy', 0),
                'balanced_acc': report.get('balanced_accuracy', 0),
                'auc': report.get('auc', 0),
            }

    print(f"\n  Internal test accuracy (from training):")
    for name, res in sorted(internal_results.items(), key=lambda x: -x[1]['test_acc']):
        print(f"    {name:20s}: acc={res['test_acc']:.4f}, bacc={res['balanced_acc']:.4f}, auc={res['auc']:.4f}")

    # Run predictions for each model
    print(f"\n[2] Running predictions on external dataset...")

    all_results = {}

    # --- ResNet50 ---
    print("\n  --- ResNet50 ---")
    from torchvision import models
    import torch.nn as nn
    def create_resnet50():
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 10)
        return m
    preds = predict_cnn_model("ResNet50", create_resnet50, samples, str(baseline_dir / "resnet50"), gpu)
    if preds:
        all_results['resnet50'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['resnet50']['exact_acc']:.4f}, Loose: {all_results['resnet50']['loose_acc']:.4f}, Superclass: {all_results['resnet50']['superclass_acc']:.4f}")

    # --- DenseNet121 ---
    print("\n  --- DenseNet121 ---")
    def create_densenet121():
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 10)
        return m
    preds = predict_cnn_model("DenseNet121", create_densenet121, samples, str(baseline_dir / "densenet121"), gpu)
    if preds:
        all_results['densenet121'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['densenet121']['exact_acc']:.4f}, Loose: {all_results['densenet121']['loose_acc']:.4f}, Superclass: {all_results['densenet121']['superclass_acc']:.4f}")

    # --- EfficientNet-B0 ---
    print("\n  --- EfficientNet-B0 ---")
    def create_efficientnet():
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 10)
        return m
    preds = predict_cnn_model("EfficientNet-B0", create_efficientnet, samples, str(baseline_dir / "efficientnet_b0"), gpu)
    if preds:
        all_results['efficientnet_b0'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['efficientnet_b0']['exact_acc']:.4f}, Loose: {all_results['efficientnet_b0']['loose_acc']:.4f}, Superclass: {all_results['efficientnet_b0']['superclass_acc']:.4f}")

    # --- Swin-Tiny (ViT-B/16) ---
    print("\n  --- Swin-Tiny (ViT-B/16) ---")
    from torchvision import models as tv_models
    def create_swin():
        m = tv_models.vit_b_16(weights=None)
        m.heads.head = nn.Linear(m.heads.head.in_features, 10)
        return m
    preds = predict_cnn_model("Swin-Tiny", create_swin, samples, str(baseline_dir / "swin_tiny"), gpu)
    if preds:
        all_results['swin_tiny'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['swin_tiny']['exact_acc']:.4f}, Loose: {all_results['swin_tiny']['loose_acc']:.4f}, Superclass: {all_results['swin_tiny']['superclass_acc']:.4f}")

    # --- DINOv2 + Linear ---
    print("\n  --- DINOv2 + Linear ---")
    preds = predict_dinov2(samples, str(baseline_dir / "dinov2_linear"), gpu)
    if preds:
        all_results['dinov2_linear'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['dinov2_linear']['exact_acc']:.4f}, Loose: {all_results['dinov2_linear']['loose_acc']:.4f}, Superclass: {all_results['dinov2_linear']['superclass_acc']:.4f}")

    # --- RuleFit ---
    print("\n  --- RuleFit ---")
    preds = predict_rulefit(samples, str(baseline_dir / "rulefit"), gpu)
    if preds:
        all_results['rulefit'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['rulefit']['exact_acc']:.4f}, Loose: {all_results['rulefit']['loose_acc']:.4f}, Superclass: {all_results['rulefit']['superclass_acc']:.4f}")

    # --- CBM ---
    print("\n  --- CBM ---")
    preds = predict_cbm(samples, str(baseline_dir / "cbm"), gpu)
    if preds:
        all_results['cbm'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['cbm']['exact_acc']:.4f}, Loose: {all_results['cbm']['loose_acc']:.4f}, Superclass: {all_results['cbm']['superclass_acc']:.4f}")

    # --- CRL ---
    print("\n  --- CRL ---")
    preds = predict_crl(samples, str(baseline_dir / "crl"), gpu)
    if preds:
        all_results['crl'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['crl']['exact_acc']:.4f}, Loose: {all_results['crl']['loose_acc']:.4f}, Superclass: {all_results['crl']['superclass_acc']:.4f}")

    # --- GradCAM (same as ResNet50) ---
    print("\n  --- GradCAM (ResNet50) ---")
    preds = predict_cnn_model("GradCAM", create_resnet50, samples, str(baseline_dir / "gradcam"), gpu)
    if preds:
        all_results['gradcam'] = evaluate_predictions(preds, samples)
        print(f"    Exact: {all_results['gradcam']['exact_acc']:.4f}, Loose: {all_results['gradcam']['loose_acc']:.4f}, Superclass: {all_results['gradcam']['superclass_acc']:.4f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("  SUMMARY: External Generalization Comparison")
    print("=" * 70)

    # Add symbolic model results for reference
    symbolic_ref = {
        'symbolic_v3': {
            'exact_acc': 0.0584,
            'loose_acc': 0.1168,
            'superclass_acc': 0.1314,
        }
    }

    print(f"\n  {'Model':<22s} {'Internal Acc':>12s} {'Exact':>8s} {'Loose':>8s} {'Superclass':>11s} {'Drop':>8s}")
    print("  " + "-" * 75)

    for name in ['densenet121', 'resnet50', 'efficientnet_b0', 'swin_tiny',
                 'dinov2_linear', 'cbm', 'crl', 'rulefit', 'gradcam']:
        if name not in all_results:
            continue
        r = all_results[name]
        internal_acc = internal_results.get(name, {}).get('test_acc', 0)
        drop = internal_acc - r['exact_acc']
        print(f"  {name:<22s} {internal_acc:>12.4f} {r['exact_acc']:>8.4f} {r['loose_acc']:>8.4f} {r['superclass_acc']:>11.4f} {drop:>+8.4f}")

    # Symbolic reference
    print(f"  {'symbolic_v3 (ref)':<22s} {'0.9585':>12s} {0.0584:>8.4f} {0.1168:>8.4f} {0.1314:>11.4f} {'+0.9001':>8s}")

    # Per-class breakdown
    print(f"\n  Per-class Exact Accuracy:")
    print(f"  {'Class':<25s}", end="")
    for name in sorted(all_results.keys()):
        short = name[:10]
        print(f" {short:>10s}", end="")
    print()
    print("  " + "-" * (25 + 11 * len(all_results)))

    all_classes = set()
    for r in all_results.values():
        all_classes.update(r['per_class'].keys())
    for cls in sorted(all_classes):
        print(f"  {cls:<25s}", end="")
        for name in sorted(all_results.keys()):
            acc = all_results[name]['per_class'].get(cls, {}).get('exact', 0)
            print(f" {acc:>10.4f}", end="")
        print()

    # Save results
    output = {
        'external_dir': external_dir,
        'total_samples': len(samples),
        'internal_results': internal_results,
        'external_results': all_results,
        'symbolic_v3_reference': symbolic_ref,
    }

    output_path = args.output or str(baseline_dir / "external_generalization_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
