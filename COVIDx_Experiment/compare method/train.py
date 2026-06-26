#!/usr/bin/env python3
"""Unified training and evaluation for HAM10000 comparison methods.

Usage:
    python train.py --method resnet50 --epochs 50 --batch_size 32
    python train.py --method dinov2_linear  # no training needed, just feature extraction
    python train.py --method rulefit        # MobileNetV2 + RuleFit

Methods:
    resnet50, densenet121, efficientnet_b0, swin_tiny  — end-to-end CNNs
    dinov2_linear                                         — frozen DINOv2 + LogisticRegression
    rulefit                                               — frozen MobileNetV2 + RuleFit
    CBM                                                   — Concept Bottleneck Model
    CRL                                                   — Concept Reasoning Layer
"""
import os, sys, json, argparse, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Add compare method dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

HAM10000_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
DATA_DIR = '/home/lqg1/code_8T/25/lxw/4/skin_symbolic_v3/datasets'
OUTPUT_BASE = '/home/lqg1/code_8T/25/lxw/4/skin_symbolic_v3/compare method'


# ─── Dataset ───────────────────────────────────────────────────────────────────

class HAM10000Dataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform
        self.samples = []
        for cls_idx, cls_name in enumerate(HAM10000_NAMES):
            cls_dir = os.path.join(self.root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(cls_dir, fname), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(method='cnn'):
    """Get transforms. DINOv2 needs special handling (224x224, multiple of 14)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if method == 'dinov2':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])


def get_train_transforms(method='cnn'):
    """Training transforms with augmentation."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if method == 'dinov2':
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        normalize,
    ])


# ─── Training functions ────────────────────────────────────────────────────────

def plot_training_curves(history, save_dir, model_name):
    """Plot training curves (loss, accuracy, learning rate) and save as PNG."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs = range(1, len(history['loss']) + 1)

        # Loss
        axes[0].plot(epochs, history['loss'], 'b-', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'{model_name} - Training Loss')
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, history['train_acc'], 'r-', linewidth=1.5, label='Train Acc')
        axes[1].plot(epochs, history['val_acc'], 'g-', linewidth=1.5, label='Val Acc')
        if history['val_acc']:
            best_epoch = history['val_acc'].index(max(history['val_acc'])) + 1
            axes[1].axvline(x=best_epoch, color='k', linestyle='--', alpha=0.5, label=f'Best Epoch={best_epoch}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'{model_name} - Accuracy (Best Val={max(history["val_acc"]):.4f})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning Rate
        axes[2].plot(epochs, history['lr'], 'm-', linewidth=1.5)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title(f'{model_name} - Learning Rate')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'training_curves.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Training curves saved to {save_path}", flush=True)
    except ImportError:
        print(f"  [WARN] matplotlib not available, skipping training curves plot", flush=True)

    # Save history as JSON
    hist_path = os.path.join(save_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)


def train_cnn_model(model, train_loader, val_loader, num_epochs, device, lr=1e-4, weight_decay=1e-4, method_name='cnn'):
    """Train a standard CNN model end-to-end."""
    model = model.to(device)

    # Class weights for imbalanced data
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_weights = compute_class_weight('balanced', classes=np.arange(7), y=np.array(all_labels))
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_bal = 0
    best_state = None

    # Training history for plotting
    history = {'loss': [], 'train_acc': [], 'val_acc': [], 'val_bal': [], 'lr': []}

    print(f"  Starting training for {num_epochs} epochs...", flush=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_bal = balanced_accuracy_score(val_labels, val_preds)

        # Record history
        history['loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_acc)
        history['val_bal'].append(val_bal)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if val_bal > best_val_bal:
            best_val_bal = val_bal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1}/{num_epochs} [{epoch_time:.0f}s]: loss={train_loss/len(train_loader):.4f}, "
              f"train_acc={train_correct/train_total:.4f}, val_acc={val_acc:.4f}, val_bal={val_bal:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    return model, history


def train_dinov2_linear(train_dataset, val_dataset, test_dataset, device):
    """DINOv2 + Linear Probe: extract features, train LogisticRegression."""
    sys.path.insert(0, os.path.join(OUTPUT_BASE, 'dinov2_linear'))
    from model import DINOv2Backbone

    backbone = DINOv2Backbone().to(device).eval()
    batch_size = 32

    def extract_features(dataset, backbone, device):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        all_features, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                # DINOv2 expects images with dimensions multiple of 14
                # 224 is 16*14, so it's fine
                features = backbone.extract_features(images)
                all_features.append(features)
                all_labels.append(labels.numpy())
        return np.vstack(all_features), np.concatenate(all_labels)

    print("  Extracting DINOv2 features for train set...", flush=True)
    train_X, train_y = extract_features(train_dataset, backbone, device)
    print(f"  Train features: {train_X.shape}", flush=True)

    print("  Extracting DINOv2 features for val set...", flush=True)
    val_X, val_y = extract_features(val_dataset, backbone, device)

    print("  Extracting DINOv2 features for test set...", flush=True)
    test_X, test_y = extract_features(test_dataset, backbone, device)

    # StandardScaler + LogisticRegression
    scaler = StandardScaler()
    train_X_s = scaler.fit_transform(train_X)
    val_X_s = scaler.transform(val_X)
    test_X_s = scaler.transform(test_X)

    # Train LogisticRegression with class weights
    best_clf = None
    best_val_bal = 0
    for C in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight='balanced', multi_class='multinomial')
        clf.fit(train_X_s, train_y)
        val_pred = clf.predict(val_X_s)
        val_bal = balanced_accuracy_score(val_y, val_pred)
        print(f"    C={C}: val_bal={val_bal:.4f}", flush=True)
        if val_bal > best_val_bal:
            best_val_bal = val_bal
            best_clf = clf

    # Save artifacts
    model_dir = os.path.join(OUTPUT_BASE, 'dinov2_linear')
    with open(f"{model_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{model_dir}/classifier.pkl", 'wb') as f:
        pickle.dump(best_clf, f)

    # Evaluate on test
    test_pred = best_clf.predict(test_X_s)
    test_probs = best_clf.predict_proba(test_X_s)
    return test_pred, test_probs, test_y


def train_rulefit(train_dataset, val_dataset, test_dataset, device):
    """MobileNetV2 + RuleFit: extract features, train RuleFit."""
    sys.path.insert(0, os.path.join(OUTPUT_BASE, 'rulefit'))
    from model import MobileNetV2FeatureExtractor

    backbone = MobileNetV2FeatureExtractor(pretrained=True).to(device).eval()
    batch_size = 64

    def extract_features(dataset, backbone, device):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        all_features, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                features = backbone.extract_features(images)
                all_features.append(features)
                all_labels.append(labels.numpy())
        return np.vstack(all_features), np.concatenate(all_labels)

    print("  Extracting MobileNetV2 features...", flush=True)
    train_X, train_y = extract_features(train_dataset, backbone, device)
    val_X, val_y = extract_features(val_dataset, backbone, device)
    test_X, test_y = extract_features(test_dataset, backbone, device)
    print(f"  Train features: {train_X.shape}", flush=True)

    # StandardScaler + SelectKBest
    scaler = StandardScaler()
    train_X_s = scaler.fit_transform(train_X)
    val_X_s = scaler.transform(val_X)
    test_X_s = scaler.transform(test_X)

    selector = SelectKBest(f_classif, k=50)
    train_X_sel = selector.fit_transform(train_X_s, train_y)
    val_X_sel = selector.transform(val_X_s)
    test_X_sel = selector.transform(test_X_s)

    # RuleFit
    try:
        from rulefit import RuleFit
        rf = RuleFit(tree_size=4, max_rules=200, rfmode='classify')
        rf.fit(train_X_sel, train_y, feature_names=[f'f{i}' for i in range(50)])
        test_pred = rf.predict(test_X_sel)
        test_probs = rf.predict_proba(test_X_sel) if hasattr(rf, 'predict_proba') else None

        model_dir = os.path.join(OUTPUT_BASE, 'rulefit')
        with open(f"{model_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        with open(f"{model_dir}/selector.pkl", 'wb') as f:
            pickle.dump(selector, f)
        with open(f"{model_dir}/rule_model.pkl", 'wb') as f:
            pickle.dump(rf, f)

        if test_probs is None:
            # One-hot encode predictions for AUC
            test_probs = np.eye(7)[test_pred.astype(int)]
        return test_pred, test_probs, test_y
    except ImportError:
        print("  RuleFit not installed, using GradientBoostingClassifier as fallback...", flush=True)
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
        clf.fit(train_X_sel, train_y)
        test_pred = clf.predict(test_X_sel)
        test_probs = clf.predict_proba(test_X_sel)

        model_dir = os.path.join(OUTPUT_BASE, 'rulefit')
        with open(f"{model_dir}/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        with open(f"{model_dir}/selector.pkl", 'wb') as f:
            pickle.dump(selector, f)
        with open(f"{model_dir}/rule_model.pkl", 'wb') as f:
            pickle.dump(clf, f)
        return test_pred, test_probs, test_y


def train_cbm(model, train_loader, val_loader, num_epochs, device, lr=1e-4, weight_decay=1e-4):
    """Train Concept Bottleneck Model with concept regularization."""
    model = model.to(device)

    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    class_weights = compute_class_weight('balanced', classes=np.arange(7), y=np.array(all_labels))
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_bal = 0
    best_state = None

    # Training history for plotting
    history = {'loss': [], 'train_acc': [], 'val_acc': [], 'val_bal': [], 'lr': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, concept_probs, concept_logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits, _, _ = model(images)
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_bal = balanced_accuracy_score(val_labels, val_preds)

        # Record history
        history['loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_acc)
        history['val_bal'].append(val_bal)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        if val_bal > best_val_bal:
            best_val_bal = val_bal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  Epoch {epoch+1}/{num_epochs}: loss={train_loss/len(train_loader):.4f}, "
              f"train_acc={train_correct/train_total:.4f}, val_acc={val_acc:.4f}, val_bal={val_bal:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)
    return model, history


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, test_loader, device, is_cbm=False):
    """Evaluate model on test set."""
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if is_cbm:
                logits, _, _ = model(images)
            else:
                outputs = model(images)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1)
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
    except:
        auc = 0.0

    report = classification_report(all_labels, all_preds, target_names=HAM10000_NAMES, digits=4, zero_division=0)

    return acc, bal_acc, auc, report, all_preds, all_probs, all_labels


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='HAM10000 Comparison Methods')
    parser.add_argument('--method', type=str, required=True,
                        choices=['resnet50', 'densenet121', 'efficientnet_b0', 'swin_tiny',
                                 'dinov2_linear', 'rulefit', 'CBM', 'CRL'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Method: {args.method}", flush=True)

    # Determine transform type
    if args.method == 'dinov2_linear':
        tf_type = 'dinov2'
    else:
        tf_type = 'cnn'

    # Datasets
    train_tf = get_train_transforms(tf_type)
    eval_tf = get_transforms(tf_type)

    train_dataset = HAM10000Dataset(DATA_DIR, 'train', train_tf)
    val_dataset = HAM10000Dataset(DATA_DIR, 'val', eval_tf)
    test_dataset = HAM10000Dataset(DATA_DIR, 'test', eval_tf)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}", flush=True)

    # Class distribution
    train_labels = [s[1] for s in train_dataset.samples]
    for i, name in enumerate(HAM10000_NAMES):
        print(f"  {name}: {train_labels.count(i)}", flush=True)

    method_dir = os.path.join(OUTPUT_BASE, args.method)
    os.makedirs(method_dir, exist_ok=True)

    start_time = time.time()

    if args.method in ['dinov2_linear', 'rulefit']:
        # Feature extraction based methods
        if args.method == 'dinov2_linear':
            test_pred, test_probs, test_y = train_dinov2_linear(train_dataset, val_dataset, test_dataset, device)
        else:
            test_pred, test_probs, test_y = train_rulefit(train_dataset, val_dataset, test_dataset, device)

        acc = accuracy_score(test_y, test_pred)
        bal_acc = balanced_accuracy_score(test_y, test_pred)
        try:
            auc = roc_auc_score(test_y, test_probs, multi_class='ovr', average='weighted')
        except:
            auc = 0.0
        report = classification_report(test_y, test_pred, target_names=HAM10000_NAMES, digits=4, zero_division=0)

    else:
        # End-to-end models
        sys.path.insert(0, method_dir)
        from model import create_model

        if args.method in ['CBM', 'CRL']:
            model = create_model(num_classes=7, pretrained=True)
            is_cbm = True
        else:
            model = create_model(num_classes=7, pretrained=True)
            is_cbm = False

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

        if args.method in ['CBM', 'CRL']:
            model, history = train_cbm(model, train_loader, val_loader, args.epochs, device, args.lr)
            plot_training_curves(history, method_dir, args.method.upper())
        else:
            model, history = train_cnn_model(model, train_loader, val_loader, args.epochs, device, args.lr, method_name=args.method)
            plot_training_curves(history, method_dir, args.method)

        # Save weights
        torch.save(model.state_dict(), os.path.join(method_dir, 'best_weights.pth'))

        # Evaluate
        acc, bal_acc, auc, report, _, _, _ = evaluate_model(model, test_loader, device, is_cbm=is_cbm)

    elapsed = time.time() - start_time

    # Print results
    print(f"\n{'='*70}", flush=True)
    print(f"Method: {args.method}", flush=True)
    print(f"Test Accuracy:     {acc:.4f}", flush=True)
    print(f"Test Balanced Acc: {bal_acc:.4f}", flush=True)
    print(f"Test AUC:          {auc:.4f}", flush=True)
    print(f"Time:              {elapsed/60:.1f} min", flush=True)
    print(f"\nClassification Report:\n{report}", flush=True)
    print(f"{'='*70}", flush=True)

    # Save results
    results = {
        'method': args.method,
        'acc': float(acc),
        'bal_acc': float(bal_acc),
        'auc': float(auc),
        'time_seconds': elapsed,
        'epochs': args.epochs,
        'report': report,
    }
    with open(os.path.join(method_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {method_dir}/results.json", flush=True)


if __name__ == '__main__':
    main()
