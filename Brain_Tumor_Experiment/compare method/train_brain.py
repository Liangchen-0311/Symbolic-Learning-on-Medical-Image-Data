#!/usr/bin/env python3
"""Brain Tumor MRI comparison baselines (adapted from the HAM10000 train.py).

Same methods (resnet50/densenet121/efficientnet_b0/swin_tiny/dinov2_linear/
rulefit/CBM/CRL), but on the Brain Tumor MRI dataset:
  - grayscale MRI replicated to 3 channels for ImageNet-pretrained backbones
  - 4 classes (glioma/meningioma/pituitary/notumor)
  - SAME split as our symbolic pipeline: val carved from Training/ (seed 42,
    val_fraction 0.15), Testing/ used in full as the test set -> identical test
    set, so accuracies are directly comparable to the symbolic 0.916.

Usage:
    python train_brain.py --method resnet50 --epochs 30 --gpu 0
"""
import os, sys, json, argparse, time, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = SCRIPT_DIR
DATA_DIR = '/home/ET/lctan/Symbolic-Learning/Symbolic-Learning-on-Medical-Image-Data/Brain Tumor MRI Dataset'
BRAIN_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']
NUM_CLASSES = len(BRAIN_NAMES)
VAL_FRACTION = 0.15
SEED = 42
_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


class BrainDataset(Dataset):
    """Grayscale->RGB Brain MRI dataset. Split matches brain_tumor_loader exactly."""

    def __init__(self, split='train', transform=None):
        self.transform = transform
        self.samples = []
        disk = 'Testing' if split == 'test' else 'Training'
        rng = np.random.RandomState(SEED)
        for cls_idx, cls in enumerate(BRAIN_NAMES):
            cdir = os.path.join(DATA_DIR, disk, cls)
            if not os.path.isdir(cdir):
                continue
            files = sorted(f for f in os.listdir(cdir) if f.lower().endswith(_EXTS))
            if split in ('train', 'val'):
                perm = rng.permutation(len(files))
                n_val = int(round(len(files) * VAL_FRACTION))
                val_idx = set(perm[:n_val].tolist())
                keep = ([i for i in range(len(files)) if i in val_idx] if split == 'val'
                        else [i for i in range(len(files)) if i not in val_idx])
                files = [files[i] for i in keep]
            for f in files:
                self.samples.append((os.path.join(cdir, f), cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')   # grayscale duplicated to 3 channels
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(method='cnn', train=False):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if method == 'dinov2' or not train:
        return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                   transforms.ToTensor(), norm])
    # train aug: MRI orientation matters -> horizontal flip + mild affine only
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(), norm,
    ])


def train_cnn(model, train_loader, val_loader, epochs, device, lr, is_cbm=False):
    model = model.to(device)
    all_labels = []
    for _, lb in train_loader:
        all_labels.extend(lb.numpy())
    cw = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=np.array(all_labels))
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_bal, best_state = 0, None
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            out = model(images)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, labels)
            loss.backward(); opt.step()
        sched.step()
        model.eval(); vp, vl = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                out = model(images.to(device))
                logits = out[0] if isinstance(out, tuple) else out
                vp.extend(logits.argmax(1).cpu().numpy()); vl.extend(labels.numpy())
        vbal = balanced_accuracy_score(vl, vp); vacc = accuracy_score(vl, vp)
        print(f"  epoch {ep+1}/{epochs} [{time.time()-t0:.0f}s] val_acc={vacc:.4f} val_bal={vbal:.4f}", flush=True)
        if vbal > best_bal:
            best_bal = vbal
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, loader, device, is_cbm=False):
    model.eval(); preds, probs, labs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            out = model(images.to(device))
            logits = out[0] if isinstance(out, tuple) else out
            p = torch.softmax(logits, 1)
            preds.extend(logits.argmax(1).cpu().numpy()); probs.extend(p.cpu().numpy()); labs.extend(labels.numpy())
    preds, probs, labs = np.array(preds), np.array(probs), np.array(labs)
    acc = accuracy_score(labs, preds); bal = balanced_accuracy_score(labs, preds)
    try:
        auc = roc_auc_score(labs, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    per = {BRAIN_NAMES[c]: round(float((preds[labs == c] == c).mean()), 4) for c in range(NUM_CLASSES)}
    rep = classification_report(labs, preds, target_names=BRAIN_NAMES, digits=4, zero_division=0)
    return acc, bal, auc, per, rep


def feat_extract(backbone, dataset, device, bs=64):
    loader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=4)
    X, y = [], []
    with torch.no_grad():
        for images, labels in loader:
            X.append(backbone.extract_features(images.to(device))); y.append(labels.numpy())
    return np.vstack(X), np.concatenate(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True,
                    choices=['resnet50', 'densenet121', 'efficientnet_b0', 'swin_tiny',
                             'dinov2_linear', 'rulefit', 'CBM', 'CRL'])
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Method: {args.method}", flush=True)

    tf_type = 'dinov2' if args.method == 'dinov2_linear' else 'cnn'
    train_ds = BrainDataset('train', get_transforms(tf_type, train=True))
    val_ds = BrainDataset('val', get_transforms(tf_type, train=False))
    test_ds = BrainDataset('test', get_transforms(tf_type, train=False))
    print(f"Train {len(train_ds)}  Val {len(val_ds)}  Test {len(test_ds)}", flush=True)

    mdir = os.path.join(OUTPUT_BASE, args.method)
    os.makedirs(mdir, exist_ok=True)
    t0 = time.time()

    if args.method in ('dinov2_linear', 'rulefit'):
        if args.method == 'dinov2_linear':
            sys.path.insert(0, os.path.join(OUTPUT_BASE, 'dinov2_linear'))
            from model import DINOv2Backbone
            bb = DINOv2Backbone().to(device).eval()
        else:
            sys.path.insert(0, os.path.join(OUTPUT_BASE, 'rulefit'))
            from model import MobileNetV2FeatureExtractor
            bb = MobileNetV2FeatureExtractor(pretrained=True).to(device).eval()
        trX, trY = feat_extract(bb, train_ds, device)
        vaX, vaY = feat_extract(bb, val_ds, device)
        teX, teY = feat_extract(bb, test_ds, device)
        sc = StandardScaler().fit(trX)
        trX, vaX, teX = sc.transform(trX), sc.transform(vaX), sc.transform(teX)
        if args.method == 'rulefit':
            sel = SelectKBest(f_classif, k=min(50, trX.shape[1])).fit(trX, trY)
            trX, vaX, teX = sel.transform(trX), sel.transform(vaX), sel.transform(teX)
        best, best_bal = None, -1
        for C in [0.01, 0.1, 1.0, 10.0]:
            clf = LogisticRegression(C=C, max_iter=2000, class_weight='balanced')
            clf.fit(trX, trY)
            b = balanced_accuracy_score(vaY, clf.predict(vaX))
            if b > best_bal:
                best_bal, best = b, clf
        pred = best.predict(teX); prob = best.predict_proba(teX)
        acc = accuracy_score(teY, pred); bal = balanced_accuracy_score(teY, pred)
        try:
            auc = roc_auc_score(teY, prob, multi_class='ovr', average='macro')
        except Exception:
            auc = 0.0
        per = {BRAIN_NAMES[c]: round(float((pred[teY == c] == c).mean()), 4) for c in range(NUM_CLASSES)}
        rep = classification_report(teY, pred, target_names=BRAIN_NAMES, digits=4, zero_division=0)
    else:
        sys.path.insert(0, mdir)
        from model import create_model
        model = create_model(num_classes=NUM_CLASSES, pretrained=True)
        is_cbm = args.method in ('CBM', 'CRL')
        tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        te = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        model = train_cnn(model, tl, vl, args.epochs, device, args.lr, is_cbm=is_cbm)
        torch.save(model.state_dict(), os.path.join(mdir, 'best_weights.pth'))
        acc, bal, auc, per, rep = evaluate(model, te, device, is_cbm=is_cbm)

    elapsed = time.time() - t0
    print(f"\n{'='*60}\nMethod {args.method}: acc={acc:.4f} bal={bal:.4f} AUC={auc:.4f}  ({elapsed/60:.1f} min)", flush=True)
    print(f"per-class: {per}\n{rep}\n{'='*60}", flush=True)
    json.dump({'method': args.method, 'acc': float(acc), 'bal_acc': float(bal), 'auc': float(auc),
               'per_class': per, 'time_seconds': elapsed, 'epochs': args.epochs},
              open(os.path.join(mdir, 'results.json'), 'w'), indent=2)
    print(f"saved {mdir}/results.json", flush=True)


if __name__ == '__main__':
    main()
