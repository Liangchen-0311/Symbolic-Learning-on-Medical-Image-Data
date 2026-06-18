#!/usr/bin/env python3
"""
Fracture Prediction Script v2 — Raw Image Prediction

Predict bone fracture types from raw X-ray images.
Loads trained symbolic formulas, extracts features from raw images on-the-fly,
then classifies using the trained sklearn Pipeline.

Difference from v1:
  - v1: --split loads pre-extracted features from features.npz (fast, no GPU needed)
  - v2: ALL modes load raw images and extract features on-the-fly (slower but more flexible)

Usage:
    python scripts/predict_fracture_v2.py --config configs/fracture_v3_expanded.yaml --image xray.jpg
    python scripts/predict_fracture_v2.py --config configs/fracture_v3_expanded.yaml --image_dir /path/to/images/
    python scripts/predict_fracture_v2.py --config configs/fracture_v3_expanded.yaml --split test
    python scripts/predict_fracture_v2.py --config configs/fracture_v3_expanded.yaml --version 6
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.data.fracture_loader import (
    HBFMIDDataModule, HBFMIDDataset, build_fracture_data_batch,
    FRACTURE_NAMES, FRACTURE_SUPERCLASS,
)
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS
from src.symbolic.fracture_operators import register_fracture_operators
from src.symbolic.feature_encoding import encode_body_distribution_v2

register_fracture_operators(TENSOR_OPERATORS)

SUPERCLASS_NAMES = ['simple', 'displaced', 'complex']


def load_trained_model(output_dir, version=None):
    if version is not None:
        version_dir = output_dir / f'v{version}'
        classifier_path = version_dir / f'best_classifier_v{version}.pkl'
        results_path = version_dir / f'classifier_results_v{version}.json'
        if not classifier_path.exists():
            classifier_path = output_dir / f'best_classifier_v{version}.pkl'
            results_path = output_dir / f'classifier_results_v{version}.json'
    else:
        # Auto-detect latest version
        classifier_path = output_dir / 'best_classifier.pkl'
        results_path = output_dir / 'classifier_results.json'
        if not classifier_path.exists():
            version_dirs = sorted(output_dir.glob('v*'))
            latest_v = None
            for vd in reversed(version_dirs):
                if vd.is_dir():
                    v_num = vd.name[1:]
                    if v_num.isdigit():
                        candidate = vd / f'best_classifier_v{v_num}.pkl'
                        if candidate.exists():
                            latest_v = int(v_num)
                            classifier_path = candidate
                            results_path = vd / f'classifier_results_v{v_num}.json'
                            break
            if latest_v is None:
                for p in sorted(output_dir.glob('best_classifier_v*.pkl'), reverse=True):
                    v_str = p.stem.replace('best_classifier_v', '')
                    if v_str.isdigit():
                        latest_v = int(v_str)
                        classifier_path = p
                        results_path = output_dir / f'classifier_results_v{latest_v}.json'
                        break
            if latest_v is not None:
                version = latest_v
                print(f"[Predict-v2] Auto-detected latest version: v{version}")

    features_path = output_dir / 'features.npz'
    validated_path = output_dir / 'validated_formulas.json'
    class_names_path = output_dir / 'class_names.json'

    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Trained classifier not found at {classifier_path}\n"
            f"Please run the full pipeline first: python experiments/run_fracture_pipeline.py\n"
            f"Or specify a version with --version (e.g. --version 6)"
        )

    clf_data = joblib.load(classifier_path)
    pipe = clf_data['pipe']
    non_const_mask = clf_data.get('non_const', None)
    scaler_for_pca = clf_data.get('scaler', None)
    anova_selector = clf_data.get('anova_selector', None)
    mi_pool_selector = clf_data.get('mi_pool_selector', None)
    mi_selector = clf_data.get('mi_selector', None)
    method = clf_data.get('method', 'unknown')
    model_version = clf_data.get('version', None)

    results = json.load(open(results_path)) if results_path.exists() else {}

    active_names = None
    if class_names_path.exists():
        active_names = json.load(open(class_names_path))

    data = np.load(features_path, allow_pickle=True)
    bodies = list(data['bodies']) if 'bodies' in data else []

    if not bodies and validated_path.exists():
        formulas = json.load(open(validated_path))
        bodies_set = set()
        for f in formulas:
            tokens = f['str'].strip().split()
            if tokens[-1] in ROOT_OPERATORS:
                bodies_set.add(' '.join(tokens[:-1]))
            else:
                bodies_set.add(f['str'])
        bodies = sorted(bodies_set)

    num_classes = len(active_names) if active_names else 10
    ver_str = f"v{version}" if version else "latest"
    print(f"[Predict-v2] Loaded {len(bodies)} formula bodies")
    print(f"[Predict-v2] Classifier: sklearn Pipeline ({method}), {num_classes} classes [{ver_str}]")
    feat_sel_info = "ANOVA"
    if mi_selector is not None:
        feat_sel_info = f"ANOVA({anova_selector.k if anova_selector else '?'})→MI_pool({mi_pool_selector.k if mi_pool_selector else '?'})→MI({mi_selector.k if mi_selector else '?'})"
    elif anova_selector is not None:
        feat_sel_info = f"ANOVA({anova_selector.k})"
    print(f"[Predict-v2] Feature selection: {feat_sel_info}")
    print(f"[Predict-v2] Classifier results: acc={results.get('test_accuracy', 'N/A')}")

    return bodies, pipe, non_const_mask, scaler_for_pca, anova_selector, mi_pool_selector, mi_selector, results, active_names, num_classes


def execute_body(body_str, data_batch):
    tokens = body_str.strip().split()
    stack = []
    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                return None
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
            stack.append(result)
        else:
            return None
    if len(stack) != 1:
        return None
    out = torch.clamp(stack[0], -1e4, 1e4)
    return out if out.dim() >= 2 else None


def extract_features_from_images(images, bodies, device, n_stats=12, n_regions=5, batch_size=16):
    features_per_body = n_stats * n_regions
    n_images = images.shape[0]
    all_feats = []

    for start in range(0, n_images, batch_size):
        end = min(start + batch_size, n_images)
        batch = images[start:end]
        data_batch = build_fracture_data_batch(batch, device)

        batch_feats = []
        for body_str in bodies:
            fm = execute_body(body_str, data_batch)
            if fm is not None:
                stats = encode_body_distribution_v2(fm, n_stats=n_stats, n_regions=n_regions)
                batch_feats.append(stats)
            else:
                batch_feats.append(torch.zeros(batch.shape[0], features_per_body, device=device))

        feats = torch.cat(batch_feats, dim=1)
        all_feats.append(feats.cpu().numpy())

        del data_batch
        torch.cuda.empty_cache()

    return np.concatenate(all_feats, axis=0)


def predict_with_pipeline(raw_feats, pipe, non_const_mask=None, scaler=None, anova_selector=None,
                          mi_pool_selector=None, mi_selector=None):
    if non_const_mask is not None:
        feats = raw_feats[:, non_const_mask]
    else:
        feats = raw_feats.copy()

    if scaler is not None:
        feats = scaler.transform(feats)

    if mi_selector is not None and mi_pool_selector is not None:
        feats_pool = mi_pool_selector.transform(feats)
        feats = mi_selector.transform(feats_pool)
    elif anova_selector is not None:
        feats = anova_selector.transform(feats)

    preds = pipe.predict(feats)
    if hasattr(pipe, 'predict_proba'):
        probs = pipe.predict_proba(feats)
    else:
        probs = None
    return preds, probs


def predict_single_image(image_path, config, device, version=None):
    output_dir = Path(config['output_dir'])
    bodies, pipe, non_const_mask, scaler, anova_sel, mi_pool_sel, mi_sel, results, active_names, num_classes = \
        load_trained_model(output_dir, version)

    resolution = config['dataset_options'].get('resolution_full', 640)
    dist_cfg = config.get('phase3', {}).get('distribution_stats', {})
    n_stats = dist_cfg.get('n_stats', 12)
    n_regions = dist_cfg.get('n_regions', 5)

    tf = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"[Error] Cannot open image: {e}")
        return None

    image_tensor = tf(image).unsqueeze(0)
    print(f"[Predict-v2] Extracting features from raw image: {image_path}")
    print(f"[Predict-v2] Image size: {image.size} -> tensor: {image_tensor.shape}")

    raw_feats = extract_features_from_images(image_tensor, bodies, device, n_stats, n_regions)
    preds, probs = predict_with_pipeline(raw_feats, pipe, non_const_mask, scaler, anova_sel,
                                          mi_pool_selector=mi_pool_sel, mi_selector=mi_sel)

    pred_class = int(preds[0])
    name = active_names[pred_class] if active_names and pred_class < len(active_names) else f"class_{pred_class}"

    print(f"\n{'='*70}")
    print(f"  PREDICTION FOR: {image_path}")
    print(f"{'='*70}")

    if probs is not None:
        prob_vec = probs[0]
        sorted_indices = np.argsort(prob_vec)[::-1]
        for idx in sorted_indices:
            n = active_names[idx] if active_names and idx < len(active_names) else f"class_{idx}"
            bar = '█' * int(prob_vec[idx] * 40)
            print(f"    {n:25s}: {prob_vec[idx]:.3f}  {bar}")
        pred_conf = float(prob_vec[pred_class])
    else:
        pred_conf = 1.0

    superclass_map = {}
    for sup_name, cls_list in FRACTURE_SUPERCLASS.items():
        for cls_id in cls_list:
            superclass_map[cls_id] = sup_name

    sup_name = superclass_map.get(pred_class, 'simple')
    print(f"\n  -> Predicted: {name} (confidence: {pred_conf:.1%})")
    print(f"  -> Superclass: {sup_name}")

    validated_path = output_dir / 'validated_formulas.json'
    if validated_path.exists():
        formulas = json.load(open(validated_path))
        formulas.sort(key=lambda f: f.get('full_res_accuracy', f.get('accuracy', 0)), reverse=True)
        print(f"\n  Top-5 contributing formulas:")
        for i, f in enumerate(formulas[:5]):
            acc = f.get('full_res_accuracy', f.get('accuracy', 0))
            print(f"    {i+1}. acc={acc:.3f}  {f['str']}")

    return {
        'image': image_path,
        'predicted_class': name,
        'confidence': pred_conf,
        'superclass': sup_name,
        'all_probs': {
            (active_names[i] if active_names and i < len(active_names) else f"class_{i}"): float(prob_vec[i])
            for i in range(len(prob_vec))
        } if probs is not None else {},
    }


def _load_yolo_labels(labels_dir, image_files, active_names):
    gt = {}
    if not labels_dir or not os.path.isdir(labels_dir):
        return gt
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_dir, base + '.txt')
        if os.path.exists(label_file):
            try:
                with open(label_file) as f:
                    lines = f.read().strip().split('\n')
                cls_ids = set()
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        cls_ids.add(int(parts[0]))
                if cls_ids:
                    cls_id = sorted(cls_ids)[0]
                    cls_name = active_names[cls_id] if active_names and cls_id < len(active_names) else f"class_{cls_id}"
                    gt[img_file] = cls_name
            except Exception:
                pass
    return gt


def predict_image_dir(image_dir, config, device, version=None, labels_dir=None):
    output_dir = Path(config['output_dir'])
    bodies, pipe, non_const_mask, scaler, anova_sel, mi_pool_sel, mi_sel, results, active_names, num_classes = \
        load_trained_model(output_dir, version)

    resolution = config['dataset_options'].get('resolution_full', 640)
    dist_cfg = config.get('phase3', {}).get('distribution_stats', {})
    n_stats = dist_cfg.get('n_stats', 12)
    n_regions = dist_cfg.get('n_regions', 5)

    tf = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])

    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    if not image_files:
        print(f"[Error] No images found in {image_dir}")
        return

    print(f"[Predict-v2] Found {len(image_files)} images in {image_dir}")

    superclass_map = {}
    for sup_name, cls_list in FRACTURE_SUPERCLASS.items():
        for cls_id in cls_list:
            superclass_map[cls_id] = sup_name

    gt_labels = _load_yolo_labels(labels_dir, image_files, active_names)
    if gt_labels:
        print(f"[Predict-v2] Loaded {len(gt_labels)} ground truth labels from {labels_dir}")

    all_results = []
    correct = 0
    total_with_gt = 0
    per_class_correct = {}
    per_class_total = {}

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue

        image_tensor = tf(image).unsqueeze(0)
        raw_feats = extract_features_from_images(image_tensor, bodies, device, n_stats, n_regions)
        preds, probs = predict_with_pipeline(raw_feats, pipe, non_const_mask, scaler, anova_sel,
                                              mi_pool_selector=mi_pool_sel, mi_selector=mi_sel)

        pred_class = int(preds[0])
        name = active_names[pred_class] if active_names and pred_class < len(active_names) else f"class_{pred_class}"
        conf = float(probs[0][pred_class]) if probs is not None else 1.0

        result = {
            'file': img_file,
            'predicted_class': name,
            'confidence': conf,
            'superclass': superclass_map.get(pred_class, 'simple'),
        }

        if img_file in gt_labels:
            true_name = gt_labels[img_file]
            result['true_class'] = true_name
            is_correct = (name == true_name)
            result['correct'] = is_correct
            if is_correct:
                correct += 1
            total_with_gt += 1

            per_class_total[true_name] = per_class_total.get(true_name, 0) + 1
            if is_correct:
                per_class_correct[true_name] = per_class_correct.get(true_name, 0) + 1

        all_results.append(result)

        mark = "✓" if result.get('correct', None) is True else ("✗" if result.get('correct', None) is False else " ")
        print(f"  [{mark}] {img_file}: {name} ({conf:.1%})")

    results_path = output_dir / 'batch_predictions_v2.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved batch predictions to {results_path}")

    pred_dist = {}
    for r in all_results:
        cls = r['predicted_class']
        pred_dist[cls] = pred_dist.get(cls, 0) + 1

    summary = {
        'total_images': len(image_files),
        'prediction_distribution': dict(sorted(pred_dist.items(), key=lambda x: -x[1])),
    }

    if total_with_gt > 0:
        overall_acc = correct / total_with_gt
        per_class_acc = {}
        for cls_name in sorted(per_class_total.keys()):
            cls_acc = per_class_correct.get(cls_name, 0) / per_class_total[cls_name]
            per_class_acc[cls_name] = round(cls_acc, 4)

        summary['images_with_ground_truth'] = total_with_gt
        summary['correct_predictions'] = correct
        summary['overall_accuracy'] = round(overall_acc, 4)
        summary['per_class_accuracy'] = per_class_acc
        summary['balanced_accuracy'] = round(np.mean(list(per_class_acc.values())), 4) if per_class_acc else 0

    print(f"\n{'='*70}")
    print(f"  FINAL PREDICTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total images: {len(image_files)}")
    print(f"\n  Prediction Distribution:")
    for cls_name, count in sorted(pred_dist.items(), key=lambda x: -x[1]):
        pct = count / len(image_files) * 100
        print(f"    {cls_name:25s}: {count:4d} ({pct:.1f}%)")

    if total_with_gt > 0:
        overall_acc = correct / total_with_gt
        print(f"\n  --- With Ground Truth ---")
        print(f"  Images with GT: {total_with_gt}")
        print(f"  Correct: {correct}")
        print(f"  Overall Accuracy: {overall_acc:.4f}")
        print(f"\n  Per-class Accuracy:")
        per_class_acc = summary.get('per_class_accuracy', {})
        for cls_name in sorted(per_class_acc.keys()):
            cls_acc = per_class_acc[cls_name]
            print(f"    {cls_name:25s}: {cls_acc:.4f} ({per_class_correct.get(cls_name, 0)}/{per_class_total[cls_name]})")
        print(f"\n  Balanced Accuracy: {summary['balanced_accuracy']:.4f}")

    summary_path = output_dir / 'final_accuracy_v2.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary to {summary_path}")


def predict_split_from_raw(config, device, split='test', version=None, show_top3=False):
    output_dir = Path(config['output_dir'])
    bodies, pipe, non_const_mask, scaler, anova_sel, mi_pool_sel, mi_sel, results, active_names, num_classes = \
        load_trained_model(output_dir, version)

    resolution = config['dataset_options'].get('resolution_full', 640)
    dist_cfg = config.get('phase3', {}).get('distribution_stats', {})
    n_stats = dist_cfg.get('n_stats', 12)
    n_regions = dist_cfg.get('n_regions', 5)

    dataset_dir = '/home/lqg1/code_8T/25/lxw/4/fracture_symbolic_v2/dataset'

    ds = HBFMIDDataset(
        dataset_dir, split=split, resolution=resolution,
        augment=False, task='classification',
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    all_images, all_labels = [], []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    print(f"[Predict-v2] {split} set: {all_images.shape[0]} images, extracting features from raw images...")

    raw_feats = extract_features_from_images(all_images, bodies, device, n_stats, n_regions)
    preds, probs = predict_with_pipeline(raw_feats, pipe, non_const_mask, scaler, anova_sel,
                                          mi_pool_selector=mi_pool_sel, mi_selector=mi_sel)

    print(f"\n{'='*70}")
    print(f"  PREDICTION RESULTS ON {split.upper()} SET (from raw images)")
    print(f"{'='*70}")

    overall_acc = (preds == all_labels).mean()
    print(f"\n  Overall Accuracy: {overall_acc:.3f}")

    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            acc = (preds[mask] == c).mean()
            name = active_names[c] if active_names and c < len(active_names) else f"class_{c}"
            per_class_acc[name] = acc
            print(f"    {name:25s}: {acc:.3f} ({mask.sum()} samples)")

    balanced_acc = np.mean(list(per_class_acc.values())) if per_class_acc else 0
    print(f"\n  Balanced Accuracy: {balanced_acc:.3f}")

    superclass_map = {}
    for sup_name, cls_list in FRACTURE_SUPERCLASS.items():
        for cls_id in cls_list:
            superclass_map[cls_id] = sup_name

    sup_preds = np.array([superclass_map.get(int(p), 'simple') for p in preds])
    sup_labels = np.array([superclass_map.get(int(l), 'simple') for l in all_labels])
    for sup_name in SUPERCLASS_NAMES:
        mask = sup_labels == sup_name
        if mask.sum() > 0:
            acc = (sup_preds[mask] == sup_name).mean()
            print(f"    Superclass {sup_name:12s}: {acc:.3f}")

    if show_top3 and probs is not None:
        print(f"\n  Top-3 predictions (first 30 samples):")
        top3_idx = np.argsort(probs, axis=1)[:, ::-1][:, :3]
        for i in range(min(30, len(all_labels))):
            true_name = active_names[all_labels[i]] if active_names and all_labels[i] < len(active_names) else f"class_{all_labels[i]}"
            top3 = [(active_names[j] if active_names and j < len(active_names) else f"class_{j}", probs[i, j])
                     for j in top3_idx[i]]
            top3_str = " > ".join([f"{n}({p:.2f})" for n, p in top3])
            mark = "✓" if preds[i] == all_labels[i] else "✗"
            print(f"    [{mark}] True={true_name:25s} | {top3_str}")

    save_results = {
        'split': split,
        'version': version,
        'mode': 'raw_image',
        'overall_accuracy': float(overall_acc),
        'balanced_accuracy': float(balanced_acc),
        'per_class_accuracy': {k: float(v) for k, v in per_class_acc.items()},
    }
    if version is not None:
        version_dir = output_dir / f'v{version}'
        version_dir.mkdir(exist_ok=True)
        save_path = version_dir / f'{split}_predictions_raw_v{version}.json'
    else:
        save_path = output_dir / f'{split}_predictions_raw.json'
    with open(save_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  Saved: {save_path}")

    return save_results


def main():
    parser = argparse.ArgumentParser(description='Fracture Prediction v2 (Raw Image)')
    parser.add_argument('--config', type=str, default='configs/fracture_v3_expanded.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--version', type=int, default=None, help='Model version (e.g. 6). Latest if not specified.')
    parser.add_argument('--image', type=str, default=None, help='Single image path')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory of images')
    parser.add_argument('--labels_dir', type=str, default=None, help='Directory of YOLO label files (.txt)')
    parser.add_argument('--split', type=str, default=None, choices=['train', 'val', 'test'],
                        help='Predict on split from raw images (extract features on-the-fly)')
    parser.add_argument('--show_top3', action='store_true', help='Show top-3 predictions (only with --split)')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.image:
        predict_single_image(args.image, config, device, version=args.version)
    elif args.image_dir:
        predict_image_dir(args.image_dir, config, device, version=args.version, labels_dir=args.labels_dir)
    elif args.split is not None:
        predict_split_from_raw(config, device, split=args.split, version=args.version, show_top3=args.show_top3)
    else:
        predict_split_from_raw(config, device, split='test', version=args.version)


if __name__ == '__main__':
    main()
