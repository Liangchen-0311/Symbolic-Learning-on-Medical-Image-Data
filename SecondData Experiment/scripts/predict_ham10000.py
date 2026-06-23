#!/usr/bin/env python3
"""
Prediction script for HAM10000 skin lesion classification.

Loads a trained classifier and runs inference on new images.

Usage:
    python scripts/predict_ham10000.py --config configs/ham10000.yaml --image_dir /path/to/images
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import yaml

from src.data.ham10000_loader import HAM10000_NAMES, HAM10000_FULL_NAMES
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS
from src.symbolic.ham10000_operators import register_ham10000_operators
from src.symbolic.feature_encoding import encode_body_distribution_v2

register_ham10000_operators(TENSOR_OPERATORS)


def predict(config_path, image_dir, output_path=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config['output_dir'])

    # Load classifier
    clf_path = output_dir / 'best_classifier.pkl'
    if not clf_path.exists():
        print(f"ERROR: Classifier not found at {clf_path}")
        return

    clf_data = joblib.load(clf_path)
    pipe = clf_data['pipe']
    method = clf_data['method']
    non_const = clf_data['non_const']
    scaler = clf_data['scaler']
    anova_selector = clf_data['anova_selector']
    mi_pool_selector = clf_data.get('mi_pool_selector')
    mi_selector = clf_data.get('mi_selector')

    print(f"Loaded classifier: method={method}")

    # Load formulas
    validated_path = output_dir / 'validated_formulas.json'
    formulas = json.load(open(validated_path))

    # Build body list
    bodies = set()
    for f in formulas:
        tokens = f['str'].strip().split()
        if tokens[-1] in ROOT_OPERATORS:
            bodies.add(' '.join(tokens[:-1]))
        else:
            bodies.add(f['str'])
    bodies = sorted(bodies)

    # Load images
    from PIL import Image
    resolution = config['dataset_options']['resolution']

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(Path(image_dir).glob(ext))
    image_files = sorted(image_files)
    print(f"Found {len(image_files)} images")

    # Process images
    all_features = []
    for img_path in image_files:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((resolution, resolution))
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).cuda()

        # Build terminal data using same naming as environment
        I_R = img_tensor[:, 0, :, :]
        I_G = img_tensor[:, 1, :, :]
        I_B = img_tensor[:, 2, :, :]
        I_GRAY = 0.2989 * I_R + 0.5870 * I_G + 0.1140 * I_B
        Cmax, _ = img_tensor.max(dim=1)
        Cmin, _ = img_tensor.min(dim=1)
        delta = Cmax - Cmin + 1e-8
        H = torch.zeros_like(I_R)
        mask_r = (Cmax == I_R)
        mask_g = (Cmax == I_G) & ~mask_r
        mask_b = ~mask_r & ~mask_g
        H[mask_r] = (((I_G[mask_r] - I_B[mask_r]) / delta[mask_r]) % 6)
        H[mask_g] = ((I_B[mask_g] - I_R[mask_g]) / delta[mask_g]) + 2
        H[mask_b] = ((I_R[mask_b] - I_G[mask_b]) / delta[mask_b]) + 4
        H = H / 6.0
        S = torch.where(Cmax > 1e-8, delta / (Cmax + 1e-8), torch.zeros_like(Cmax))
        total = I_R + I_G + I_B + 1e-8
        terminal_data = {
            'I_R': I_R, 'I_G': I_G, 'I_B': I_B, 'I_GRAY': I_GRAY,
            'I_H': H, 'I_S': S,
            'I_r': I_R / total, 'I_g': I_G / total,
            'I_RG': I_R - I_G, 'I_BY': I_B - (I_R + I_G) / 2,
        }

        sample_features = []
        for body_str in bodies:
            tokens = body_str.strip().split()
            stack = []
            valid = True
            for token in tokens:
                if token in terminal_data:
                    stack.append(terminal_data[token])
                elif token in TENSOR_OPERATORS:
                    op_func, arity, _ = TENSOR_OPERATORS[token]
                    if len(stack) < arity:
                        valid = False
                        break
                    operands = [stack.pop() for _ in range(arity)]
                    operands.reverse()
                    result = op_func(*operands)
                    result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                    stack.append(result)
                else:
                    valid = False
                    break
            if not valid or len(stack) != 1:
                sample_features.append(np.zeros(1))
                continue
            result = torch.clamp(stack[0], -1e4, 1e4)
            stats = encode_body_distribution_v2(result)
            sample_features.append(stats.cpu().numpy().flatten())

        feat = np.concatenate(sample_features)
        all_features.append(feat)

    X = np.array(all_features)
    X_nc = X[:, non_const]
    X_scaled = scaler.transform(X_nc)

    use_mi = method in ['hgb_mi', 'hgb_mi_sw', 'stacking_mi']
    if use_mi:
        X_pool = mi_pool_selector.transform(X_scaled)
        X_final = mi_selector.transform(X_pool)
    else:
        X_final = anova_selector.transform(X_scaled)

    # Predict
    preds = pipe.predict(X_final)
    try:
        probs = pipe.predict_proba(X_final)
    except Exception:
        probs = None

    # Output results
    results = []
    for i, img_path in enumerate(image_files):
        pred_class = int(preds[i])
        result = {
            'image': img_path.name,
            'predicted_class': HAM10000_NAMES[pred_class],
            'full_name': HAM10000_FULL_NAMES[HAM10000_NAMES[pred_class]],
        }
        if probs is not None:
            result['probabilities'] = {
                HAM10000_NAMES[j]: float(probs[i, j]) for j in range(len(HAM10000_NAMES))
            }
        results.append(result)

    if output_path:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    else:
        for r in results:
            print(f"  {r['image']}: {r['predicted_class']} ({r['full_name']})")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    predict(args.config, args.image_dir, args.output)
