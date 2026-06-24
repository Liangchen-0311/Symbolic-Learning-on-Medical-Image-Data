#!/usr/bin/env python3
"""Parameter-count comparison: our symbolic+HGB model vs the deep baselines.

Ours: formulas = 0 params (fixed symbolic ops), distribution stats = 0 params.
All learned parameters live in the HGB classifier (decision-tree ensemble), whose
"parameter" count we take as the total number of tree nodes (each split stores a
feature index + threshold; each leaf stores a value). RL search params excluded.
"""
import os, sys, json
import numpy as np
import joblib

SD = os.path.dirname(os.path.abspath(__file__))
BRAIN = os.path.dirname(SD)


def count_hgb(pkl_path):
    data = joblib.load(pkl_path)
    pipe = data['pipe']
    clf = pipe.named_steps['clf'] if hasattr(pipe, 'named_steps') else pipe
    sel = pipe.named_steps.get('select', None) if hasattr(pipe, 'named_steps') else None
    nodes = leaves = trees = 0
    for iteration in clf._predictors:          # list over boosting iterations
        for tree in iteration:                 # one tree per class (multiclass)
            trees += 1
            n = tree.nodes
            nodes += len(n)
            leaves += int(n['is_leaf'].sum()) if 'is_leaf' in n.dtype.names else 0
    k = sel.k if sel is not None and hasattr(sel, 'k') else '?'
    return {'trees': trees, 'nodes': nodes, 'leaves': leaves, 'selected_features': k}


def _load_model_module(mdir):
    """Import the per-method model.py in isolation (avoid the cached 'model')."""
    import importlib.util
    sys.modules.pop('model', None)
    spec = importlib.util.spec_from_file_location('model', os.path.join(mdir, 'model.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_torch(method, num_classes=4):
    mdir = os.path.join(SD, method)
    mod = _load_model_module(mdir)
    if method == 'dinov2_linear':
        m = mod.DINOv2Backbone()
        total = sum(p.numel() for p in m.parameters())
        return {'total': total, 'trainable': 0, 'note': 'backbone frozen; head=LogReg'}
    if method == 'rulefit':
        m = mod.MobileNetV2FeatureExtractor(pretrained=False)
        total = sum(p.numel() for p in m.parameters())
        return {'total': total, 'trainable': 0, 'note': 'backbone frozen; head=rules/GBM'}
    try:
        m = mod.create_model(num_classes=num_classes, pretrained=False)
    except TypeError:
        m = mod.create_model(num_classes=num_classes)
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'note': ''}


print("=" * 78)
print("OUR SYMBOLIC + HGB MODEL")
print("=" * 78)
hgb = count_hgb(os.path.join(BRAIN, 'outputs/brain_dir3_fine/best_classifier.pkl'))
print(f"  formulas:            0 params (fixed symbolic operations)")
print(f"  distribution stats:  0 params (fixed statistics)")
print(f"  HGB trees:           {hgb['trees']}")
print(f"  HGB total nodes:     {hgb['nodes']:,}   (= splits + leaves; the learned params)")
print(f"  HGB leaves:          {hgb['leaves']:,}")
print(f"  input features used: {hgb['selected_features']}")
print(f"  => TOTAL learned params ~= {hgb['nodes']:,} (tree nodes)")

print("\n" + "=" * 78)
print(f"{'BASELINE':18s} {'total params':>16s} {'trainable':>16s}  note")
print("=" * 78)
for m in ['resnet50', 'densenet121', 'efficientnet_b0', 'swin_tiny',
          'dinov2_linear', 'rulefit', 'CBM', 'CRL']:
    try:
        r = count_torch(m)
        tr = f"{r['trainable']:,}" if r['trainable'] else "~0 (frozen)"
        print(f"{m:18s} {r['total']:>16,} {tr:>16s}  {r['note']}")
    except Exception as e:
        print(f"{m:18s}  ERROR: {e}")
print("=" * 78)
print(f"{'OURS (HGB nodes)':18s} {hgb['nodes']:>16,} {hgb['nodes']:>16,}  formulas/features = 0 params")
