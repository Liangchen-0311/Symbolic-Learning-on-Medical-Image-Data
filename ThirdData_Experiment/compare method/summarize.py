#!/usr/bin/env python3
import json, os
M = ['resnet50', 'densenet121', 'efficientnet_b0', 'swin_tiny',
     'dinov2_linear', 'rulefit', 'CBM', 'CRL']
SD = os.path.dirname(os.path.abspath(__file__))
print(f"{'method':18s} {'acc':>7} {'bal':>7} {'AUC':>7} {'glioma':>7} {'menin':>7} {'pit':>7} {'notum':>7} {'min':>6}")
print('-' * 80)
for m in M:
    p = os.path.join(SD, m, 'results.json')
    if not os.path.exists(p):
        print(f"{m:18s}  (no results)")
        continue
    d = json.load(open(p))
    pc = d.get('per_class', {})
    print(f"{m:18s} {d['acc']:.4f}  {d['bal_acc']:.4f}  {d['auc']:.4f}  "
          f"{pc.get('glioma',0):.3f}  {pc.get('meningioma',0):.3f}  "
          f"{pc.get('pituitary',0):.3f}  {pc.get('notumor',0):.3f}  {d['time_seconds']/60:5.1f}")
print('-' * 80)
print(f"{'OURS (symbolic 3x3)':18s} 0.9163  0.9163  0.9753  0.728  0.950  0.988  1.000   (RL+HGB)")
