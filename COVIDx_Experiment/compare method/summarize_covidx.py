#!/usr/bin/env python3
import json, os
M = ['resnet50', 'densenet121', 'efficientnet_b0', 'swin_tiny',
     'dinov2_linear', 'rulefit', 'CBM', 'CRL']
SD = os.path.dirname(os.path.abspath(__file__))
print(f"{'method':18s} {'acc':>7} {'bal':>7} {'AUC':>7} {'normal':>7} {'pneu':>7} {'covid':>7} {'min':>6}")
print('-' * 78)
for m in M:
    p = os.path.join(SD, m, 'results.json')
    if not os.path.exists(p):
        print(f"{m:18s}  (no results)"); continue
    d = json.load(open(p)); pc = d.get('per_class', {})
    print(f"{m:18s} {d['acc']:.4f}  {d['bal_acc']:.4f}  {d['auc']:.4f}  "
          f"{pc.get('normal',0):.3f}  {pc.get('pneumonia',0):.3f}  {pc.get('covid',0):.3f}  {d['time_seconds']/60:5.1f}")
print('-' * 78)
print(f"{'OURS (symbolic)':18s} 0.8949  0.8949  0.9787  0.917  0.930  0.837   (RL+HGB)")
