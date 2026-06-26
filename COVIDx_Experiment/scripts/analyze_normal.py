#!/usr/bin/env python3
"""What characterizes NORMAL-vs-rest discriminative formulas? (ThirdData/BUSI)

Uses the reg16 run's features.npz. Per-formula normal-vs-rest AUC, then operator/
terminal enrichment in the top formulas. Guides how to boost the weak 'normal' class.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')

OUT = Path('outputs/thirddata_reg16')
NORMAL = 2  # benign=0, malignant=1, normal=2

d = np.load(OUT / 'features.npz', allow_pickle=True)
Xtr, ytr = np.nan_to_num(d['train_features']), d['train_labels']
Xva, yva = np.nan_to_num(d['val_features']), d['val_labels']
Xte, yte = np.nan_to_num(d['test_features']), d['test_labels']
bodies = [str(b) for b in d['bodies']]
nf = len(bodies); stats = Xtr.shape[1] // nf
print(f"{nf} formulas x {stats} stats; analyzing NORMAL-vs-rest\n")

comb = np.concatenate([Xtr, Xva]); comby = np.concatenate([ytr, yva])
combyb = (comby == NORMAL).astype(int); yteb = (yte == NORMAL).astype(int)

aucs = np.zeros(nf)
for i in range(nf):
    sl = slice(i * stats, (i + 1) * stats)
    sc = StandardScaler().fit(comb[:, sl])
    lr = LogisticRegression(max_iter=200, C=0.5).fit(sc.transform(comb[:, sl]), combyb)
    aucs[i] = roc_auc_score(yteb, lr.predict_proba(sc.transform(Xte[:, sl]))[:, 1])
order = np.argsort(aucs)[::-1]; topK = order[:40]
print(f"normal-vs-rest single-formula AUC: max={aucs.max():.3f}, "
      f"top40 mean={aucs[topK].mean():.3f}, all mean={aucs.mean():.3f}\n")
print("TOP-12 normal-discriminative formulas:")
for i in topK[:12]:
    print(f"  {aucs[i]:.3f}  {bodies[i]}")

def presence(idxs):
    c = {}
    for i in idxs:
        for t in set(bodies[i].split()):
            c[t] = c.get(t, 0) + 1
    return c
allc, topc = presence(range(nf)), presence(topK)
rows = []
for t, ac in allc.items():
    af, tf = ac / nf, topc.get(t, 0) / len(topK)
    rows.append((t, topc.get(t, 0), ac, tf / af if af > 0 else 0))
rows.sort(key=lambda r: -r[3])
print("\nENRICHED in top-40 normal formulas (token | top/40 | all | enrichment):")
for t, tn, an, enr in rows[:16]:
    if tn >= 3:
        print(f"  {t:16s} {tn:3d}/40  {an:4d}/{nf}  x{enr:.2f}")
