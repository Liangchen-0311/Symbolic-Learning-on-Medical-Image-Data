#!/usr/bin/env python3
"""What characterizes glioma-discriminative formulas?

Uses the default (focus) run's features.npz. For each formula (a 12-stat x
5-region = 60-col block) we measure its glioma-vs-rest test AUC, rank, then:
  (A) operator / terminal enrichment in the top formulas vs all formulas
  (B) which (region, statistic) cells carry the most glioma signal (region prior)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import f_classif
import warnings; warnings.filterwarnings('ignore')

OUT = Path('outputs/brain_dir3')   # the default (focus) run
GLIOMA = 0
NSTAT, NREG = 12, 5
STATN = ['mean', 'std', 'max', 'skew', 'kurt', 'q10', 'q25', 'q50', 'q75', 'q90', 'ratio>mu', 'range']
REGN = ['whole', 'top-left', 'top-right', 'bot-left', 'bot-right']

d = np.load(OUT / 'features.npz', allow_pickle=True)
Xtr, ytr = d['train_features'], d['train_labels']
Xva, yva = d['val_features'], d['val_labels']
Xte, yte = d['test_features'], d['test_labels']
bodies = [str(b) for b in d['bodies']]
nf = len(bodies); stats = Xtr.shape[1] // nf
print(f"{nf} formulas x {stats} stats; analyzing glioma-vs-rest\n")

comb = np.nan_to_num(np.concatenate([Xtr, Xva]))
comby = np.concatenate([ytr, yva])
combyb = (comby == GLIOMA).astype(int)
Xte = np.nan_to_num(Xte); yteb = (yte == GLIOMA).astype(int)

# ---- (A) per-formula glioma-vs-rest AUC ----
aucs = np.zeros(nf)
for i in range(nf):
    sl = slice(i * stats, (i + 1) * stats)
    sc = StandardScaler().fit(comb[:, sl])
    lr = LogisticRegression(max_iter=200, C=0.5).fit(sc.transform(comb[:, sl]), combyb)
    aucs[i] = roc_auc_score(yteb, lr.predict_proba(sc.transform(Xte[:, sl]))[:, 1])
order = np.argsort(aucs)[::-1]
topK = order[:40]
print(f"glioma-vs-rest single-formula AUC: max={aucs.max():.3f}, "
      f"top40 mean={aucs[topK].mean():.3f}, all mean={aucs.mean():.3f}\n")

print("TOP-15 glioma-discriminative formulas (AUC | body):")
for i in topK[:15]:
    print(f"  {aucs[i]:.3f}  {bodies[i]}")

# ---- enrichment of operators/terminals in top-40 vs all ----
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
print("\nENRICHED in top-40 glioma formulas (token | in_top/40 | in_all/%d | enrichment x):" % nf)
for t, tn, an, enr in rows[:18]:
    if tn >= 3:
        print(f"  {t:16s} {tn:3d}/40  {an:4d}/{nf}  x{enr:.2f}")
print("\nDEPLETED (present overall but rare in top-40):")
for t, tn, an, enr in sorted(rows, key=lambda r: r[3])[:8]:
    if an >= nf * 0.1:
        print(f"  {t:16s} {tn:3d}/40  {an:4d}/{nf}  x{enr:.2f}")

# ---- (B) region x stat glioma signal (averaged over all formulas) ----
F, _ = f_classif(comb, combyb)
F = np.nan_to_num(F)
grid = np.zeros((NREG, NSTAT)); cnt = np.zeros((NREG, NSTAT))
for j in range(F.shape[0]):
    loc = j % stats; r, s = loc // NSTAT, loc % NSTAT
    if r < NREG and s < NSTAT:
        grid[r, s] += F[j]; cnt[r, s] += 1
grid /= np.maximum(cnt, 1)
print("\n(B) mean glioma F-score by REGION (higher = more glioma signal):")
for r in np.argsort(grid.mean(1))[::-1]:
    print(f"  {REGN[r]:10s}: {grid[r].mean():7.2f}")
print("\n    by STATISTIC:")
for s in np.argsort(grid.mean(0))[::-1][:6]:
    print(f"  {STATN[s]:10s}: {grid[:, s].mean():7.2f}")
print("\n    top 8 (region, stat) cells:")
flat = [(REGN[j // NSTAT], STATN[j % NSTAT], grid[j // NSTAT, j % NSTAT])
        for j in range(NREG * NSTAT)]
for rn, sn, v in sorted(flat, key=lambda x: -x[2])[:8]:
    print(f"  {rn:10s} {sn:10s}: {v:7.2f}")
