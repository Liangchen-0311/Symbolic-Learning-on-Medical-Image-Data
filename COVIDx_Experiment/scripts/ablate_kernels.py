#!/usr/bin/env python3
"""Cheap ablation: how much test accuracy depends on the learnable conv kernels.

Reuses the ALREADY-VALID features.npz from the full run (computed when the kernel
bank was correctly on GPU). For each formula (= a 60-col block) we know its body
string, so we can keep only the feature columns whose formula avoids certain
operator groups, retrain HGB, and compare test accuracy. No RL, no GPU, no bug.

Groups:
  full      : all formulas (baseline, should reproduce ~0.899)
  B1        : drop only RANDOM learnable kernels (conv3x3_*, conv5x5_*); keep classic_*
  B2        : drop ALL kernel-bank ops (classic_* and conv*) -> pure base operators
"""
import sys, os, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings; warnings.filterwarnings('ignore')

OUT = Path('outputs/brain_tumor')
NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']
d = np.load(OUT / 'features.npz', allow_pickle=True)
Xtr, ytr = d['train_features'], d['train_labels']
Xva, yva = d['val_features'],   d['val_labels']
Xte, yte = d['test_features'],  d['test_labels']
bodies = [str(b) for b in d['bodies']]
n_bodies = len(bodies)
stats = Xtr.shape[1] // n_bodies
print(f"{n_bodies} bodies x {stats} stats = {Xtr.shape[1]} dims")

RND = re.compile(r'conv3x3_|conv5x5_')          # random learnable
ANYK = re.compile(r'conv3x3_|conv5x5_|classic_')  # any kernel-bank op

def col_mask(keep_body):
    """Boolean column mask keeping the 60-col block of each body where keep_body(body)."""
    m = np.zeros(Xtr.shape[1], dtype=bool)
    kept = 0
    for i, b in enumerate(bodies):
        if keep_body(b):
            m[i * stats:(i + 1) * stats] = True
            kept += 1
    return m, kept

def run(tag, keep_body):
    m, kept = col_mask(keep_body)
    if kept == 0:
        print(f"\n[{tag}] no formulas -> skip"); return
    Atr, Ava, Ate = Xtr[:, m], Xva[:, m], Xte[:, m]
    comb = np.concatenate([Atr, Ava], 0); comby = np.concatenate([ytr, yva], 0)
    var = comb.var(0) > 1e-12
    comb, Ate2 = comb[:, var], Ate[:, var]
    sc = StandardScaler().fit(comb)
    comb_s, Ate_s = sc.transform(comb), sc.transform(Ate2)
    k = min(1000, comb_s.shape[1])
    pipe = Pipeline([('sel', SelectKBest(f_classif, k=min(300, k))),
                     ('clf', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                            max_depth=5, random_state=42))])
    # quick ANOVA preselect to k then the pipe
    pre = SelectKBest(f_classif, k=k).fit(comb_s, comby)
    pipe.fit(pre.transform(comb_s), comby)
    pred = pipe.predict(pre.transform(Ate_s))
    acc = accuracy_score(yte, pred); bal = balanced_accuracy_score(yte, pred)
    per = {NAMES[c]: round(float((pred[yte == c] == c).mean()), 3) for c in range(4)}
    print(f"\n[{tag}] formulas={kept}/{n_bodies}  dims={int(m.sum())}")
    print(f"  test acc={acc:.4f}  bal={bal:.4f}  per-class={per}")

run("full (all kernels)",        lambda b: True)
run("B1 (drop random conv)",     lambda b: RND.search(b) is None)
run("B2 (drop ALL kernel ops)",  lambda b: ANYK.search(b) is None)
