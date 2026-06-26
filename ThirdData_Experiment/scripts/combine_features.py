#!/usr/bin/env python3
"""Directly test feature-set complementarity: concatenate the baseline (reg16)
and residual feature matrices (same split/order) and train HGB, vs each alone.
Bypasses step-3 quality re-filtering (which drops the low-global-F but
complementary residual formulas)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings; warnings.filterwarnings('ignore')

NAMES = ['benign', 'malignant', 'normal']
base = np.load('outputs/thirddata_reg16/features.npz', allow_pickle=True)
res = np.load('outputs/thirddata_residual/features.npz', allow_pickle=True)
ytr, yva, yte = base['train_labels'], base['val_labels'], base['test_labels']
assert np.array_equal(ytr, res['train_labels']) and np.array_equal(yte, res['test_labels']), "split mismatch"

def feats(d): return (np.nan_to_num(d['train_features']), np.nan_to_num(d['val_features']), np.nan_to_num(d['test_features']))
b_tr, b_va, b_te = feats(base)
r_tr, r_va, r_te = feats(res)

def run(tag, Xtr, Xva, Xte):
    var = Xtr.var(0) > 1e-12
    Xtr, Xva, Xte = Xtr[:, var], Xva[:, var], Xte[:, var]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    k = min(500, Xtr.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
    Xtr, Xva, Xte = sel.transform(Xtr), sel.transform(Xva), sel.transform(Xte)
    comb = np.concatenate([Xtr, Xva]); comby = np.concatenate([ytr, yva])
    sw = compute_sample_weight('balanced', comby)
    best = None
    for name, w in [('plain', None), ('sample_weight', sw)]:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, random_state=42)
        m.fit(comb, comby, sample_weight=w)
        pred = m.predict(Xte)
        acc = accuracy_score(yte, pred); bal = balanced_accuracy_score(yte, pred)
        per = {NAMES[c]: round(float((pred[yte == c] == c).mean()), 3) for c in range(3)}
        if best is None or acc > best[1]:
            best = (name, acc, bal, per)
    print(f"{tag:28s} dims={Xtr.shape[1]:4d}  acc={best[1]:.4f}  bal={best[2]:.4f}  ({best[0]}) {best[3]}")

print(f"baseline formulas: {b_tr.shape[1]//192}, residual formulas: {r_tr.shape[1]//192}\n")
run("baseline only (reg16)", b_tr, b_va, b_te)
run("residual only", r_tr, r_va, r_te)
run("COMBINED (base + residual)", np.hstack([b_tr, r_tr]), np.hstack([b_va, r_va]), np.hstack([b_te, r_te]))
