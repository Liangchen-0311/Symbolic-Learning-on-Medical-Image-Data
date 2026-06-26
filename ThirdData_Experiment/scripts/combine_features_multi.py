#!/usr/bin/env python3
"""Multi-set feature combination test for the residual-boosting rounds.
Concatenates feature matrices from several runs (same split/order) and trains
HGB, comparing cumulative combinations to see if acc keeps climbing."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings; warnings.filterwarnings('ignore')

NAMES = ['benign', 'malignant', 'normal']
RUNS = {
    'baseline':  'outputs/thirddata_reg16/features.npz',
    'residual1': 'outputs/thirddata_residual/features.npz',
    'residual2': 'outputs/thirddata_res2/features.npz',
}
D = {k: np.load(v, allow_pickle=True) for k, v in RUNS.items() if os.path.exists(v)}
ref = D['baseline']
ytr, yva, yte = ref['train_labels'], ref['val_labels'], ref['test_labels']

def feats(d, split):
    return np.nan_to_num(d[f'{split}_features'])

def run(tag, keys):
    Xtr = np.hstack([feats(D[k], 'train') for k in keys])
    Xva = np.hstack([feats(D[k], 'val') for k in keys])
    Xte = np.hstack([feats(D[k], 'test') for k in keys])
    var = Xtr.var(0) > 1e-12
    Xtr, Xva, Xte = Xtr[:, var], Xva[:, var], Xte[:, var]
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    k = min(500, Xtr.shape[1])
    sel = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
    Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
    comb = np.vstack([Xtr]); comby = ytr
    # fit on train+val
    Xva2 = sel.transform(sc.transform(np.hstack([feats(D[kk], 'val') for kk in keys])[:, var]))
    comb = np.vstack([Xtr, Xva2]); comby = np.concatenate([ytr, yva])
    sw = compute_sample_weight('balanced', comby)
    best = None
    for nm, w in [('plain', None), ('sw', sw)]:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=5, random_state=42)
        m.fit(comb, comby, sample_weight=w)
        pred = m.predict(Xte)
        acc = accuracy_score(yte, pred); bal = balanced_accuracy_score(yte, pred)
        per = {NAMES[c]: round(float((pred[yte == c] == c).mean()), 3) for c in range(3)}
        if best is None or acc > best[1]:
            best = (nm, acc, bal, per)
    print(f"{tag:34s} acc={best[1]:.4f}  bal={best[2]:.4f}  ({best[0]:5s}) {best[3]}")

print("available runs:", list(D.keys()), "\n")
run("baseline", ['baseline'])
if 'residual1' in D:
    run("baseline + residual1", ['baseline', 'residual1'])
if 'residual2' in D:
    run("baseline + residual2", ['baseline', 'residual2'])
if 'residual1' in D and 'residual2' in D:
    run("baseline + residual1 + residual2", ['baseline', 'residual1', 'residual2'])
