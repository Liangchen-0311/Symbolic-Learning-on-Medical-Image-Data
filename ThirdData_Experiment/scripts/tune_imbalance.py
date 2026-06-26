#!/usr/bin/env python3
"""Classifier-side imbalance handling for ThirdData/BUSI on the reg16 features.

Analysis showed NORMAL is highly separable per-formula (AUC up to 0.95) yet its
recall is only ~0.54 — the bottleneck is multiclass competition with the
majority classes (benign 310 / malignant 148 / normal 87), not missing signal.
So we attack it at the classifier: class weights, per-class decision thresholds,
SMOTE oversampling. Discipline: fit on train, SELECT on val, REPORT on test.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import warnings; warnings.filterwarnings('ignore')

OUT = Path('outputs/thirddata_reg16')
NAMES = ['benign', 'malignant', 'normal']
NORMAL = 2
d = np.load(OUT / 'features.npz', allow_pickle=True)
Xtr, ytr = np.nan_to_num(d['train_features']), d['train_labels']
Xva, yva = np.nan_to_num(d['val_features']), d['val_labels']
Xte, yte = np.nan_to_num(d['test_features']), d['test_labels']

# Train-only preprocessing: variance filter -> scale -> ANOVA top-300.
var = Xtr.var(0) > 1e-12
Xtr, Xva, Xte = Xtr[:, var], Xva[:, var], Xte[:, var]
sc = StandardScaler().fit(Xtr)
Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
k = min(300, Xtr.shape[1])
sel = SelectKBest(f_classif, k=k).fit(Xtr, ytr)
Xtr, Xva, Xte = sel.transform(Xtr), sel.transform(Xva), sel.transform(Xte)

HGB = dict(max_iter=300, learning_rate=0.05, max_depth=5, random_state=42)

def report(tag, model, thr=None):
    Pva, Pte = model.predict_proba(Xva), model.predict_proba(Xte)
    def ev(P, y):
        pred = (P * thr).argmax(1) if thr is not None else P.argmax(1)
        per = {NAMES[c]: round(float((pred[y == c] == c).mean()), 3) for c in range(3)}
        return accuracy_score(y, pred), balanced_accuracy_score(y, pred), per
    va, vb, vper = ev(Pva, yva); ta, tb, tper = ev(Pte, yte)
    print(f"\n[{tag}]")
    print(f"  val : acc={va:.4f} bal={vb:.4f} {vper}")
    print(f"  test: acc={ta:.4f} bal={tb:.4f} {tper}")
    return ta, tb

# 0) Baseline (plain HGB)
m = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr)
report("0 baseline (argmax)", m)

# 1) sample_weight = balanced
from sklearn.utils.class_weight import compute_sample_weight
sw = compute_sample_weight('balanced', ytr)
m_sw = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr, sample_weight=sw)
report("1 sample_weight=balanced", m_sw)

# 2) stronger normal weight (sweep on val)
best = (1.0, -1)
for w in [1, 2, 3, 4, 6, 8]:
    sw2 = np.where(ytr == NORMAL, float(w), 1.0)
    mm = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr, sample_weight=sw2)
    vb = balanced_accuracy_score(yva, mm.predict(Xva))
    if vb > best[1]:
        best = (w, vb)
sw3 = np.where(ytr == NORMAL, float(best[0]), 1.0)
m_nw = HistGradientBoostingClassifier(**HGB).fit(Xtr, ytr, sample_weight=sw3)
report(f"2 normal-weight x{best[0]} (val-tuned)", m_nw)

# 3) per-class threshold multiplier on the balanced-sw model (tune normal mult on val)
Pva = m_sw.predict_proba(Xva)
best = (1.0, -1)
for t in np.linspace(1.0, 3.0, 41):
    thr = np.array([1.0, 1.0, t])
    vb = balanced_accuracy_score(yva, (Pva * thr).argmax(1))
    if vb > best[1]:
        best = (t, vb)
report(f"3 sample_weight + normal-threshold x{best[0]:.2f} (val-tuned)",
       m_sw, thr=np.array([1.0, 1.0, best[0]]))

# 4) SMOTE oversampling (if available)
try:
    from imblearn.over_sampling import SMOTE
    Xr, yr = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
    m_sm = HistGradientBoostingClassifier(**HGB).fit(Xr, yr)
    report("4 SMOTE oversample", m_sm)
except Exception as e:
    print(f"\n[4 SMOTE] unavailable: {e}")
