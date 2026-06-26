#!/usr/bin/env python3
"""Post-hoc glioma threshold / class-weight tuning on the FIXED 800-formula
features. Discipline: fit on train, SELECT on val, REPORT on test.

Reuses the saved best_classifier.pkl only for (a) the train-only preprocessing
(non_const mask, StandardScaler, ANOVA selector) and (b) the winning HGB
hyper-parameters. The final estimator is refit on TRAIN ONLY so the val set is
clean for threshold/weight selection.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score

OUT = Path('outputs/brain_tumor')
NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']
GLIOMA = 0

d = np.load(OUT / 'features.npz', allow_pickle=True)
Xtr, ytr = d['train_features'], d['train_labels']
Xva, yva = d['val_features'],   d['val_labels']
Xte, yte = d['test_features'],  d['test_labels']

clf = joblib.load(OUT / 'best_classifier.pkl')
non_const, scaler, anova = clf['non_const'], clf['scaler'], clf['anova_selector']
sel_k = clf['pipe'].named_steps['select'].k
hp = clf['pipe'].named_steps['clf'].get_params()
hgb_kw = {k: hp[k] for k in ('max_iter', 'learning_rate', 'max_depth', 'random_state')}
print(f"Reusing: ANOVA->SelectKBest(k={sel_k}), HGB={hgb_kw}")

def tf(X):
    return anova.transform(scaler.transform(X[:, non_const]))
Atr, Ava, Ate = tf(Xtr), tf(Xva), tf(Xte)

def make():
    return Pipeline([('select', SelectKBest(f_classif, k=sel_k)),
                     ('clf', HistGradientBoostingClassifier(**hgb_kw))])

def evl(P, y, w=None):
    pred = (P if w is None else P * w).argmax(1)
    per = {NAMES[c]: round(float((pred[y == c] == c).mean()), 3) for c in range(4)}
    return accuracy_score(y, pred), balanced_accuracy_score(y, pred), per

def show(tag, va, te):
    print(f"\n[{tag}]")
    print(f"  val : acc={va[0]:.4f} bal={va[1]:.4f} {va[2]}")
    print(f"  test: acc={te[0]:.4f} bal={te[1]:.4f} {te[2]}")

# ---- Baseline: final estimator refit on TRAIN ONLY ----
base = make(); base.fit(Atr, ytr)
Pva, Pte = base.predict_proba(Ava), base.predict_proba(Ate)
show("Baseline (train-only fit, argmax)", evl(Pva, yva), evl(Pte, yte))

# ---- A: glioma posterior multiplier, tuned on val ----
best = (1.0, -1)
for a in np.linspace(1.0, 3.0, 41):
    acc = evl(Pva, yva, np.array([a, 1, 1, 1]))[0]
    if acc > best[1]: best = (a, acc)
wa = np.array([best[0], 1, 1, 1])
show(f"A: glioma x{best[0]:.2f} (val-tuned)", evl(Pva, yva, wa), evl(Pte, yte, wa))

# ---- A2: glioma up + meningioma down, tuned on val ----
best = (1.0, 1.0, -1)
for a in np.linspace(1.0, 3.0, 21):
    for b in np.linspace(0.5, 1.0, 11):
        acc = evl(Pva, yva, np.array([a, b, 1, 1]))[0]
        if acc > best[2]: best = (a, b, acc)
wa2 = np.array([best[0], best[1], 1, 1])
show(f"A2: glioma x{best[0]:.2f}, mening x{best[1]:.2f} (val-tuned)",
     evl(Pva, yva, wa2), evl(Pte, yte, wa2))

# ---- B: class weight via sample_weight (refit), tuned on val ----
best = (1.0, -1, None)
for bw in [1, 1.5, 2, 3, 4, 6]:
    sw = np.where(ytr == GLIOMA, float(bw), 1.0)
    p = make(); p.fit(Atr, ytr, clf__sample_weight=sw)
    acc = evl(p.predict_proba(Ava), yva)[0]
    if acc > best[1]: best = (bw, acc, p)
pB = best[2]
show(f"B: glioma class-weight x{best[0]} (val-tuned, refit)",
     evl(pB.predict_proba(Ava), yva), evl(pB.predict_proba(Ate), yte))
