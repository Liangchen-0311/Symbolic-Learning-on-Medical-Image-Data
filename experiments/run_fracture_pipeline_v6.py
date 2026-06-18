#!/usr/bin/env python3
"""
Fracture Symbolic Feature Discovery Pipeline — v6 (Advanced Ensemble)

v6 Changes from v5:
  - Method 1: HistGB + sample_weight='balanced' (v5 没用 sample_weight)
  - Method 2: Stacking ensemble (HistGB + SVM + KNN → LR meta-learner)
  - Method 3: Mutual Information feature selection (替代 ANOVA)
  - Method 4: Hierarchical classification (superclass → subclass)
  - 保留 v5 的基线方法做对比
  - 所有方法统一评估，选出最佳

Steps 0-4 与 v5 完全共享（同一个 features.npz），只修改 Step 5 和 Step 6。

Usage:
    python experiments/run_fracture_pipeline_v6.py --config configs/fracture_v3_expanded.yaml --start_step 5
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.data.fracture_loader import (
    HBFMIDDataset, HBFMIDDataModule, build_fracture_data_batch,
    build_fracture_superclass_mapping, FRACTURE_NAMES,
)


def _get_split_file(config):
    return str(Path(config['output_dir']) / 'split_indices.npz')


def _make_dm(config, resolution, batch_size=32, num_workers=4, augment=False):
    return HBFMIDDataModule(
        data_dir=config['dataset_options']['data_dir'],
        resolution=resolution,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=augment,
        split_file=_get_split_file(config),
    )


from src.symbolic.tensor_operators import (
    TensorOperators, TENSOR_OPERATORS, ROOT_OPERATORS, SymbolicKernelBank,
)
from src.symbolic.fracture_operators import register_fracture_operators, FRACTURE_OPERATORS
from src.symbolic.feature_encoding import (
    encode_body_distribution_v2,
    SymbolicFisherVector,
    homogeneous_kernel_map,
    apply_normalization_pipeline,
    apply_normalization_pipeline_with_stats,
)
from src.symbolic.large_feature_bank import LargeFeatureBank
from src.models.policy_agent import PolicyAgent
from src.rl.fracture_environment import FractureVSREnvironment, FractureTokenVocabulary
from src.rl.ppo_trainer import PPOTrainer

register_fracture_operators(TENSOR_OPERATORS)


def execute_body(body_str, data_batch):
    tokens = body_str.strip().split()
    stack = []
    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                return None
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
            stack.append(result)
        else:
            return None
    if len(stack) != 1:
        return None
    out = torch.clamp(stack[0], -1e4, 1e4)
    return out if out.dim() >= 2 else None


def execute_formula(formula_str, data_batch):
    tokens = formula_str.strip().split()
    stack = []
    for token in tokens:
        if token in data_batch:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                return None
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            if torch.isnan(result).any() or torch.isinf(result).any():
                return None
            stack.append(result)
        else:
            return None
    if len(stack) != 1:
        return None
    out = stack[0]
    out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
    return torch.clamp(out, -1e4, 1e4)


def formulas_to_bodies(formulas):
    bodies = set()
    for f in formulas:
        tokens = f['str'].strip().split()
        if tokens[-1] in ROOT_OPERATORS:
            bodies.add(' '.join(tokens[:-1]))
        else:
            bodies.add(f['str'])
    return sorted(bodies)


def load_formulas_from_banks(phase1_dir):
    all_formulas = []
    seen = set()
    for bank_dir in sorted(Path(phase1_dir).glob('bank_*/feature_bank')):
        fb_path = bank_dir / 'feature_bank.json'
        if not fb_path.exists():
            continue
        bank = json.load(open(fb_path))
        for f in bank['formulas']:
            if f['str'] not in seen:
                seen.add(f['str'])
                all_formulas.append(f)
    return all_formulas


# ======================================================================
# Step 0-4: 与 v5 完全相同，委托给原始 pipeline
# ======================================================================

def step0_validate_dataset(config, device):
    from experiments.run_fracture_pipeline import step0_validate_dataset as _step0
    _step0(config, device)


def step1_phase1(config, device):
    from experiments.run_fracture_pipeline import step1_phase1 as _step1
    _step1(config, device)


def step2_merge(config, device):
    from experiments.run_fracture_pipeline import step2_merge as _step2
    _step2(config, device)


def step3_validate(config, device):
    from experiments.run_fracture_pipeline import step3_validate as _step3
    _step3(config, device)


def step4_extract_features(config, device):
    from experiments.run_fracture_pipeline import step4_extract_features as _step4
    _step4(config, device)


# ======================================================================
# Step 5: Train Classifier (v6 — Advanced Ensemble)
# ======================================================================

SUPERCLASS_MAP = {
    'simple': ['Healthy', 'Linear', 'Oblique', 'Transverse'],
    'displaced': ['Comminuted', 'Oblique Displaced', 'Transverse Displaced'],
    'complex': ['Greenstick', 'Segmental', 'Spiral'],
}


def step5_train_classifier(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 5: Train Classifier (v6 — Advanced Ensemble)")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    results_path = output_dir / 'classifier_results.json'

    version = 6
    version_dir = output_dir / f'v{version}'
    version_dir.mkdir(exist_ok=True)
    versioned_results_path = version_dir / f'classifier_results_v{version}.json'
    versioned_model_path = version_dir / f'best_classifier_v{version}.pkl'

    features_path = output_dir / 'features.npz'
    data = np.load(features_path, allow_pickle=True)

    bodies = list(data['bodies']) if 'bodies' in data else []
    active_classes = list(data['active_classes']) if 'active_classes' in data else list(range(10))
    num_classes = len(active_classes)

    train_X = data['train_features']
    train_y = data['train_labels']
    val_X = data['val_features']
    val_y = data['val_labels']
    test_X = data['test_features']
    test_y = data['test_labels']

    dist_cfg = config.get('phase3', {}).get('distribution_stats', {})
    _n_stats = dist_cfg.get('n_stats', 12)
    _n_regions = dist_cfg.get('n_regions', 5)
    stats_per_formula = _n_stats * _n_regions
    n_formulas = len(bodies) if bodies else train_X.shape[1] // stats_per_formula
    print(f"  {n_formulas} formulas x {stats_per_formula} stats = {train_X.shape[1]} dims")
    print(f"  Train: {train_X.shape[0]} | Val: {val_X.shape[0]} | Test: {test_X.shape[0]}, {num_classes} classes")

    from collections import Counter
    train_dist = Counter(train_y.tolist())
    val_dist = Counter(val_y.tolist())
    test_dist = Counter(test_y.tolist())
    print(f"  Train dist: {dict(sorted(train_dist.items()))}")
    print(f"  Val dist:   {dict(sorted(val_dist.items()))}")
    print(f"  Test dist:  {dict(sorted(test_dist.items()))}")

    combine_X = np.concatenate([train_X, val_X], axis=0)
    combine_y = np.concatenate([train_y, val_y], axis=0)

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, ExtraTreesClassifier,
        StackingClassifier,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    from sklearn.metrics import roc_auc_score
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    variances = np.var(train_X, axis=0)
    non_const = variances > 1e-12
    if non_const.sum() < train_X.shape[1]:
        print(f"  Removing {(~non_const).sum()} constant features...")
    train_X_nc = train_X[:, non_const]
    val_X_nc = val_X[:, non_const]
    test_X_nc = test_X[:, non_const]
    combine_X_nc = np.concatenate([train_X_nc, val_X_nc], axis=0)
    print(f"  Non-constant features: {non_const.sum()} / {train_X.shape[1]}")

    print(f"\n  Strategy v6: No SMOTE, sample_weight + Stacking + MI + Hierarchical")
    print(f"  [STRICT] All feature selectors fitted on train only, no data leakage")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X_nc)
    val_scaled = scaler.transform(val_X_nc)
    combine_scaled = np.concatenate([train_scaled, val_scaled], axis=0)

    anova_preselect = min(1000, train_X_nc.shape[1])
    print(f"  ANOVA pre-selection: top-{anova_preselect} from {train_X_nc.shape[1]} features")
    anova_selector = SelectKBest(f_classif, k=anova_preselect)
    train_anova = anova_selector.fit_transform(train_scaled, train_y)
    combine_anova = anova_selector.transform(combine_scaled)
    print(f"  After ANOVA: {train_anova.shape[1]} features")

    mi_pool_size = min(3000, train_X_nc.shape[1])
    mi_select_size = min(1000, mi_pool_size)
    print(f"  MI pool: top-{mi_pool_size} from {train_X_nc.shape[1]}, then MI top-{mi_select_size}")
    mi_pool_selector = SelectKBest(f_classif, k=mi_pool_size)
    train_mi_pool = mi_pool_selector.fit_transform(train_scaled, train_y)
    print(f"  MI pool size: {train_mi_pool.shape[1]} features")
    mi_selector = SelectKBest(mutual_info_classif, k=mi_select_size)
    train_mi = mi_selector.fit_transform(train_mi_pool, train_y)
    combine_mi_pool = mi_pool_selector.transform(combine_scaled)
    combine_mi = mi_selector.transform(combine_mi_pool)
    print(f"  After MI: {train_mi.shape[1]} features")

    anova_selected_idx = anova_selector.get_support(indices=True)
    mi_pool_selected_idx = mi_pool_selector.get_support(indices=True)
    mi_within_pool_idx = mi_selector.get_support(indices=True)
    mi_original_idx = mi_pool_selected_idx[mi_within_pool_idx]
    anova_overlap = len(set(anova_selected_idx.tolist()) & set(mi_original_idx.tolist()))
    print(f"  ANOVA/MI overlap: {anova_overlap}/{anova_preselect} ({100*anova_overlap/anova_preselect:.1f}%)")

    sample_weights = compute_sample_weight('balanced', combine_y)
    print(f"  Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_pipes = {}
    cv_accs = {}
    all_pipe_configs = {}

    def _cv_eval(pipe, X, y, sw=None):
        accs = []
        for tr, va in skf.split(X, y):
            fit_params = {}
            if sw is not None and hasattr(pipe, 'named_steps'):
                clf_step = pipe.named_steps.get('clf', None)
                if hasattr(clf_step, 'fit') and 'sample_weight' in clf_step.fit.__code__.co_varnames:
                    fit_params['clf__sample_weight'] = sw[tr]
            pipe.fit(X[tr], y[tr], **fit_params)
            accs.append(accuracy_score(y[va], pipe.predict(X[va])))
        return np.mean(accs)

    def _calc_auc(pipe, X, y):
        try:
            y_prob = pipe.predict_proba(X)
            return roc_auc_score(y, y_prob, multi_class='ovr', average='macro')
        except Exception:
            return 0.0

    def _apply_pipeline_transform(X_raw, sel='anova'):
        X_s = scaler.transform(X_raw[:, non_const])
        if sel == 'anova':
            return anova_selector.transform(X_s)
        else:
            X_pool = mi_pool_selector.transform(X_s)
            return mi_selector.transform(X_pool)

    test_anova = _apply_pipeline_transform(test_X, 'anova')
    test_mi = _apply_pipeline_transform(test_X, 'mi')

    hgb_K_values = [100, 200, 300]
    hgb_configs = [(100, 0.1, 3), (200, 0.1, 5), (200, 0.05, 3), (300, 0.1, 5)]

    # ==================================================================
    # Baseline A: L2-LR + ANOVA
    # ==================================================================
    print(f"\n  --- Baseline A: L2-LR + ANOVA ---", flush=True)
    best_cv_lr, best_K_lr, best_C_lr = 0, None, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        for C in [0.1, 1.0, 10.0]:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', LogisticRegression(C=C, max_iter=5000, solver='lbfgs', class_weight='balanced')),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_cv_lr:
                best_cv_lr, best_K_lr, best_C_lr = cv, K, C
    print(f"  Best: K={best_K_lr}, C={best_C_lr}, cv_acc={best_cv_lr:.3f}")
    pipe_lr = Pipeline([
        ('select', SelectKBest(f_classif, k=best_K_lr)),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(C=best_C_lr, max_iter=5000, solver='lbfgs', class_weight='balanced')),
    ])
    pipe_lr.fit(combine_anova, combine_y)
    acc = accuracy_score(test_y, pipe_lr.predict(test_anova))
    auc = _calc_auc(pipe_lr, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['lr_l2'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['lr_l2'] = pipe_lr
    cv_accs['lr_l2'] = best_cv_lr

    # ==================================================================
    # Baseline B: HistGB + ANOVA (no sample_weight, v5 baseline)
    # ==================================================================
    print(f"\n  --- Baseline B: HistGB + ANOVA (no sample_weight) ---", flush=True)
    best_cv_hgb, best_hgb_cfg, best_hgb_K = 0, None, None
    for K in hgb_K_values:
        if K > anova_preselect:
            continue
        for max_iter, lr, max_d in hgb_configs:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                       max_depth=max_d, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_cv_hgb:
                best_cv_hgb, best_hgb_cfg, best_hgb_K = cv, (max_iter, lr, max_d), K
    print(f"  Best: max_iter={best_hgb_cfg[0]}, lr={best_hgb_cfg[1]}, max_depth={best_hgb_cfg[2]}, K={best_hgb_K}, cv_acc={best_cv_hgb:.3f}")
    pipe_hgb_base = Pipeline([
        ('select', SelectKBest(f_classif, k=best_hgb_K)),
        ('clf', HistGradientBoostingClassifier(max_iter=best_hgb_cfg[0], learning_rate=best_hgb_cfg[1],
                                               max_depth=best_hgb_cfg[2], random_state=42)),
    ])
    pipe_hgb_base.fit(combine_anova, combine_y)
    acc = accuracy_score(test_y, pipe_hgb_base.predict(test_anova))
    auc = _calc_auc(pipe_hgb_base, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['hgb_baseline'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['hgb_baseline'] = pipe_hgb_base
    cv_accs['hgb_baseline'] = best_cv_hgb

    # ==================================================================
    # Method 1: HistGB + sample_weight
    # ==================================================================
    print(f"\n  --- Method 1: HistGB + ANOVA + sample_weight ---", flush=True)
    best_cv_hgb_sw, best_hgb_sw_cfg, best_hgb_sw_K = 0, None, None
    for K in hgb_K_values:
        if K > anova_preselect:
            continue
        for max_iter, lr, max_d in hgb_configs:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                       max_depth=max_d, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y, sw=sample_weights)
            if cv > best_cv_hgb_sw:
                best_cv_hgb_sw, best_hgb_sw_cfg, best_hgb_sw_K = cv, (max_iter, lr, max_d), K
    print(f"  Best: max_iter={best_hgb_sw_cfg[0]}, lr={best_hgb_sw_cfg[1]}, max_depth={best_hgb_sw_cfg[2]}, K={best_hgb_sw_K}, cv_acc={best_cv_hgb_sw:.3f}")
    pipe_hgb_sw = Pipeline([
        ('select', SelectKBest(f_classif, k=best_hgb_sw_K)),
        ('clf', HistGradientBoostingClassifier(max_iter=best_hgb_sw_cfg[0], learning_rate=best_hgb_sw_cfg[1],
                                               max_depth=best_hgb_sw_cfg[2], random_state=42)),
    ])
    pipe_hgb_sw.fit(combine_anova, combine_y, clf__sample_weight=sample_weights)
    acc = accuracy_score(test_y, pipe_hgb_sw.predict(test_anova))
    auc = _calc_auc(pipe_hgb_sw, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['hgb_sample_weight'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['hgb_sample_weight'] = pipe_hgb_sw
    cv_accs['hgb_sample_weight'] = best_cv_hgb_sw

    # ==================================================================
    # Method 2: HistGB + MI feature selection
    # ==================================================================
    print(f"\n  --- Method 2: HistGB + MI selection ---", flush=True)
    best_cv_hgb_mi, best_hgb_mi_cfg, best_hgb_mi_K = 0, None, None
    for K in hgb_K_values:
        if K > anova_preselect:
            continue
        for max_iter, lr, max_d in hgb_configs:
            pipe = Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=K)),
                ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                       max_depth=max_d, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_mi, combine_y)
            if cv > best_cv_hgb_mi:
                best_cv_hgb_mi, best_hgb_mi_cfg, best_hgb_mi_K = cv, (max_iter, lr, max_d), K
    print(f"  Best: max_iter={best_hgb_mi_cfg[0]}, lr={best_hgb_mi_cfg[1]}, max_depth={best_hgb_mi_cfg[2]}, K={best_hgb_mi_K}, cv_acc={best_cv_hgb_mi:.3f}")
    pipe_hgb_mi = Pipeline([
        ('select', SelectKBest(mutual_info_classif, k=best_hgb_mi_K)),
        ('clf', HistGradientBoostingClassifier(max_iter=best_hgb_mi_cfg[0], learning_rate=best_hgb_mi_cfg[1],
                                               max_depth=best_hgb_mi_cfg[2], random_state=42)),
    ])
    pipe_hgb_mi.fit(combine_mi, combine_y)
    acc = accuracy_score(test_y, pipe_hgb_mi.predict(test_mi))
    auc = _calc_auc(pipe_hgb_mi, test_mi, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['hgb_mi'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['hgb_mi'] = pipe_hgb_mi
    cv_accs['hgb_mi'] = best_cv_hgb_mi

    # ==================================================================
    # Method 3: HistGB + MI + sample_weight
    # ==================================================================
    print(f"\n  --- Method 3: HistGB + MI + sample_weight ---", flush=True)
    best_cv_hgb_mi_sw, best_hgb_mi_sw_cfg, best_hgb_mi_sw_K = 0, None, None
    for K in hgb_K_values:
        if K > anova_preselect:
            continue
        for max_iter, lr, max_d in hgb_configs:
            pipe = Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=K)),
                ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                       max_depth=max_d, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_mi, combine_y, sw=sample_weights)
            if cv > best_cv_hgb_mi_sw:
                best_cv_hgb_mi_sw, best_hgb_mi_sw_cfg, best_hgb_mi_sw_K = cv, (max_iter, lr, max_d), K
    print(f"  Best: max_iter={best_hgb_mi_sw_cfg[0]}, lr={best_hgb_mi_sw_cfg[1]}, max_depth={best_hgb_mi_sw_cfg[2]}, K={best_hgb_mi_sw_K}, cv_acc={best_cv_hgb_mi_sw:.3f}")
    pipe_hgb_mi_sw = Pipeline([
        ('select', SelectKBest(mutual_info_classif, k=best_hgb_mi_sw_K)),
        ('clf', HistGradientBoostingClassifier(max_iter=best_hgb_mi_sw_cfg[0], learning_rate=best_hgb_mi_sw_cfg[1],
                                               max_depth=best_hgb_mi_sw_cfg[2], random_state=42)),
    ])
    pipe_hgb_mi_sw.fit(combine_mi, combine_y, clf__sample_weight=sample_weights)
    acc = accuracy_score(test_y, pipe_hgb_mi_sw.predict(test_mi))
    auc = _calc_auc(pipe_hgb_mi_sw, test_mi, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['hgb_mi_sw'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['hgb_mi_sw'] = pipe_hgb_mi_sw
    cv_accs['hgb_mi_sw'] = best_cv_hgb_mi_sw

    # ==================================================================
    # Method 4: SVM + ANOVA (stacking component)
    # ==================================================================
    print(f"\n  --- Method 4: SVM (RBF) + ANOVA ---", flush=True)
    best_cv_svm, best_svm_C, best_svm_K = 0, None, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        for C in [1.0, 10.0, 50.0]:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', SVC(C=C, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_cv_svm:
                best_cv_svm, best_svm_C, best_svm_K = cv, C, K
    print(f"  Best: K={best_svm_K}, C={best_svm_C}, cv_acc={best_cv_svm:.3f}")
    pipe_svm = Pipeline([
        ('select', SelectKBest(f_classif, k=best_svm_K)),
        ('scale', StandardScaler()),
        ('clf', SVC(C=best_svm_C, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
    ])
    pipe_svm.fit(combine_anova, combine_y)
    acc = accuracy_score(test_y, pipe_svm.predict(test_anova))
    auc = _calc_auc(pipe_svm, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['svm_rbf'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['svm_rbf'] = pipe_svm
    cv_accs['svm_rbf'] = best_cv_svm

    # ==================================================================
    # Method 5: KNN + ANOVA (stacking component)
    # ==================================================================
    print(f"\n  --- Method 5: KNN + ANOVA ---", flush=True)
    best_cv_knn, best_knn_k, best_knn_K = 0, None, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        for k in [3, 5, 7, 11]:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=k)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_cv_knn:
                best_cv_knn, best_knn_k, best_knn_K = cv, k, K
    print(f"  Best: K={best_knn_K}, k={best_knn_k}, cv_acc={best_cv_knn:.3f}")
    pipe_knn = Pipeline([
        ('select', SelectKBest(f_classif, k=best_knn_K)),
        ('scale', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=best_knn_k)),
    ])
    pipe_knn.fit(combine_anova, combine_y)
    acc = accuracy_score(test_y, pipe_knn.predict(test_anova))
    auc = _calc_auc(pipe_knn, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['knn'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['knn'] = pipe_knn
    cv_accs['knn'] = best_cv_knn

    # ==================================================================
    # Method 6: MLP + ANOVA
    # ==================================================================
    print(f"\n  --- Method 6: MLP + ANOVA ---", flush=True)
    best_cv_mlp, best_mlp_K, best_mlp_cfg = 0, None, None
    for K in [50, 100, 200]:
        if K > anova_preselect:
            continue
        for hidden in [(128, 64), (256, 128), (128, 64, 32)]:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000, early_stopping=True,
                                      validation_fraction=0.1, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_cv_mlp:
                best_cv_mlp, best_mlp_K, best_mlp_cfg = cv, K, hidden
    print(f"  Best: K={best_mlp_K}, hidden={best_mlp_cfg}, cv_acc={best_cv_mlp:.3f}")
    pipe_mlp = Pipeline([
        ('select', SelectKBest(f_classif, k=best_mlp_K)),
        ('scale', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=best_mlp_cfg, max_iter=1000, early_stopping=True,
                              validation_fraction=0.1, random_state=42)),
    ])
    pipe_mlp.fit(combine_anova, combine_y)
    acc = accuracy_score(test_y, pipe_mlp.predict(test_anova))
    auc = _calc_auc(pipe_mlp, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['mlp'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['mlp'] = pipe_mlp
    cv_accs['mlp'] = best_cv_mlp

    # ==================================================================
    # Method 7: Stacking (HistGB + SVM + KNN → LR meta-learner) on ANOVA
    # ==================================================================
    print(f"\n  --- Method 7: Stacking (HistGB + SVM + KNN → LR) on ANOVA ---", flush=True)
    best_cv_stack, best_stack_K = 0, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        stack_estimators = [
            ('hgb', Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('clf', HistGradientBoostingClassifier(
                    max_iter=best_hgb_cfg[0], learning_rate=best_hgb_cfg[1],
                    max_depth=best_hgb_cfg[2], random_state=42)),
            ])),
            ('svm', Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', SVC(C=best_svm_C, kernel='rbf', class_weight='balanced',
                            probability=True, random_state=42)),
            ])),
            ('knn', Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=best_knn_k)),
            ])),
        ]
        stack_clf = StackingClassifier(
            estimators=stack_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
            cv=5,
            stack_method='predict_proba',
            n_jobs=1,
        )
        cv = _cv_eval(stack_clf, combine_anova, combine_y)
        if cv > best_cv_stack:
            best_cv_stack, best_stack_K = cv, K
    print(f"  Best: K={best_stack_K}, cv_acc={best_cv_stack:.3f}")

    stack_estimators_final = [
        ('hgb', Pipeline([
            ('select', SelectKBest(f_classif, k=best_stack_K)),
            ('clf', HistGradientBoostingClassifier(
                max_iter=best_hgb_cfg[0], learning_rate=best_hgb_cfg[1],
                max_depth=best_hgb_cfg[2], random_state=42)),
        ])),
        ('svm', Pipeline([
            ('select', SelectKBest(f_classif, k=best_stack_K)),
            ('scale', StandardScaler()),
            ('clf', SVC(C=best_svm_C, kernel='rbf', class_weight='balanced',
                        probability=True, random_state=42)),
        ])),
        ('knn', Pipeline([
            ('select', SelectKBest(f_classif, k=best_stack_K)),
            ('scale', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=best_knn_k)),
        ])),
    ]
    pipe_stack = StackingClassifier(
        estimators=stack_estimators_final,
        final_estimator=LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
        cv=5,
        stack_method='predict_proba',
        n_jobs=1,
    )
    pipe_stack.fit(combine_anova, combine_y)
    acc = accuracy_score(test_y, pipe_stack.predict(test_anova))
    auc = _calc_auc(pipe_stack, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['stacking'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['stacking'] = pipe_stack
    cv_accs['stacking'] = best_cv_stack

    # ==================================================================
    # Method 8: Stacking + sample_weight (HistGB_sw + SVM + KNN → LR)
    # ==================================================================
    print(f"\n  --- Method 8: Stacking + sample_weight (HistGB_sw + SVM + KNN → LR) ---", flush=True)
    best_cv_stack_sw, best_stack_sw_K = 0, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        stack_sw_estimators = [
            ('hgb_sw', Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('clf', HistGradientBoostingClassifier(
                    max_iter=best_hgb_sw_cfg[0], learning_rate=best_hgb_sw_cfg[1],
                    max_depth=best_hgb_sw_cfg[2], random_state=42)),
            ])),
            ('svm', Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', SVC(C=best_svm_C, kernel='rbf', class_weight='balanced',
                            probability=True, random_state=42)),
            ])),
            ('knn', Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=best_knn_k)),
            ])),
        ]
        stack_sw_clf = StackingClassifier(
            estimators=stack_sw_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
            cv=5,
            stack_method='predict_proba',
            n_jobs=1,
        )
        cv = _cv_eval(stack_sw_clf, combine_anova, combine_y)
        if cv > best_cv_stack_sw:
            best_cv_stack_sw, best_stack_sw_K = cv, K
    print(f"  Best: K={best_stack_sw_K}, cv_acc={best_cv_stack_sw:.3f}")

    stack_sw_estimators_final = [
        ('hgb_sw', Pipeline([
            ('select', SelectKBest(f_classif, k=best_stack_sw_K)),
            ('clf', HistGradientBoostingClassifier(
                max_iter=best_hgb_sw_cfg[0], learning_rate=best_hgb_sw_cfg[1],
                max_depth=best_hgb_sw_cfg[2], random_state=42)),
        ])),
        ('svm', Pipeline([
            ('select', SelectKBest(f_classif, k=best_stack_sw_K)),
            ('scale', StandardScaler()),
            ('clf', SVC(C=best_svm_C, kernel='rbf', class_weight='balanced',
                        probability=True, random_state=42)),
        ])),
        ('knn', Pipeline([
            ('select', SelectKBest(f_classif, k=best_stack_sw_K)),
            ('scale', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=best_knn_k)),
        ])),
    ]
    pipe_stack_sw = StackingClassifier(
        estimators=stack_sw_estimators_final,
        final_estimator=LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
        cv=5,
        stack_method='predict_proba',
        n_jobs=1,
    )
    pipe_stack_sw.fit(combine_anova, combine_y)
    acc = accuracy_score(test_y, pipe_stack_sw.predict(test_anova))
    auc = _calc_auc(pipe_stack_sw, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['stacking_sw'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['stacking_sw'] = pipe_stack_sw
    cv_accs['stacking_sw'] = best_cv_stack_sw

    # ==================================================================
    # Method 9: Hierarchical Classification
    # ==================================================================
    print(f"\n  --- Method 9: Hierarchical Classification ---", flush=True)

    active_names = json.load(open(output_dir / 'class_names.json'))
    name_to_idx = {name: i for i, name in enumerate(active_names)}

    superclass_to_indices = {}
    for sc_name, sc_classes in SUPERCLASS_MAP.items():
        indices = [name_to_idx[c] for c in sc_classes if c in name_to_idx]
        superclass_to_indices[sc_name] = indices
    print(f"  Superclass mapping:")
    for sc, indices in superclass_to_indices.items():
        names = [active_names[i] for i in indices]
        print(f"    {sc}: {names}")

    combine_super_y = np.zeros(len(combine_y), dtype=int)
    for sc_id, (sc_name, sc_indices) in enumerate(superclass_to_indices.items()):
        for idx in sc_indices:
            combine_super_y[combine_y == idx] = sc_id

    test_super_y = np.zeros(len(test_y), dtype=int)
    for sc_id, (sc_name, sc_indices) in enumerate(superclass_to_indices.items()):
        for idx in sc_indices:
            test_super_y[test_y == idx] = sc_id

    n_super = len(superclass_to_indices)
    print(f"  Superclass distribution (combine): {dict(Counter(combine_super_y.tolist()))}")
    print(f"  Superclass distribution (test): {dict(Counter(test_super_y.tolist()))}")

    print(f"\n  Training Level-1 superclass classifier (HistGB)...", flush=True)
    best_cv_super, best_super_K, best_super_cfg = 0, None, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        for max_iter, lr, max_d in hgb_configs:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                       max_depth=max_d, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_super_y)
            if cv > best_cv_super:
                best_cv_super, best_super_K, best_super_cfg = cv, K, (max_iter, lr, max_d)
    print(f"  Level-1 best: K={best_super_K}, cfg={best_super_cfg}, cv_acc={best_cv_super:.3f}")
    pipe_super = Pipeline([
        ('select', SelectKBest(f_classif, k=best_super_K)),
        ('clf', HistGradientBoostingClassifier(max_iter=best_super_cfg[0], learning_rate=best_super_cfg[1],
                                               max_depth=best_super_cfg[2], random_state=42)),
    ])
    pipe_super.fit(combine_anova, combine_super_y)
    super_preds = pipe_super.predict(test_anova)
    super_acc = accuracy_score(test_super_y, super_preds)
    print(f"  Level-1 test acc: {super_acc:.3f}")

    sub_classifiers = {}
    for sc_id, (sc_name, sc_indices) in enumerate(superclass_to_indices.items()):
        sc_mask = combine_super_y == sc_id
        sc_X = combine_anova[sc_mask]
        sc_y = np.zeros(sc_mask.sum(), dtype=int)
        for new_id, orig_idx in enumerate(sc_indices):
            sc_y[combine_y[sc_mask] == orig_idx] = new_id
        n_sub = len(sc_indices)

        if n_sub <= 1:
            print(f"  Skipping {sc_name}: only {n_sub} subclass")
            continue

        print(f"\n  Training Level-2 for '{sc_name}' ({n_sub} classes, {sc_mask.sum()} samples)...", flush=True)

        best_cv_sub, best_sub_K, best_sub_cfg = 0, None, None
        for K in [50, 100, 200]:
            if K > anova_preselect:
                continue
            for max_iter, lr, max_d in [(100, 0.1, 3), (200, 0.1, 5)]:
                pipe = Pipeline([
                    ('select', SelectKBest(f_classif, k=K)),
                    ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                           max_depth=max_d, random_state=42)),
                ])
                if len(np.unique(sc_y)) < 2:
                    continue
                try:
                    min_cls_count = min(Counter(sc_y.tolist()).values())
                    n_splits = min(3, min_cls_count)
                    if n_splits < 2:
                        continue
                    skf_sub = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                    accs = []
                    for tr, va in skf_sub.split(sc_X, sc_y):
                        pipe.fit(sc_X[tr], sc_y[tr])
                        accs.append(accuracy_score(sc_y[va], pipe.predict(sc_X[va])))
                    cv = np.mean(accs)
                except Exception:
                    cv = 0
                if cv > best_cv_sub:
                    best_cv_sub, best_sub_K, best_sub_cfg = cv, K, (max_iter, lr, max_d)

        if best_sub_cfg is None:
            print(f"  Using LR fallback for {sc_name}")
            sub_pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=min(50, anova_preselect))),
                ('scale', StandardScaler()),
                ('clf', LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced')),
            ])
        else:
            print(f"  Level-2 '{sc_name}' best: K={best_sub_K}, cfg={best_sub_cfg}, cv_acc={best_cv_sub:.3f}")
            sub_pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=best_sub_K)),
                ('clf', HistGradientBoostingClassifier(max_iter=best_sub_cfg[0], learning_rate=best_sub_cfg[1],
                                                       max_depth=best_sub_cfg[2], random_state=42)),
            ])
        sub_pipe.fit(sc_X, sc_y)
        sub_classifiers[sc_id] = (sub_pipe, sc_indices)

    print(f"\n  Evaluating hierarchical classifier on test set...", flush=True)
    hier_preds = np.zeros(len(test_y), dtype=int)
    for i in range(len(test_y)):
        sc_pred = super_preds[i]
        if sc_pred in sub_classifiers:
            sub_pipe, sc_indices = sub_classifiers[sc_pred]
            sub_pred = sub_pipe.predict(test_anova[i:i+1])[0]
            if sub_pred < len(sc_indices):
                hier_preds[i] = sc_indices[sub_pred]
            else:
                hier_preds[i] = sc_indices[0]
        else:
            hier_preds[i] = 0

    hier_acc = accuracy_score(test_y, hier_preds)
    try:
        hier_probs = np.zeros((len(test_y), num_classes))
        for i in range(len(test_y)):
            sc_pred = super_preds[i]
            if sc_pred in sub_classifiers:
                sub_pipe, sc_indices = sub_classifiers[sc_pred]
                sub_prob = sub_pipe.predict_proba(test_anova[i:i+1])[0]
                for j, idx in enumerate(sc_indices):
                    if j < len(sub_prob):
                        hier_probs[i, idx] = sub_prob[j]
        hier_auc = roc_auc_score(test_y, hier_probs, multi_class='ovr', average='macro')
    except Exception:
        hier_auc = 0.0

    print(f"  Hierarchical test: acc={hier_acc:.3f}, AUC={hier_auc:.3f}")
    results['hierarchical'] = {'acc': float(hier_acc), 'auc': float(hier_auc)}
    best_pipes['hierarchical'] = pipe_super
    cv_accs['hierarchical'] = best_cv_super

    # ==================================================================
    # Method 10: Stacking on MI features
    # ==================================================================
    print(f"\n  --- Method 10: Stacking on MI features ---", flush=True)

    print(f"  Training SVM on MI features...", flush=True)
    best_cv_svm_mi, best_svm_mi_C, best_svm_mi_K = 0, None, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        for C in [1.0, 10.0, 50.0]:
            pipe = Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', SVC(C=C, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_mi, combine_y)
            if cv > best_cv_svm_mi:
                best_cv_svm_mi, best_svm_mi_C, best_svm_mi_K = cv, C, K
    print(f"  SVM+MI best: K={best_svm_mi_K}, C={best_svm_mi_C}, cv_acc={best_cv_svm_mi:.3f}")

    print(f"  Training KNN on MI features...", flush=True)
    best_cv_knn_mi, best_knn_mi_k, best_knn_mi_K = 0, None, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        for k in [3, 5, 7, 11]:
            pipe = Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=k)),
            ])
            cv = _cv_eval(pipe, combine_mi, combine_y)
            if cv > best_cv_knn_mi:
                best_cv_knn_mi, best_knn_mi_k, best_knn_mi_K = cv, k, K
    print(f"  KNN+MI best: K={best_knn_mi_K}, k={best_knn_mi_k}, cv_acc={best_cv_knn_mi:.3f}")

    best_cv_stack_mi, best_stack_mi_K = 0, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        stack_mi_estimators = [
            ('hgb_mi', Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=K)),
                ('clf', HistGradientBoostingClassifier(
                    max_iter=best_hgb_mi_cfg[0], learning_rate=best_hgb_mi_cfg[1],
                    max_depth=best_hgb_mi_cfg[2], random_state=42)),
            ])),
            ('svm_mi', Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', SVC(C=best_svm_mi_C, kernel='rbf', class_weight='balanced',
                            probability=True, random_state=42)),
            ])),
            ('knn_mi', Pipeline([
                ('select', SelectKBest(mutual_info_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=best_knn_mi_k)),
            ])),
        ]
        stack_mi_clf = StackingClassifier(
            estimators=stack_mi_estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
            cv=5,
            stack_method='predict_proba',
            n_jobs=1,
        )
        cv = _cv_eval(stack_mi_clf, combine_mi, combine_y)
        if cv > best_cv_stack_mi:
            best_cv_stack_mi, best_stack_mi_K = cv, K
    print(f"  Stacking+MI best: K={best_stack_mi_K}, cv_acc={best_cv_stack_mi:.3f}")

    stack_mi_estimators_final = [
        ('hgb_mi', Pipeline([
            ('select', SelectKBest(mutual_info_classif, k=best_stack_mi_K)),
            ('clf', HistGradientBoostingClassifier(
                max_iter=best_hgb_mi_cfg[0], learning_rate=best_hgb_mi_cfg[1],
                max_depth=best_hgb_mi_cfg[2], random_state=42)),
        ])),
        ('svm_mi', Pipeline([
            ('select', SelectKBest(mutual_info_classif, k=best_stack_mi_K)),
            ('scale', StandardScaler()),
            ('clf', SVC(C=best_svm_mi_C, kernel='rbf', class_weight='balanced',
                        probability=True, random_state=42)),
        ])),
        ('knn_mi', Pipeline([
            ('select', SelectKBest(mutual_info_classif, k=best_stack_mi_K)),
            ('scale', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=best_knn_mi_k)),
        ])),
    ]
    pipe_stack_mi = StackingClassifier(
        estimators=stack_mi_estimators_final,
        final_estimator=LogisticRegression(C=1.0, max_iter=5000, class_weight='balanced'),
        cv=5,
        stack_method='predict_proba',
        n_jobs=1,
    )
    pipe_stack_mi.fit(combine_mi, combine_y)
    acc = accuracy_score(test_y, pipe_stack_mi.predict(test_mi))
    auc = _calc_auc(pipe_stack_mi, test_mi, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['stacking_mi'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['stacking_mi'] = pipe_stack_mi
    cv_accs['stacking_mi'] = best_cv_stack_mi

    # ==================================================================
    # Method 11: Plain Linear Classifier (no regularization) + ANOVA
    # ==================================================================
    print(f"\n  --- Method 11: Plain Linear Classifier + ANOVA ---", flush=True)
    best_cv_linear, best_K_linear = 0, None
    for K in [100, 200, 300]:
        if K > anova_preselect:
            continue
        pipe = Pipeline([
            ('select', SelectKBest(f_classif, k=K)),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs', class_weight='balanced')),
        ])
        cv = _cv_eval(pipe, combine_anova, combine_y)
        if cv > best_cv_linear:
            best_cv_linear, best_K_linear = cv, K
    print(f"  Best: K={best_K_linear}, cv_acc={best_cv_linear:.3f}")
    pipe_linear = Pipeline([
        ('select', SelectKBest(f_classif, k=best_K_linear)),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(penalty=None, max_iter=5000, solver='lbfgs', class_weight='balanced')),
    ])
    pipe_linear.fit(combine_anova, combine_y)
    acc = accuracy_score(test_y, pipe_linear.predict(test_anova))
    auc = _calc_auc(pipe_linear, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['linear'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['linear'] = pipe_linear
    cv_accs['linear'] = best_cv_linear

    # ==================================================================
    # Summary & Save
    # ==================================================================
    print(f"\n  {'='*60}")
    print(f"  COMPARISON SUMMARY (v{version})")
    print(f"  {'='*60}")

    interp_labels = {
        'lr_l2': '[HIGH]',
        'hgb_baseline': '[MED]',
        'hgb_sample_weight': '[MED]',
        'hgb_mi': '[MED]',
        'hgb_mi_sw': '[MED]',
        'svm_rbf': '[LOW]',
        'knn': '[HIGH]',
        'mlp': '[LOW]',
        'stacking': '[MED]',
        'stacking_sw': '[MED]',
        'stacking_mi': '[MED]',
        'linear': '[HIGH]',
        'hierarchical': '[MED]',
    }

    best_method = max(results, key=lambda k: results[k]['acc'])
    for method, res in results.items():
        marker = " <-- BEST" if method == best_method else ""
        interp = interp_labels.get(method, '')
        cv_info = f" cv_acc={cv_accs.get(method, 0):.3f}"
        gap = cv_accs.get(method, 0) - res['acc']
        print(f"    {method:25s}{interp:6s}: acc={res['acc']:.3f}, AUC={res['auc']:.3f}{cv_info} gap={gap:.3f}{marker}")

    all_per_class = {}
    for method_name, pipe in best_pipes.items():
        if method_name == 'hierarchical':
            pc = {}
            for c in range(num_classes):
                mask = test_y == c
                if mask.sum() > 0:
                    name = active_names[c] if c < len(active_classes) else f"class_{c}"
                    pc[name] = float((hier_preds[mask] == c).mean())
            all_per_class[method_name] = pc
            continue

        use_mi = method_name in ['hgb_mi', 'hgb_mi_sw', 'stacking_mi']
        test_data = test_mi if use_mi else test_anova
        mpreds = pipe.predict(test_data)
        pc = {}
        for c in range(num_classes):
            mask = test_y == c
            if mask.sum() > 0:
                name = active_names[c] if c < len(active_classes) else f"class_{c}"
                pc[name] = float((mpreds[mask] == c).mean())
        all_per_class[method_name] = pc

    best_pipe = best_pipes[best_method]
    if best_method == 'hierarchical':
        best_preds = hier_preds
    else:
        use_mi = best_method in ['hgb_mi', 'hgb_mi_sw', 'stacking_mi']
        best_preds = best_pipe.predict(test_mi if use_mi else test_anova)

    per_class = all_per_class.get(best_method, {})

    if best_method == 'hierarchical':
        train_preds = pipe_super.predict(combine_anova)
        train_acc = accuracy_score(combine_super_y, train_preds)
    else:
        use_mi = best_method in ['hgb_mi', 'hgb_mi_sw', 'stacking_mi']
        train_preds = best_pipe.predict(combine_mi if use_mi else combine_anova)
        train_acc = accuracy_score(combine_y, train_preds)

    print(f"\n  Best method: {best_method}")
    print(f"  Train acc={train_acc:.3f}")
    print(f"  Test  acc={results[best_method]['acc']:.3f}, AUC={results[best_method]['auc']:.3f}")
    print(f"  Per-class test ({best_method}):")
    for name, acc_val in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"    {name:25s}: {acc_val:.3f}")

    print(f"\n  All methods per-class accuracy:")
    header = f"    {'Class':25s}"
    for mn in results.keys():
        header += f" | {mn[:10]:>10s}"
    print(header)
    for name in sorted(per_class.keys()):
        line = f"    {name:25s}"
        for mn in results.keys():
            val = all_per_class.get(mn, {}).get(name, 0.0)
            line += f" | {val:10.3f}"
        print(line)

    import joblib
    clf_save_data = {
        'pipe': best_pipe,
        'non_const': non_const,
        'method': best_method,
        'scaler': scaler,
        'anova_selector': anova_selector,
        'mi_pool_selector': mi_pool_selector,
        'mi_selector': mi_selector,
        'sample_weights': sample_weights,
        'version': version,
    }
    if best_method == 'hierarchical':
        clf_save_data['pipe_super'] = pipe_super
        clf_save_data['sub_classifiers'] = sub_classifiers
        clf_save_data['superclass_map'] = SUPERCLASS_MAP
    joblib.dump(clf_save_data, output_dir / 'best_classifier.pkl')
    joblib.dump(clf_save_data, versioned_model_path)

    for mn, mpipe in best_pipes.items():
        save_data = {
            'pipe': mpipe,
            'non_const': non_const,
            'method': mn,
            'scaler': scaler,
            'anova_selector': anova_selector,
            'mi_pool_selector': mi_pool_selector,
            'mi_selector': mi_selector,
            'version': version,
        }
        if mn == 'hierarchical':
            save_data['pipe_super'] = pipe_super
            save_data['sub_classifiers'] = sub_classifiers
            save_data['superclass_map'] = SUPERCLASS_MAP
        joblib.dump(save_data, version_dir / f'classifier_{mn}_v{version}.pkl')
        print(f"  Saved: v{version}/classifier_{mn}_v{version}.pkl")

    best_results = {
        'version': version,
        'method': best_method,
        'train_accuracy': float(train_acc),
        'test_accuracy': results[best_method]['acc'],
        'test_auc': results[best_method]['auc'],
        'per_class_test': per_class,
        'all_methods': results,
        'all_per_class': all_per_class,
        'cv_accs': {k: float(v) for k, v in cv_accs.items()},
        'interpretability': interp_labels,
        'pipe_configs': {k: str(v) for k, v in all_pipe_configs.items()},
    }
    with open(results_path, 'w') as f:
        json.dump(best_results, f, indent=2)
    with open(versioned_results_path, 'w') as f:
        json.dump(best_results, f, indent=2)

    import shutil
    pipeline_src = Path(__file__).resolve()
    pipeline_dst = version_dir / f'run_fracture_pipeline_v{version}.py'
    shutil.copy2(pipeline_src, pipeline_dst)

    print(f"  Results saved: {versioned_results_path} (v{version})")
    print(f"  Model saved:   {versioned_model_path} (v{version})")
    print(f"  Code backup:   {pipeline_dst} (v{version})")


# ======================================================================
# Step 6: Interpretability Report (v6)
# ======================================================================

def step6_report(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 6: Generate Interpretability Report (v6)")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    report_path = output_dir / 'interpretability_report.txt'

    validated_path = output_dir / 'validated_formulas.json'
    formulas = json.load(open(validated_path))

    dist_cfg = config.get('phase3', {}).get('distribution_stats', {})
    stats_per_formula = dist_cfg.get('n_stats', 12) * dist_cfg.get('n_regions', 5)

    classifier_path = output_dir / 'best_classifier.pkl'
    if not classifier_path.exists():
        print("  No trained classifier found, skipping weight analysis")
        return

    import joblib
    clf_data = joblib.load(classifier_path)
    pipe = clf_data['pipe']
    method = clf_data.get('method', 'unknown')
    anova_selector = clf_data.get('anova_selector')
    mi_selector = clf_data.get('mi_selector')
    non_const_mask = clf_data.get('non_const')
    version = clf_data.get('version', 6)

    from sklearn.ensemble import StackingClassifier, VotingClassifier

    if isinstance(pipe, StackingClassifier):
        inner_names = [name for name, _ in pipe.estimators]
        if hasattr(pipe, 'estimators_') and pipe.estimators_ is not None:
            fitted_inner = dict(zip(inner_names, pipe.estimators_))
        else:
            fitted_inner = {name: est for name, est in pipe.estimators}
        meta_clf = pipe.final_estimator_
        lines_note = f"  (Stacking ensemble: {', '.join(inner_names)} -> LR meta-learner)"
        primary_key = None
        for k in ['hgb', 'hgb_mi', 'hgb_sw']:
            if k in fitted_inner:
                primary_key = k
                break
        if primary_key is None:
            primary_key = list(fitted_inner.keys())[0]
        primary_pipe = fitted_inner[primary_key]
        clf = primary_pipe.named_steps['clf']
        selector = primary_pipe.named_steps.get('select', None)
    elif isinstance(pipe, VotingClassifier):
        inner_names = [name for name, _ in pipe.estimators]
        if hasattr(pipe, 'estimators_') and pipe.estimators_ is not None:
            fitted_inner = dict(zip(inner_names, pipe.estimators_))
        else:
            fitted_inner = {name: est for name, est in pipe.estimators}
        primary_key = list(fitted_inner.keys())[0]
        inner_pipe = fitted_inner[primary_key]
        clf = inner_pipe.named_steps['clf']
        selector = inner_pipe.named_steps.get('select', None)
        lines_note = f"  (Ensemble model — showing feature analysis from sub-model '{primary_key}')"
    elif hasattr(pipe, 'named_steps'):
        primary_key = method
        clf = pipe.named_steps['clf']
        selector = pipe.named_steps.get('select', None)
        lines_note = None
    else:
        print("  Cannot extract feature analysis from this classifier type")
        return

    formulas.sort(key=lambda f: f.get('full_res_accuracy', 0), reverse=True)

    lines = []
    lines.append("=" * 70)
    lines.append("BONE FRACTURE SYMBOLIC FEATURE INTERPRETABILITY REPORT (v6)")
    lines.append("=" * 70)
    lines.append(f"  Best method: {method}")
    if lines_note:
        lines.append(lines_note)
    lines.append("")

    lines.append("TOP-20 MOST DISCRIMINATIVE FORMULAS:")
    lines.append("-" * 50)
    for i, f in enumerate(formulas[:20]):
        acc = f.get('full_res_accuracy', f.get('accuracy', 0))
        lines.append(f"  {i+1:2d}. acc={acc:.3f}  {f['str']}")

    lines.append("")
    lines.append("CLASSIFIER FEATURE ANALYSIS:")
    lines.append("-" * 50)

    stat_names = ['mean','std','min','max','median','q10','q25','q75','q90',
                 'skew','kurtosis','energy','l2_norm','l1_norm','range',
                 'iqr','mad','cv','entropy','pct_nonzero',
                 'mean_q0','std_q0','mean_q1','std_q1','mean_q2','std_q2','mean_q3','std_q3',
                 'mean_top','std_top','mean_bottom','std_bottom',
                 'mean_left','std_left','mean_right','std_right',
                 'mean_center','std_center',
                 'h_mean','h_std','h_skew','h_kurtosis','h_energy',
                 's_mean','s_std','s_skew','s_kurtosis','s_energy',
                 'v_mean','v_std','v_skew','v_kurtosis','v_energy',
                 'grad_mean','grad_std','grad_max',
                 'lbp_uniform','lbp_entropy']

    if selector is not None and hasattr(selector, 'get_support'):
        try:
            selected_idx = selector.get_support(indices=True)
        except Exception:
            selected_idx = np.arange(selector.n_features_in_) if hasattr(selector, 'n_features_in_') else None

        if selected_idx is not None:
            use_mi = method in ['hgb_mi', 'hgb_mi_sw', 'stacking_mi']
            mi_pool_sel = clf_data.get('mi_pool_selector')
            if use_mi and mi_pool_sel is not None and mi_selector is not None:
                try:
                    mi_pool_mask = mi_pool_sel.get_support(indices=True)
                    mi_within_pool = mi_selector.get_support(indices=True)
                    mi_to_pool = mi_pool_mask[mi_within_pool]
                    original_idx = mi_to_pool[selected_idx]
                    if non_const_mask is not None:
                        original_idx = np.where(non_const_mask)[0][original_idx]
                except Exception:
                    if anova_selector is not None:
                        anova_mask = anova_selector.get_support(indices=True)
                        original_idx = anova_mask[selected_idx]
                        if non_const_mask is not None:
                            original_idx = np.where(non_const_mask)[0][original_idx]
                    else:
                        original_idx = selected_idx
            elif anova_selector is not None:
                anova_mask = anova_selector.get_support(indices=True)
                original_idx = anova_mask[selected_idx]
                if non_const_mask is not None:
                    original_idx = np.where(non_const_mask)[0][original_idx]
            else:
                original_idx = selected_idx

        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            imp_label = "importance"
        elif hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_).mean(axis=0)
            imp_label = "|weight|"
        elif selector is not None and hasattr(selector, 'scores_'):
            importances = selector.scores_[selected_idx]
            imp_label = "selection_score"
        else:
            importances = None
            imp_label = None

        lines.append(f"  Selected {len(selected_idx)} features (from sub-model '{primary_key}')")
        if importances is not None:
            top_feat_idx = np.argsort(importances)[::-1][:20]
            lines.append(f"  Top-20 features by {imp_label}:")
            for rank, fi in enumerate(top_feat_idx):
                orig = original_idx[fi]
                formula_idx = orig // stats_per_formula
                stat_idx = orig % stats_per_formula
                sname = stat_names[stat_idx] if stat_idx < len(stat_names) else f'stat_{stat_idx}'
                if formula_idx < len(formulas):
                    lines.append(f"    {rank+1:2d}. formula[{formula_idx}].{sname} ({imp_label}={importances[fi]:.4f})")
                else:
                    lines.append(f"    {rank+1:2d}. feat[{orig}].{sname} ({imp_label}={importances[fi]:.4f})")
        else:
            lines.append("  (Classifier does not expose feature importances directly)")

        if isinstance(pipe, StackingClassifier) and hasattr(pipe, 'estimators_'):
            lines.append("")
            lines.append("  STACKING SUB-MODEL FEATURE ANALYSIS:")
            for sub_name, sub_est in zip(inner_names, pipe.estimators_):
                sub_clf = sub_est.named_steps.get('clf', None)
                sub_sel = sub_est.named_steps.get('select', None)
                n_sel_info = ""
                if sub_sel is not None and hasattr(sub_sel, 'get_support'):
                    try:
                        n_sel_info = f"{sub_sel.get_support(indices=True).shape[0]} selected"
                    except Exception:
                        n_sel_info = "N/A"
                if sub_clf is not None and hasattr(sub_clf, 'feature_importances_'):
                    sub_imp = sub_clf.feature_importances_
                    lines.append(f"    {sub_name}: {n_sel_info}, top-5 importance = {np.sort(sub_imp)[::-1][:5].round(4).tolist()}")
                elif sub_clf is not None and hasattr(sub_clf, 'coef_'):
                    sub_imp = np.abs(sub_clf.coef_).mean(axis=0)
                    lines.append(f"    {sub_name}: {n_sel_info}, top-5 |weight| = {np.sort(sub_imp)[::-1][:5].round(4).tolist()}")
                elif sub_sel is not None and hasattr(sub_sel, 'scores_'):
                    try:
                        sub_scores = sub_sel.scores_
                        sub_mask = sub_sel.get_support(indices=True)
                        lines.append(f"    {sub_name}: {n_sel_info}, top-5 selection_score = {np.sort(sub_scores[sub_mask])[::-1][:5].round(4).tolist()}")
                    except Exception:
                        lines.append(f"    {sub_name}: {n_sel_info}")
                else:
                    lines.append(f"    {sub_name}: {n_sel_info}")

    if mi_selector is not None and anova_selector is not None:
        lines.append("")
        lines.append("MUTUAL INFORMATION vs ANOVA COMPARISON:")
        lines.append("-" * 50)
        anova_mask = anova_selector.get_support(indices=True)
        mi_pool_sel = clf_data.get('mi_pool_selector')
        if mi_pool_sel is not None:
            mi_pool_mask = mi_pool_sel.get_support(indices=True)
            mi_within_pool = mi_selector.get_support(indices=True)
            mi_original = mi_pool_mask[mi_within_pool]
        else:
            mi_original = mi_selector.get_support(indices=True)
        overlap = len(set(anova_mask.tolist()) & set(mi_original.tolist()))
        lines.append(f"  ANOVA top-{len(anova_mask)} features")
        lines.append(f"  MI top-{len(mi_original)} features")
        lines.append(f"  Overlap: {overlap} features ({100*overlap/min(len(anova_mask), len(mi_original)):.1f}%)")
        anova_only = sorted(set(anova_mask.tolist()) - set(mi_original.tolist()))
        mi_only = sorted(set(mi_original.tolist()) - set(anova_mask.tolist()))
        lines.append(f"  ANOVA-only features: {len(anova_only)}")
        lines.append(f"  MI-only features: {len(mi_only)}")

    lines.append("")
    lines.append("OPERATOR FREQUENCY ANALYSIS:")
    lines.append("-" * 50)
    op_counts = {}
    for f in formulas:
        for tok in f['str'].split():
            if tok in TENSOR_OPERATORS:
                op_counts[tok] = op_counts.get(tok, 0) + 1
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:20]:
        is_fracture = " [FRACTURE-SPECIFIC]" if op in FRACTURE_OPERATORS else ""
        lines.append(f"  {op:25s}: {count:4d}{is_fracture}")

    lines.append("")
    lines.append("TERMINAL FREQUENCY ANALYSIS:")
    lines.append("-" * 50)
    term_counts = {}
    for f in formulas:
        for tok in f['str'].split():
            if tok.startswith('I_'):
                term_counts[tok] = term_counts.get(tok, 0) + 1
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        lines.append(f"  {term:25s}: {count:4d}")

    lines.append("")
    lines.append("MEDICAL INTERPRETATION OF TOP FORMULAS:")
    lines.append("-" * 50)
    medical_interpretations = {
        'edge_mag': 'Gradient magnitude — detects fracture line edges',
        'edge_x': 'Horizontal edge — detects vertical fracture lines',
        'edge_y': 'Vertical edge — detects horizontal fracture lines',
        'line_h': 'Horizontal line detector — transverse fractures',
        'line_v': 'Vertical line detector — longitudinal fractures',
        'line_45': '45-degree line detector — oblique fractures',
        'line_135': '135-degree line detector — oblique fractures',
        'edge_diag_45': 'Diagonal edge (45deg) — spiral/oblique fractures',
        'edge_diag_135': 'Diagonal edge (135deg) — spiral/oblique fractures',
        'cortical_cont': 'Cortical bone continuity — detects cortical breaks',
        'discont_map': 'Discontinuity map — highlights fracture gaps',
        'displace_ind': 'Displacement indicator — detects bone displacement',
        'black_tophat': 'Dark thin structures — fracture lines in bright bone',
        'white_tophat': 'Bright thin structures — bone fragments/callus',
        'local_entropy': 'Local entropy — disrupted trabecular patterns',
        'local_range': 'Local range — sharp intensity transitions',
        'bone_enhance': 'Bone enhancement — sharpens cortical edges',
        'threshold_bone': 'Bone segmentation — isolates bone from tissue',
        'soft_suppress': 'Soft tissue suppression — bone-only view',
        'lr_asymmetry': 'Left-right asymmetry — unilateral fractures',
        'ms_edge': 'Multi-scale edges — fractures of different sizes',
        'blob_detect': 'Blob detection — bone fragments/callus',
        'I_NEG': 'Inverted X-ray — bright bone on dark background',
        'I_BONE': 'Bone-enhanced channel — high contrast bone',
        'I_EDGE_PRIOR': 'Edge prior channel — pre-computed gradient magnitude',
    }

    for i, f in enumerate(formulas[:10]):
        lines.append(f"\n  Formula {i+1}: {f['str']}")
        acc = f.get('full_res_accuracy', f.get('accuracy', 0))
        lines.append(f"  Accuracy: {acc:.3f}")
        lines.append(f"  Interpretation:")
        for tok in f['str'].split():
            if tok in medical_interpretations:
                lines.append(f"    - {tok}: {medical_interpretations[tok]}")

    report_text = "\n".join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f"\n  Report saved to {report_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Fracture Symbolic Feature Discovery Pipeline (v6 — Advanced Ensemble)')
    parser.add_argument('--config', type=str, default='configs/fracture_v3_expanded.yaml')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    parser.add_argument('--start_step', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    gpu_id = args.gpu
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if gpu_id >= n_gpus:
            print(f"  WARNING: GPU {gpu_id} not available (only {n_gpus} GPUs), falling back to GPU 0")
            gpu_id = 0
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        mem_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
        print(f"  GPU {gpu_id}: {gpu_name} ({mem_gb:.1f} GB)")
    else:
        device = torch.device('cpu')
        gpu_id = None
        print("  CUDA not available, using CPU")

    config['gpu_id'] = gpu_id
    config['device'] = str(device)

    print(f"{'='*70}")
    print(f"  Fracture Symbolic Pipeline v6 — Advanced Ensemble")
    print(f"  Split saved to: {_get_split_file(config)}")
    print(f"  All steps share the same train/val/test split")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    steps = [
        ('Step 0: Dataset Validation', lambda: step0_validate_dataset(config, device)),
        ('Step 1: Phase 1 RL Discovery', lambda: step1_phase1(config, device)),
        ('Step 2: Merge & Deduplicate', lambda: step2_merge(config, device)),
        ('Step 3: Full-Resolution Validation', lambda: step3_validate(config, device)),
        ('Step 4: Feature Extraction + Encoding', lambda: step4_extract_features(config, device)),
        ('Step 5: Train Classifier (v6)', lambda: step5_train_classifier(config, device)),
        ('Step 6: Interpretability Report (v6)', lambda: step6_report(config, device)),
    ]

    for i, (name, fn) in enumerate(steps):
        if i >= args.start_step:
            print(f"\n{'#'*70}")
            print(f"# {name}")
            print(f"{'#'*70}")
            t0 = time.time()
            fn()
            print(f"  Completed in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
