#!/usr/bin/env python3
"""
Fracture Symbolic Feature Discovery Pipeline (v2 — One-Time Stratified Split)

v2 Changes from v1:
  - 合并所有原始数据后做一次分层划分（70/15/15），保存到 split_indices.npz
  - 所有步骤（RL搜索、特征提取、分类器）共享同一份划分
  - RL 搜索只看 train split → 无数据泄露
  - 测试集类别丰富（每类至少 2 张）→ 评估更可靠
  - Step 5 不再重新划分，直接使用 features.npz 中的 train/val/test

Pipeline Steps:
  Step 0: Dataset validation & statistics (also generates split_indices.npz)
  Step 1: Phase 1 — RL formula discovery (4 banks, low resolution 128x128)
  Step 2: Merge & deduplicate formulas across banks
  Step 3: Full-resolution validation (640x640)
  Step 4: Feature extraction + encoding (distribution stats + Fisher Vector)
  Step 5: Train classifier (using shared split from features.npz)
  Step 6: Evaluate on test set + generate interpretability report

Usage:
    python experiments/run_fracture_pipeline.py --config configs/fracture_v1.yaml
    python experiments/run_fracture_pipeline.py --config configs/fracture_v1.yaml --start_step 3
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


# ======================================================================
# Shared helpers
# ======================================================================

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
# Step 0: Dataset Validation
# ======================================================================

def step0_validate_dataset(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 0: Dataset Validation & Statistics")
    print(f"{'='*70}")

    data_dir = config['dataset_options']['data_dir']
    dm = _make_dm(
        config,
        resolution=config['dataset_options'].get('resolution_quick', 128),
        batch_size=32, num_workers=2,
    )
    dm.setup()

    from collections import Counter
    cnt = Counter()
    for i in range(len(dm.train_dataset)):
        _, lbl = dm.train_dataset[i]
        cnt[int(lbl)] += 1
    dist = {k: v for k, v in sorted(cnt.items())}
    print(f"\n  Class distribution (train):")
    for cls_id in sorted(dist.keys()):
        print(f"    {FRACTURE_NAMES[dm._active_classes[cls_id]]:25s}: {dist[cls_id]:4d} images")

    total = sum(dist.values())
    print(f"\n  Total boxes: {total}")
    print(f"  Imbalance ratio: {max(dist.values()) / max(min(dist.values()), 1):.1f}:1")

    loader = dm.get_train_loader()
    images, labels = next(iter(loader))
    data_batch = build_fracture_data_batch(images, device)
    print(f"\n  Terminal channels:")
    for name, tensor in data_batch.items():
        print(f"    {name:20s}: shape={tensor.shape}, "
              f"range=[{tensor.min().item():.3f}, {tensor.max().item():.3f}]")

    print(f"\n  Fracture operators: {len(FRACTURE_OPERATORS)}")
    print(f"  Total operators: {len(TENSOR_OPERATORS)}")
    print(f"  Root operators: {len(ROOT_OPERATORS)}")

    out_dir = Path(config['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {FRACTURE_NAMES[dm._active_classes[k]]: v for k, v in dist.items()}
    with open(out_dir / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved dataset stats to {out_dir / 'dataset_stats.json'}")


# ======================================================================
# Step 1: Phase 1 — RL Formula Discovery
# ======================================================================

def step1_phase1(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 1: Phase 1 — RL Formula Discovery")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir']) / 'phase1'
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / 'phase1_meta.json'
    if meta_path.exists():
        meta = json.load(open(meta_path))
        print(f"  Already done: {meta.get('total_formulas', '?')} formulas")
        return

    data_dir = config['dataset_options']['data_dir']
    resolution = config['dataset_options'].get('resolution_quick', 128)

    dm = _make_dm(
        config, resolution=resolution,
        batch_size=config['training']['batch_size'],
        num_workers=4, augment=True,
    )
    dm.setup()
    train_loader = dm.get_train_loader()

    multi_bank_cfg = config.get('multi_bank', {})
    num_banks = multi_bank_cfg.get('num_banks', 4) if multi_bank_cfg.get('enabled', False) else 1
    bank_configs = multi_bank_cfg.get('bank_configs', [])

    total_formulas = 0

    def _find_latest_checkpoint(ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        if not ckpt_dir.exists():
            return None, 0
        ckpts = list(ckpt_dir.glob('checkpoint_iter_*.pth'))
        if not ckpts:
            return None, 0
        ckpts.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest = ckpts[-1]
        iter_num = int(latest.stem.split('_')[-1])
        return str(latest), iter_num

    for bank_id in range(num_banks):
        bank_dir = output_dir / f'bank_{bank_id}'
        bank_dir.mkdir(parents=True, exist_ok=True)

        fb_path = bank_dir / 'feature_bank' / 'feature_bank.json'
        if fb_path.exists():
            existing = json.load(open(fb_path))
            n_existing = len(existing.get('formulas', []))
            print(f"\n--- Bank {bank_id}/{num_banks} --- SKIPPED (already has {n_existing} formulas)")
            total_formulas += n_existing
            continue

        ckpt_dir = bank_dir / 'checkpoints'
        latest_ckpt, resume_iter = _find_latest_checkpoint(ckpt_dir)

        if latest_ckpt:
            print(f"\n--- Bank {bank_id}/{num_banks} --- RESUMING from iter {resume_iter} ---")
        else:
            print(f"\n--- Bank {bank_id}/{num_banks} ---")

        bank_config = bank_configs[bank_id] if bank_id < len(bank_configs) else {}
        bank_full_config = _merge_bank_config(config, bank_config, bank_id)

        env = FractureVSREnvironment(
            data_loader=train_loader,
            config=bank_full_config,
            device=device,
        )

        if latest_ckpt:
            fb_resume_path = bank_dir / 'feature_bank_resume'
            if fb_resume_path.exists():
                env.feature_bank = LargeFeatureBank.load(str(fb_resume_path), device=device)
                print(f"  Resumed feature_bank: {env.feature_bank.size()} formulas")

        vocab_size = len(env.vocabulary)
        policy = PolicyAgent(
            vocab_size=vocab_size,
            embedding_dim=bank_full_config['model'].get('embedding_dim', 128),
            hidden_size=bank_full_config['model'].get('hidden_size', 256),
            num_layers=bank_full_config['model'].get('num_layers', 2),
            dropout=bank_full_config['model'].get('dropout', 0.1),
        ).to(device)

        trainer = PPOTrainer(
            policy=policy, env=env,
            learning_rate=bank_full_config['training']['learning_rate'],
            gamma=bank_full_config['training']['gamma'],
            gae_lambda=bank_full_config['training']['gae_lambda'],
            clip_epsilon=bank_full_config['training']['clip_epsilon'],
            value_coef=bank_full_config['training']['value_coef'],
            entropy_coef=bank_full_config['training'].get('entropy_coef_start', 0.08),
            entropy_coef_end=bank_full_config['training'].get('entropy_coef_end', 0.005),
            entropy_decay_fraction=bank_full_config['training'].get('entropy_decay_fraction', 0.5),
            max_grad_norm=bank_full_config['training']['max_grad_norm'],
            n_epochs=bank_full_config['training']['n_epochs_ppo'],
            batch_size=bank_full_config['training']['batch_size_ppo'],
            device=device,
            lr_warmup_iterations=bank_full_config['training'].get('lr_warmup_iterations', 20),
            total_iterations=bank_full_config['training']['iterations'],
        )

        binary_bias = bank_full_config['training'].get('binary_op_bias', 0.0)
        if binary_bias > 0:
            trainer.set_binary_op_bias(binary_bias, env.vocabulary)

        n_iters = bank_full_config['training']['iterations']
        episodes_per_iter = bank_full_config['training']['episodes_per_iteration']

        print(f"  Training: {n_iters} iterations x {episodes_per_iter} episodes")
        print(f"  Vocab size: {vocab_size}, Bank focus: {bank_config.get('focus', 'general')}")

        save_dir = str(bank_dir / 'checkpoints')
        trainer.train(
            n_iterations=n_iters,
            episodes_per_iteration=episodes_per_iter,
            save_dir=save_dir,
            start_iteration=resume_iter,
            resume_checkpoint=latest_ckpt,
        )

        env.feature_bank.save(str(bank_dir / 'feature_bank'))
        total_formulas += env.feature_bank.size()
        print(f"  Bank {bank_id} final size: {env.feature_bank.size()}")

        del policy, trainer, env
        gc.collect()
        torch.cuda.empty_cache()

    meta = {'total_formulas': total_formulas, 'num_banks': num_banks}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Phase 1 complete: {total_formulas} formulas across {num_banks} banks")


def _merge_bank_config(base_config, bank_config, bank_id):
    merged = json.loads(json.dumps(base_config))
    if 'max_depth' in bank_config:
        merged['model']['max_depth'] = bank_config['max_depth']
    if 'max_sequence_length' in bank_config:
        merged['model']['max_sequence_length'] = bank_config['max_sequence_length']
    if 'binary_op_bias' in bank_config:
        merged['training']['binary_op_bias'] = bank_config['binary_op_bias']
    return merged


# ======================================================================
# Step 2: Merge & Deduplicate
# ======================================================================

def step2_merge(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 2: Merge & Deduplicate Formulas")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    phase1_dir = output_dir / 'phase1'
    merged_path = output_dir / 'merged_formulas.json'

    if merged_path.exists():
        merged = json.load(open(merged_path))
        print(f"  Already done: {len(merged)} formulas")
        return

    formulas = load_formulas_from_banks(phase1_dir)
    print(f"  Loaded {len(formulas)} formulas from Phase 1")

    seen = set()
    unique = []
    for f in formulas:
        if f['str'] not in seen:
            seen.add(f['str'])
            unique.append(f)
    print(f"  After dedup: {len(unique)} formulas")

    op_counts = {}
    for f in unique:
        for tok in f['str'].split():
            if tok in TENSOR_OPERATORS:
                op_counts[tok] = op_counts.get(tok, 0) + 1
    print(f"\n  Top-15 operators:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    {op}: {count}")

    fracture_op_count = sum(1 for f in unique for tok in f['str'].split() if tok in FRACTURE_OPERATORS)
    print(f"\n  Fracture-specific operator usage: {fracture_op_count}")

    with open(merged_path, 'w') as f:
        json.dump(unique, f, indent=2)
    print(f"  Saved {len(unique)} merged formulas to {merged_path}")


# ======================================================================
# Step 3: Full-Resolution Validation
# ======================================================================

def _save_phase_b_ckpt(ckpt_path, validated, next_ci):
    tmp_path = str(ckpt_path) + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump({'validated': validated, 'next_ci': next_ci}, f)
    os.replace(tmp_path, ckpt_path)


def step3_validate(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 3: Full-Resolution Validation")
    print(f"{'='*70}")

    val_cfg = config.get('validation', {})
    prefilter_threshold = val_cfg.get('prefilter_acc_threshold', 0.15)
    fullres_threshold = val_cfg.get('fullres_acc_threshold', 0.15)
    print(f"  Thresholds: prefilter={prefilter_threshold}, fullres={fullres_threshold}")

    output_dir = Path(config['output_dir'])
    validated_path = output_dir / 'validated_formulas.json'
    candidates_ckpt_path = output_dir / 'step3_candidates_ckpt.json'
    phase_b_ckpt_path = output_dir / 'step3_phase_b_ckpt.json'

    if validated_path.exists():
        validated = json.load(open(validated_path))
        print(f"  Already done: {len(validated)} formulas")
        return

    merged_path = output_dir / 'merged_formulas.json'
    formulas = json.load(open(merged_path))
    print(f"  Validating {len(formulas)} formulas at full resolution")

    data_dir = config['dataset_options']['data_dir']
    resolution = config['dataset_options'].get('resolution_full', 640)

    # ---- Phase A: Quick pre-filter (with checkpoint) ----
    candidates = None
    if candidates_ckpt_path.exists():
        candidates = json.load(open(candidates_ckpt_path))
        print(f"  Resumed Phase A checkpoint: {len(candidates)} candidates")

    if candidates is None:
        dm = _make_dm(
            config, resolution=resolution,
            batch_size=32, num_workers=4,
        )
        dm.setup()
        val_loader = dm.get_val_loader()

        val_labels_list = []
        for _, labels in val_loader:
            val_labels_list.append(labels)
        val_labels = torch.cat(val_labels_list, dim=0).long()
        print(f"  Val set: {val_labels.shape[0]} images")

        print(f"  Phase A: Quick pre-filter {len(formulas)} formulas (train on train_subset, eval on val)...")
        pre_res = 320
        dm_sub = _make_dm(config, resolution=pre_res, batch_size=128, num_workers=4)
        dm_sub.setup()
        sub_images, sub_labels = [], []
        for images, labels in dm_sub.get_train_loader():
            sub_images.append(images)
            sub_labels.append(labels)
            if sum(i.shape[0] for i in sub_images) >= 128:
                break
        sub_images = torch.cat(sub_images, dim=0)
        sub_labels = torch.cat(sub_labels, dim=0).long().to(device)
        print(f"  Train subset: {sub_images.shape[0]} images @ {pre_res}")

        sub_data_batch = build_fracture_data_batch(sub_images, device)
        del sub_images
        gc.collect()
        torch.cuda.empty_cache()

        dm_val_sub = _make_dm(config, resolution=pre_res, batch_size=128, num_workers=4)
        dm_val_sub.setup()
        val_sub_images, val_sub_labels = [], []
        for images, labels in dm_val_sub.get_val_loader():
            val_sub_images.append(images)
            val_sub_labels.append(labels)
        val_sub_images = torch.cat(val_sub_images, dim=0)
        val_sub_labels = torch.cat(val_sub_labels, dim=0).long().to(device)
        val_data_batch = build_fracture_data_batch(val_sub_images, device)
        del val_sub_images
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Val set: {val_sub_labels.shape[0]} images @ {pre_res}")

        candidates = []
        for i, f in enumerate(formulas):
            if i % 200 == 0:
                print(f"    {i}/{len(formulas)}")

            train_out = execute_formula(f['str'], sub_data_batch)
            if train_out is None:
                continue
            if train_out.dim() == 1:
                train_out = train_out.unsqueeze(1)
            if train_out.dim() > 2:
                train_out = train_out.flatten(1)

            val_out = execute_formula(f['str'], val_data_batch)
            if val_out is None:
                continue
            if val_out.dim() == 1:
                val_out = val_out.unsqueeze(1)
            if val_out.dim() > 2:
                val_out = val_out.flatten(1)

            train_mean = train_out.mean(dim=0, keepdim=True)
            train_std = train_out.std(dim=0, keepdim=True) + 1e-8
            train_norm = (train_out - train_mean) / train_std
            val_norm = (val_out - train_mean) / train_std

            classifier = nn.Linear(train_norm.shape[1], dm_sub.num_classes).to(device)
            class_counts = torch.bincount(sub_labels, minlength=dm_sub.num_classes).float()
            class_weights = 1.0 / (class_counts + 1e-6)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
            classifier.train()
            for _ in range(30):
                optimizer.zero_grad()
                loss = criterion(classifier(train_norm), sub_labels)
                loss.backward()
                optimizer.step()
            classifier.eval()
            with torch.no_grad():
                val_acc = (classifier(val_norm).argmax(dim=1) == val_sub_labels).float().mean().item()
            del train_out, val_out, train_norm, val_norm, classifier
            torch.cuda.empty_cache()

            if val_acc >= prefilter_threshold:
                candidates.append({'orig_idx': i, 'formula': f, 'quick_acc': val_acc})

        del sub_data_batch, val_data_batch, sub_labels, val_sub_labels
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Pre-filter: {len(candidates)}/{len(formulas)} passed")

        with open(candidates_ckpt_path, 'w') as f:
            json.dump(candidates, f)
        print(f"  Phase A checkpoint saved to {candidates_ckpt_path}")
    else:
        print(f"  Skipping Phase A (checkpoint found with {len(candidates)} candidates)")

    # ---- Phase B: Full train+eval (with incremental checkpoint) ----
    dm = _make_dm(
        config, resolution=resolution,
        batch_size=32, num_workers=4,
    )
    dm.setup()
    val_loader = dm.get_val_loader()
    val_labels_list = []
    for _, labels in val_loader:
        val_labels_list.append(labels)
    val_labels = torch.cat(val_labels_list, dim=0).long()
    train_loader = dm.get_train_loader()
    train_labels_list = []
    for _, labels in train_loader:
        train_labels_list.append(labels)
    train_labels = torch.cat(train_labels_list, dim=0).long()
    val_labels_cpu = val_labels
    n_classes = dm.num_classes

    dm2 = _make_dm(config, resolution=resolution, batch_size=32, num_workers=4)
    dm2.setup()

    validated = []
    start_ci = 0

    if phase_b_ckpt_path.exists():
        ckpt = json.load(open(phase_b_ckpt_path))
        validated = ckpt.get('validated', [])
        start_ci = ckpt.get('next_ci', 0)
        print(f"  Resumed Phase B from candidate {start_ci}/{len(candidates)} ({len(validated)} already validated)")

    print(f"  Phase B: Full train+eval for {len(candidates)} candidates (starting from {start_ci})...")

    for ci in range(start_ci, len(candidates)):
        entry = candidates[ci]
        orig_idx = entry['orig_idx']
        f = entry['formula']
        quick_acc = entry['quick_acc']

        if ci % 5 == 0:
            print(f"    Candidate {ci}/{len(candidates)}")

        train_feats = []
        ok = True
        for images, _ in dm2.get_train_loader():
            data_batch = build_fracture_data_batch(images, device)
            out = execute_formula(f['str'], data_batch)
            del data_batch
            torch.cuda.empty_cache()
            if out is None:
                ok = False
                break
            if out.dim() == 1:
                out = out.unsqueeze(1)
            if out.dim() > 2:
                out = out.flatten(1)
            train_feats.append(out.cpu())
        if not ok or not train_feats:
            _save_phase_b_ckpt(phase_b_ckpt_path, validated, ci + 1)
            continue
        train_feat = torch.cat(train_feats, dim=0)
        del train_feats

        val_feats = []
        for images, _ in val_loader:
            data_batch = build_fracture_data_batch(images, device)
            out = execute_formula(f['str'], data_batch)
            del data_batch
            torch.cuda.empty_cache()
            if out is None:
                ok = False
                break
            if out.dim() == 1:
                out = out.unsqueeze(1)
            if out.dim() > 2:
                out = out.flatten(1)
            val_feats.append(out.cpu())
        if not ok or not val_feats:
            _save_phase_b_ckpt(phase_b_ckpt_path, validated, ci + 1)
            continue
        val_feat = torch.cat(val_feats, dim=0)
        del val_feats

        train_mean = train_feat.mean(dim=0, keepdim=True)
        train_std = train_feat.std(dim=0, keepdim=True) + 1e-8
        train_feat_norm = (train_feat - train_mean) / train_std
        val_feat_norm = (val_feat - train_mean) / train_std

        train_feat_norm = train_feat_norm.to(device)
        val_feat_norm = val_feat_norm.to(device)
        train_labels_dev = train_labels.to(device)
        val_labels_dev = val_labels_cpu.to(device)

        best_val_bacc = 0
        best_val_acc = 0
        for wd in [1, 10, 50, 100]:
            classifier = nn.Linear(train_feat_norm.shape[1], n_classes).to(device)
            class_counts = torch.bincount(train_labels, minlength=n_classes).float()
            class_weights = 1.0 / (class_counts + 1e-6)
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.005, weight_decay=wd)

            best_epoch_val_bacc = 0
            best_epoch_val_acc = 0
            classifier.train()
            for epoch in range(100):
                optimizer.zero_grad()
                loss = criterion(classifier(train_feat_norm), train_labels_dev)
                loss.backward()
                optimizer.step()

                if epoch % 10 == 9:
                    classifier.eval()
                    with torch.no_grad():
                        v_preds = classifier(val_feat_norm).argmax(dim=1)
                        v_acc = (v_preds == val_labels_dev).float().mean().item()
                        per_cls = []
                        for c in range(n_classes):
                            m = val_labels_dev == c
                            if m.sum() > 0:
                                per_cls.append((v_preds[m] == c).float().mean().item())
                        v_bacc = np.mean(per_cls) if per_cls else 0
                    if v_bacc > best_epoch_val_bacc:
                        best_epoch_val_bacc = v_bacc
                        best_epoch_val_acc = v_acc
                    classifier.train()

            if best_epoch_val_bacc > best_val_bacc:
                best_val_bacc = best_epoch_val_bacc
                best_val_acc = best_epoch_val_acc

            del classifier
            torch.cuda.empty_cache()

        del train_feat, val_feat, train_feat_norm, val_feat_norm, train_labels_dev, val_labels_dev
        torch.cuda.empty_cache()

        if best_val_acc >= fullres_threshold:
            f['full_res_accuracy'] = best_val_acc
            f['full_res_balanced_accuracy'] = best_val_bacc
            validated.append(f)

        _save_phase_b_ckpt(phase_b_ckpt_path, validated, ci + 1)

    print(f"\n  Validated: {len(validated)}/{len(formulas)} formulas passed")

    with open(validated_path, 'w') as f:
        json.dump(validated, f, indent=2)
    print(f"  Saved to {validated_path}")

    if candidates_ckpt_path.exists():
        candidates_ckpt_path.unlink()
    if phase_b_ckpt_path.exists():
        phase_b_ckpt_path.unlink()

    gc.collect()
    torch.cuda.empty_cache()


# ======================================================================
# Step 4: Feature Extraction + Encoding
# ======================================================================

def step4_extract_features(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 4: Feature Extraction + Encoding")
    print(f"{'='*70}")

    dist_cfg = config.get('phase3', {}).get('distribution_stats', {})
    n_stats = dist_cfg.get('n_stats', 12)
    n_regions = dist_cfg.get('n_regions', 5)
    features_per_body = n_stats * n_regions
    print(f"  Encoding: {n_stats} stats x {n_regions} regions = {features_per_body} per body")

    output_dir = Path(config['output_dir'])
    features_path = output_dir / 'features.npz'

    if features_path.exists():
        print(f"  Already done: {features_path}")
        return

    validated_path = output_dir / 'validated_formulas.json'
    formulas = json.load(open(validated_path))
    bodies = formulas_to_bodies(formulas)
    print(f"  Extracting features from {len(bodies)} formula bodies")

    v2_dataset_dir = '/home/lqg1/code_8T/25/lxw/4/fracture_symbolic_v2/dataset'
    resolution = config['dataset_options'].get('resolution_full', 640)
    print(f"  Using v2 dataset directory: {v2_dataset_dir}")

    all_features = {split: [] for split in ['train', 'val', 'test']}
    all_labels = {split: [] for split in ['train', 'val', 'test']}

    for split in ['train', 'val', 'test']:
        print(f"\n  Processing {split} split...")
        ds = HBFMIDDataset(
            v2_dataset_dir, split=split, resolution=resolution,
            augment=False, task='classification',
        )
        loader = DataLoader(
            ds, batch_size=16, shuffle=False,
            num_workers=4, pin_memory=True,
        )

        split_feats = []
        split_labels = []

        for batch_idx, (images, labels) in enumerate(loader):
            data_batch = build_fracture_data_batch(images, device)
            batch_feats = []

            for body_str in bodies:
                fm = execute_body(body_str, data_batch)
                if fm is not None:
                    stats = encode_body_distribution_v2(fm, n_stats=n_stats, n_regions=n_regions)
                    batch_feats.append(stats)
                else:
                    batch_feats.append(torch.zeros(images.shape[0], features_per_body, device=device))

            feats = torch.cat(batch_feats, dim=1)
            split_feats.append(feats.cpu().numpy())
            split_labels.append(labels if isinstance(labels, np.ndarray) else np.array(labels))

            if batch_idx % 20 == 0:
                print(f"    Batch {batch_idx}: {feats.shape}")

            del data_batch
            torch.cuda.empty_cache()

        all_features[split] = np.concatenate(split_feats, axis=0)
        all_labels[split] = np.concatenate(split_labels, axis=0)
        print(f"  {split}: {all_features[split].shape}")

    active_classes = list(range(10))
    active_names = [FRACTURE_NAMES[c] for c in active_classes]

    np.savez(
        features_path,
        train_features=all_features['train'],
        train_labels=all_labels['train'],
        val_features=all_features['val'],
        val_labels=all_labels['val'],
        test_features=all_features['test'],
        test_labels=all_labels['test'],
        bodies=bodies,
        active_classes=np.array(active_classes),
    )
    with open(output_dir / 'class_names.json', 'w') as f:
        json.dump(active_names, f)
    print(f"  Saved features to {features_path}")
    print(f"  Active classes: {active_names}")

    from collections import Counter
    for split_name in ['train', 'val', 'test']:
        dist = Counter(all_labels[split_name].tolist())
        print(f"  {split_name} dist: {dict(sorted(dist.items()))}")


# ======================================================================
# Step 5: Train Classifier
# ======================================================================

def step5_train_classifier(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 5: Train Classifier (v5 — Interpretable ML Focus)")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    results_path = output_dir / 'classifier_results.json'

    version = 5
    versioned_results_path = output_dir / f'classifier_results_v{version}.json'
    versioned_model_path = output_dir / f'best_classifier_v{version}.pkl'

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
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import balanced_accuracy_score, accuracy_score
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    variances = np.var(combine_X, axis=0)
    non_const = variances > 1e-12
    if non_const.sum() < combine_X.shape[1]:
        print(f"  Removing {(~non_const).sum()} constant features...")
    combine_X_nc = combine_X[:, non_const]
    test_X_nc = test_X[:, non_const]
    print(f"  Non-constant features: {non_const.sum()} / {combine_X.shape[1]}")

    print(f"\n  Strategy: No SMOTE, class_weight='balanced', PCA after ANOVA pre-selection")
    print(f"  Rationale: SMOTE in {combine_X_nc.shape[1]}-dim space inflates CV bacc by ~0.3")

    anova_preselect = min(1000, combine_X_nc.shape[1])
    print(f"  ANOVA pre-selection: top-{anova_preselect} from {combine_X_nc.shape[1]} features")

    scaler_for_pca = StandardScaler()
    combine_scaled = scaler_for_pca.fit_transform(combine_X_nc)
    anova_selector = SelectKBest(f_classif, k=anova_preselect)
    combine_anova = anova_selector.fit_transform(combine_scaled, combine_y)
    print(f"  After ANOVA: {combine_anova.shape[1]} features")

    pca_full = PCA(n_components=0.99, random_state=42)
    pca_full.fit(combine_anova)
    n_components_99 = pca_full.n_components_
    pca_95 = np.searchsorted(np.cumsum(pca_full.explained_variance_ratio_), 0.95) + 1
    print(f"  PCA 99% variance: {n_components_99} components")
    print(f"  PCA 95% variance: {pca_95} components")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_pipes = {}
    cv_accs = {}

    from sklearn.metrics import roc_auc_score
    from sklearn.svm import SVC, LinearSVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    def _cv_eval(pipe, X, y):
        accs = []
        for tr, va in skf.split(X, y):
            pipe.fit(X[tr], y[tr])
            accs.append(accuracy_score(y[va], pipe.predict(X[va])))
        return np.mean(accs)

    def _calc_auc(pipe, X, y):
        try:
            y_prob = pipe.predict_proba(X)
            return roc_auc_score(y, y_prob, multi_class='ovr', average='macro')
        except Exception:
            return 0.0

    def _apply_pipeline_transform(X_raw):
        X_s = scaler_for_pca.transform(X_raw[:, non_const])
        X_a = anova_selector.transform(X_s)
        return X_a

    test_anova = _apply_pipeline_transform(test_X)

    print(f"\n  --- Method 1: ElasticNet LR (L1+L2) + ANOVA ---", flush=True)
    K_values = [50, 100, 200, 300, 500]
    en_C_values = [0.01, 0.1, 1.0, 10.0]
    en_l1_ratios = [0.3, 0.5, 0.7, 0.9]
    best_cv1, best_K1, best_C1, best_l1r = 0, None, None, None
    for K in K_values:
        if K > anova_preselect:
            continue
        for C in en_C_values:
            for l1r in en_l1_ratios:
                pipe = Pipeline([
                    ('select', SelectKBest(f_classif, k=K)),
                    ('scale', StandardScaler()),
                    ('clf', LogisticRegression(C=C, penalty='elasticnet', solver='saga',
                                               l1_ratio=l1r, max_iter=5000, class_weight='balanced')),
                ])
                cv = _cv_eval(pipe, combine_anova, combine_y)
                if cv > best_cv1:
                    best_cv1, best_K1, best_C1, best_l1r = cv, K, C, l1r
    print(f"  Best: K={best_K1}, C={best_C1}, l1_ratio={best_l1r}, cv_acc={best_cv1:.3f}")
    pipe_lr1 = Pipeline([
        ('select', SelectKBest(f_classif, k=best_K1)),
        ('scale', StandardScaler()),
        ('clf', LogisticRegression(C=best_C1, penalty='elasticnet', solver='saga',
                                   l1_ratio=best_l1r, max_iter=5000, class_weight='balanced')),
    ])
    pipe_lr1.fit(combine_anova, combine_y)
    preds = pipe_lr1.predict(test_anova)
    acc = accuracy_score(test_y, preds)
    auc = _calc_auc(pipe_lr1, test_anova, test_y)
    n_nonzero = np.sum(pipe_lr1.named_steps['clf'].coef_ != 0)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}, nonzero_coefs={n_nonzero}")
    results['lr_elasticnet'] = {'acc': float(acc), 'auc': float(auc), 'nonzero_coefs': int(n_nonzero)}
    best_pipes['lr_elasticnet'] = pipe_lr1
    cv_accs['lr_elasticnet'] = best_cv1

    print(f"\n  --- Method 2: GradientBoosting + ANOVA ---", flush=True)
    gb_K_values = [50, 100, 200, 300]
    gb_configs = [(100, 0.05, 3), (200, 0.05, 3), (200, 0.1, 5), (300, 0.05, 5), (300, 0.1, 3)]
    best_gb_cv, best_gb_cfg, best_gb_K = 0, None, None
    for K in gb_K_values:
        if K > anova_preselect:
            continue
        for n_est, lr, max_d in gb_configs:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', GradientBoostingClassifier(n_estimators=n_est, learning_rate=lr,
                                                   max_depth=max_d, subsample=0.8,
                                                   random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_gb_cv:
                best_gb_cv, best_gb_cfg, best_gb_K = cv, (n_est, lr, max_d), K
    print(f"  Best: n_est={best_gb_cfg[0]}, lr={best_gb_cfg[1]}, max_depth={best_gb_cfg[2]}, K={best_gb_K}, cv_acc={best_gb_cv:.3f}")
    pipe_gb = Pipeline([
        ('select', SelectKBest(f_classif, k=best_gb_K)),
        ('scale', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=best_gb_cfg[0], learning_rate=best_gb_cfg[1],
                                           max_depth=best_gb_cfg[2], subsample=0.8,
                                           random_state=42)),
    ])
    pipe_gb.fit(combine_anova, combine_y)
    preds = pipe_gb.predict(test_anova)
    acc = accuracy_score(test_y, preds)
    auc = _calc_auc(pipe_gb, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['gradient_boosting'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['gradient_boosting'] = pipe_gb
    cv_accs['gradient_boosting'] = best_gb_cv

    print(f"\n  --- Method 3: ExtraTrees + ANOVA ---", flush=True)
    et_K_values = [50, 100, 200, 300]
    et_configs = [(200, 5), (300, 5), (300, 10), (500, 10)]
    best_et_cv, best_et_cfg, best_et_K = 0, None, None
    for K in et_K_values:
        if K > anova_preselect:
            continue
        for n_est, min_s in et_configs:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', ExtraTreesClassifier(n_estimators=n_est, min_samples_leaf=min_s,
                                             class_weight='balanced', random_state=42, n_jobs=-1)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_et_cv:
                best_et_cv, best_et_cfg, best_et_K = cv, (n_est, min_s), K
    print(f"  Best: n_est={best_et_cfg[0]}, min_samples={best_et_cfg[1]}, K={best_et_K}, cv_acc={best_et_cv:.3f}")
    pipe_et = Pipeline([
        ('select', SelectKBest(f_classif, k=best_et_K)),
        ('scale', StandardScaler()),
        ('clf', ExtraTreesClassifier(n_estimators=best_et_cfg[0], min_samples_leaf=best_et_cfg[1],
                                     class_weight='balanced', random_state=42, n_jobs=-1)),
    ])
    pipe_et.fit(combine_anova, combine_y)
    preds = pipe_et.predict(test_anova)
    acc = accuracy_score(test_y, preds)
    auc = _calc_auc(pipe_et, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['extra_trees'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['extra_trees'] = pipe_et
    cv_accs['extra_trees'] = best_et_cv

    print(f"\n  --- Method 4: KNN + ANOVA ---", flush=True)
    knn_K_values = [50, 100, 200, 300]
    knn_n_values = [3, 5, 7, 11, 15]
    best_knn_cv, best_knn_n, best_knn_K = 0, None, None
    for K in knn_K_values:
        if K > anova_preselect:
            continue
        for n in knn_n_values:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', KNeighborsClassifier(n_neighbors=n, weights='distance', n_jobs=-1)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_knn_cv:
                best_knn_cv, best_knn_n, best_knn_K = cv, n, K
    print(f"  Best: n_neighbors={best_knn_n}, K={best_knn_K}, cv_acc={best_knn_cv:.3f}")
    pipe_knn = Pipeline([
        ('select', SelectKBest(f_classif, k=best_knn_K)),
        ('scale', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=best_knn_n, weights='distance', n_jobs=-1)),
    ])
    pipe_knn.fit(combine_anova, combine_y)
    preds = pipe_knn.predict(test_anova)
    acc = accuracy_score(test_y, preds)
    auc = _calc_auc(pipe_knn, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['knn'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['knn'] = pipe_knn
    cv_accs['knn'] = best_knn_cv

    print(f"\n  --- Method 5: SVM (RBF) + ANOVA ---", flush=True)
    svm_K_values = [50, 100, 200]
    svm_C_values = [0.1, 1.0, 10.0]
    best_svm_cv, best_svm_K, best_svm_C = 0, None, None
    for K in svm_K_values:
        if K > anova_preselect:
            continue
        for C in svm_C_values:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', SVC(C=C, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_svm_cv:
                best_svm_cv, best_svm_K, best_svm_C = cv, K, C
    print(f"  Best: K={best_svm_K}, C={best_svm_C}, cv_acc={best_svm_cv:.3f}")
    pipe_svm = Pipeline([
        ('select', SelectKBest(f_classif, k=best_svm_K)),
        ('scale', StandardScaler()),
        ('clf', SVC(C=best_svm_C, kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
    ])
    pipe_svm.fit(combine_anova, combine_y)
    preds = pipe_svm.predict(test_anova)
    acc = accuracy_score(test_y, preds)
    auc = _calc_auc(pipe_svm, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['svm_rbf'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['svm_rbf'] = pipe_svm
    cv_accs['svm_rbf'] = best_svm_cv

    print(f"\n  --- Method 6: MLP + ANOVA ---", flush=True)
    mlp_K_values = [50, 100, 200]
    mlp_configs = [(128, 64), (256, 128), (128, 64, 32)]
    best_mlp_cv, best_mlp_K, best_mlp_cfg = 0, None, None
    for K in mlp_K_values:
        if K > anova_preselect:
            continue
        for hidden in mlp_configs:
            pipe = Pipeline([
                ('select', SelectKBest(f_classif, k=K)),
                ('scale', StandardScaler()),
                ('clf', MLPClassifier(hidden_layer_sizes=hidden, max_iter=1000, early_stopping=True,
                                      validation_fraction=0.1, random_state=42)),
            ])
            cv = _cv_eval(pipe, combine_anova, combine_y)
            if cv > best_mlp_cv:
                best_mlp_cv, best_mlp_K, best_mlp_cfg = cv, K, hidden
    print(f"  Best: K={best_mlp_K}, hidden={best_mlp_cfg}, cv_acc={best_mlp_cv:.3f}")
    pipe_mlp = Pipeline([
        ('select', SelectKBest(f_classif, k=best_mlp_K)),
        ('scale', StandardScaler()),
        ('clf', MLPClassifier(hidden_layer_sizes=best_mlp_cfg, max_iter=1000, early_stopping=True,
                              validation_fraction=0.1, random_state=42)),
    ])
    pipe_mlp.fit(combine_anova, combine_y)
    preds = pipe_mlp.predict(test_anova)
    acc = accuracy_score(test_y, preds)
    auc = _calc_auc(pipe_mlp, test_anova, test_y)
    print(f"  Test: acc={acc:.3f}, AUC={auc:.3f}")
    results['mlp'] = {'acc': float(acc), 'auc': float(auc)}
    best_pipes['mlp'] = pipe_mlp
    cv_accs['mlp'] = best_mlp_cv

    print(f"\n  {'='*60}")
    print(f"  COMPARISON SUMMARY (v{version})")
    print(f"  {'='*60}")
    interp_labels = {
        'lr_elasticnet': '[HIGH]',
        'gradient_boosting': '[MED]',
        'extra_trees': '[MED]',
        'knn': '[HIGH]',
        'svm_rbf': '[LOW]',
        'mlp': '[LOW]',
    }
    best_method = max(results, key=lambda k: results[k]['acc'])
    for method, res in results.items():
        marker = " <-- BEST" if method == best_method else ""
        interp = interp_labels.get(method, '')
        cv_info = f" cv_acc={cv_accs.get(method, 0):.3f}"
        gap = cv_accs.get(method, 0) - res['acc']
        print(f"    {method:20s}{interp:6s}: acc={res['acc']:.3f}, AUC={res['auc']:.3f}{cv_info} gap={gap:.3f}{marker}")

    all_per_class = {}
    active_names = json.load(open(output_dir / 'class_names.json'))
    for method_name, pipe in best_pipes.items():
        mpreds = pipe.predict(test_anova)
        pc = {}
        for c in range(num_classes):
            mask = test_y == c
            if mask.sum() > 0:
                name = active_names[c] if c < len(active_classes) else f"class_{c}"
                pc[name] = float((mpreds[mask] == c).mean())
        all_per_class[method_name] = pc

    best_pipe = best_pipes[best_method]
    best_preds = best_pipe.predict(test_anova)
    per_class = all_per_class[best_method]

    train_preds = best_pipe.predict(combine_anova)
    train_acc = accuracy_score(combine_y, train_preds)
    train_auc = _calc_auc(best_pipe, combine_anova, combine_y)

    print(f"\n  Best method: {best_method}")
    print(f"  Train acc={train_acc:.3f}, AUC={train_auc:.3f}")
    print(f"  Test  acc={results[best_method]['acc']:.3f}, AUC={results[best_method]['auc']:.3f}")
    print(f"  Per-class test ({best_method}):")
    for name, acc_val in sorted(per_class.items(), key=lambda x: -x[1]):
        print(f"    {name:25s}: {acc_val:.3f}")

    print(f"\n  All methods per-class accuracy:")
    header = f"    {'Class':25s}"
    for mn in results.keys():
        header += f" | {mn:>10s}"
    print(header)
    for name in sorted(per_class.keys()):
        line = f"    {name:25s}"
        for mn in results.keys():
            val = all_per_class[mn].get(name, 0.0)
            line += f" | {val:10.3f}"
        print(line)

    import joblib
    joblib.dump({
        'pipe': best_pipe,
        'non_const': non_const,
        'method': best_method,
        'scaler': scaler_for_pca,
        'anova_selector': anova_selector,
    }, output_dir / 'best_classifier.pkl')
    joblib.dump({
        'pipe': best_pipe,
        'non_const': non_const,
        'method': best_method,
        'scaler': scaler_for_pca,
        'anova_selector': anova_selector,
    }, versioned_model_path)

    for mn, mpipe in best_pipes.items():
        joblib.dump({
            'pipe': mpipe,
            'non_const': non_const,
            'method': mn,
            'scaler': scaler_for_pca,
            'anova_selector': anova_selector,
        }, output_dir / f'classifier_{mn}_v{version}.pkl')
        print(f"  Saved: classifier_{mn}_v{version}.pkl")

    best_results = {
        'version': version,
        'method': best_method,
        'train_accuracy': float(train_acc),
        'train_auc': float(train_auc),
        'test_accuracy': results[best_method]['acc'],
        'test_auc': results[best_method]['auc'],
        'per_class_test': per_class,
        'all_methods': results,
        'all_per_class': all_per_class,
        'cv_accs': {k: float(v) for k, v in cv_accs.items()},
        'interpretability': interp_labels,
    }
    with open(results_path, 'w') as f:
        json.dump(best_results, f, indent=2)
    with open(versioned_results_path, 'w') as f:
        json.dump(best_results, f, indent=2)
    print(f"  Results saved: {versioned_results_path} (v{version})")
    print(f"  Model saved:   {versioned_model_path} (v{version})")


# ======================================================================
# Step 6: Interpretability Report
# ======================================================================

def step6_report(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 6: Generate Interpretability Report")
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

    from sklearn.ensemble import VotingClassifier
    scaler_for_pca = clf_data.get('scaler')
    anova_selector = clf_data.get('anova_selector')
    non_const_mask = clf_data.get('non_const')

    if isinstance(pipe, VotingClassifier):
        inner_pipes = {name: est for name, est in pipe.estimators}
        best_inner_name = max(inner_pipes.keys(), key=lambda n: 1)
        inner_pipe = inner_pipes[best_inner_name]
        clf = inner_pipe.named_steps['clf']
        selector = inner_pipe.named_steps.get('select', None)
        pca_step = inner_pipe.named_steps.get('pca', None)
        lines_note = f"  (Ensemble model — showing feature analysis from sub-model '{best_inner_name}')"
    elif hasattr(pipe, 'named_steps'):
        clf = pipe.named_steps['clf']
        selector = pipe.named_steps.get('select', None)
        pca_step = pipe.named_steps.get('pca', None)
        lines_note = None
    else:
        print("  Cannot extract feature analysis from this classifier type, skipping weight analysis")
        return

    formulas.sort(key=lambda f: f.get('full_res_accuracy', 0), reverse=True)

    lines = []
    lines.append("=" * 70)
    lines.append("BONE FRACTURE SYMBOLIC FEATURE INTERPRETABILITY REPORT")
    lines.append("=" * 70)
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

    if pca_step is not None:
        lines.append(f"  Pipeline uses PCA ({pca_step.n_components_} components) — showing ANOVA top features instead")
        if anova_selector is not None:
            anova_scores = anova_selector.scores_
            anova_mask = anova_selector.get_support(indices=True)
            top_anova = np.argsort(anova_scores[anova_mask])[::-1][:20]
            for rank, fi in enumerate(top_anova):
                orig_idx = anova_mask[fi]
                if non_const_mask is not None:
                    orig_idx = np.where(non_const_mask)[0][orig_idx]
                formula_idx = orig_idx // stats_per_formula
                stat_idx = orig_idx % stats_per_formula
                sname = stat_names[stat_idx] if stat_idx < len(stat_names) else f'stat_{stat_idx}'
                if formula_idx < len(formulas):
                    lines.append(f"    {rank+1:2d}. formula[{formula_idx}].{sname} (F-score={anova_scores[anova_mask[fi]]:.2f})")
                else:
                    lines.append(f"    {rank+1:2d}. feat[{orig_idx}].{sname} (F-score={anova_scores[anova_mask[fi]]:.2f})")
    elif selector is not None and hasattr(selector, 'get_support'):
        selected_idx = selector.get_support(indices=True)
        if anova_selector is not None:
            anova_mask = anova_selector.get_support(indices=True)
            original_idx = anova_mask[selected_idx]
            if non_const_mask is not None:
                original_idx = np.where(non_const_mask)[0][original_idx]
        else:
            original_idx = selected_idx
        if hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_).mean(axis=0)
            imp_label = "|weight|"
        elif hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            imp_label = "importance"
        else:
            importances = None
            imp_label = None
        lines.append(f"  Selected {len(selected_idx)} features from ANOVA pre-selection")
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
            lines.append("  (Classifier does not expose feature importances)")
    else:
        lines.append("  (No feature selector available in pipeline)")

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
        'edge_diag_45': 'Diagonal edge (45°) — spiral/oblique fractures',
        'edge_diag_135': 'Diagonal edge (135°) — spiral/oblique fractures',
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
    parser = argparse.ArgumentParser(description='Fracture Symbolic Feature Discovery Pipeline (v3 — Expanded)')
    parser.add_argument('--config', type=str, default='configs/fracture_v3_expanded.yaml')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id (0 or 1)')
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
    print(f"  Fracture Symbolic Pipeline v3 — Expanded Feature Discovery")
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
        ('Step 5: Train Classifier', lambda: step5_train_classifier(config, device)),
        ('Step 6: Interpretability Report', lambda: step6_report(config, device)),
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
