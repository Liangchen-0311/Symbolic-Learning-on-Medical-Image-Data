#!/usr/bin/env python3
"""
ThirdData (BUSI breast ultrasound) Symbolic Feature Discovery Pipeline (grayscale).

Ported from the HAM10000 v2 pipeline. Same RL -> formula discovery -> HGB
classification structure, adapted for single-channel MRI:
  - Grayscale terminals (I_GRAY/I_BLUR/I_GRAD/I_LOCALSTD/I_LOG/I_LBP) instead
    of RGB/HSV/color channels.
  - 4 balanced classes (glioma / meningioma / pituitary / notumor); a tumor-vs-
    notumor superclass is used for hierarchical RL evaluation.
  - val split is carved from Training/ (Testing/ is the held-out test set).

Steps:
  0. Validate dataset
  1. Phase 1: Symbolic feature discovery (multi-bank RL search, early stop)
  2. Merge feature banks
  3. Validate formulas (Phase A quick multi-root linear probe)
  4. Extract features (distribution statistics per body)
  5. Train classifiers (4 HGB variants; sample_weight kept though data is balanced)
  6. Generate interpretability report

Usage:
    python experiments/run_brain_tumor_pipeline.py --config configs/brain_tumor.yaml --start_step 0
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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from src.data.thirddata_loader import (
    ThirdDataDataset, ThirdDataDataModule, build_third_data_batch,
    build_third_superclass_mapping, THIRD_NAMES, THIRD_FULL_NAMES,
    THIRD_SUPERCLASS, SUPERCLASS_NAMES,
)

from src.symbolic.tensor_operators import (
    TensorOperators, TENSOR_OPERATORS, ROOT_OPERATORS, SymbolicKernelBank,
)
from src.symbolic.brain_operators import register_brain_operators, BRAIN_OPERATORS
from src.symbolic.feature_encoding import (
    encode_body_distribution_v2,
    SymbolicFisherVector,
    homogeneous_kernel_map,
    apply_normalization_pipeline,
    apply_normalization_pipeline_with_stats,
)
from src.symbolic.large_feature_bank import LargeFeatureBank
from src.models.policy_agent import PolicyAgent
from src.rl.brain_tumor_environment import (
    BrainTumorVSREnvironment, build_brain_terminals, BRAIN_TERMINALS,
)
from src.rl.ppo_trainer import PPOTrainer

# Register the brain-MRI symmetry/texture operators. Color-only ops are absent.
register_brain_operators(TENSOR_OPERATORS)

# Register learnable convolution kernels (classic_edge_x, conv3x3_0, etc.)
# These operators are used by formulas discovered during RL training.
_kernel_bank = SymbolicKernelBank(device='cpu')
_kernel_bank.register_operators(TENSOR_OPERATORS)


# ======================================================================
# Helper functions
# ======================================================================

def _build_terminal_data(images, device='cuda'):
    """Build grayscale terminal channel dict from images [B,1,H,W] (or [B,C,H,W];
    channel 0 is used).

    Delegates to the single source of truth in brain_tumor_environment so the
    terminals used during feature extraction match exactly those the RL agent
    searched over: I_GRAY, I_BLUR, I_GRAD, I_LOCALSTD, I_LOG, I_LBP.
    """
    return build_brain_terminals(images)


def execute_body(body_str, terminal_data):
    """Execute a formula body (without root operator) on terminal data.
    Returns spatial map [B, 1, H, W] or None.
    """
    tokens = body_str.strip().split()
    stack = []
    for token in tokens:
        if token in terminal_data:
            stack.append(terminal_data[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                return None
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            try:
                result = op_func(*operands)
                result = torch.nan_to_num(result, nan=0.0, posinf=1e4, neginf=-1e4)
                stack.append(result)
            except Exception:
                return None
        else:
            return None
    if len(stack) != 1:
        return None
    out = torch.clamp(stack[0], -1e4, 1e4)
    return out if out.dim() >= 2 else None


def execute_formula(formula_str, terminal_data):
    """Execute a full formula (body + root operator) on terminal data.
    Returns scalar features [B] or None.
    """
    tokens = formula_str.strip().split()
    stack = []
    for token in tokens:
        if token in terminal_data:
            stack.append(terminal_data[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                return None
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            try:
                result = op_func(*operands)
                if torch.isnan(result).any() or torch.isinf(result).any():
                    return None
                stack.append(result)
            except Exception:
                return None
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
# Step 0: Validate Dataset
# ======================================================================

def step0_validate_dataset(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 0: Validate ThirdData (BUSI) Dataset")
    print(f"{'='*70}")

    data_dir = config['dataset_options']['data_dir']
    num_classes = config['dataset_options']['num_classes']
    class_names = config['dataset_options']['class_names']
    val_fraction = config['dataset_options'].get('val_fraction', 0.15)
    seed = config['dataset_options'].get('split_seed', 42)

    print(f"  Data dir: {data_dir}")
    print(f"  Num classes: {num_classes}")
    print(f"  Class names: {class_names}")
    print(f"  Val carved from Training (fraction={val_fraction}, seed={seed})")

    # On-disk layout is Training/ + Testing/; val is a virtual split of Training.
    for split in ['train', 'val', 'test']:
        try:
            ds = ThirdDataDataset(
                data_dir, split=split,
                resolution=config['dataset_options']['resolution'],
                augment=False, val_fraction=val_fraction, seed=seed,
            )
        except Exception as e:
            print(f"  ERROR building split={split}: {e}")
            return
        print(f"  {split} total: {len(ds)} images")

    # Quick data batch test
    print(f"\n  Testing data batch creation...")
    batch = build_third_data_batch(
        data_dir, resolution=config['dataset_options']['resolution'],
        batch_size=4, num_workers=0, val_fraction=val_fraction, seed=seed,
    )
    print(f"  images shape: {batch['images'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  Sample labels: {batch['labels'].tolist()}")
    print(f"  Label names: {[THIRD_NAMES[l] for l in batch['labels'].tolist()]}")

    print(f"\n  Dataset validation PASSED")


# ======================================================================
# Step 1: Phase 1 — Symbolic Feature Discovery (Multi-bank RL)
# ======================================================================

def _merge_bank_config(base_config, bank_config, bank_id):
    """Merge bank-specific config overrides into base config."""
    merged = json.loads(json.dumps(base_config))
    if 'max_depth' in bank_config:
        merged['model']['max_depth'] = bank_config['max_depth']
    if 'max_sequence_length' in bank_config:
        merged['model']['max_sequence_length'] = bank_config['max_sequence_length']
    if 'binary_op_bias' in bank_config:
        merged['training']['binary_op_bias'] = bank_config['binary_op_bias']
    return merged


def _find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint in a directory."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None, 0
    ckpts = list(ckpt_dir.glob('checkpoint_iter_*.pth'))
    if not ckpts:
        return None, 0
    ckpts.sort(key=lambda x: int(x.stem.split('_')[-1]))
    latest = ckpts[-1]
    iter_num = int(latest.stem.split('_')[-1])
    return latest, iter_num


def step1_phase1(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 1: Phase 1 — Symbolic Feature Discovery (grayscale ultrasound + early stop)")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    phase1_dir = output_dir / 'phase1'
    phase1_dir.mkdir(parents=True, exist_ok=True)

    meta_path = phase1_dir / 'phase1_meta.json'
    if meta_path.exists():
        meta = json.load(open(meta_path))
        print(f"  Already done: {meta.get('total_formulas', '?')} formulas")
        return

    data_dir = config['dataset_options']['data_dir']
    # unified resolution for all steps (config['dataset_options']['resolution'])
    resolution = config['dataset_options']['resolution']

    # Build DataLoader for the environment
    train_dataset = ThirdDataDataset(
        data_dir, split='train', resolution=resolution, augment=config.get('augment', True),
        val_fraction=config['dataset_options'].get('val_fraction', 0.15),
        seed=config['dataset_options'].get('split_seed', 42),
        class_subset=config['dataset_options'].get('class_subset', None),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )

    multi_bank_cfg = config.get('multi_bank', {})
    if not multi_bank_cfg.get('enabled', False):
        bank_configs = [{}]
    else:
        bank_configs = multi_bank_cfg['bank_configs']

    # v3: early stopping patience
    early_stop_patience = config['training'].get('early_stop_patience', 150)

    total_formulas = 0

    for bank_id, bank_cfg in enumerate(bank_configs):
        bank_dir = phase1_dir / f'bank_{bank_id}'
        bank_dir.mkdir(exist_ok=True)

        ckpt_dir = bank_dir / 'checkpoints'
        latest_ckpt, resume_iter = _find_latest_checkpoint(ckpt_dir)

        if latest_ckpt:
            print(f"\n--- Bank {bank_id}/{len(bank_configs)} --- RESUMING from iter {resume_iter} ---")
        else:
            print(f"\n--- Bank {bank_id}/{len(bank_configs)} ---")

        bank_full_config = _merge_bank_config(config, bank_cfg, bank_id)

        env = BrainTumorVSREnvironment(
            data_loader=train_loader,
            config=bank_full_config,
            device=device,
        )

        # Tumor-vs-notumor superclass for hierarchical RL evaluation. Only valid
        # for the full 4-class problem; skip for focused sub-problem searches
        # (e.g. binary glioma-vs-meningioma, where it would be degenerate).
        if config['dataset_options']['num_classes'] >= 3 and \
                config.get('strategy', {}).get('use_hierarchical_eval', False):
            superclass_groups, superclass_names = build_third_superclass_mapping()
            class_to_super = {}
            for super_id, (sname, members) in enumerate(superclass_groups.items()):
                for cls_id in members:
                    class_to_super[cls_id] = super_id
            env.superclass_mapping = class_to_super
            env.num_superclasses = len(superclass_groups)

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
            policy=policy,
            env=env,
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

        # Soft operator prior toward the glioma-discriminative family. Banks marked
        # neutral=true in their bank_config skip it (keep a general-purpose pool).
        op_bias = bank_full_config['training'].get('operator_bias', None)
        if op_bias and not bank_cfg.get('neutral', False):
            trainer.set_operator_bias(op_bias, env.vocabulary)
        else:
            print(f"  [Bias] bank {bank_id}: neutral (no operator prior)")

        n_iters = bank_full_config['training']['iterations']
        episodes_per_iter = bank_full_config['training']['episodes_per_iteration']

        print(f"  Training: {n_iters} iterations x {episodes_per_iter} episodes @ {resolution}px")
        print(f"  Vocab size: {vocab_size}, Bank focus: {bank_cfg.get('focus', 'general')}")
        print(f"  Early stop patience: {early_stop_patience} iters")

        # v3: custom training loop with early stopping
        save_dir = str(bank_dir / 'checkpoints')
        if latest_ckpt:
            extra = trainer.load_checkpoint(latest_ckpt)
            trainer.iteration_count = extra.get('iteration_count', resume_iter)
            trainer.best_reward = extra.get('best_reward', float('-inf'))
            trainer.best_program = extra.get('best_program', 'None')

        last_bank_size = env.feature_bank.size()
        iters_without_new = 0
        early_stopped = False

        pbar = tqdm(range(resume_iter, n_iters),
                     desc=f"Bank {bank_id}",
                     initial=resume_iter, total=n_iters,
                     unit="iter",
                     dynamic_ncols=True)

        for iteration in pbar:
            start_time = time.time()

            metrics = trainer.update(n_episodes=episodes_per_iter)
            trainer.iteration_count = iteration + 1

            current_bank_size = env.feature_bank.size()
            new_formulas_this_iter = current_bank_size - last_bank_size

            duration = time.time() - start_time

            # v3: early stopping check
            if new_formulas_this_iter > 0:
                iters_without_new = 0
                last_bank_size = current_bank_size
            else:
                iters_without_new += 1

            pbar.set_postfix({
                'bank_sz': current_bank_size,
                'new': new_formulas_this_iter,
                'reward': f"{metrics['avg_reward']:.3f}",
                'best': f"{trainer.best_reward:.3f}",
                'no_new': iters_without_new,
                'dur': f"{duration:.1f}s",
            })

            if iters_without_new >= early_stop_patience:
                pbar.close()
                print(f"  *** EARLY STOP: No new formula in {early_stop_patience} iters → skip to next bank ***")
                early_stopped = True

            # Save checkpoints
            if metrics['avg_reward'] > trainer.best_reward:
                trainer.best_reward = metrics['avg_reward']
                trainer.save_checkpoint(save_dir, "best_model.pth", {
                    'iteration_count': trainer.iteration_count,
                    'best_reward': trainer.best_reward,
                    'best_program': trainer.best_program,
                })

            if (iteration + 1) % 10 == 0:
                trainer.save_checkpoint(save_dir, f"checkpoint_iter_{iteration+1}.pth", {
                    'iteration_count': trainer.iteration_count,
                    'best_reward': trainer.best_reward,
                    'best_program': trainer.best_program,
                })
                if hasattr(env, 'feature_bank') and hasattr(env.feature_bank, 'save'):
                    fb_resume_dir = str(Path(save_dir).parent / 'feature_bank_resume')
                    env.feature_bank.save(fb_resume_dir)

            if early_stopped:
                break

        if not early_stopped:
            pbar.close()
        env.feature_bank.save(str(bank_dir / 'feature_bank'))
        total_formulas += env.feature_bank.size()
        status = f"early-stopped @ iter {iteration+1}" if early_stopped else f"completed {n_iters} iters"
        print(f"  Bank {bank_id} final size: {env.feature_bank.size()} ({status})")

        del policy, trainer, env
        gc.collect()
        torch.cuda.empty_cache()

    meta = {'total_formulas': total_formulas, 'num_banks': len(bank_configs)}
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Phase 1 complete: {total_formulas} total formulas across {len(bank_configs)} banks")


# ======================================================================
# Step 2: Merge Feature Banks
# ======================================================================

def step2_merge(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 2: Merge Feature Banks")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    phase1_dir = output_dir / 'phase1'
    merged_path = output_dir / 'merged_formulas.json'

    if merged_path.exists():
        formulas = json.load(open(merged_path))
        print(f"  Already merged: {len(formulas)} formulas")
        return

    formulas = load_formulas_from_banks(phase1_dir)
    print(f"  Merged {len(formulas)} unique formulas from all banks")

    with open(merged_path, 'w') as f:
        json.dump(formulas, f, indent=2)
    print(f"  Saved to {merged_path}")


# ======================================================================
# Step 3: Validate Formulas on Full Resolution
# ======================================================================

def _stratified_subset(dataset, num_classes, max_total, seed=0):
    """Return (images [N,1,H,W], labels [N]) — a class-balanced subset."""
    labels_all = []
    tmp = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    for _, lab in tmp:
        labels_all.extend(lab.tolist())
    labels_all = np.array(labels_all)
    per_class = max(1, max_total // num_classes)
    rng = np.random.RandomState(seed)
    idx = []
    for c in range(num_classes):
        cls_idx = np.where(labels_all == c)[0]
        if len(cls_idx) > per_class:
            cls_idx = rng.choice(cls_idx, per_class, replace=False)
        idx.extend(cls_idx.tolist())
    idx = sorted(idx)
    loader = DataLoader(torch.utils.data.Subset(dataset, idx),
                        batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    imgs, labs = [], []
    for im, lb in loader:
        imgs.append(im)
        labs.append(lb)
    return torch.cat(imgs, 0), torch.cat(labs, 0)


def step3_validate(config, device):
    """Step 3: post-merge de-duplication + quality gating.

    After several banks are merged, formulas are numerous. Each was already
    diversity-filtered *within* its own bank, but there is no cross-bank
    redundancy control: different banks trained on different mini-batches, so
    their per-bank correlation gates are not mutually comparable. Here every
    unique body is re-evaluated on ONE common image subset so signatures ARE
    comparable, then:

      Phase 1  Compute a cheap pooled signature per body and an ANOVA-F quality
               score (signal vs. label). No per-body model training.
      Phase 2  Quality gate — keep the top `min_quality_keep_fraction` of bodies,
               dropping near-chance / label-irrelevant ones.
      Phase 3  Greedy correlation de-duplication: scan bodies best-first and drop
               any whose primary signal correlates (|Pearson| >= threshold) with
               an already-kept body — removes cross-bank near-duplicates.
      Phase 4  Cap to `max_formulas` by quality.

    Writes validated_formulas.json: a compact, decorrelated, high-quality set.
    """
    print(f"\n{'='*70}")
    print(f"  STEP 3: Merge de-duplication + quality gating")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    merged_path = output_dir / 'merged_formulas.json'
    validated_path = output_dir / 'validated_formulas.json'

    if validated_path.exists():
        formulas = json.load(open(validated_path))
        print(f"  Already validated: {len(formulas)} formulas")
        return

    formulas = json.load(open(merged_path))
    num_classes = config['dataset_options']['num_classes']
    data_dir = config['dataset_options']['data_dir']

    vcfg = config.get('validation', {})
    dedup_thr = vcfg.get('dedup_correlation_threshold', 0.95)
    keep_frac = vcfg.get('min_quality_keep_fraction', 0.7)
    max_formulas = vcfg.get('max_formulas', vcfg.get('top_k_formulas', 800))
    sig_subset = vcfg.get('signature_subset_size', 1200)

    # Unique bodies (strip root operators).
    body_set = set()
    for f in formulas:
        tokens = f['str'].strip().split()
        body = ' '.join(tokens[:-1]) if tokens[-1] in ROOT_OPERATORS else f['str']
        body_set.add(body)
    unique_bodies = sorted(body_set)
    print(f"  {len(unique_bodies)} unique bodies from {len(formulas)} formulas")
    print(f"  dedup |r|>={dedup_thr}, keep top {keep_frac:.0%} by quality, cap={max_formulas}")

    # One common, class-balanced subset (signatures comparable across banks).
    res = config['dataset_options'].get('resolution', 224)
    vf = config['dataset_options'].get('val_fraction', 0.15)
    seed = config['dataset_options'].get('split_seed', 42)
    train_ds = ThirdDataDataset(data_dir, split='train', resolution=res,
                                 augment=False, val_fraction=vf, seed=seed)
    images, labels = _stratified_subset(train_ds, num_classes, sig_subset, seed=seed)
    labels_np = labels.numpy()
    # Precompute per-chunk grayscale terminals on GPU. Executing each formula on
    # the WHOLE subset at once peaks GPU memory and, under GPU contention,
    # silently OOM-culls deep formulas (execute_body catches and returns None);
    # chunking bounds the per-body intermediate so step 3 is reproducible
    # regardless of what else shares the card.
    sig_chunk = vcfg.get('signature_chunk', 256)
    chunk_terminals = [
        _build_terminal_data(images[i:i + sig_chunk].to(device), device)
        for i in range(0, images.shape[0], sig_chunk)
    ]
    del images
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Signature subset: {labels_np.shape[0]} images @ {res}px "
          f"({len(chunk_terminals)} chunks of <= {sig_chunk})")

    # Optional focus on a hard class PAIR: blend global ANOVA-F quality with the
    # pair's binary discriminability so step 3 preferentially keeps formulas that
    # separate e.g. glioma vs meningioma. focus_weight in [0,1] (0 = global only).
    class_names = config['dataset_options']['class_names']
    focus_pair = vcfg.get('focus_pair', None)
    focus_weight = float(vcfg.get('focus_weight', 0.0))
    pair_mask = None
    if focus_pair and focus_weight > 0:
        pair_ids = [class_names.index(c) if isinstance(c, str) else int(c) for c in focus_pair]
        pmask = np.isin(labels_np, pair_ids)
        pair_y = labels_np[pmask]
        if len(np.unique(pair_y)) >= 2:
            pair_mask = pmask
            print(f"  Focus pair: {focus_pair} (ids {pair_ids}), weight={focus_weight}")
        else:
            print(f"  Focus pair {focus_pair} not both present; ignoring")

    # Pooled signature operators (cheap [B,H,W] -> [B] descriptors).
    sig_op_names = ['global_avg_pool', 'global_std_pool', 'percentile_90', 'spatial_entropy']
    sig_funcs = [TENSOR_OPERATORS[o][0] for o in sig_op_names if o in TENSOR_OPERATORS]

    from sklearn.feature_selection import f_classif
    import warnings
    warnings.filterwarnings('ignore')

    # ---- Phase 1: signature + ANOVA-F quality per body ----
    records = []  # {body, quality, primary (unit-norm [N] np)}
    pbar = tqdm(unique_bodies, desc="Step3 sig", unit="body", dynamic_ncols=True)
    for body in pbar:
        # Execute the body chunk-by-chunk; collect per-sample pooled signatures
        # on CPU. A body is kept only if every chunk yields all k descriptors.
        chunk_sigs = []
        ok = True
        for ct in chunk_terminals:
            try:
                out = execute_body(body, ct)
            except Exception:
                torch.cuda.empty_cache()
                ok = False
                break
            if out is None or out.dim() < 2:
                ok = False
                break
            if out.dim() == 4:            # [B,1,H,W] -> [B,H,W]
                out = out[:, 0]
            cols = []
            for fn in sig_funcs:
                try:
                    v = fn(out)
                except Exception:
                    v = None
                if v is None:
                    break
                if v.dim() > 1:
                    v = v.flatten(1).mean(dim=1)
                cols.append(v)
            del out
            if len(cols) != len(sig_funcs):
                ok = False
                break
            chunk_sigs.append(torch.stack(cols, dim=1).detach().cpu())  # [chunk, k]
            del cols
        if not ok or not chunk_sigs:
            continue
        sig = torch.nan_to_num(torch.cat(chunk_sigs, dim=0), nan=0.0, posinf=0.0, neginf=0.0)  # [N,k]
        std = sig.std(dim=0)
        if (std < 1e-8).all():
            continue                                         # dead body
        sig_np = sig.numpy()
        try:
            F, _ = f_classif(sig_np, labels_np)
            F = np.nan_to_num(F, nan=0.0)
        except Exception:
            continue
        best = int(np.argmax(F))
        quality = float(F[best])
        # Pair-focused discriminability (binary F on the masked subset).
        quality_pair = quality
        if pair_mask is not None:
            try:
                Fp, _ = f_classif(sig_np[pair_mask], labels_np[pair_mask])
                quality_pair = float(np.nan_to_num(Fp, nan=0.0).max())
            except Exception:
                quality_pair = 0.0
        primary = sig_np[:, best].astype(np.float64)
        primary = primary - primary.mean()
        nrm = np.linalg.norm(primary)
        if nrm < 1e-8:
            continue
        records.append({'body': body, 'quality_global': quality,
                        'quality_pair': quality_pair, 'quality': quality,
                        'primary': primary / nrm})
        pbar.set_postfix({'kept': len(records)})
    pbar.close()
    del chunk_terminals
    gc.collect()
    torch.cuda.empty_cache()

    if not records:
        with open(validated_path, 'w') as f:
            json.dump([], f)
        print("  No usable bodies after signature pass!")
        return

    # Blend global + pair quality (population z-scores) into the ranking score.
    if pair_mask is not None:
        def _z(vals):
            a = np.array(vals, dtype=np.float64)
            s = a.std()
            return (a - a.mean()) / s if s > 1e-12 else np.zeros_like(a)
        zg = _z([r['quality_global'] for r in records])
        zp = _z([r['quality_pair'] for r in records])
        blended = (1 - focus_weight) * zg + focus_weight * zp
        for r, b in zip(records, blended):
            r['quality'] = float(b)

    # ---- Phase 2: quality gate (keep top keep_frac) ----
    records.sort(key=lambda r: r['quality'], reverse=True)
    keep_n = max(1, int(round(len(records) * keep_frac)))
    gated = records[:keep_n]
    print(f"  Quality gate: {len(gated)}/{len(records)} kept "
          f"(min F={gated[-1]['quality']:.2f})")

    # ---- Phase 3: greedy correlation de-duplication ----
    kept = []
    kept_mat = None  # [K, N] unit vectors
    n_dup = 0
    for r in gated:
        v = r['primary']
        if kept_mat is None:
            kept.append(r)
            kept_mat = v[None, :]
            continue
        max_corr = float(np.abs(kept_mat @ v).max())
        if max_corr < dedup_thr:
            kept.append(r)
            kept_mat = np.vstack([kept_mat, v[None, :]])
        else:
            n_dup += 1
    print(f"  De-dup: removed {n_dup}, kept {len(kept)}")

    # ---- Phase 4: cap to max_formulas ----
    kept = kept[:max_formulas]

    validated = [{
        'str': r['body'],
        'body': r['body'],
        'root_op': '',
        'quality_anova_f': r.get('quality_global', r['quality']),
        'quality_pair': r.get('quality_pair', 0.0),
        'rank_score': r['quality'],
        'full_res_balanced_accuracy': 0.0,
    } for r in kept]

    with open(validated_path, 'w') as f:
        json.dump(validated, f, indent=2)
    print(f"\n  Validated: {len(validated)} bodies "
          f"(from {len(unique_bodies)} unique; quality range "
          f"{kept[-1]['quality']:.2f}–{kept[0]['quality']:.2f})")


# ======================================================================
# Step 4: Extract Features
# ======================================================================

def step4_extract_features(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 4: Extract Features")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    features_path = output_dir / 'features.npz'

    if features_path.exists():
        print(f"  Features already extracted: {features_path}")
        return

    validated_path = output_dir / 'validated_formulas.json'
    formulas = json.load(open(validated_path))
    bodies = formulas_to_bodies(formulas)
    print(f"  {len(bodies)} unique bodies from {len(formulas)} formulas")

    data_dir = config['dataset_options']['data_dir']
    resolution = config['dataset_options']['resolution']

    # Distribution-stats encoding granularity (defaults reproduce the 12x5=60
    # baseline). Raising n_regions (e.g. 10 = whole + 3x3 grid) keeps finer
    # spatial detail before pooling.
    _dist = config.get('phase3', {}).get('distribution_stats', {})
    n_stats = _dist.get('n_stats', 12)
    n_regions = _dist.get('n_regions', 5)
    stats_per_body = n_stats * n_regions
    print(f"  Distribution encoding: {n_stats} stats x {n_regions} regions = {stats_per_body}/body")

    # Extract features for each split
    all_features = {}
    all_labels = {}

    for split in ['train', 'val', 'test']:
        print(f"\n  Extracting features for {split}...")
        dataset = ThirdDataDataset(
            data_dir, split=split, resolution=resolution, augment=False,
            val_fraction=config['dataset_options'].get('val_fraction', 0.15),
            seed=config['dataset_options'].get('split_seed', 42),
        )
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        split_features = []
        split_labels = []

        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Step4 {split}", unit="batch", dynamic_ncols=True)):
            images = images.to(device)
            batch_size = images.shape[0]

            terminal_data = _build_terminal_data(images, device)

            sample_features = []

            for body_str in bodies:
                result = execute_body(body_str, terminal_data)
                if result is None:
                    sample_features.append(np.zeros((batch_size, stats_per_body)))
                    continue
                if result.dim() == 4:
                    result = result[:, 0]
                # Encode distribution statistics
                stats = encode_body_distribution_v2(result, n_stats=n_stats, n_regions=n_regions)
                sample_features.append(stats.cpu().numpy())

            # Stack: [n_bodies * stats_per_body, batch_size]
            if sample_features:
                feat_matrix = np.concatenate(sample_features, axis=1) if len(sample_features[0].shape) > 1 else np.column_stack(sample_features)
                split_features.append(feat_matrix)
            split_labels.append(labels.numpy())

        all_features[split] = np.concatenate(split_features, axis=0)
        all_labels[split] = np.concatenate(split_labels, axis=0)
        print(f"  {split}: {all_features[split].shape}")

    # Save
    np.savez(
        features_path,
        train_features=all_features['train'],
        train_labels=all_labels['train'],
        val_features=all_features['val'],
        val_labels=all_labels['val'],
        test_features=all_features['test'],
        test_labels=all_labels['test'],
        bodies=np.array(bodies),
        active_classes=np.arange(config['dataset_options']['num_classes']),
    )
    print(f"  Features saved: {features_path}")


# ======================================================================
# Step 5: Train Classifier (v1 — 11 methods, same as fracture v6)
# ======================================================================

SUPERCLASS_MAP = {
    'tumor': ['glioma', 'meningioma', 'pituitary'],
    'notumor': ['notumor'],
}


def step5_train_classifier(config, device):
    print(f"\n{'='*70}")
    print(f"  STEP 5: Train Classifier (v2 — HGB Variants with sample_weight, no SMOTE)")
    print(f"{'='*70}")

    output_dir = Path(config['output_dir'])
    features_path = output_dir / 'features.npz'
    data = np.load(features_path, allow_pickle=True)

    bodies = list(data['bodies']) if 'bodies' in data else []
    active_classes = list(data['active_classes']) if 'active_classes' in data else list(range(7))
    num_classes = len(active_classes)
    class_names = config['dataset_options']['class_names']

    train_X = data['train_features']
    train_y = data['train_labels']
    val_X = data['val_features']
    val_y = data['val_labels']
    test_X = data['test_features']
    test_y = data['test_labels']

    # Derive stats-per-formula from the actual feature matrix (encode_body_
    # distribution_v2 defaults to 12 stats x 5 regions = 60, regardless of the
    # phase3 config values).
    n_formulas = len(bodies) if bodies else 1
    stats_per_formula = train_X.shape[1] // max(1, n_formulas)
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
    from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.metrics import roc_auc_score
    from sklearn.utils.class_weight import compute_sample_weight
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Remove constant features
    variances = np.var(train_X, axis=0)
    non_const = variances > 1e-12
    if non_const.sum() < train_X.shape[1]:
        print(f"  Removing {(~non_const).sum()} constant features...")
    train_X_nc = train_X[:, non_const]
    val_X_nc = val_X[:, non_const]
    test_X_nc = test_X[:, non_const]
    combine_X_nc = np.concatenate([train_X_nc, val_X_nc], axis=0)
    print(f"  Non-constant features: {non_const.sum()} / {train_X.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_X_nc)
    val_scaled = scaler.transform(val_X_nc)
    combine_scaled = np.concatenate([train_scaled, val_scaled], axis=0)

    # ANOVA pre-selection
    anova_preselect = min(1000, train_X_nc.shape[1])
    print(f"  ANOVA pre-selection: top-{anova_preselect} from {train_X_nc.shape[1]} features")
    anova_selector = SelectKBest(f_classif, k=anova_preselect)
    train_anova = anova_selector.fit_transform(train_scaled, train_y)
    combine_anova = anova_selector.transform(combine_scaled)
    print(f"  After ANOVA: {train_anova.shape[1]} features")

    # ------------------------------------------------------------------
    # Method / CV grid selection (needed early so MI is only fit when used).
    # MI (mutual_info_classif) selection is dropped by default: it is ~20-100x
    # slower than ANOVA (kNN entropy estimator, single-threaded, recomputed inside
    # every CV fold) yet gives statistically indistinguishable accuracy here. It
    # can be re-enabled via classifier.methods: [..., hgb_mi, hgb_mi_sw].
    # ------------------------------------------------------------------
    clf_cfg = config.get('classifier', {})
    light_cv = clf_cfg.get('light_cv', False)
    if light_cv:
        default_methods = ['hgb_baseline']
        default_K = [200]
        default_configs = [(200, 0.1, 3), (300, 0.05, 5)]
    else:
        default_methods = ['hgb_baseline', 'hgb_sample_weight']
        default_K = [100, 200, 300, 500]
        default_configs = [(100, 0.1, 3), (200, 0.1, 5), (200, 0.05, 3), (300, 0.1, 5), (300, 0.05, 5)]
    enabled_methods = clf_cfg.get('methods', default_methods)
    if 'hgb_baseline' not in enabled_methods:
        enabled_methods = ['hgb_baseline'] + list(enabled_methods)  # always need a fallback best
    hgb_K_values = clf_cfg.get('hgb_K_values', default_K)
    hgb_configs = [tuple(c) for c in clf_cfg.get('hgb_configs', default_configs)]
    need_mi = any(m in enabled_methods for m in ('hgb_mi', 'hgb_mi_sw'))
    print(f"  CV: light_cv={light_cv}, methods={enabled_methods}, "
          f"K={hgb_K_values}, n_configs={len(hgb_configs)}, MI={need_mi}")

    # MI feature selection (only when an MI-based method is enabled).
    mi_pool_size = min(3000, train_X_nc.shape[1])
    mi_select_size = min(1000, mi_pool_size)
    mi_pool_selector = mi_selector = None
    train_mi = combine_mi = None
    if need_mi:
        print(f"  MI pool: top-{mi_pool_size}, then MI top-{mi_select_size}")
        mi_pool_selector = SelectKBest(f_classif, k=mi_pool_size)
        train_mi_pool = mi_pool_selector.fit_transform(train_scaled, train_y)
        mi_selector = SelectKBest(mutual_info_classif, k=mi_select_size)
        train_mi = mi_selector.fit_transform(train_mi_pool, train_y)
        combine_mi_pool = mi_pool_selector.transform(combine_scaled)
        combine_mi = mi_selector.transform(combine_mi_pool)
        print(f"  After MI: {train_mi.shape[1]} features")

    # Sample weights for class imbalance (no SMOTE)
    sample_weights = compute_sample_weight('balanced', combine_y)
    print(f"  Sample weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    best_pipes = {}
    cv_accs = {}
    cv_bal_accs = {}

    def _cv_eval(pipe, X, y, sw=None):
        """Returns (mean_accuracy, mean_balanced_accuracy) across folds.
        If sw is not None, applies sample_weight to training fold only."""
        accs, bal_accs = [], []
        for tr, va in skf.split(X, y):
            fit_params = {}
            if sw is not None and hasattr(pipe, 'named_steps'):
                clf_step = pipe.named_steps.get('clf', None)
                if hasattr(clf_step, 'fit') and 'sample_weight' in clf_step.fit.__code__.co_varnames:
                    fit_params['clf__sample_weight'] = sw[tr]
            pipe.fit(X[tr], y[tr], **fit_params)
            preds = pipe.predict(X[va])
            accs.append(accuracy_score(y[va], preds))
            bal_accs.append(balanced_accuracy_score(y[va], preds))
        return np.mean(accs), np.mean(bal_accs)

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
    test_mi = _apply_pipeline_transform(test_X, 'mi') if need_mi else None

    # --- Method 1: HistGB + ANOVA (baseline) ---
    if 'hgb_baseline' in enabled_methods:
        print(f"\n  --- Method 1: HistGB + ANOVA (baseline) ---", flush=True)
        best_cv_hgb, best_bal_hgb, best_hgb_cfg, best_hgb_K = 0, 0, None, None
        for K in hgb_K_values:
            if K > anova_preselect: continue
            for max_iter, lr, max_d in hgb_configs:
                pipe = Pipeline([
                    ('select', SelectKBest(f_classif, k=K)),
                    ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                           max_depth=max_d, random_state=42)),
                ])
                cv_acc, cv_bal = _cv_eval(pipe, combine_anova, combine_y)
                if cv_bal > best_bal_hgb:
                    best_cv_hgb, best_bal_hgb, best_hgb_cfg, best_hgb_K = cv_acc, cv_bal, (max_iter, lr, max_d), K
        print(f"  Best: max_iter={best_hgb_cfg[0]}, lr={best_hgb_cfg[1]}, max_depth={best_hgb_cfg[2]}, K={best_hgb_K}, cv_acc={best_cv_hgb:.3f}, cv_bal={best_bal_hgb:.3f}")
        pipe_hgb_base = Pipeline([
            ('select', SelectKBest(f_classif, k=best_hgb_K)),
            ('clf', HistGradientBoostingClassifier(max_iter=best_hgb_cfg[0], learning_rate=best_hgb_cfg[1],
                                                   max_depth=best_hgb_cfg[2], random_state=42)),
        ])
        pipe_hgb_base.fit(combine_anova, combine_y)
        preds = pipe_hgb_base.predict(test_anova)
        acc = accuracy_score(test_y, preds)
        bal_acc = balanced_accuracy_score(test_y, preds)
        auc = _calc_auc(pipe_hgb_base, test_anova, test_y)
        print(f"  Test: acc={acc:.3f}, bal_acc={bal_acc:.3f}, AUC={auc:.3f}")
        results['hgb_baseline'] = {'acc': float(acc), 'bal_acc': float(bal_acc), 'auc': float(auc)}
        best_pipes['hgb_baseline'] = pipe_hgb_base
        cv_accs['hgb_baseline'] = best_cv_hgb
        cv_bal_accs['hgb_baseline'] = best_bal_hgb

    # --- Method 2: HistGB + ANOVA + sample_weight ---
    if 'hgb_sample_weight' in enabled_methods:
        print(f"\n  --- Method 2: HistGB + ANOVA + sample_weight ---", flush=True)
        best_cv_hgb_sw, best_bal_hgb_sw, best_hgb_sw_cfg, best_hgb_sw_K = 0, 0, None, None
        for K in hgb_K_values:
            if K > anova_preselect: continue
            for max_iter, lr, max_d in hgb_configs:
                pipe = Pipeline([
                    ('select', SelectKBest(f_classif, k=K)),
                    ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                           max_depth=max_d, random_state=42)),
                ])
                cv_acc, cv_bal = _cv_eval(pipe, combine_anova, combine_y, sw=sample_weights)
                if cv_bal > best_bal_hgb_sw:
                    best_cv_hgb_sw, best_bal_hgb_sw, best_hgb_sw_cfg, best_hgb_sw_K = cv_acc, cv_bal, (max_iter, lr, max_d), K
        print(f"  Best: cfg={best_hgb_sw_cfg}, K={best_hgb_sw_K}, cv_acc={best_cv_hgb_sw:.3f}, cv_bal={best_bal_hgb_sw:.3f}")
        pipe_hgb_sw = Pipeline([
            ('select', SelectKBest(f_classif, k=best_hgb_sw_K)),
            ('clf', HistGradientBoostingClassifier(max_iter=best_hgb_sw_cfg[0], learning_rate=best_hgb_sw_cfg[1],
                                                   max_depth=best_hgb_sw_cfg[2], random_state=42)),
        ])
        pipe_hgb_sw.fit(combine_anova, combine_y, clf__sample_weight=sample_weights)
        preds = pipe_hgb_sw.predict(test_anova)
        acc = accuracy_score(test_y, preds)
        bal_acc = balanced_accuracy_score(test_y, preds)
        auc = _calc_auc(pipe_hgb_sw, test_anova, test_y)
        print(f"  Test: acc={acc:.3f}, bal_acc={bal_acc:.3f}, AUC={auc:.3f}")
        results['hgb_sample_weight'] = {'acc': float(acc), 'bal_acc': float(bal_acc), 'auc': float(auc)}
        best_pipes['hgb_sample_weight'] = pipe_hgb_sw
        cv_accs['hgb_sample_weight'] = best_cv_hgb_sw
        cv_bal_accs['hgb_sample_weight'] = best_bal_hgb_sw

    # --- Method 3: HistGB + MI ---
    if 'hgb_mi' in enabled_methods:
        print(f"\n  --- Method 3: HistGB + MI ---", flush=True)
        best_cv_hgb_mi, best_bal_hgb_mi, best_hgb_mi_cfg, best_hgb_mi_K = 0, 0, None, None
        for K in hgb_K_values:
            if K > mi_select_size: continue
            for max_iter, lr, max_d in hgb_configs:
                pipe = Pipeline([
                    ('select', SelectKBest(mutual_info_classif, k=K)),
                    ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                           max_depth=max_d, random_state=42)),
                ])
                cv_acc, cv_bal = _cv_eval(pipe, combine_mi, combine_y)
                if cv_bal > best_bal_hgb_mi:
                    best_cv_hgb_mi, best_bal_hgb_mi, best_hgb_mi_cfg, best_hgb_mi_K = cv_acc, cv_bal, (max_iter, lr, max_d), K
        print(f"  Best: cfg={best_hgb_mi_cfg}, K={best_hgb_mi_K}, cv_acc={best_cv_hgb_mi:.3f}, cv_bal={best_bal_hgb_mi:.3f}")
        pipe_hgb_mi = Pipeline([
            ('select', SelectKBest(mutual_info_classif, k=best_hgb_mi_K)),
            ('clf', HistGradientBoostingClassifier(max_iter=best_hgb_mi_cfg[0], learning_rate=best_hgb_mi_cfg[1],
                                                   max_depth=best_hgb_mi_cfg[2], random_state=42)),
        ])
        pipe_hgb_mi.fit(combine_mi, combine_y)
        preds = pipe_hgb_mi.predict(test_mi)
        acc = accuracy_score(test_y, preds)
        bal_acc = balanced_accuracy_score(test_y, preds)
        auc = _calc_auc(pipe_hgb_mi, test_mi, test_y)
        print(f"  Test: acc={acc:.3f}, bal_acc={bal_acc:.3f}, AUC={auc:.3f}")
        results['hgb_mi'] = {'acc': float(acc), 'bal_acc': float(bal_acc), 'auc': float(auc)}
        best_pipes['hgb_mi'] = pipe_hgb_mi
        cv_accs['hgb_mi'] = best_cv_hgb_mi
        cv_bal_accs['hgb_mi'] = best_bal_hgb_mi

    # --- Method 4: HistGB + MI + sample_weight ---
    if 'hgb_mi_sw' in enabled_methods:
        print(f"\n  --- Method 4: HistGB + MI + sample_weight ---", flush=True)
        best_cv_hgb_mi_sw, best_bal_hgb_mi_sw, best_hgb_mi_sw_cfg, best_hgb_mi_sw_K = 0, 0, None, None
        for K in hgb_K_values:
            if K > mi_select_size: continue
            for max_iter, lr, max_d in hgb_configs:
                pipe = Pipeline([
                    ('select', SelectKBest(mutual_info_classif, k=K)),
                    ('clf', HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr,
                                                           max_depth=max_d, random_state=42)),
                ])
                cv_acc, cv_bal = _cv_eval(pipe, combine_mi, combine_y, sw=sample_weights)
                if cv_bal > best_bal_hgb_mi_sw:
                    best_cv_hgb_mi_sw, best_bal_hgb_mi_sw, best_hgb_mi_sw_cfg, best_hgb_mi_sw_K = cv_acc, cv_bal, (max_iter, lr, max_d), K
        print(f"  Best: cfg={best_hgb_mi_sw_cfg}, K={best_hgb_mi_sw_K}, cv_acc={best_cv_hgb_mi_sw:.3f}, cv_bal={best_bal_hgb_mi_sw:.3f}")
        pipe_hgb_mi_sw = Pipeline([
            ('select', SelectKBest(mutual_info_classif, k=best_hgb_mi_sw_K)),
            ('clf', HistGradientBoostingClassifier(max_iter=best_hgb_mi_sw_cfg[0], learning_rate=best_hgb_mi_sw_cfg[1],
                                                   max_depth=best_hgb_mi_sw_cfg[2], random_state=42)),
        ])
        pipe_hgb_mi_sw.fit(combine_mi, combine_y, clf__sample_weight=sample_weights)
        preds = pipe_hgb_mi_sw.predict(test_mi)
        acc = accuracy_score(test_y, preds)
        bal_acc = balanced_accuracy_score(test_y, preds)
        auc = _calc_auc(pipe_hgb_mi_sw, test_mi, test_y)
        print(f"  Test: acc={acc:.3f}, bal_acc={bal_acc:.3f}, AUC={auc:.3f}")
        results['hgb_mi_sw'] = {'acc': float(acc), 'bal_acc': float(bal_acc), 'auc': float(auc)}
        best_pipes['hgb_mi_sw'] = pipe_hgb_mi_sw
        cv_accs['hgb_mi_sw'] = best_cv_hgb_mi_sw
        cv_bal_accs['hgb_mi_sw'] = best_bal_hgb_mi_sw

    # ==================================================================
    # Summary & Save
    # ==================================================================
    print(f"\n  {'='*60}")
    print(f"  COMPARISON SUMMARY (v2 — HGB Variants with sample_weight)")
    print(f"  {'='*60}")

    # Select best by balanced accuracy
    best_method = max(results, key=lambda k: results[k].get('bal_acc', results[k].get('acc', 0)))
    for method, res in results.items():
        marker = " <-- BEST" if method == best_method else ""
        bal_str = f", bal_acc={res['bal_acc']:.3f}" if 'bal_acc' in res else ""
        cv_bal_str = f", cv_bal={cv_bal_accs.get(method, 0):.3f}" if method in cv_bal_accs else ""
        cv_info = f" cv_acc={cv_accs.get(method, 0):.3f}{cv_bal_str}"
        print(f"    {method:25s}: acc={res['acc']:.3f}{bal_str}, AUC={res['auc']:.3f}{cv_info}{marker}")

    # Per-class accuracy
    all_per_class = {}
    for method_name, pipe in best_pipes.items():
        use_mi = 'mi' in method_name
        test_data = test_mi if use_mi else test_anova
        mpreds = pipe.predict(test_data)
        pc = {}
        for c in range(num_classes):
            mask = test_y == c
            if mask.sum() > 0:
                name = class_names[c]
                pc[name] = float((mpreds[mask] == c).mean())
        all_per_class[method_name] = pc

    # Save results
    import joblib
    version = 2
    version_dir = output_dir / f'v{version}'
    version_dir.mkdir(exist_ok=True)

    best_pipe = best_pipes[best_method]
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
    joblib.dump(clf_save_data, output_dir / 'best_classifier.pkl')
    joblib.dump(clf_save_data, output_dir / 'best_classifier_balanced.pkl')
    joblib.dump(clf_save_data, version_dir / f'best_classifier_v{version}.pkl')

    for mn, mpipe in best_pipes.items():
        save_data = {
            'pipe': mpipe, 'non_const': non_const, 'method': mn,
            'scaler': scaler, 'anova_selector': anova_selector,
            'mi_pool_selector': mi_pool_selector, 'mi_selector': mi_selector,
            'sample_weights': sample_weights,
            'version': version,
        }
        joblib.dump(save_data, version_dir / f'classifier_{mn}_v{version}.pkl')

    best_results = {
        'version': version,
        'method': best_method,
        'test_accuracy': results[best_method]['acc'],
        'test_balanced_accuracy': results[best_method].get('bal_acc', 0),
        'test_auc': results[best_method]['auc'],
        'per_class_test': all_per_class.get(best_method, {}),
        'all_methods': results,
        'all_per_class': all_per_class,
        'cv_accs': {k: float(v) for k, v in cv_accs.items()},
        'cv_bal_accs': {k: float(v) for k, v in cv_bal_accs.items()},
    }
    with open(output_dir / 'classifier_results.json', 'w') as f:
        json.dump(best_results, f, indent=2)
    with open(version_dir / f'classifier_results_v{version}.json', 'w') as f:
        json.dump(best_results, f, indent=2)

    # Save class names
    with open(output_dir / 'class_names.json', 'w') as f:
        json.dump(class_names, f)

    print(f"\n  Results saved to {output_dir}")
    best_res = results[best_method]
    bal_str = f", bal_acc={best_res['bal_acc']:.3f}" if 'bal_acc' in best_res else ""
    print(f"  Best method: {best_method} (acc={best_res['acc']:.3f}{bal_str}, AUC={best_res['auc']:.3f})")


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
    if not validated_path.exists():
        print("  No validated formulas found, skipping report")
        return

    formulas = json.load(open(validated_path))
    formulas.sort(key=lambda f: f.get('quality_anova_f', f.get('full_res_balanced_accuracy', 0)), reverse=True)

    dist_cfg = config.get('phase3', {}).get('distribution_stats', {})
    n_stats = dist_cfg.get('n_stats', 16)
    n_regions = dist_cfg.get('n_regions', 7)
    stats_per_formula = n_stats * n_regions

    class_names = config['dataset_options']['class_names']

    # Load classifier for feature analysis
    classifier_path = output_dir / 'best_classifier.pkl'
    clf_data = None
    if classifier_path.exists():
        import joblib
        clf_data = joblib.load(classifier_path)

    lines = []
    lines.append("=" * 70)
    lines.append("THIRDDATA (BUSI) SYMBOLIC FEATURE INTERPRETABILITY REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total validated formulas: {len(formulas)}")
    lines.append(f"Feature dimensions: {len(formulas)} formulas x {stats_per_formula} stats ({n_stats} stats x {n_regions} regions) = {len(formulas) * stats_per_formula}")
    if clf_data is not None:
        lines.append(f"Best classifier method: {clf_data.get('method', 'unknown')}")
    lines.append("")

    # ---- Section 1: Top-20 formulas ----
    lines.append("TOP-20 FORMULAS BY ANOVA-F QUALITY (step 3):")
    lines.append("-" * 50)
    for i, f in enumerate(formulas[:20]):
        q = f.get('quality_anova_f', 0)
        lines.append(f"  {i+1:2d}. F={q:8.2f}  {f['str']}")

    # ---- Section 2: Classifier feature analysis ----
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

    if clf_data is not None:
        pipe = clf_data['pipe']
        method = clf_data.get('method', 'unknown')
        non_const_mask = clf_data.get('non_const')
        anova_selector = clf_data.get('anova_selector')
        mi_pool_selector = clf_data.get('mi_pool_selector')
        mi_selector = clf_data.get('mi_selector')

        if hasattr(pipe, 'named_steps'):
            clf = pipe.named_steps['clf']
            selector = pipe.named_steps.get('select', None)
        else:
            clf = pipe
            selector = None

        lines.append(f"  Method: {method}")

        # Determine which feature set was used
        use_mi = 'mi' in method
        if use_mi and mi_selector is not None:
            top_selector = mi_selector
            pre_selector = mi_pool_selector
            lines.append(f"  Feature selection: MI (top-{selector.get_params().get('k', '?') if selector else '?'})")
        elif anova_selector is not None:
            top_selector = anova_selector
            pre_selector = None
            lines.append(f"  Feature selection: ANOVA (top-{selector.get_params().get('k', '?') if selector else '?'})")
        else:
            top_selector = None
            pre_selector = None

        # Permutation importance (since HistGB doesn't have feature_importances_)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            imp_label = "importance"
        else:
            importances = None
            imp_label = None

        if selector is not None and hasattr(selector, 'get_support'):
            selected_idx = selector.get_support(indices=True)
            lines.append(f"  Selected {len(selected_idx)} features")

            # Compute permutation importance if no direct importances
            # NOTE: This is slow (500 features × 5 repeats = 2500 model predictions)
            # Configurable via config['step6']['permutation_importance'] (default: false)
            step6_cfg = config.get('step6', {})
            use_perm_imp = step6_cfg.get('permutation_importance', False)

            if importances is None and use_perm_imp:
                lines.append(f"  (HistGB has no feature_importances_; computing Permutation Importance...)")
                try:
                    from sklearn.inspection import permutation_importance
                    features_path = output_dir / 'features.npz'
                    if features_path.exists():
                        print(f"  Loading features for permutation importance...", flush=True)
                        data = np.load(features_path, allow_pickle=True)
                        test_X = data['test_features']
                        test_y = data['test_labels']
                        print(f"  test_X shape: {test_X.shape}, applying transforms...", flush=True)

                        # Apply same transforms: non_const -> scaler -> anova/mi -> select
                        scaler = clf_data.get('scaler')
                        if scaler is not None and non_const_mask is not None:
                            print(f"  Applying non_const mask + scaler transform...", flush=True)
                            test_X_t = scaler.transform(test_X[:, non_const_mask])
                        else:
                            test_X_t = test_X
                        del test_X
                        gc.collect()

                        if pre_selector is not None:
                            print(f"  Applying pre_selector transform...", flush=True)
                            test_X_t = pre_selector.transform(test_X_t)
                        if top_selector is not None:
                            print(f"  Applying top_selector transform...", flush=True)
                            test_X_t = top_selector.transform(test_X_t)
                        # Now apply the final selector in the pipe
                        test_X_final = test_X_t
                        if hasattr(pipe, 'named_steps') and 'select' in pipe.named_steps:
                            print(f"  Applying pipe.select transform...", flush=True)
                            test_X_final = pipe.named_steps['select'].transform(test_X_t)
                        del test_X_t
                        gc.collect()

                        n_reps = step6_cfg.get('perm_n_repeats', 3)
                        print(f"  Computing permutation importance (n_jobs=1, n_repeats={n_reps}, may take a few minutes)...", flush=True)
                        # n_jobs=1: joblib subprocesses crash due to NumPy 1.x vs 2.0.2 conflict
                        perm_result = permutation_importance(
                            clf, test_X_final, test_y, n_repeats=n_reps, random_state=42, n_jobs=1
                        )
                        importances = perm_result.importances_mean
                        imp_label = "perm_importance"
                        lines.append(f"  Computed Permutation Importance on {test_X_final.shape[0]} test samples")
                        del test_X_final, test_y
                        gc.collect()
                except Exception as e:
                    lines.append(f"  (Permutation Importance failed: {e})")
            elif importances is None:
                # Fallback: use ANOVA/MI scores as feature ranking (fast, no model prediction needed)
                lines.append(f"  (HistGB has no feature_importances_; using selection scores as ranking)")
                if use_mi and mi_selector is not None:
                    sel_scores = mi_selector.scores_
                    sel_mask = mi_selector.get_support(indices=True)
                    # Map through pipe.select
                    if selector is not None:
                        pipe_sel_mask = selector.get_support(indices=True)
                        importances = sel_scores[sel_mask][pipe_sel_mask]
                        imp_label = "mi_score"
                elif anova_selector is not None:
                    sel_scores = anova_selector.scores_
                    sel_mask = anova_selector.get_support(indices=True)
                    if selector is not None:
                        pipe_sel_mask = selector.get_support(indices=True)
                        importances = sel_scores[sel_mask][pipe_sel_mask]
                        imp_label = "anova_fscore"

            if importances is not None:
                top_feat_idx = np.argsort(importances)[::-1][:20]
                lines.append(f"  Top-20 features by {imp_label}:")

                # Build the full reverse mapping chain:
                # pipe.select input -> [mi_selector -> mi_pool_selector] -> non_const -> original
                # For ANOVA: pipe.select input -> anova_selector -> non_const -> original
                # For MI:    pipe.select input -> mi_selector -> mi_pool_selector -> non_const -> original
                non_const_indices = np.where(non_const_mask)[0] if non_const_mask is not None else None

                for rank, fi in enumerate(top_feat_idx):
                    # Step 1: index in pipe.select input space
                    idx = selected_idx[fi]

                    # Step 2: map through top_selector (mi_selector or anova_selector)
                    if top_selector is not None and hasattr(top_selector, 'get_support'):
                        top_mask = top_selector.get_support(indices=True)
                        if idx < len(top_mask):
                            idx = top_mask[idx]

                    # Step 3: map through pre_selector (mi_pool_selector) if exists
                    if pre_selector is not None and hasattr(pre_selector, 'get_support'):
                        pre_mask = pre_selector.get_support(indices=True)
                        if idx < len(pre_mask):
                            idx = pre_mask[idx]

                    # Step 4: map through non_const mask
                    if non_const_indices is not None and idx < len(non_const_indices):
                        orig_idx = non_const_indices[idx]
                    else:
                        orig_idx = idx

                    formula_idx = orig_idx // stats_per_formula
                    stat_idx = orig_idx % stats_per_formula
                    sname = stat_names[stat_idx] if stat_idx < len(stat_names) else f'stat_{stat_idx}'
                    if formula_idx < len(formulas):
                        lines.append(f"    {rank+1:2d}. formula[{formula_idx}].{sname} ({imp_label}={importances[fi]:.4f})")
                    else:
                        lines.append(f"    {rank+1:2d}. feat[{orig_idx}].{sname} ({imp_label}={importances[fi]:.4f})")
        else:
            lines.append("  (No feature selector available in pipeline)")
    else:
        lines.append("  (No trained classifier found)")

    # ---- Section 3: Operator frequency analysis ----
    lines.append("")
    lines.append("OPERATOR FREQUENCY ANALYSIS:")
    lines.append("-" * 50)
    op_counts = {}
    for f in formulas:
        for tok in f['str'].split():
            if tok in TENSOR_OPERATORS:
                op_counts[tok] = op_counts.get(tok, 0) + 1
    total_ops = sum(op_counts.values())
    lines.append(f"  Total operator occurrences: {total_ops}")
    lines.append(f"  Unique operators: {len(op_counts)}")
    lines.append("")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        is_ham = " [BRAIN-MRI]" if op in BRAIN_OPERATORS else ""
        pct = 100.0 * count / total_ops if total_ops > 0 else 0
        lines.append(f"  {op:25s}: {count:4d} ({pct:5.1f}%){is_ham}")

    # ---- Section 4: Terminal frequency analysis ----
    lines.append("")
    lines.append("TERMINAL FREQUENCY ANALYSIS:")
    lines.append("-" * 50)
    term_counts = {}
    for f in formulas:
        for tok in f['str'].split():
            if tok.startswith('I_'):
                term_counts[tok] = term_counts.get(tok, 0) + 1
    total_terms = sum(term_counts.values())
    lines.append(f"  Total terminal occurrences: {total_terms}")
    lines.append(f"  Unique terminals: {len(term_counts)}")
    lines.append("")
    for term, count in sorted(term_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total_terms if total_terms > 0 else 0
        lines.append(f"  {term:25s}: {count:4d} ({pct:5.1f}%)")

    # ---- Section 5: Root operator analysis ----
    lines.append("")
    lines.append("ROOT OPERATOR FREQUENCY ANALYSIS:")
    lines.append("-" * 50)
    root_counts = {}
    for f in formulas:
        root = f.get('root_op', '')
        if root:
            root_counts[root] = root_counts.get(root, 0) + 1
    for root, count in sorted(root_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / len(formulas) if formulas else 0
        lines.append(f"  {root:25s}: {count:4d} ({pct:5.1f}%)")

    # ---- Section 6: Medical interpretation of top formulas ----
    lines.append("")
    lines.append("RADIOLOGICAL INTERPRETATION OF TOP FORMULAS:")
    lines.append("-" * 50)
    medical_interpretations = {
        # Brain-MRI symmetry/texture operators
        'lr_symmetry': 'Left-right mirror difference — tumors break bilateral brain symmetry',
        'tb_symmetry': 'Top-bottom mirror difference — superior/inferior asymmetry',
        'diag_symmetry': 'Diagonal (transpose) asymmetry',
        'border_sharp': 'Gradient magnitude — mass effect / tumor margin sharpness',
        'local_range': 'Local intensity range — tissue heterogeneity (necrosis/edema vs. solid)',
        'texture_entropy': 'Local texture entropy — disordered tumor microstructure',
        'local_var': 'Local variance — fine texture heterogeneity',
        # Generic spatial/texture operators
        'edge_mag': 'Gradient magnitude — detects tumor / ventricle borders',
        'edge_x': 'Horizontal edge — vertical borders',
        'edge_y': 'Vertical edge — horizontal borders',
        'blur': 'Gaussian blur — smoothing for noise reduction',
        'normalize': 'Normalization — standardize intensity range',
        'sigmoid': 'Sigmoid activation — nonlinear mapping',
        'relu': 'ReLU activation — nonlinear mapping',
        'abs': 'Absolute value — magnitude features',
        'sqrt_abs': 'Sqrt of absolute — compress dynamic range',
        'log1p_abs': 'Log1p of absolute — compress dynamic range',
        'negate': 'Negation — invert signal',
        'opening': 'Morphological opening — remove small bright specks',
        'closing': 'Morphological closing — fill small dark holes',
        'dilate': 'Dilation — expand bright (hyperintense) regions',
        'downsample_2x': 'Downsample 2x — multi-scale analysis',
        'downsample_4x': 'Downsample 4x — multi-scale analysis',
        'gabor_0': 'Gabor filter 0° — directional texture',
        'gabor_45': 'Gabor filter 45° — directional texture',
        'gabor_90': 'Gabor filter 90° — directional texture',
        'gabor_mag': 'Gabor magnitude — texture energy',
        'laplacian': 'Laplacian — second-derivative edges / ring structures',
        'dog': 'Difference of Gaussians — blob / lesion detection',
        'lbp_like': 'Local binary pattern — fine texture descriptor',
        'local_std_5x5': 'Local std — texture heterogeneity',
        'local_contrast': 'Local contrast — lesion conspicuity',
        'flip_h': 'Horizontal flip — symmetry analysis',
        'flip_v': 'Vertical flip — symmetry analysis',
        # Grayscale terminals
        'I_GRAY': 'Raw intensity — overall signal (hyper/hypointensity)',
        'I_BLUR': 'Smoothed intensity — coarse anatomy',
        'I_GRAD': 'Gradient magnitude channel — borders / mass margins',
        'I_LOCALSTD': 'Local std channel — texture heterogeneity',
        'I_LOG': 'Laplacian channel — blob / ring-enhancing structures',
        'I_LBP': 'LBP channel — fine texture',
    }

    for i, f in enumerate(formulas[:10]):
        lines.append(f"\n  Formula {i+1}: {f['str']}")
        lines.append(f"  ANOVA-F quality (step 3): {f.get('quality_anova_f', 0):.2f}")
        lines.append(f"  Interpretation:")
        tokens = f['str'].split()
        interpreted = []
        for tok in tokens:
            if tok in medical_interpretations:
                interpreted.append(f"    - {tok}: {medical_interpretations[tok]}")
        if interpreted:
            lines.extend(interpreted)
        else:
            lines.append("    (No specific medical interpretation available)")

    # ---- Section 7: Summary statistics ----
    lines.append("")
    lines.append("FORMULA STATISTICS:")
    lines.append("-" * 50)
    formula_lengths = [len(f['str'].split()) for f in formulas]
    quals = [f.get('quality_anova_f', 0) for f in formulas]
    lines.append(f"  Formula length (tokens): min={min(formula_lengths)}, max={max(formula_lengths)}, avg={np.mean(formula_lengths):.1f}")
    lines.append(f"  ANOVA-F quality: min={min(quals):.2f}, max={max(quals):.2f}, avg={np.mean(quals):.2f}")

    report_text = "\n".join(lines)
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f"\n  Report saved to {report_path}")


# ======================================================================
# Main
# ======================================================================

STEPS = {
    0: step0_validate_dataset,
    1: step1_phase1,
    2: step2_merge,
    3: step3_validate,
    4: step4_extract_features,
    5: step5_train_classifier,
    6: step6_report,
}


def main():
    parser = argparse.ArgumentParser(description='ThirdData (BUSI) Symbolic Pipeline')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--start_step', type=int, default=0, help='Start from this step')
    parser.add_argument('--end_step', type=int, default=6, help='End at this step')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cuda:0'))

    print(f"\n{'='*70}")
    print(f"  ThirdData (BUSI) Symbolic Feature Discovery Pipeline")
    print(f"  Dataset: {config['dataset']}")
    print(f"  Device: {device}")
    print(f"  Steps: {args.start_step} → {args.end_step}")
    print(f"{'='*70}")

    for step_id in range(args.start_step, args.end_step + 1):
        if step_id in STEPS:
            STEPS[step_id](config, device)
        else:
            print(f"\n  Step {step_id} not found, skipping")

    print(f"\n{'='*70}")
    print(f"  Pipeline Complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
