"""
Fracture-specific RL Environment for Symbolic Feature Discovery.

Extends TensorVSREnvironmentLargeBank with:
  - HBFMID dataset integration (YOLO format, 10 fracture classes)
  - Fracture-specific terminals (I_NEG, I_BONE, I_SOFT, I_EDGE_PRIOR)
  - Fracture-specific operators (from fracture_operators.py)
  - Class-balanced evaluation for imbalanced dataset
  - Hierarchical reward (superclass → fine-grained)
  - Medical interpretability bonus
"""

import os
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any, Optional
from collections import Counter

from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS
from src.symbolic.fracture_operators import register_fracture_operators, FRACTURE_OPERATORS
from src.symbolic.large_feature_bank import LargeFeatureBank
from src.rl.tensor_action_masking import TensorActionMasker
from src.rl.rpn_grammar_mask import RPNGrammarMask
from src.data.fracture_loader import (
    HBFMIDDataModule, build_fracture_data_batch,
    build_fracture_superclass_mapping, FRACTURE_NAMES,
)


register_fracture_operators(TENSOR_OPERATORS)


class FractureTokenVocabulary:
    """Vocabulary for fracture-specific VSR.

    Terminals include X-ray specific channels:
      I_R, I_G, I_B, I_GRAY, I_NEG, I_BONE, I_SOFT, I_EDGE_PRIOR,
      I_H, I_S, I_r, I_g, I_RG, I_BY
    """

    def __init__(self, exclude_operators=None, extra_terminals=None):
        special_tokens = ['START', 'END', 'PAD']

        exclude = set(exclude_operators or [])
        operator_tokens = [op for op in TENSOR_OPERATORS.keys() if op not in exclude]

        terminal_tokens = [
            'I_R', 'I_G', 'I_B', 'I_GRAY',
            'I_NEG', 'I_BONE', 'I_SOFT', 'I_EDGE_PRIOR',
            'I_H', 'I_S', 'I_r', 'I_g', 'I_RG', 'I_BY',
        ]

        if extra_terminals:
            terminal_tokens.extend(extra_terminals)

        self.tokens = special_tokens + operator_tokens + terminal_tokens
        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(self.tokens)}

    def encode(self, token: str) -> int:
        return self.token_to_idx[token]

    def decode(self, idx: int) -> str:
        return self.idx_to_token[idx]

    def __len__(self):
        return len(self.tokens)


class FractureVSREnvironment(gym.Env):
    """RL environment for discovering symbolic fracture features.

    Key differences from ImageNet environment:
    1. 10 fracture classes (not 1000)
    2. X-ray specific terminals (I_NEG, I_BONE, I_SOFT, I_EDGE_PRIOR)
    3. Fracture-specific operators (line detectors, cortical continuity, etc.)
    4. Class-balanced evaluation (weighted accuracy for imbalanced data)
    5. Hierarchical reward (superclass first, then fine-grained)
    6. Medical interpretability bonus in reward
    """

    def __init__(
        self,
        data_loader: torch.utils.data.DataLoader,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        super().__init__()

        self.data_loader = data_loader
        self.config = config
        self.device = device

        model_cfg = config['model']
        train_cfg = config['training']
        strategy_cfg = config.get('strategy', {})

        self.max_depth = model_cfg.get('max_depth', 6)
        self.max_sequence_length = model_cfg.get('max_sequence_length', 15)
        self.num_classes = train_cfg.get('num_classes', 10)
        self.feature_bank_size = strategy_cfg.get('feature_bank_size', 500)
        self.min_accuracy = strategy_cfg.get('min_accuracy_threshold', 0.10)
        self.correlation_threshold = strategy_cfg.get('correlation_threshold', 0.90)
        self.length_penalty = train_cfg.get('length_penalty', 0.01)
        self.diversity_penalty_coef = strategy_cfg.get('diversity_penalty', 0.15)

        self.reward_type = strategy_cfg.get('reward_type', 'balanced_accuracy')
        self.use_hierarchical_eval = strategy_cfg.get('use_hierarchical_eval', True)
        self.medical_bonus = strategy_cfg.get('medical_interpretability_bonus', 0.05)

        self.superclass_mapping = build_fracture_superclass_mapping()
        self.num_superclasses = 3

        self.vocabulary = FractureTokenVocabulary(
            exclude_operators=config.get('exclude_operators', []),
        )

        self.action_space = gym.spaces.Discrete(len(self.vocabulary))
        self.observation_space = gym.spaces.Box(
            low=0, high=len(self.vocabulary),
            shape=(self.max_sequence_length,), dtype=int
        )

        self.action_masker = TensorActionMasker(self.vocabulary, max_depth=self.max_depth)
        self.rpn_masker = RPNGrammarMask(self.vocabulary, max_sequence_length=self.max_sequence_length)

        self.feature_bank = LargeFeatureBank(
            max_size=self.feature_bank_size,
            min_accuracy=self.min_accuracy,
            correlation_threshold=self.correlation_threshold,
            num_classes=self.num_classes,
            device=self.device,
            adaptive_threshold=strategy_cfg.get('adaptive_threshold', True),
            threshold_warmup_fraction=strategy_cfg.get('threshold_warmup_fraction', 0.5),
        )

        self.cached_images = None
        self.cached_labels = None
        self._cache_validation_set()

        self.current_sequence = []
        self.step_count = 0
        self.episode_count = 0

        self._medical_operators = set(FRACTURE_OPERATORS.keys()) | {
            'edge_mag', 'edge_x', 'edge_y', 'local_std_5x5',
            'dog', 'laplacian', 'corner_harris', 'gabor_mag',
        }

    def _cache_validation_set(self):
        cached_images = []
        cached_labels = []
        for images, labels in self.data_loader:
            cached_images.append(images)
            cached_labels.append(labels)
        if not cached_images:
            raise RuntimeError("Empty validation cache")
        self.cached_images = torch.cat(cached_images, dim=0)
        self.cached_labels = torch.cat(cached_labels, dim=0).long()
        print(f"[FractureEnv] Cached {self.cached_images.shape[0]} images")

    def get_data_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.config['training'].get('batch_size', 32)
        batch_size = min(batch_size, self.cached_images.size(0))
        indices = torch.randint(0, self.cached_images.size(0), (batch_size,))
        images = self.cached_images[indices].to(self.device)
        labels = self.cached_labels[indices].to(self.device)
        terminal_values = build_fracture_data_batch(images, self.device)
        return terminal_values, labels

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_sequence = [self.vocabulary.encode('START')]
        self.step_count = 0
        return self._get_observation(), {}

    def step(self, action):
        self.current_sequence.append(action)
        self.step_count += 1
        token_str = self.vocabulary.decode(action)
        terminated = token_str == 'END'
        truncated = self.step_count >= self.max_sequence_length
        reward = 0.0
        info = {}
        if terminated or truncated:
            reward, info = self._calculate_reward()
            self.episode_count += 1
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        obs = np.array(self.current_sequence, dtype=int)
        if len(obs) < self.max_sequence_length:
            obs = np.pad(obs, (0, self.max_sequence_length - len(obs)),
                         mode='constant', constant_values=self.vocabulary.encode('PAD'))
        else:
            obs = obs[:self.max_sequence_length]
        return obs

    def get_action_mask(self):
        current_tokens = [
            t for t in self.current_sequence
            if self.vocabulary.decode(t) not in ['START', 'PAD']
        ]
        mask = self.rpn_masker.get_valid_actions(current_tokens, device=self.device)
        return mask

    def _execute_formula(self, formula_tokens, data_batch):
        decoded = [self.vocabulary.decode(t) for t in formula_tokens]
        stack = []
        for token in decoded:
            if token in data_batch:
                stack.append(data_batch[token])
            elif token in TENSOR_OPERATORS:
                op_func, arity, _ = TENSOR_OPERATORS[token]
                if len(stack) < arity:
                    raise ValueError(f"Not enough operands for {token}")
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
        output = stack[0]
        output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
        return torch.clamp(output, -1e4, 1e4)

    def _compute_balanced_accuracy(self, features, labels):
        """Compute class-balanced accuracy for imbalanced fracture dataset."""
        labels = labels.long()
        n_samples = features.size(0)
        n_train = int(0.7 * n_samples)
        indices = torch.randperm(n_samples, device=self.device)
        train_feat = features[indices[:n_train]]
        train_labels = labels[indices[:n_train]].long()
        val_feat = features[indices[n_train:]]
        val_labels = labels[indices[n_train:]].long()

        classifier = nn.Linear(features.size(1), self.num_classes).to(self.device)
        train_labels = train_labels.to(torch.int64)
        val_labels = val_labels.to(torch.int64)
        class_counts = torch.bincount(train_labels, minlength=self.num_classes).float()
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[train_labels]
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)
        classifier.train()
        for _ in range(30):
            optimizer.zero_grad()
            logits = classifier(train_feat)
            loss = criterion(logits, train_labels)
            loss.backward()
            optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_feat)
            val_preds = val_logits.argmax(dim=1)

            per_class_acc = []
            for c in range(self.num_classes):
                mask = val_labels == c
                if mask.sum() > 0:
                    acc = (val_preds[mask] == c).float().mean().item()
                    per_class_acc.append(acc)
            balanced_acc = np.mean(per_class_acc) if per_class_acc else 0.0

            standard_acc = (val_preds == val_labels).float().mean().item()

        return balanced_acc, standard_acc

    def _compute_hierarchical_accuracy(self, features, labels):
        """Compute hierarchical accuracy: superclass first, then fine-grained."""
        labels = labels.long()
        n_samples = features.size(0)
        n_train = int(0.7 * n_samples)
        indices = torch.randperm(n_samples, device=self.device)
        train_feat = features[indices[:n_train]]
        train_labels = labels[indices[:n_train]].long()
        val_feat = features[indices[n_train:]]
        val_labels = labels[indices[n_train:]].long()

        sup_labels_train = torch.tensor(
            [self.superclass_mapping.get(l.item(), 0) for l in train_labels],
            device=self.device, dtype=torch.int64
        )
        sup_labels_val = torch.tensor(
            [self.superclass_mapping.get(l.item(), 0) for l in val_labels],
            device=self.device, dtype=torch.int64
        )

        sup_classifier = nn.Linear(features.size(1), self.num_superclasses).to(self.device)
        optimizer = torch.optim.Adam(sup_classifier.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        sup_classifier.train()
        for _ in range(20):
            optimizer.zero_grad()
            logits = sup_classifier(train_feat)
            loss = criterion(logits, sup_labels_train)
            loss.backward()
            optimizer.step()

        sup_classifier.eval()
        with torch.no_grad():
            sup_acc = (sup_classifier(val_feat).argmax(dim=1) == sup_labels_val).float().mean().item()

        return sup_acc

    def _calculate_medical_bonus(self, formula_str):
        """Bonus for formulas containing medically interpretable operators."""
        tokens = formula_str.split()
        bonus = 0.0
        for token in tokens:
            if token in self._medical_operators:
                bonus += self.medical_bonus
                break
        if any(t.startswith('I_BONE') or t.startswith('I_NEG') or t.startswith('I_EDGE') for t in tokens):
            bonus += self.medical_bonus * 0.5
        return min(bonus, self.medical_bonus * 2)

    def _calculate_reward(self):
        expression_tokens = [
            t for t in self.current_sequence
            if self.vocabulary.decode(t) not in ["START", "END", "PAD"]
        ]

        try:
            if not expression_tokens:
                return -1.0, {"valid": False, "reason": "empty"}

            formula_str = ' '.join([self.vocabulary.decode(t) for t in expression_tokens])
            data_batch, labels = self.get_data_batch()

            formula_output = self._execute_formula(expression_tokens, data_batch)
            if formula_output is None:
                has_pooling = any(self.vocabulary.decode(t) in ROOT_OPERATORS for t in expression_tokens)
                has_terminal = any(self.vocabulary.decode(t).startswith('I_') for t in expression_tokens)
                partial = -1.0
                if has_pooling and has_terminal:
                    partial += 0.4
                elif has_pooling:
                    partial += 0.2
                elif has_terminal:
                    partial += 0.2
                if expression_tokens and self.vocabulary.decode(expression_tokens[-1]) in ROOT_OPERATORS:
                    partial += 0.2
                return partial, {"valid": False, "reason": "execution_error"}

            if formula_output.dim() == 1:
                features = formula_output.unsqueeze(1)
            else:
                features = formula_output

            if features.dim() > 2:
                features = features.flatten(1)

            feat_mean = features.mean(dim=0, keepdim=True)
            feat_std = features.std(dim=0, keepdim=True) + 1e-8
            features_norm = (features - feat_mean) / feat_std

            balanced_acc, standard_acc = self._compute_balanced_accuracy(features_norm, labels)

            accuracy = balanced_acc

            if self.use_hierarchical_eval:
                sup_acc = self._compute_hierarchical_accuracy(features_norm, labels)
                accuracy = 0.6 * balanced_acc + 0.4 * sup_acc

            output_vector = formula_output.detach().cpu()
            accepted, reason = self.feature_bank.add_formula(
                formula=None, formula_str=formula_str,
                length=len(expression_tokens), accuracy=accuracy,
                output_vector=output_vector,
            )

            medical_bonus = self._calculate_medical_bonus(formula_str)

            reward = (
                accuracy
                - self.length_penalty * len(expression_tokens)
                + medical_bonus
            )

            if accepted:
                reward += 0.05

            reward = max(-1.0, min(1.0, reward))

            metrics = {
                "valid": True,
                "formula": formula_str,
                "balanced_accuracy": balanced_acc,
                "standard_accuracy": standard_acc,
                "reward": reward,
                "medical_bonus": medical_bonus,
                "bank_size": self.feature_bank.size(),
                "accepted": accepted,
            }

            if self.episode_count % 10 == 0:
                print(f"[FractureEnv] BAcc={balanced_acc:.3f} SAcc={standard_acc:.3f} "
                      f"Reward={reward:.3f} Bank={self.feature_bank.size()} "
                      f"Formula: {formula_str}")

            return reward, metrics

        except Exception as e:
            import traceback
            print(f"[FractureEnv Error] {e}")
            traceback.print_exc()
            return -1.0, {"valid": False, "error": str(e)}
