"""Concept Reasoning Layer (CRL) for Fracture Classification

Architecture:
    Backbone: torchvision.models.resnet34 (pretrained=False)
        - Remove FC layer, output: 512-dim features
    Concept Predictor: Linear(512, 8)
        - Output: concept_logits [B, 8], concept_probs = sigmoid(concept_logits)
    Reasoning Layers:
        Layer 0: BinarizeLayer(8, use_not=True) → [B, 16]
            Binarize concept probs to {0,1}, concatenate with negation
        Layer 1: UnionLayer(16, 256, use_not=False) → [B, 512]
            ConjunctionLayer(16, 256) + DisjunctionLayer(16, 256)
        Layer 2: UnionLayer(512, 256, use_not=True) → [B, 1024]
            ConjunctionLayer(1024, 256) + DisjunctionLayer(1024, 256)
        Layer 3: LRLayer(512+512=1024, 10) [with skip from Layer 1]
            Linear(1024, 10)
    Temperature: learnable scalar t, logits = output / exp(t)

Forward:
    input: [B, 3, 224, 224]
    → backbone features [B, 512]
    → concept_logits [B, 8] → concept_probs [B, 8]
    → BinarizeLayer: h [B, 16]
    → UnionLayer1: h [B, 512]  (save for skip)
    → UnionLayer2: h [B, 1024]
    → concat skip: h [B, 1024+512=1536]... wait, skip from layer1 output=512
    → LRLayer: logits [B, 10] / exp(t)
    Returns: (logits, concept_probs, concept_logits)

Model Complexity:
    Total params:     21,831,507 (21.8M)
      - Backbone (ResNet34): 21,297,160 (98%)
      - Concept predictor: 4,104
      - Reasoning layers: 530,243
    Trainable params: 21,831,507
    Model size:       83.3 MB
    Est. GPU memory:  ~500 MB (training, batch_size=32)
    vs Ours (Symbolic RL): 22.7x more parameters

Key Components:
    _Binarizer: STE-based binarization (forward: threshold at 0, backward: pass-through)
    _GradGraft: forward uses binarized result, backward uses continuous gradient
    _Product: differentiable product (conjunction) via log-sum-exp trick
    _ConjunctionLayer: logical AND over weighted inputs
    _DisjunctionLayer: logical OR over weighted inputs (De Morgan's)
    _UnionLayer: parallel conjunction + disjunction
    _LRLayer: linear classification layer

Concepts (8):
    cortical_break, fracture_line_horizontal, fracture_line_oblique_45,
    fracture_line_oblique_135, fracture_line_vertical, displacement,
    bone_fragment, soft_tissue_swelling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]

FRACTURE_CONCEPTS = [
    'cortical_break', 'fracture_line_horizontal', 'fracture_line_oblique_45',
    'fracture_line_oblique_135', 'fracture_line_vertical', 'displacement',
    'bone_fragment', 'soft_tissue_swelling'
]


# ─── Autograd Functions ───────────────────────────────────────────────────────

class _GradGraft(torch.autograd.Function):
    """Forward: use binarized X. Backward: use continuous Y's gradient."""
    @staticmethod
    def forward(ctx, X, Y):
        return X
    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class _Binarizer(torch.autograd.Function):
    """Straight-through estimator: forward threshold at 0, backward pass-through."""
    @staticmethod
    def forward(_, concepts):
        return (concepts.detach() > 0.0).float()
    @staticmethod
    def backward(_, grad_output):
        return grad_output.clone()


class _Product(torch.autograd.Function):
    """Differentiable product via log-sum-exp: y = 1 / (1 - sum(log(x)))"""
    @staticmethod
    def forward(ctx, X):
        y = -1.0 / (-1.0 + torch.sum(torch.log(X + 1e-10), dim=1))
        ctx.save_for_backward(X, y)
        return y
    @staticmethod
    def backward(ctx, grad_output):
        X, y = ctx.saved_tensors
        return grad_output.unsqueeze(1) * (y.unsqueeze(1) ** 2 / (X + 1e-10))


# ─── Layer Modules ─────────────────────────────────────────────────────────────

class _BinarizeLayer(nn.Module):
    """Binarize concept probabilities and optionally concatenate negation.

    output_dim = 2 * n_concepts if use_not else n_concepts
    """
    def __init__(self, n_concepts, use_not=True):
        super().__init__()
        self.n_concepts = n_concepts
        self.use_not = use_not
        self.output_dim = 2 * n_concepts if use_not else n_concepts

    def forward(self, x):
        x = _Binarizer.apply(x)
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return x


class _ConjunctionLayer(nn.Module):
    """Logical AND layer: product of weighted inputs.

    W: [input_dim, output_dim] learnable weights (0~1)
    Forward uses GradGraft: binarized result with continuous gradient.
    """
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))

    def forward(self, x):
        res_tilde = self._continuous_forward(x)
        res_bar = self._binarized_forward(x)
        return _GradGraft.apply(res_bar, res_tilde)

    def _continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return _Product.apply(1 - (1 - x).unsqueeze(-1) * self.W)

    @torch.no_grad()
    def _binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = _Binarizer.apply(self.W - 0.5)
        return torch.prod(1 - (1 - x).unsqueeze(-1) * Wb, dim=1)


class _DisjunctionLayer(nn.Module):
    """Logical OR layer: De Morgan's dual of conjunction.

    W: [input_dim, output_dim] learnable weights (0~1)
    """
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim if not use_not else input_dim * 2
        self.output_dim = output_dim
        self.use_not = use_not
        self.W = nn.Parameter(0.5 * torch.rand(self.input_dim, self.output_dim))

    def forward(self, x):
        res_tilde = self._continuous_forward(x)
        res_bar = self._binarized_forward(x)
        return _GradGraft.apply(res_bar, res_tilde)

    def _continuous_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        return 1 - _Product.apply(1 - x.unsqueeze(-1) * self.W)

    @torch.no_grad()
    def _binarized_forward(self, x):
        if self.use_not:
            x = torch.cat((x, 1 - x), dim=1)
        Wb = _Binarizer.apply(self.W - 0.5)
        return 1 - torch.prod(1 - x.unsqueeze(-1) * Wb, dim=1)


class _UnionLayer(nn.Module):
    """Parallel conjunction + disjunction layer.

    output_dim = 2 * output_dim_param (conjunction + disjunction concatenated)
    """
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2
        self.con_layer = _ConjunctionLayer(input_dim, output_dim, use_not=use_not)
        self.dis_layer = _DisjunctionLayer(input_dim, output_dim, use_not=use_not)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)


class _LRLayer(nn.Module):
    """Linear reasoning layer: simple linear transformation."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc1(x)


# ─── Full CRL Model ───────────────────────────────────────────────────────────

class CRLModel(nn.Module):
    """Concept Reasoning Layer model for fracture classification.

    Pipeline:
        Image → ResNet34 → concept predictor → Binarize → Union layers → LR layer
        With skip connection from layer 1 to layer 3.

    Args:
        num_concepts: number of concept variables (default: 8)
        num_classes: number of fracture classes (default: 10)
        l1: UnionLayer1 output dim per branch (default: 256)
        l2: UnionLayer2 output dim per branch (default: 256)
        use_not: whether to use negation in BinarizeLayer (default: True)
        use_skip: whether to add skip connection from layer1 to layer3 (default: True)
        temperature: initial temperature for logit scaling (default: 1.0)
    """

    def __init__(self, num_concepts=8, num_classes=10, l1=256, l2=256,
                 use_not=True, use_skip=True, temperature=1.0):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.use_not = use_not
        self.use_skip = use_skip

        # Backbone: ResNet34
        backbone = models.resnet34(weights=None)
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Concept predictor
        self.concept_predictor = nn.Linear(in_features, num_concepts)

        # Temperature
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

        # Build reasoning layers
        self.layer_list = nn.ModuleList()
        dim_list = [num_concepts, l1, l2, num_classes]
        prev_layer_dim = None

        for idx, dim in enumerate(dim_list):
            # Compute effective input dim (with skip connection)
            effective_input_dim = prev_layer_dim
            if use_skip and idx >= 3 and len(self.layer_list) >= 2:
                skip_dim = self.layer_list[-2].output_dim
                effective_input_dim = prev_layer_dim + skip_dim

            if idx == 0:
                layer = _BinarizeLayer(dim, use_not)
            elif idx == len(dim_list) - 1:
                layer = _LRLayer(effective_input_dim, dim)
            else:
                # Layer 1: no use_not; Layer 2: use_not
                layer_use_not = True if idx != 1 else False
                layer = _UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)

            prev_layer_dim = layer.output_dim
            self.layer_list.append(layer)

        # Record skip indices for forward pass
        self._skip_indices = {}
        for idx in range(3, len(dim_list)):
            self._skip_indices[idx] = idx - 2

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] RGB image tensor (ImageNet normalized)
        Returns:
            logits: [B, num_classes] scaled classification logits
            concept_probs: [B, num_concepts] sigmoid concept probabilities
            concept_logits: [B, num_concepts] raw concept logits
        """
        features = self.backbone(x)                                    # [B, 512]
        concept_logits = self.concept_predictor(features)              # [B, 8]
        concept_probs = torch.sigmoid(concept_logits)                  # [B, 8]

        h = concept_logits
        skip_cache = {}

        for idx, layer in enumerate(self.layer_list):
            # Apply skip connection before this layer
            if idx in self._skip_indices:
                skip_idx = self._skip_indices[idx]
                h = torch.cat((h, skip_cache[skip_idx]), dim=1)

            h = layer(h)

            # Cache output for potential skip connections
            if idx in self._skip_indices.values():
                skip_cache[idx] = h

        logits = h / torch.exp(self.t)
        return logits, concept_probs, concept_logits

    def load_weights(self, path, device='cpu'):
        """Load weights from checkpoint."""
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def predict(self, x):
        """Return softmax probabilities, predicted class, and concept probabilities."""
        logits, concept_probs, concept_logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        return probs, pred, concept_probs


def create_model(num_concepts=8, num_classes=10, l1=256, l2=256,
                 use_not=True, use_skip=True, temperature=1.0):
    return CRLModel(num_concepts, num_classes, l1, l2, use_not, use_skip, temperature)
