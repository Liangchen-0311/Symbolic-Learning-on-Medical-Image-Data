"""Concept Reasoning Layer (CRL) for HAM10000 Skin Lesion Classification

Architecture:
    Backbone: torchvision.models.resnet34 (pretrained=True)
    Concept Predictor: Linear(512, 8)
    Reasoning Layers: Binarize → Union → Union → LR (with skip connection)

Concepts (8 dermatology-specific):
    asymmetric_pigmentation, irregular_border, color_variation,
    large_diameter, blue_white_veil, vascular_pattern,
    scale_crust, papular_surface
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

HAM10000_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

HAM10000_CONCEPTS = [
    'asymmetric_pigmentation', 'irregular_border', 'color_variation',
    'large_diameter', 'blue_white_veil', 'vascular_pattern',
    'scale_crust', 'papular_surface'
]


# ─── Autograd Functions ───────────────────────────────────────────────────────

class _GradGraft(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, Y):
        return X
    @staticmethod
    def backward(ctx, grad_output):
        return None, grad_output.clone()


class _Binarizer(torch.autograd.Function):
    @staticmethod
    def forward(_, concepts):
        return (concepts.detach() > 0.0).float()
    @staticmethod
    def backward(_, grad_output):
        return grad_output.clone()


class _Product(torch.autograd.Function):
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
    def __init__(self, input_dim, output_dim, use_not=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim * 2
        self.con_layer = _ConjunctionLayer(input_dim, output_dim, use_not=use_not)
        self.dis_layer = _DisjunctionLayer(input_dim, output_dim, use_not=use_not)

    def forward(self, x):
        return torch.cat([self.con_layer(x), self.dis_layer(x)], dim=1)


class _LRLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc1(x)


# ─── Full CRL Model ───────────────────────────────────────────────────────────

class CRLModel(nn.Module):
    def __init__(self, num_concepts=8, num_classes=7, l1=256, l2=256,
                 use_not=True, use_skip=True, temperature=1.0, pretrained=True):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.use_not = use_not
        self.use_skip = use_skip

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.concept_predictor = nn.Linear(in_features, num_concepts)
        self.t = nn.Parameter(torch.log(torch.tensor([temperature])))

        self.layer_list = nn.ModuleList()
        dim_list = [num_concepts, l1, l2, num_classes]
        prev_layer_dim = None

        for idx, dim in enumerate(dim_list):
            effective_input_dim = prev_layer_dim
            if use_skip and idx >= 3 and len(self.layer_list) >= 2:
                skip_dim = self.layer_list[-2].output_dim
                effective_input_dim = prev_layer_dim + skip_dim

            if idx == 0:
                layer = _BinarizeLayer(dim, use_not)
            elif idx == len(dim_list) - 1:
                layer = _LRLayer(effective_input_dim, dim)
            else:
                layer_use_not = True if idx != 1 else False
                layer = _UnionLayer(prev_layer_dim, dim, use_not=layer_use_not)

            prev_layer_dim = layer.output_dim
            self.layer_list.append(layer)

        self._skip_indices = {}
        for idx in range(3, len(dim_list)):
            self._skip_indices[idx] = idx - 2

    def forward(self, x):
        features = self.backbone(x)
        concept_logits = self.concept_predictor(features)
        concept_probs = torch.sigmoid(concept_logits)

        h = concept_logits
        skip_cache = {}

        for idx, layer in enumerate(self.layer_list):
            if idx in self._skip_indices:
                skip_idx = self._skip_indices[idx]
                h = torch.cat((h, skip_cache[skip_idx]), dim=1)
            h = layer(h)
            if idx in self._skip_indices.values():
                skip_cache[idx] = h

        logits = h / torch.exp(self.t)
        return logits, concept_probs, concept_logits

    def load_weights(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def predict(self, x):
        logits, concept_probs, concept_logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        return probs, pred, concept_probs


def create_model(num_concepts=8, num_classes=7, l1=256, l2=256,
                 use_not=True, use_skip=True, temperature=1.0, pretrained=True):
    return CRLModel(num_concepts, num_classes, l1, l2, use_not, use_skip, temperature, pretrained)
