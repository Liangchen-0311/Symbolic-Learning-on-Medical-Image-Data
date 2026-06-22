"""Concept Bottleneck Model (CBM) for HAM10000 Skin Lesion Classification

Architecture:
    Backbone: torchvision.models.resnet34 (pretrained=True)
    Concept Heads: 8 independent MLP heads (dermatology concepts)
    Classifier: MLP (concepts → 7 classes)

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


class CBMModel(nn.Module):
    def __init__(self, num_concepts=8, num_classes=7, expand_dim=64, pretrained=True):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, expand_dim),
                nn.ReLU(),
                nn.Linear(expand_dim, 1)
            )
            for _ in range(num_concepts)
        ])

        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, expand_dim),
            nn.ReLU(),
            nn.Linear(expand_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        concept_logits = torch.cat([head(features) for head in self.concept_heads], dim=1)
        concept_probs = torch.sigmoid(concept_logits)
        logits = self.classifier(concept_probs)
        return logits, concept_probs, concept_logits

    def load_weights(self, path, device='cpu'):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def predict(self, x):
        logits, concept_probs, concept_logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        return probs, pred, concept_probs


def create_model(num_concepts=8, num_classes=7, expand_dim=64, pretrained=True):
    return CBMModel(num_concepts, num_classes, expand_dim, pretrained)
