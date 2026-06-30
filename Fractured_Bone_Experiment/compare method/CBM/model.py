"""Concept Bottleneck Model (CBM) for Fracture Classification

Architecture:
    Backbone: torchvision.models.resnet34 (pretrained=False)
        - Remove FC layer, output: 512-dim features
    Concept Heads: 8 independent MLP heads
        - Each: Linear(512, 64) → ReLU → Linear(64, 1)
        - Output: concept_logits [B, 8], concept_probs = sigmoid(concept_logits)
    Classifier: MLP
        - Linear(8, 64) → ReLU → Linear(64, 10)

Forward:
    input: [B, 3, 224, 224]
    → backbone features [B, 512]
    → concept_logits [B, 8] → concept_probs = sigmoid(concept_logits) [B, 8]
    → logits [B, 10]
    Returns: (logits, concept_probs, concept_logits)

Model Complexity:
    Total params:     21,549,074 (21.5M)
      - Backbone (ResNet34): 21,297,160 (99%)
      - Concept heads: 196,616
      - Classifier: 55,298
    Trainable params: 21,549,074
    Model size:       82.2 MB
    Est. GPU memory:  ~490 MB (training, batch_size=32)
    vs Ours (Symbolic RL): 22.4x more parameters

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


class CBMModel(nn.Module):
    """Concept Bottleneck Model for fracture classification.

    Predicts 8 interpretable concepts first, then classifies based on concepts.
    """

    def __init__(self, num_concepts=8, num_classes=10, expand_dim=64):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        # Backbone: ResNet34 without FC
        backbone = models.resnet34(weights=None)
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Concept heads: one MLP per concept
        self.concept_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, expand_dim),  # 512 → 64
                nn.ReLU(),
                nn.Linear(expand_dim, 1)              # 64 → 1
            )
            for _ in range(num_concepts)
        ])

        # Classifier: concept probabilities → class logits
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, expand_dim),  # 8 → 64
            nn.ReLU(),
            nn.Linear(expand_dim, num_classes)     # 64 → 10
        )

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] RGB image tensor (ImageNet normalized)
        Returns:
            logits: [B, num_classes] raw classification logits
            concept_probs: [B, num_concepts] sigmoid concept probabilities
            concept_logits: [B, num_concepts] raw concept logits
        """
        features = self.backbone(x)                                    # [B, 512]
        concept_logits = torch.cat([head(features) for head in self.concept_heads], dim=1)  # [B, 8]
        concept_probs = torch.sigmoid(concept_logits)                  # [B, 8]
        logits = self.classifier(concept_probs)                        # [B, 10]
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


def create_model(num_concepts=8, num_classes=10, expand_dim=64):
    return CBMModel(num_concepts, num_classes, expand_dim)
