"""DenseNet121 for Fracture Classification

Architecture:
    torchvision.models.densenet121 (pretrained=False)
    - Replace classifier: Linear(1024, 10)

Forward:
    input: [B, 3, 224, 224]
    output: logits [B, 10]

Model Complexity:
    Total params:     6,964,106 (7.0M)
    Trainable params: 6,964,106
    Model size:       26.6 MB
    Est. GPU memory:  ~160 MB (training, batch_size=32)
    vs Ours (Symbolic RL): 7.2x more parameters

Weight Keys:
    classifier.weight, classifier.bias (torchvision native format)
"""
import torch
import torch.nn as nn
from torchvision import models

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]


class DenseNet121Fracture(nn.Module):
    """DenseNet-121 with 10-class fracture classification head."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.densenet121(weights=None)
        in_features = self.model.classifier.in_features  # 1024
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] RGB image tensor (ImageNet normalized)
        Returns:
            logits: [B, num_classes] raw classification logits
        """
        return self.model(x)

    def load_weights(self, path, device='cpu'):
        """Load weights from checkpoint, handling key prefix mismatch."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        mapped = {f'model.{k}': v for k, v in ckpt.items()}
        self.load_state_dict(mapped)

    def predict(self, x):
        """Return softmax probabilities and predicted class."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        return probs, pred


def create_model(num_classes=10):
    return DenseNet121Fracture(num_classes)
