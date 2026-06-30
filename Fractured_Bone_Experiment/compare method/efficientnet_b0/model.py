"""EfficientNet-B0 for Fracture Classification

Architecture:
    torchvision.models.efficientnet_b0 (pretrained=False)
    - Replace classifier[1]: Linear(1280, 10)

Forward:
    input: [B, 3, 224, 224]
    output: logits [B, 10]

Model Complexity:
    Total params:     4,020,358 (4.0M)
    Trainable params: 4,020,358
    Model size:       15.3 MB
    Est. GPU memory:  ~90 MB (training, batch_size=32)
    vs Ours (Symbolic RL): 4.2x more parameters

Weight Keys:
    classifier.1.weight, classifier.1.bias (torchvision native format)
"""
import torch
import torch.nn as nn
from torchvision import models

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]


class EfficientNetB0Fracture(nn.Module):
    """EfficientNet-B0 with 10-class fracture classification head."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.efficientnet_b0(weights=None)
        in_features = self.model.classifier[1].in_features  # 1280
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

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
    return EfficientNetB0Fracture(num_classes)
