"""ResNet50 for Fracture Classification

Architecture:
    torchvision.models.resnet50 (pretrained=False)
    - Replace final FC layer: Linear(2048, 10)

Forward:
    input: [B, 3, 224, 224]
    output: logits [B, 10]

Model Complexity:
    Total params:     23,528,522 (23.5M)
    Trainable params: 23,528,522
    Model size:       89.8 MB
    Est. GPU memory:  ~540 MB (training, batch_size=32)
    vs Ours (Symbolic RL): 24.4x more parameters

Weight Keys:
    Saved checkpoint uses torchvision's native key names:
    conv1.weight, bn1.weight, ..., layer4.*, fc.weight, fc.bias
    So we wrap the full ResNet50 as self.model (not self.backbone)
    to keep key names compatible with load_state_dict().
"""
import torch
import torch.nn as nn
from torchvision import models

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]


class ResNet50Fracture(nn.Module):
    """ResNet-50 with 10-class fracture classification head.

    Note: self.model is the full ResNet50 (not just backbone) so that
    state_dict keys match the saved checkpoint (fc.weight, not backbone.fc.weight).
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet50(weights=None)
        in_features = self.model.fc.in_features  # 2048
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] RGB image tensor (ImageNet normalized)
        Returns:
            logits: [B, num_classes] raw classification logits
        """
        return self.model(x)

    def load_weights(self, path, device='cpu'):
        """Load weights from checkpoint, handling key prefix mismatch.

        Checkpoint keys: fc.weight, layer1.*, ...
        Model keys: model.fc.weight, model.layer1.*, ...
        """
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
    return ResNet50Fracture(num_classes)
