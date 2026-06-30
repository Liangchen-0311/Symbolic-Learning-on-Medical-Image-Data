"""ViT-B/16 for Fracture Classification

Architecture:
    timm.create_model('vit_b_16', pretrained=False, num_classes=10)
    - Vision Transformer Base, patch size 16x16
    - Embedding dim: 768, Depth: 12, Heads: 12

Forward:
    input: [B, 3, 224, 224]
    output: logits [B, 10]

Model Complexity:
    Total params:     86,569,000 (86.6M)
    Trainable params: 86,569,000
    Model size:       330.4 MB
    Est. GPU memory:  ~4000 MB (training, batch_size=32)
    vs Ours (Symbolic RL): 89.9x more parameters

Weight Keys:
    heads.head.weight, heads.head.bias (timm native format)
"""
import torch
import torch.nn as nn

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]


class ViTB16Fracture(nn.Module):
    """Vision Transformer Base (patch16) with 10-class fracture classification head."""

    def __init__(self, num_classes=10):
        super().__init__()
        import timm
        self.model = timm.create_model('vit_b_16', pretrained=False, num_classes=num_classes)

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
    return ViTB16Fracture(num_classes)
