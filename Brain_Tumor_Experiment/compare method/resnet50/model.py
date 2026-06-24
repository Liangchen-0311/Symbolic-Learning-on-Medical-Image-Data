"""ResNet50 for HAM10000 Skin Lesion Classification

Architecture:
    torchvision.models.resnet50 (pretrained=True)
    - Replace final FC layer: Linear(2048, 7)

Forward:
    input: [B, 3, 224, 224]
    output: logits [B, 7]
"""
import torch
import torch.nn as nn
from torchvision import models

HAM10000_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


class ResNet50HAM10000(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.model = models.resnet50(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def load_weights(self, path, device='cpu'):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(ckpt)

    def predict(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        return probs, pred


def create_model(num_classes=7, pretrained=True):
    return ResNet50HAM10000(num_classes, pretrained)
