"""EfficientNet-B0 for HAM10000 Skin Lesion Classification"""
import torch
import torch.nn as nn
from torchvision import models

HAM10000_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


class EfficientNetB0HAM10000(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

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
    return EfficientNetB0HAM10000(num_classes, pretrained)
