"""DenseNet121 for HAM10000 Skin Lesion Classification"""
import torch
import torch.nn as nn
from torchvision import models

HAM10000_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


class DenseNet121HAM10000(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.densenet121(weights=weights)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

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
    return DenseNet121HAM10000(num_classes, pretrained)
