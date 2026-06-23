"""Swin-Tiny for HAM10000 Skin Lesion Classification"""
import torch
import torch.nn as nn
from torchvision import models

HAM10000_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


class SwinTinyHAM10000(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.swin_t(weights=weights)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

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
    return SwinTinyHAM10000(num_classes, pretrained)
