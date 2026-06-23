"""MobileNetV2 + RuleFit for HAM10000 Skin Lesion Classification

Architecture:
    Backbone: torchvision.models.mobilenet_v2 (pretrained=True), frozen
    Feature pipeline: StandardScaler → SelectKBest(k=50) → RuleFit
"""
import torch
import torch.nn as nn
import pickle
import numpy as np
from torchvision import models

HAM10000_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


class MobileNetV2FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.model = models.mobilenet_v2(weights=weights)
        in_features = self.model.last_channel  # 1280
        self.model.classifier = nn.Identity()
        self.feature_dim = in_features

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def extract_features(self, x):
        features = self.forward(x)
        return features.cpu().numpy()


class RuleFitPipeline:
    def __init__(self, backbone=None, scaler=None, selector=None, rulefit=None):
        self.backbone = backbone or MobileNetV2FeatureExtractor()
        self.scaler = scaler
        self.selector = selector
        self.rulefit = rulefit

    def predict(self, x, device='cpu'):
        self.backbone = self.backbone.to(device).eval()
        features = self.backbone.extract_features(x.to(device))
        features_scaled = self.scaler.transform(features)
        features_selected = self.selector.transform(features_scaled)
        preds = self.rulefit.predict(features_selected)
        results = []
        for i in range(len(preds)):
            pred = int(preds[i])
            results.append({
                'pred_class': pred,
                'pred_name': HAM10000_NAMES[pred],
                'confidence': 1.0,
            })
        return results

    @classmethod
    def load(cls, model_dir, device='cpu'):
        backbone = MobileNetV2FeatureExtractor()
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open(f"{model_dir}/selector.pkl", 'rb') as f:
            selector = pickle.load(f)
        with open(f"{model_dir}/rule_model.pkl", 'rb') as f:
            rulefit = pickle.load(f)
        return cls(backbone=backbone, scaler=scaler, selector=selector, rulefit=rulefit)
