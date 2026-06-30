"""MobileNetV2 + RuleFit for Fracture Classification

Architecture:
    Backbone: torchvision.models.mobilenet_v2 (pretrained=False), frozen
        - Inverted residuals, width_mult=1.0
        - Feature dim: 1280 (last conv output before classifier)
    Feature pipeline: StandardScaler → SelectKBest(k=50) → RuleFit
        - StandardScaler: normalize 1280-dim features
        - SelectKBest: select top-50 most informative features
        - RuleFit: tree-based rule extraction + linear model

Forward (two-stage):
    Stage 1 - Feature Extraction:
        input: [B, 3, 224, 224]
        output: features [B, 1280] (frozen MobileNetV2)

    Stage 2 - Classification:
        input: features [B, 1280]
        pipeline: StandardScaler → SelectKBest → RuleFit
        output: class predictions

Model Complexity:
    Backbone params:   2,223,872 (2.2M)
    Trainable params:  0 (backbone frozen, RuleFit is sklearn)
    Backbone size:     8.5 MB
    Est. GPU memory:   ~50 MB (inference only, no GPU training)
    vs Ours (Symbolic RL): 2.3x more parameters (backbone only)

Saved Artifacts:
    - backbone_weights.pth: MobileNetV2 backbone weights
    - scaler.pkl: fitted StandardScaler
    - selector.pkl: fitted SelectKBest
    - rule_model.pkl: fitted RuleFit classifier
"""
import torch
import torch.nn as nn
import pickle
import numpy as np
from torchvision import models

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]


class MobileNetV2FeatureExtractor(nn.Module):
    """Frozen MobileNetV2 feature extractor.

    Extracts 1280-dim features from the last convolutional layer.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.mobilenet_v2(weights=None)
        # Remove the original classifier, use as feature extractor
        in_features = self.model.last_channel  # 1280
        self.model.classifier = nn.Identity()
        self.feature_dim = in_features

    def forward(self, x):
        """
        Args:
            x: [B, 3, 224, 224] RGB image tensor (ImageNet normalized)
        Returns:
            features: [B, 1280] feature vector
        """
        return self.model(x)

    def load_weights(self, path, device='cpu'):
        """Load backbone weights from checkpoint, handling key prefix mismatch.

        Uses strict=False because the saved checkpoint includes the original
        classifier head which we replaced with Identity.
        """
        ckpt = torch.load(path, map_location=device, weights_only=True)
        mapped = {f'model.{k}': v for k, v in ckpt.items()}
        self.load_state_dict(mapped, strict=False)

    @torch.no_grad()
    def extract_features(self, x):
        """Extract features as numpy array for sklearn pipeline."""
        features = self.forward(x)
        return features.cpu().numpy()


class RuleFitPipeline:
    """Full MobileNetV2 + RuleFit pipeline.

    Combines frozen MobileNetV2 backbone with sklearn
    StandardScaler → SelectKBest → RuleFit.
    """

    def __init__(self, backbone=None, scaler=None, selector=None, rulefit=None):
        self.backbone = backbone or MobileNetV2FeatureExtractor()
        self.scaler = scaler
        self.selector = selector
        self.rulefit = rulefit

    def predict(self, x, device='cpu'):
        """
        Args:
            x: [B, 3, 224, 224] RGB image tensor
            device: torch device
        Returns:
            predictions: list of dicts with 'pred_class', 'pred_name', 'confidence'
        """
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
                'pred_name': FRACTURE_NAMES[pred],
                'confidence': 1.0,  # RuleFit doesn't provide probabilities
            })
        return results

    @classmethod
    def load(cls, model_dir, device='cpu'):
        """Load saved pipeline from directory."""
        backbone = MobileNetV2FeatureExtractor()
        backbone.load_weights(f"{model_dir}/backbone_weights.pth", device='cpu')
        backbone.eval()
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open(f"{model_dir}/selector.pkl", 'rb') as f:
            selector = pickle.load(f)
        with open(f"{model_dir}/rule_model.pkl", 'rb') as f:
            rulefit = pickle.load(f)
        return cls(backbone=backbone, scaler=scaler, selector=selector, rulefit=rulefit)
