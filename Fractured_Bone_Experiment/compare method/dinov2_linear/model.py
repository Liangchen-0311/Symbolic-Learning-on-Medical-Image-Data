"""DINOv2 + Linear Probe for Fracture Classification

Architecture:
    Backbone: facebookresearch/dinov2 (dinov2_vitb14), frozen
        - Vision Transformer Base, patch size 14x14
        - Embedding dim: 768, Depth: 12, Heads: 12
    Feature extraction: [CLS] token output (768-dim)
    Classifier: StandardScaler + LogisticRegression (sklearn)

Forward (two-stage):
    Stage 1 - Feature Extraction:
        input: [B, 3, 224, 224]
        output: features [B, 768] (CLS token, frozen backbone)

    Stage 2 - Classification:
        input: features [B, 768]
        output: class predictions (sklearn pipeline)

Model Complexity:
    Backbone params:   86,580,480 (86.6M)
    Trainable params:  0 (backbone frozen, classifier is sklearn)
    Backbone size:     330.3 MB
    Est. GPU memory:   ~2000 MB (inference only, no GPU training)
    vs Ours (Symbolic RL): 89.9x more parameters (backbone only)

Saved Artifacts:
    - scaler.pkl: fitted StandardScaler
    - classifier.pkl: fitted LogisticRegression
"""
import torch
import torch.nn as nn
import pickle
import numpy as np

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 ViT-B/14 feature extractor.

    Extracts 768-dim CLS token features from images.
    """

    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_dim = 768

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] RGB image tensor (ImageNet normalized, H/W multiple of 14)
        Returns:
            cls_features: [B, 768] CLS token features
        """
        return self.model(x)

    @torch.no_grad()
    def extract_features(self, x):
        """Extract features as numpy array for sklearn classifier."""
        features = self.forward(x)
        return features.cpu().numpy()


class DINOv2LinearProbe:
    """Full DINOv2 + Linear Probe pipeline.

    Combines frozen DINOv2 backbone with sklearn StandardScaler + LogisticRegression.
    """

    def __init__(self, backbone=None, scaler=None, classifier=None):
        self.backbone = backbone or DINOv2Backbone()
        self.scaler = scaler
        self.classifier = classifier

    def predict(self, x, device='cpu'):
        """
        Args:
            x: [B, 3, H, W] RGB image tensor
            device: torch device
        Returns:
            predictions: list of dicts with 'pred_class', 'pred_name', 'confidence'
        """
        self.backbone = self.backbone.to(device)
        features = self.backbone.extract_features(x.to(device))
        features_scaled = self.scaler.transform(features)
        probs = self.classifier.predict_proba(features_scaled)
        preds = np.argmax(probs, axis=1)

        results = []
        for i in range(len(preds)):
            results.append({
                'pred_class': int(preds[i]),
                'pred_name': FRACTURE_NAMES[preds[i]],
                'confidence': float(probs[i, preds[i]]),
            })
        return results

    @classmethod
    def load(cls, model_dir, device='cpu'):
        """Load saved pipeline from directory."""
        backbone = DINOv2Backbone()
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open(f"{model_dir}/classifier.pkl", 'rb') as f:
            classifier = pickle.load(f)
        return cls(backbone=backbone, scaler=scaler, classifier=classifier)
