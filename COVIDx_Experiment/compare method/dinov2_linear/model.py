"""DINOv2 + Linear Probe for HAM10000 Skin Lesion Classification

Architecture:
    Backbone: facebookresearch/dinov2 (dinov2_vitb14), frozen
    Feature extraction: [CLS] token output (768-dim)
    Classifier: StandardScaler + LogisticRegression (sklearn)
"""
import torch
import torch.nn as nn
import pickle
import numpy as np

HAM10000_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']


class DINOv2Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.embed_dim = 768

    def forward(self, x):
        return self.model(x)

    @torch.no_grad()
    def extract_features(self, x):
        features = self.forward(x)
        return features.cpu().numpy()


class DINOv2LinearProbe:
    def __init__(self, backbone=None, scaler=None, classifier=None):
        self.backbone = backbone or DINOv2Backbone()
        self.scaler = scaler
        self.classifier = classifier

    def predict(self, x, device='cpu'):
        self.backbone = self.backbone.to(device)
        features = self.backbone.extract_features(x.to(device))
        features_scaled = self.scaler.transform(features)
        probs = self.classifier.predict_proba(features_scaled)
        preds = np.argmax(probs, axis=1)
        results = []
        for i in range(len(preds)):
            results.append({
                'pred_class': int(preds[i]),
                'pred_name': HAM10000_NAMES[preds[i]],
                'confidence': float(probs[i, preds[i]]),
            })
        return results

    @classmethod
    def load(cls, model_dir, device='cpu'):
        backbone = DINOv2Backbone()
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open(f"{model_dir}/classifier.pkl", 'rb') as f:
            classifier = pickle.load(f)
        return cls(backbone=backbone, scaler=scaler, classifier=classifier)
