"""Optimized Tsetlin Machine (OTM) for Brain Tumor MRI Classification

Faithful to OTM toolbox (compare method/OTM/An-Optimized-Toolbox-...).

Architecture (following CIFAR103x3ColorThermometerScoring.py):
    Image binarization: Color Thermometer encoding
        - RGB image (3 channels) -> resize(32, 32)
        - For each channel, threshold at (z+1)*255/(resolution+1) for z in [0, resolution)
        - Produces binary tensor [N, 32, 32, 3*resolution] = [N, 32, 32, 24] for resolution=8
        - 24 bits per pixel (preserves color information)
    Classifier: TMClassifier (tmu library, CUDA platform)
        - patch_dim=(3, 3): 3x3 pixel patches as clause inputs
        - weighted_clauses=True: clauses have real-valued weights

Forward (single-stage, no neural backbone):
    input: MRI image -> resize(32, 32)
    -> Color Thermometer encoding (fixed thresholds (z+1)*255/9)
    -> [32, 32, 24] binary tensor
    -> TMClassifier with patch_dim=(3, 3) learns local pattern clauses
    -> class prediction via clause voting

Model Complexity:
    Backbone params:   0 (no neural network, pure logic-based)
    TM clauses:        2000
    TM T (threshold):  3000 (as in CIFAR103x3ColorThermometerScoring)
    TM s (specificity): 5.0 (as in CIFAR103x3ColorThermometerScoring)
    Patch size:        3x3
    Platform:          CUDA (GPU via pycuda)

Library:
    tmu (Tsetlin Machine Unified, v0.8.3)

Saved Artifacts:
    - tm_classifier.pkl: fitted TMClassifier state (array-only format for GPU-trained)
    - training_history.json / training_curves.png: per-epoch test accuracy
    - report.json / results.json: metrics + resource usage
    - predict.py: standalone inference script
"""
import numpy as np

BRAIN_NAMES = ['glioma', 'meningioma', 'pituitary', 'notumor']
DEFAULT_RESOLUTION = 8
DEFAULT_IMG_SIZE = 32
DEFAULT_PATCH = 3


def create_model(num_classes=4):
    """Factory function for API consistency with other baselines.

    OTM has no neural model; returns a placeholder dict of config.
    Actual TMClassifier is created during training.
    """
    return {
        'type': 'otm_tmu',
        'resolution': DEFAULT_RESOLUTION,
        'img_size': DEFAULT_IMG_SIZE,
        'patch': DEFAULT_PATCH,
        'num_classes': num_classes,
    }
