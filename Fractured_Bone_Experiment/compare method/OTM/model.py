"""Optimized Tsetlin Machine (OTM) for Fracture Classification

Faithful to OTM toolbox (compare method/OTM/An-Optimized-Toolbox-for-Advanced-Image-Processing-with-Tsetlin-Machine-Composites-main).

Architecture (following CIFAR103x3ColorThermometerScoring.py):
    Image binarization: Color Thermometer encoding
        - RGB image (3 channels) -> resize(32, 32)
        - For each channel, threshold at (z+1)*255/(resolution+1) for z in [0, resolution)
        - Produces binary tensor [N, 32, 32, 3*resolution] = [N, 32, 32, 24] for resolution=8
        - 24 bits per pixel (preserves color information, unlike grayscale CTM)
    Classifier: TMClassifier (tmu library, CUDA platform)
        - Convolutional/patch-based Tsetlin Machine
        - Learns propositional clauses over local image patches
        - patch_dim=(3, 3): 3x3 pixel patches as clause inputs (as in CIFAR10 3x3 config)
        - weighted_clauses=True: clauses have real-valued weights

Forward (single-stage, no neural backbone):
    input: RGB image -> resize(32, 32)
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
    vs Ours (Symbolic RL): 0x neural parameters (interpretable rules)

Library:
    tmu (Tsetlin Machine Unified, v0.8.3)
    Source: compare method/OTM/An-Optimized-Toolbox-...

Color Thermometer vs other OTM binarizations (all in source toolbox):
    - Color Thermometer (used here): fixed thresholds, preserves color, 24 bits/pixel
    - Adaptive Color Thermometer: per-image multilevel thresholds (mlt_temp), slower
    - Adaptive Thresholding Gaussian: cv2.adaptiveThreshold per channel, 3 bits/pixel
    - Otsu Thresholding: global Otsu per channel, 3 bits/pixel
    - Canny edge: edge-based, sparse
    - HOG: histogram of gradients features
    Color Thermometer is the canonical/default OTM method.

Key Difference from CTM:
    - OTM uses Color Thermometer (24 bits/pixel, preserves RGB color)
    - CTM uses OpenCV adaptiveThreshold (1 bit/pixel, grayscale only)
    - OTM uses TMClassifier (tmu library, GPU support)
    - CTM uses MultiClassConvolutionalTsetlinMachine2D (pyTsetlinMachineParallel, CPU)

Saved Artifacts:
    - tm_classifier.pkl: fitted TMClassifier state
        Format A (CPU platform): full __getstate__ pickle (rebuild via __setstate__)
        Format B (CUDA platform): array-only dict with 'format'='cuda_arrays_v1',
            containing clause_bank/weight_bank arrays + config; predict.py rebuilds
            a CPU TMClassifier and injects the learned arrays (pycuda Module is not
            picklable, so GPU kernels are not saved).
    - training_history.json / training_curves.png: per-epoch test accuracy
"""
import numpy as np

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]

DEFAULT_RESOLUTION = 8   # thermometer bits per channel (3*8=24 bits/pixel)
DEFAULT_IMG_SIZE = 32    # resize target (CIFAR standard, as in OTM source)
DEFAULT_PATCH = 3        # patch_dim (as in CIFAR103x3ColorThermometerScoring)


def create_model(num_classes=10):
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
