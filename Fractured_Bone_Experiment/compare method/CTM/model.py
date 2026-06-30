"""Classic Tsetlin Machine (CTM) for Fracture Classification

Faithful to pyTsetlinMachineParallel (compare method/CTM/pyTsetlinMachineParallel-master).

Architecture (following examples/FashionMNISTDemo2DConvolutionWeightedClauses.py):
    Image binarization: OpenCV adaptiveThreshold
        - Grayscale conversion -> resize(64, 64)
        - cv2.adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=11, C=2)
        - Produces binary 2D patterns [N, 64, 64] with values {0, 1}
    Classifier: MultiClassConvolutionalTsetlinMachine2D
        - Convolutional/patch-based Tsetlin Machine from pyTsetlinMachineParallel
        - Learns propositional clauses over local image patches
        - patch_dim=(10, 10): 10x10 pixel patches as clause inputs
        - weighted_clauses=True: clauses have real-valued weights

Forward (single-stage, no neural backbone):
    input: RGB image -> grayscale -> resize(64, 64)
    -> cv2.adaptiveThreshold -> binary [64, 64]
    -> MultiClassConvolutionalTsetlinMachine2D with patch_dim=(10, 10)
    -> class prediction via clause voting

Model Complexity:
    Backbone params:   0 (no neural network, pure logic-based)
    TM clauses:        2000
    TM T (threshold):  5000 (= 50 * 100, as in MNISTDemoWeightedClauses)
    TM s (specificity): 10.0
    Patch size:        10x10
    Platform:          CPU (C library with OpenMP)
    vs Ours (Symbolic RL): 0x neural parameters (interpretable rules)

Library:
    pyTsetlinMachineParallel (v0.3.0, C extension)
    Source: compare method/CTM/pyTsetlinMachineParallel-master

Key Difference from OTM:
    - CTM uses OpenCV adaptiveThreshold (single bit per pixel)
    - OTM uses Color Thermometer (24 bits per pixel, preserves color)
    - CTM uses MultiClassConvolutionalTsetlinMachine2D (pyTsetlinMachineParallel)
    - OTM uses TMClassifier (tmu library, newer with GPU support)

Saved Artifacts:
    - tm_model.pkl: fitted MultiClassConvolutionalTsetlinMachine2D (pickled state)
    - training_history.json / training_curves.png: per-epoch test accuracy
"""
import numpy as np

FRACTURE_NAMES = [
    'Comminuted', 'Greenstick', 'Healthy', 'Linear',
    'Oblique Displaced', 'Oblique', 'Segmental', 'Spiral',
    'Transverse Displaced', 'Transverse'
]

DEFAULT_IMG_SIZE = 64   # resize target (CIFAR uses 32; 64 keeps more X-ray detail)
DEFAULT_PATCH = 10      # patch_dim (as in FashionMNISTDemo2DConvolutionWeightedClauses)


def create_model(num_classes=10):
    """Factory function for API consistency with other baselines.

    CTM has no neural model; returns a placeholder dict of config.
    Actual MultiClassConvolutionalTsetlinMachine2D is created during training.
    """
    return {
        'type': 'ctm_pyTsetlinMachine',
        'img_size': DEFAULT_IMG_SIZE,
        'patch': DEFAULT_PATCH,
        'num_classes': num_classes,
    }
