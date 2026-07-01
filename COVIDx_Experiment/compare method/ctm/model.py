"""Classic Tsetlin Machine (CTM) for COVIDx CT Classification

Faithful to pyTsetlinMachineParallel.

Architecture (following examples/FashionMNISTDemo2DConvolutionalWeightedClauses.py):
    Image binarization: OpenCV adaptiveThreshold
        - Grayscale conversion -> resize(64, 64)
        - cv2.adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=11, C=2)
        - Produces binary 2D patterns [N, 64, 64] with values {0, 1}
    Classifier: MultiClassConvolutionalTsetlinMachine2D
        - Convolutional/patch-based Tsetlin Machine from pyTsetlinMachineParallel
        - patch_dim=(10, 10): 10x10 pixel patches as clause inputs
        - weighted_clauses=True: clauses have real-valued weights

Forward (single-stage, no neural backbone):
    input: CT image -> grayscale -> resize(64, 64)
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

Library:
    pyTsetlinMachineParallel (v0.3.0, C extension)

Saved Artifacts:
    - tm_model.pkl: fitted MultiClassConvolutionalTsetlinMachine2D (pickled state)
    - training_history.json / training_curves.png: per-epoch test accuracy
    - report.json / results.json: metrics + resource usage
    - predict.py: standalone inference script
"""
import numpy as np

COVIDX_NAMES = ['normal', 'pneumonia', 'covid']
DEFAULT_IMG_SIZE = 64
DEFAULT_PATCH = 10


def create_model(num_classes=3):
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
