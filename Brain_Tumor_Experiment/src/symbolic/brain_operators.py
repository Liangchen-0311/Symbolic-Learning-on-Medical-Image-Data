"""
Domain operators for Brain Tumor MRI (grayscale) symbolic feature discovery.

The base operator library (src/symbolic/tensor_operators.py) is already rich and
modality-agnostic — intensity arithmetic, spatial filters (blur, edge, laplacian,
gabor, dog, lbp_like, local_std, local_contrast, morphology), and a large set of
pooling roots. Those are registered globally and reused unchanged here.

On top of them we register a small set of *symmetry* and *texture* operators that
are especially meaningful for brain MRI — most notably a horizontal mirror
difference, which directly measures the left-right asymmetry that brain tumors
induce. These are self-contained (no skin/color dependencies); color-only
operators are deliberately absent because they collapse to constants on a single
intensity channel.

All operators here take a single-channel spatial map [B, H, W] and return one.
"""

import torch
import torch.nn.functional as F

from src.symbolic.tensor_operators import TENSOR_OPERATORS


# ======================================================================
# Operator implementations ([B, H, W] -> [B, H, W])
# ======================================================================

def _lr_symmetry(x):
    """Left-right mirror difference — bilateral asymmetry (key brain-tumor cue)."""
    return (x - torch.flip(x, [2])).abs()


def _tb_symmetry(x):
    """Top-bottom mirror difference."""
    return (x - torch.flip(x, [1])).abs()


def _diag_symmetry(x):
    """Diagonal (transpose) asymmetry."""
    transposed = x.permute(0, 2, 1)
    if transposed.shape != x.shape:
        transposed = F.interpolate(
            transposed.unsqueeze(1), size=x.shape[1:], mode='bilinear', align_corners=False
        ).squeeze(1)
    return (x - transposed).abs()


def _border_sharp(x):
    """Gradient magnitude — mass effect / tumor-margin sharpness."""
    grad_x = F.pad(x[:, :, 1:] - x[:, :, :-1], [0, 1, 0, 0])
    grad_y = F.pad(x[:, 1:, :] - x[:, :-1, :], [0, 0, 0, 1])
    return (grad_x ** 2 + grad_y ** 2).sqrt()


def _local_range(x):
    """Local max - min over a 7x7 neighbourhood — intensity heterogeneity."""
    k = 7
    pad = k // 2
    padded = F.pad(x.unsqueeze(1), [pad] * 4, mode='reflect')
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    local_max = patches.max(dim=-1)[0].max(dim=-1)[0].squeeze(1)
    local_min = patches.min(dim=-1)[0].min(dim=-1)[0].squeeze(1)
    return local_max - local_min


def _texture_entropy(x):
    """Log of local variance over a 7x7 neighbourhood — texture complexity."""
    k = 7
    pad = k // 2
    padded = F.pad(x.unsqueeze(1), [pad] * 4, mode='reflect')
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    local_var = patches.var(dim=(-1, -2)).squeeze(1)
    return (local_var + 1e-8).log()


def _local_var(x):
    """Local variance over a 3x3 neighbourhood — fine texture."""
    k = 3
    pad = k // 2
    padded = F.pad(x.unsqueeze(1), [pad] * 4, mode='reflect')
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    return patches.var(dim=(-1, -2)).squeeze(1)


BRAIN_OPERATORS = {}

# Brain-semantic name -> implementation. Names are chosen so the discovered
# formula strings and the interpretability report read in MRI terms.
_BRAIN_UNARY = [
    ('lr_symmetry',     _lr_symmetry),
    ('tb_symmetry',     _tb_symmetry),
    ('diag_symmetry',   _diag_symmetry),
    ('border_sharp',    _border_sharp),
    ('local_range',     _local_range),
    ('texture_entropy', _texture_entropy),
    ('local_var',       _local_var),
]


def register_brain_operators(tensor_ops):
    """Register brain-MRI operators into the global operator dict. Idempotent."""
    for name, func in _BRAIN_UNARY:
        tensor_ops[name] = (func, 1, f'BrainMRI: {name}')
        BRAIN_OPERATORS[name] = tensor_ops[name]
    return BRAIN_OPERATORS


# Color-only operators (absent here) that should never be registered for grayscale.
BRAIN_EXCLUDE_OPERATORS = ['blue_veil', 'color_diff', 'color_ratio', 'regression']
