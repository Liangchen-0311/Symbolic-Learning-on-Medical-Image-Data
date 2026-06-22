"""
Domain-specific operators for HAM10000 skin lesion analysis.

These operators work on single-channel spatial maps [B, H, W] (same format
as the terminal channels I_R, I_G, I_B, I_GRAY, I_H, I_S, etc. provided
by TensorVSREnvironmentLargeBank).

Channel extraction operators (red_ch, green_ch, etc.) are NOT needed here
because the environment already provides individual channels as terminals.

Operators implemented:
  - Asymmetry measures (horizontal, vertical, diagonal)
  - Border/edge analysis
  - Blue-white veil detection (needs I_R, I_G, I_B as separate inputs)
  - Pigment network detection
  - Vascular pattern detection
  - Globules/dots detection
  - Streaks detection
  - Regression structure detection
  - Local hue range
  - Color variety
"""

import torch
import torch.nn.functional as F
import numpy as np

from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS


# ======================================================================
# Single-channel Unary Operators (work on [B, H, W] spatial maps)
# ======================================================================

def _asymmetry_h(x):
    """Horizontal asymmetry measure (ABCD: A). Input: [B, H, W]."""
    flipped = torch.flip(x, [2])
    diff = (x - flipped).abs()
    return diff


def _asymmetry_v(x):
    """Vertical asymmetry measure (ABCD: A). Input: [B, H, W]."""
    flipped = torch.flip(x, [1])
    diff = (x - flipped).abs()
    return diff


def _asymmetry_diag(x):
    """Diagonal asymmetry (transpose). Input: [B, H, W]."""
    transposed = x.permute(0, 2, 1)
    if transposed.shape != x.shape:
        transposed = F.interpolate(
            transposed.unsqueeze(1), size=x.shape[1:], mode='bilinear', align_corners=False
        ).squeeze(1)
    diff = (x - transposed).abs()
    return diff


def _pigment_network(x):
    """Detect pigment network pattern via local variance. Input: [B, H, W]."""
    k = 3
    pad = k // 2
    padded = F.pad(x.unsqueeze(1), [pad]*4, mode='reflect')
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)  # [B,1,H,W,k,k]
    local_var = patches.var(dim=(-1, -2)).squeeze(1)
    return local_var


def _border_sharpness(x):
    """Measure border sharpness (ABCD: B). Input: [B, H, W]."""
    grad_x = x[:, :, 1:] - x[:, :, :-1]
    grad_y = x[:, 1:, :] - x[:, :-1, :]
    grad_x = F.pad(grad_x, [0, 1, 0, 0])
    grad_y = F.pad(grad_y, [0, 0, 0, 1])
    gradient_mag = (grad_x ** 2 + grad_y ** 2).sqrt()
    return gradient_mag


def _globules_dots(x):
    """Detect globules and dots pattern. Input: [B, H, W]."""
    dark = (1.0 - x).clamp(0, 1)
    pooled = F.max_pool2d(dark.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
    globules = (pooled == dark).float() * dark
    return globules


def _streaks(x):
    """Detect streaks (pseudopods or radial streaming). Input: [B, H, W]."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
    ky = kx.t()
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    gx = F.conv2d(F.pad(x.unsqueeze(1), [1]*4, mode='reflect'), kx).squeeze(1)
    gy = F.conv2d(F.pad(x.unsqueeze(1), [1]*4, mode='reflect'), ky).squeeze(1)
    streak_mag = gx.abs() + gy.abs()
    return streak_mag


def _local_range(x):
    """Compute local range (max - min in neighborhood). Input: [B, H, W]."""
    k = 7
    pad = k // 2
    padded = F.pad(x.unsqueeze(1), [pad]*4, mode='reflect')
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    local_max = patches.max(dim=(-1, -2))[0].squeeze(1)
    local_min = patches.min(dim=(-1, -2))[0].squeeze(1)
    return local_max - local_min


def _local_entropy(x):
    """Compute local entropy (texture complexity). Input: [B, H, W]."""
    k = 7
    pad = k // 2
    padded = F.pad(x.unsqueeze(1), [pad]*4, mode='reflect')
    patches = padded.unfold(2, k, 1).unfold(3, k, 1)
    local_var = patches.var(dim=(-1, -2)).squeeze(1)
    # Approximate entropy via log of variance
    return (local_var + 1e-8).log()


# ======================================================================
# Binary Operators (need two channels, e.g. I_R and I_B)
# ======================================================================

def _blue_veil(r, b):
    """Detect blue-white veil regions. Inputs: [B, H, W] each (R and B channels).
    Blue-dominant regions where blue > red.
    """
    blue_mask = (b > 0.4) & (b > r)
    return blue_mask.float()


def _color_diff(a, b):
    """Absolute difference between two channels. Inputs: [B, H, W] each."""
    return (a - b).abs()


def _color_ratio(a, b):
    """Ratio of two channels. Inputs: [B, H, W] each."""
    return a / (b + 1e-8)


def _regression_struct(brightness, saturation):
    """Detect regression structures (bright + low saturation).
    Inputs: [B, H, W] each (brightness/value channel and saturation channel).
    """
    regression = (brightness > 0.6) & (saturation < 0.15)
    return regression.float()


# ======================================================================
# Register all HAM10000 operators
# ======================================================================

HAM10000_OPERATORS = {}


def register_ham10000_operators(tensor_ops):
    """Register HAM10000-specific operators into the global operator dict."""

    # Unary operators (single-channel spatial → spatial)
    for name, func in [
        ('asym_h', _asymmetry_h),
        ('asym_v', _asymmetry_v),
        ('asym_diag', _asymmetry_diag),
        ('pigment_net', _pigment_network),
        ('border_sharp', _border_sharpness),
        ('globules', _globules_dots),
        ('streaks', _streaks),
        ('local_range', _local_range),
        ('local_entropy', _local_entropy),
    ]:
        tensor_ops[name] = (func, 1, f'HAM10000: {name}')
        HAM10000_OPERATORS[name] = tensor_ops[name]

    # Binary operators (two channels → spatial)
    for name, func in [
        ('blue_veil', _blue_veil),
        ('color_diff', _color_diff),
        ('color_ratio', _color_ratio),
        ('regression', _regression_struct),
    ]:
        tensor_ops[name] = (func, 2, f'HAM10000: {name}')
        HAM10000_OPERATORS[name] = tensor_ops[name]
