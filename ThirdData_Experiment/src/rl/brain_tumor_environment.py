"""
Brain Tumor MRI Symbolic Feature Discovery RL Environment (grayscale).

Adapts TensorVSREnvironmentLargeBank for single-channel MRI:
  - Grayscale terminal channels instead of RGB/HSV/color-ratio channels.
  - Registers the brain-MRI operator subset (symmetry / texture / border) and
    relies on config['exclude_operators'] to drop the color-only operators.

Grayscale terminals (single source of truth — also imported by the pipeline so
feature extraction matches exactly):
  I_GRAY      raw intensity
  I_BLUR      5x5 Gaussian-smoothed intensity (coarse anatomy)
  I_GRAD      Sobel gradient magnitude (mass / tumor-margin borders)
  I_LOCALSTD  5x5 local standard deviation (texture heterogeneity)
  I_LOG       Laplacian (blob / ring-enhancing structures)
  I_LBP       local-binary-pattern-like texture response (fine texture)
"""

import torch
import torch.nn.functional as F

from src.rl.tensor_environment_large_bank import (
    TensorVSREnvironmentLargeBank,
)
from src.symbolic.tensor_operators import TENSOR_OPERATORS
from src.symbolic.brain_operators import register_brain_operators

# Register the brain-MRI symmetry/texture operators into the global registry.
register_brain_operators(TENSOR_OPERATORS)


# Ordered list of grayscale terminal names exposed to the RL agent.
# I_LRDIFF (whole-brain left-right asymmetry) is an explicit building block: brain
# tumors break bilateral symmetry, and this signal was enriched in the most
# glioma-discriminative formulas, so we hand it to the agent directly.
BRAIN_TERMINALS = ['I_GRAY', 'I_BLUR', 'I_GRAD', 'I_LOCALSTD', 'I_LOG', 'I_LBP', 'I_LRDIFF']


def _reflect_pad(x4, p):
    return F.pad(x4, [p, p, p, p], mode='reflect')


def _gaussian_blur(x4):
    k = torch.tensor(
        [[1, 4, 6, 4, 1],
         [4, 16, 24, 16, 4],
         [6, 24, 36, 24, 6],
         [4, 16, 24, 16, 4],
         [1, 4, 6, 4, 1]],
        dtype=x4.dtype, device=x4.device,
    ) / 256.0
    return F.conv2d(_reflect_pad(x4, 2), k.view(1, 1, 5, 5))


def _sobel_mag(x4):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=x4.dtype, device=x4.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(_reflect_pad(x4, 1), kx)
    gy = F.conv2d(_reflect_pad(x4, 1), ky)
    return (gx ** 2 + gy ** 2 + 1e-12).sqrt()


def _local_std(x4, k=5):
    w = torch.ones(1, 1, k, k, dtype=x4.dtype, device=x4.device) / (k * k)
    mean = F.conv2d(_reflect_pad(x4, k // 2), w)
    mean_sq = F.conv2d(_reflect_pad(x4 * x4, k // 2), w)
    return (mean_sq - mean ** 2).clamp_min(0.0).sqrt()


def _laplacian(x4):
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                     dtype=x4.dtype, device=x4.device).view(1, 1, 3, 3)
    return F.conv2d(_reflect_pad(x4, 1), k)


def _lbp_like(x4):
    """Fraction of the 8-neighbourhood that is brighter than the centre — a
    smooth, differentiable LBP surrogate in [0, 1]."""
    center = x4
    acc = torch.zeros_like(x4)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = torch.roll(x4, shifts=(dy, dx), dims=(2, 3))
            acc = acc + (shifted >= center).to(x4.dtype)
    return acc / 8.0


def build_brain_terminals(images):
    """Build the grayscale terminal dict.

    Args:
        images: [B, 1, H, W] or [B, C, H, W] (channel 0 used) intensity in [0, 1].
    Returns:
        dict {name: [B, H, W]} keyed by BRAIN_TERMINALS.
    """
    if images.dim() != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(images.shape)}")
    x4 = images[:, :1, :, :]   # [B, 1, H, W]
    terminals = {
        'I_GRAY':     x4,
        'I_BLUR':     _gaussian_blur(x4),
        'I_GRAD':     _sobel_mag(x4),
        'I_LOCALSTD': _local_std(x4),
        'I_LOG':      _laplacian(x4),
        'I_LBP':      _lbp_like(x4),
        'I_LRDIFF':   (x4 - torch.flip(x4, dims=[3])).abs(),  # whole-brain L-R asymmetry
    }
    # Squeeze channel dim -> [B, H, W] (terminal format used by operators).
    out = {}
    for name, t in terminals.items():
        t = torch.nan_to_num(t.squeeze(1), nan=0.0, posinf=1e4, neginf=-1e4)
        out[name] = torch.clamp(t, -1e4, 1e4)
    return out


class BrainTumorVSREnvironment(TensorVSREnvironmentLargeBank):
    """RL environment for grayscale brain-MRI symbolic feature discovery."""

    def __init__(self, data_loader, config: dict, device: str = "cuda"):
        # Inject grayscale terminal names so the base vocabulary uses them.
        config = dict(config)
        config.setdefault('terminal_tokens', BRAIN_TERMINALS)
        super().__init__(data_loader=data_loader, config=config, device=device)

    def get_data_batch(self, batch_size=None, resolution=None):
        """Sample a batch and build grayscale terminals (overrides RGB base)."""
        if batch_size is None:
            batch_size = self.batch_size

        batch_size = min(batch_size, self.cached_images.size(0))
        indices = torch.randint(0, self.cached_images.size(0), (batch_size,))
        images = self.cached_images[indices].to(self.device, non_blocking=True)
        labels = self.cached_labels[indices].to(self.device, non_blocking=True)

        if self.augment:
            images = self._gpu_augment(images)

        if resolution is not None and resolution < images.shape[-1]:
            images = F.interpolate(
                images, size=(resolution, resolution),
                mode='bilinear', align_corners=False,
            )

        terminal_values = build_brain_terminals(images)

        # Layer-2 support (parity with base class): execute L1 bodies as terminals.
        if self.l1_bodies:
            zero = torch.zeros_like(terminal_values['I_GRAY'])
            for i, body_str in enumerate(self.l1_bodies):
                name = f'L1_{i}'
                try:
                    stack = []
                    for token in body_str.strip().split():
                        if token in terminal_values:
                            stack.append(terminal_values[token])
                        elif token in TENSOR_OPERATORS:
                            op_func, arity, _ = TENSOR_OPERATORS[token]
                            if len(stack) < arity:
                                stack = [zero]
                                break
                            operands = [stack.pop() for _ in range(arity)]
                            operands.reverse()
                            r = op_func(*operands)
                            stack.append(torch.nan_to_num(r, nan=0.0, posinf=1e4, neginf=-1e4))
                        else:
                            stack = [zero]
                            break
                    terminal_values[name] = (
                        torch.clamp(stack[0], -1e4, 1e4)
                        if len(stack) == 1 and stack[0].dim() >= 2 else zero
                    )
                except Exception:
                    terminal_values[name] = zero

        return terminal_values, labels
