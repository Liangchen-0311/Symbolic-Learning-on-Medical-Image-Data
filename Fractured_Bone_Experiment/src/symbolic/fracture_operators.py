"""
Medical / Bone Fracture Specific Tensor Operators.

Extends the base TensorOperators with operators specifically designed for
X-ray bone fracture detection:

  1. Fracture line detection: directional edge operators tuned for
     linear, oblique, spiral, and transverse fracture lines
  2. Bone texture analysis: local entropy, GLCM-like measures,
     cortical bone continuity
  3. Morphological fracture indicators: discontinuity detection,
     displacement measurement, cortical break detection
  4. Intensity-based: bone density estimation, soft tissue swelling
  5. Symmetry operators: left-right asymmetry for bilateral comparison

All methods work on batched tensors: [batch, H, W]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FractureOperators:
    """Bone fracture specific spatial operators for X-ray image processing."""

    # ============================================
    # Fracture Line Detection
    # ============================================

    @staticmethod
    def edge_diag_45(x):
        """Diagonal edge detection at 45 degrees.

        Detects oblique and spiral fracture lines running at 45 degrees.
        Kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] rotated 45 degrees.
        """
        kernel = torch.tensor([
            [2., -1., 0.],
            [-1., 2., -1.],
            [0., -1., 2.]
        ], device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)

    @staticmethod
    def edge_diag_135(x):
        """Diagonal edge detection at 135 degrees.

        Detects oblique fracture lines running at 135 degrees.
        """
        kernel = torch.tensor([
            [0., -1., 2.],
            [-1., 2., -1.],
            [2., -1., 0.]
        ], device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)

    @staticmethod
    def line_detector_h(x):
        """Horizontal line detector.

        Detects transverse fracture lines (horizontal breaks).
        Kernel responds to thin horizontal bright/dark lines.
        """
        kernel = torch.tensor([
            [-1., -1., -1.],
            [2., 2., 2.],
            [-1., -1., -1.]
        ], device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)

    @staticmethod
    def line_detector_v(x):
        """Vertical line detector.

        Detects vertical fracture lines.
        """
        kernel = torch.tensor([
            [-1., 2., -1.],
            [-1., 2., -1.],
            [-1., 2., -1.]
        ], device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)

    @staticmethod
    def line_detector_45(x):
        """45-degree line detector.

        Detects oblique fracture lines at 45 degrees.
        """
        kernel = torch.tensor([
            [2., -1., -1.],
            [-1., 2., -1.],
            [-1., -1., 2.]
        ], device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)

    @staticmethod
    def line_detector_135(x):
        """135-degree line detector.

        Detects oblique fracture lines at 135 degrees.
        """
        kernel = torch.tensor([
            [-1., -1., 2.],
            [-1., 2., -1.],
            [2., -1., -1.]
        ], device=x.device).view(1, 1, 3, 3)
        return F.conv2d(x.unsqueeze(1), kernel, padding=1).squeeze(1)

    # ============================================
    # Bone Texture Analysis
    # ============================================

    @staticmethod
    def local_entropy_5x5(x):
        """Local entropy over 5x5 patches.

        Measures texture randomness — fracture zones often show
        disrupted trabecular patterns with higher entropy.
        """
        B, H, W = x.shape
        n_bins = 16
        x_4d = x.unsqueeze(1)

        x_flat = x.reshape(B, -1)
        x_min = x_flat.min(dim=1, keepdim=True).values
        x_max = x_flat.max(dim=1, keepdim=True).values
        x_norm = (x - x_min.unsqueeze(-1)) / (x_max.unsqueeze(-1) - x_min.unsqueeze(-1) + 1e-8)

        bin_idx = (x_norm * (n_bins - 1e-4)).long().clamp(0, n_bins - 1)
        one_hot = F.one_hot(bin_idx, n_bins).float()

        kernel_size = 5
        pad = kernel_size // 2
        local_counts = torch.zeros(B, n_bins, H, W, device=x.device)
        for b in range(n_bins):
            local_counts[:, b] = F.avg_pool2d(one_hot[:, :, :, b].unsqueeze(1),
                                               kernel_size=kernel_size, stride=1,
                                               padding=pad).squeeze(1) * (kernel_size * kernel_size)

        local_probs = local_counts / (local_counts.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(local_probs * (local_probs + 1e-10).log()).sum(dim=1)
        return entropy

    @staticmethod
    def local_range(x):
        """Local range (max - min) over 5x5 patches.

        High range indicates sharp intensity transitions — typical
        at fracture edges and cortical bone boundaries.
        """
        x_4d = x.unsqueeze(1)
        local_max = F.max_pool2d(x_4d, kernel_size=5, stride=1, padding=2)
        local_min = -F.max_pool2d(-x_4d, kernel_size=5, stride=1, padding=2)
        return (local_max - local_min).squeeze(1)

    @staticmethod
    def cortical_continuity(x):
        """Cortical bone continuity measure.

        Detects disruptions in cortical bone by measuring local
        gradient consistency. Low values = potential fracture.
        Computed as: 1 - (local variance of gradient direction).
        """
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               device=x.device).view(1, 1, 3, 3)
        x_4d = x.unsqueeze(1)
        gx = F.conv2d(x_4d, sobel_x, padding=1).squeeze(1)
        gy = F.conv2d(x_4d, sobel_y, padding=1).squeeze(1)

        orientation = torch.atan2(gy, gx + 1e-8)
        x_4d_o = orientation.unsqueeze(1)
        local_var = F.avg_pool2d(x_4d_o * x_4d_o, 5, 1, 2) - \
                    F.avg_pool2d(x_4d_o, 5, 1, 2) ** 2
        continuity = 1.0 - torch.clamp(local_var.squeeze(1) / (math.pi ** 2), 0, 1)
        return continuity

    # ============================================
    # Morphological Fracture Indicators
    # ============================================

    @staticmethod
    def black_tophat(x):
        """Black top-hat transform: closing(x) - x.

        Extracts dark thin structures on bright background.
        In X-rays, this detects dark fracture lines within
        bright cortical bone.
        """
        x_4d = x.unsqueeze(1)
        dilated = F.max_pool2d(x_4d, kernel_size=3, stride=1, padding=1)
        closed = -F.max_pool2d(-dilated, kernel_size=3, stride=1, padding=1)
        return (closed - x_4d).squeeze(1)

    @staticmethod
    def white_tophat(x):
        """White top-hat transform: x - opening(x).

        Extracts bright thin structures on dark background.
        Detects bone fragments, callus formation, and
        bright fracture edges.
        """
        x_4d = x.unsqueeze(1)
        eroded = -F.max_pool2d(-x_4d, kernel_size=3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, kernel_size=3, stride=1, padding=1)
        return (x_4d - opened).squeeze(1)

    @staticmethod
    def discontinuity_map(x):
        """Detect intensity discontinuities that may indicate fractures.

        Combines gradient magnitude with local range to highlight
        regions where bone continuity is disrupted.
        """
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               device=x.device).view(1, 1, 3, 3)
        x_4d = x.unsqueeze(1)
        gx = F.conv2d(x_4d, sobel_x, padding=1).squeeze(1)
        gy = F.conv2d(x_4d, sobel_y, padding=1).squeeze(1)
        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)

        lr = FractureOperators.local_range(x)
        return grad_mag * lr

    @staticmethod
    def displacement_indicator(x):
        """Detect bone displacement by measuring local gradient direction changes.

        Displaced fractures show abrupt changes in gradient direction
        at the fracture site.
        """
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               device=x.device).view(1, 1, 3, 3)
        x_4d = x.unsqueeze(1)
        gx = F.conv2d(x_4d, sobel_x, padding=1).squeeze(1)
        gy = F.conv2d(x_4d, sobel_y, padding=1).squeeze(1)

        grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        safe_mag = torch.clamp(grad_mag, min=1e-4)

        cos_dir = gx / safe_mag
        sin_dir = gy / safe_mag

        cos_var = F.avg_pool2d((cos_dir * cos_dir).unsqueeze(1), 7, 1, 3).squeeze(1) - \
                  F.avg_pool2d(cos_dir.unsqueeze(1), 7, 1, 3).squeeze(1) ** 2
        sin_var = F.avg_pool2d((sin_dir * sin_dir).unsqueeze(1), 7, 1, 3).squeeze(1) - \
                  F.avg_pool2d(sin_dir.unsqueeze(1), 7, 1, 3).squeeze(1) ** 2

        return torch.clamp(cos_var + sin_var, 0, 1)

    # ============================================
    # Intensity-based Operators
    # ============================================

    @staticmethod
    def bone_enhance(x):
        """Enhance bone structures using unsharp masking.

        Sharpens bone edges and cortical boundaries while
        suppressing soft tissue.
        """
        blurred = F.avg_pool2d(x.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        enhanced = x + 1.5 * (x - blurred)
        return torch.clamp(enhanced, 0, 1)

    @staticmethod
    def threshold_bone(x):
        """Simple bone segmentation via thresholding.

        Assumes bright regions are bone. Uses adaptive threshold
        based on local statistics.
        """
        local_mean = F.avg_pool2d(x.unsqueeze(1), kernel_size=15, stride=1, padding=7).squeeze(1)
        local_std = torch.sqrt(
            F.avg_pool2d((x ** 2).unsqueeze(1), kernel_size=15, stride=1, padding=7).squeeze(1)
            - local_mean ** 2 + 1e-8
        )
        threshold = local_mean + 0.5 * local_std
        return torch.sigmoid(10 * (x - threshold))

    @staticmethod
    def soft_tissue_suppress(x):
        """Suppress soft tissue to isolate bone structures.

        Uses morphological approach: large-scale opening removes
        soft tissue gradients while preserving bone edges.
        """
        x_4d = x.unsqueeze(1)
        eroded = -F.max_pool2d(-x_4d, kernel_size=7, stride=1, padding=3)
        opened = F.max_pool2d(eroded, kernel_size=7, stride=1, padding=3)
        return opened.squeeze(1)

    # ============================================
    # Symmetry / Bilateral Comparison
    # ============================================

    @staticmethod
    def lr_asymmetry(x):
        """Left-right asymmetry measure.

        Computes absolute difference between left and right halves.
        Fractures often cause asymmetric appearance in bilateral bones.
        """
        H = x.shape[-2]
        left = x[:, :, :H // 2]
        right = x[:, :, H // 2:]
        right_flipped = right.flip(-1)

        min_w = min(left.shape[-1], right_flipped.shape[-1])
        left = left[:, :, :min_w]
        right_flipped = right_flipped[:, :, :min_w]

        return torch.abs(left - right_flipped)

    @staticmethod
    def tb_asymmetry(x):
        """Top-bottom asymmetry measure.

        Useful for detecting fractures in long bones where
        the proximal and distal segments should be symmetric.
        """
        W = x.shape[-1]
        top = x[:, :x.shape[-2] // 2, :]
        bottom = x[:, x.shape[-2] // 2:, :]
        bottom_flipped = bottom.flip(-2)

        min_h = min(top.shape[-2], bottom_flipped.shape[-2])
        top = top[:, :min_h, :]
        bottom_flipped = bottom_flipped[:, :min_h, :]

        return torch.abs(top - bottom_flipped)

    # ============================================
    # Multi-scale Fracture Detection
    # ============================================

    @staticmethod
    def multi_scale_edge(x):
        """Multi-scale edge detection for fractures of different sizes.

        Combines edges at 3x3, 5x5, and 7x7 scales.
        Small fractures respond to fine scale, comminuted fractures
        to coarse scale.
        """
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               device=x.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
                               device=x.device).view(1, 1, 3, 3)

        x_4d = x.unsqueeze(1)
        gx = F.conv2d(x_4d, sobel_x, padding=1).squeeze(1)
        gy = F.conv2d(x_4d, sobel_y, padding=1).squeeze(1)
        edge_fine = torch.sqrt(gx * gx + gy * gy + 1e-8)

        x_blur5 = F.avg_pool2d(x_4d, kernel_size=5, stride=1, padding=2).squeeze(1)
        x_4d5 = x_blur5.unsqueeze(1)
        gx5 = F.conv2d(x_4d5, sobel_x, padding=1).squeeze(1)
        gy5 = F.conv2d(x_4d5, sobel_y, padding=1).squeeze(1)
        edge_medium = torch.sqrt(gx5 * gx5 + gy5 * gy5 + 1e-8)

        x_blur7 = F.avg_pool2d(x_4d, kernel_size=7, stride=1, padding=3).squeeze(1)
        x_4d7 = x_blur7.unsqueeze(1)
        gx7 = F.conv2d(x_4d7, sobel_x, padding=1).squeeze(1)
        gy7 = F.conv2d(x_4d7, sobel_y, padding=1).squeeze(1)
        edge_coarse = torch.sqrt(gx7 * gx7 + gy7 * gy7 + 1e-8)

        return edge_fine + edge_medium + edge_coarse

    @staticmethod
    def blob_detector(x):
        """Detect blob-like structures (bone fragments, callus).

        Uses Difference of Gaussians at multiple scales.
        """
        x_4d = x.unsqueeze(1)
        g1 = F.avg_pool2d(x_4d, kernel_size=3, stride=1, padding=1)
        g3 = F.avg_pool2d(x_4d, kernel_size=7, stride=1, padding=3)
        g5 = F.avg_pool2d(x_4d, kernel_size=13, stride=1, padding=6)

        dog1 = (g1 - g3).squeeze(1)
        dog2 = (g3 - g5).squeeze(1)

        return torch.abs(dog1) + torch.abs(dog2)


# ============================================
# Register Fracture Operators into TENSOR_OPERATORS
# ============================================

FRACTURE_OPERATORS = {
    'edge_diag_45': (FractureOperators.edge_diag_45, 1, 'tensor'),
    'edge_diag_135': (FractureOperators.edge_diag_135, 1, 'tensor'),
    'line_h': (FractureOperators.line_detector_h, 1, 'tensor'),
    'line_v': (FractureOperators.line_detector_v, 1, 'tensor'),
    'line_45': (FractureOperators.line_detector_45, 1, 'tensor'),
    'line_135': (FractureOperators.line_detector_135, 1, 'tensor'),
    'local_entropy': (FractureOperators.local_entropy_5x5, 1, 'tensor'),
    'local_range': (FractureOperators.local_range, 1, 'tensor'),
    'cortical_cont': (FractureOperators.cortical_continuity, 1, 'tensor'),
    'black_tophat': (FractureOperators.black_tophat, 1, 'tensor'),
    'white_tophat': (FractureOperators.white_tophat, 1, 'tensor'),
    'discont_map': (FractureOperators.discontinuity_map, 1, 'tensor'),
    'displace_ind': (FractureOperators.displacement_indicator, 1, 'tensor'),
    'bone_enhance': (FractureOperators.bone_enhance, 1, 'tensor'),
    'threshold_bone': (FractureOperators.threshold_bone, 1, 'tensor'),
    'soft_suppress': (FractureOperators.soft_tissue_suppress, 1, 'tensor'),
    'lr_asymmetry': (FractureOperators.lr_asymmetry, 1, 'tensor'),
    'tb_asymmetry': (FractureOperators.tb_asymmetry, 1, 'tensor'),
    'ms_edge': (FractureOperators.multi_scale_edge, 1, 'tensor'),
    'blob_detect': (FractureOperators.blob_detector, 1, 'tensor'),
}


def register_fracture_operators(tensor_operators_dict):
    """Register all fracture-specific operators into the global TENSOR_OPERATORS dict."""
    for name, (func, arity, output_type) in FRACTURE_OPERATORS.items():
        tensor_operators_dict[name] = (func, arity, output_type)
    print(f"[FractureOps] Registered {len(FRACTURE_OPERATORS)} fracture-specific operators")
    return tensor_operators_dict
