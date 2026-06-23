"""
HAM10000 Symbolic Feature Discovery RL Environment.

Adapts TensorVSREnvironmentLargeBank for skin lesion analysis:
  - Registers HAM10000-specific operators (color, asymmetry, border, etc.)
  - Uses the same data_loader + config interface as the base environment
  - Terminal channels: I_R, I_G, I_B, I_GRAY, I_H, I_S, I_r, I_g, I_RG, I_BY
    (handled by the base class's get_data_batch method)
"""

import torch
import numpy as np

from src.rl.tensor_environment_large_bank import (
    TensorVSREnvironmentLargeBank,
    TensorTokenVocabulary,
)
from src.symbolic.tensor_operators import TENSOR_OPERATORS
from src.symbolic.ham10000_operators import register_ham10000_operators

# Register HAM10000 operators into the global operator dict
register_ham10000_operators(TENSOR_OPERATORS)


class HAM10000VSREnvironment(TensorVSREnvironmentLargeBank):
    """RL environment for HAM10000 symbolic feature discovery.

    Inherits from TensorVSREnvironmentLargeBank which already handles:
      - RGB/HSV/grayscale/color-ratio terminal channels
      - Large feature bank with diversity penalty
      - LASSO pruning
      - RPN grammar masking
      - GPU augmentation

    This subclass only overrides the vocabulary to include HAM10000-specific
    operator names so they appear in the token space for the RL agent.
    """

    def __init__(
        self,
        data_loader,
        config: dict,
        device: str = "cuda",
    ):
        # The base class will:
        # 1. Register SymbolicKernelBank operators
        # 2. Build vocabulary from TENSOR_OPERATORS (now includes HAM10000 ops)
        # 3. Cache validation data
        # 4. Initialize feature bank
        super().__init__(
            data_loader=data_loader,
            config=config,
            device=device,
        )
