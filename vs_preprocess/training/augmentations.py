"""
Intensity augmentations for the moving image during training.
Spatial augmentation is handled by SyntheticPairDataset (known affines).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def apply_intensity_augmentation(
    volume: torch.Tensor,
    p: float = 0.5,
) -> torch.Tensor:
    """
    Apply random intensity augmentations to a (1, D, H, W) tensor.

    Each augmentation is applied independently with probability p:
    1. Intensity shift (+/- 0.1)
    2. Intensity scale (* 0.9-1.1)
    3. Gaussian noise (sigma 0.01-0.05)
    4. Simulated smooth bias field (low-freq multiplicative, range 0.9-1.1)

    Returns augmented tensor (same shape), input unchanged.
    """
    vol = volume.clone()

    if np.random.random() < p:
        shift = np.random.uniform(-0.1, 0.1)
        vol = vol + shift

    if np.random.random() < p:
        scale = np.random.uniform(0.9, 1.1)
        vol = vol * scale

    if np.random.random() < p:
        std = np.random.uniform(0.01, 0.05)
        vol = vol + torch.randn_like(vol) * std

    if np.random.random() < p:
        small = torch.rand(1, 1, 8, 8, 8) * 0.2 + 0.9
        bias = F.interpolate(small, size=vol.shape[1:], mode="trilinear", align_corners=True)
        vol = vol * bias.squeeze(0)

    return vol
