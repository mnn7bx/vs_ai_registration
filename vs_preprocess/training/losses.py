"""
Loss functions for affine registration training.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from monai.losses import GlobalMutualInformationLoss


def get_similarity_loss(loss_type: str = "mi", **kwargs):
    """
    Factory for similarity loss.

    "mi": GlobalMutualInformationLoss (returns negative MI, minimize it).
    "ncc": LocalNormalizedCrossCorrelationLoss (returns negative NCC).
    """
    if loss_type == "mi":
        return GlobalMutualInformationLoss(
            num_bins=kwargs.get("num_bins", 32),
            sigma_ratio=kwargs.get("sigma_ratio", 0.5),
        )
    elif loss_type == "ncc":
        from monai.losses import LocalNormalizedCrossCorrelationLoss
        return LocalNormalizedCrossCorrelationLoss(
            spatial_dims=3,
            kernel_size=kwargs.get("kernel_size", 9),
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def identity_regularization(params: torch.Tensor) -> torch.Tensor:
    """L2 penalty on params deviating from zero (identity)."""
    return torch.mean(params ** 2)


def determinant_regularization(affine: torch.Tensor) -> torch.Tensor:
    """Penalize det(rotation+scale) deviating from 1.0."""
    R = affine[:, :3, :3]
    det = torch.det(R)
    return torch.mean((det - 1.0) ** 2)


def supervised_param_loss(
    predicted: torch.Tensor,
    gt: torch.Tensor,
) -> torch.Tensor:
    """
    MSE between predicted and ground-truth decomposed params.
    Applied only to synthetic pairs with known transforms.

    Weights: rotation + translation equally, scale with 0.1 weight.
    """
    rot = F.mse_loss(predicted[:, :3], gt[:, :3])
    trans = F.mse_loss(predicted[:, 3:6], gt[:, 3:6])
    loss = rot + trans
    if predicted.shape[1] >= 9 and gt.shape[1] >= 9:
        scale = F.mse_loss(predicted[:, 6:9], gt[:, 6:9])
        loss = loss + 0.1 * scale
    return loss
