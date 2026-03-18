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


class ThreeTierLoss(torch.nn.Module):
    """
    Routes similarity loss based on pair_type:
        0 (SAME_SUB):    NCC
        1 (CROSS_SUB):   MI
        2 (CROSS_SUPER): MI

    Uses EMA normalization so NCC and MI contribute equally.
    """

    def __init__(
        self,
        ncc_kernel_size: int = 9,
        mi_num_bins: int = 32,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.ncc_fn = get_similarity_loss("ncc", kernel_size=ncc_kernel_size)
        self.mi_fn = get_similarity_loss("mi", num_bins=mi_num_bins)
        self.ema_decay = ema_decay
        self.ema_ncc = 0.0
        self.ema_mi = 0.0

    def forward(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        pair_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mixed NCC/MI loss based on pair types.

        Args:
            warped: (B, 1, D, H, W)
            fixed: (B, 1, D, H, W)
            pair_types: (B,) int tensor with values 0, 1, or 2

        Returns:
            Scalar loss (mean over batch, EMA-normalized).
        """
        ncc_mask = pair_types == 0
        mi_mask = (pair_types == 1) | (pair_types == 2)

        losses = []

        if ncc_mask.any():
            ncc_val = self.ncc_fn(warped[ncc_mask], fixed[ncc_mask])
            raw_ncc = ncc_val.detach().abs().item()
            if self.ema_ncc == 0.0:
                self.ema_ncc = raw_ncc if raw_ncc > 1e-8 else 1.0
            else:
                self.ema_ncc = (
                    self.ema_decay * self.ema_ncc
                    + (1 - self.ema_decay) * raw_ncc
                )
            normalized_ncc = ncc_val / max(self.ema_ncc, 1e-8)
            losses.append(normalized_ncc * ncc_mask.sum() / pair_types.shape[0])

        if mi_mask.any():
            mi_val = self.mi_fn(warped[mi_mask], fixed[mi_mask])
            raw_mi = mi_val.detach().abs().item()
            if self.ema_mi == 0.0:
                self.ema_mi = raw_mi if raw_mi > 1e-8 else 1.0
            else:
                self.ema_mi = (
                    self.ema_decay * self.ema_mi
                    + (1 - self.ema_decay) * raw_mi
                )
            normalized_mi = mi_val / max(self.ema_mi, 1e-8)
            losses.append(normalized_mi * mi_mask.sum() / pair_types.shape[0])

        if not losses:
            return torch.tensor(0.0, device=warped.device, requires_grad=True)

        return sum(losses)


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
