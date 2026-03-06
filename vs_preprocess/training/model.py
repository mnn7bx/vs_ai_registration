"""
GlobalNet with decomposed affine head for registration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import GlobalNet

# Allow import from parent package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from affine_utils import compose_affine


class _DecomposedHead(nn.Module):
    """
    Replacement for MONAI's AffineHead.
    Receives the list of decoded feature maps from GlobalNet.forward,
    flattens the bottom-most one, and projects to num_params via an MLP
    initialized so the output is zero -> identity transform at init.
    """

    def __init__(self, in_features: int, num_params: int, hidden: int = 0, dropout: float = 0.0):
        super().__init__()
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_params),
            )
            # Zero-init final layer for identity at init
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.net = nn.Linear(in_features, num_params)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)

    def forward(self, x, image_size=None):
        # x is a list of tensors; x[0] is the bottom-most decoded feature map
        feat = x[0]
        return self.net(feat.reshape(feat.shape[0], -1))


class AffineRegNet(nn.Module):
    """
    MONAI GlobalNet encoder + decomposed affine parameter head.

    MONAI's GlobalNet default final layer outputs 12 params (raw 3x4 matrix).
    We replace it to output 9 (affine) or 6 (rigid) decomposed parameters.
    Zero initialization -> identity transform at init.
    """

    def __init__(
        self,
        image_size=(96, 96, 96),
        in_channels=2,
        num_channel_initial=16,
        depth=4,
        transform_type="affine",
        head_hidden=0,
        head_dropout=0.0,
    ):
        super().__init__()
        self.transform_type = transform_type
        self.num_params = 9 if transform_type == "affine" else 6
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_channel_initial = num_channel_initial
        self.depth = depth
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout

        self.backbone = GlobalNet(
            image_size=image_size,
            spatial_dims=3,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            depth=depth,
        )

        # Replace output_block with our decomposed head
        # The bottom-most decoded feature map fed into the head is x[0]
        # which comes from bottom_block output; its in_features matches
        # the existing AffineHead's fc in_features.
        in_features = self.backbone.output_block.fc.in_features
        self.backbone.output_block = _DecomposedHead(
            in_features, self.num_params, hidden=head_hidden, dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 2, D, H, W) concatenated [moving, fixed].

        Returns:
            params: (B, num_params) decomposed parameters.
            affine: (B, 3, 4) composed affine matrix.
        """
        raw = self.backbone(x)

        # MONAI GlobalNet may return (ddf, affine) tuple in some versions
        if isinstance(raw, (tuple, list)):
            raw = raw[0]

        # If MONAI reshaped to (B, 3, 4), flatten it
        if raw.ndim == 3:
            raw = raw.reshape(raw.shape[0], -1)

        params = raw[:, : self.num_params]
        affine = compose_affine(params, transform_type=self.transform_type)
        return params, affine

    def get_config(self) -> dict:
        return {
            "image_size": list(self.image_size),
            "in_channels": self.in_channels,
            "num_channel_initial": self.num_channel_initial,
            "depth": self.depth,
            "transform_type": self.transform_type,
            "num_params": self.num_params,
            "head_hidden": self.head_hidden,
            "head_dropout": self.head_dropout,
        }
