"""
Evaluation metrics for trained registration models.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from affine_utils import compose_affine, decompose_affine, warp_volume_torch

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    val_loader: DataLoader,
    transform_type: str = "affine",
    device: str = "cuda",
) -> dict:
    """
    Compute evaluation metrics on a validation set.

    Returns dict with:
        val_loss_mi: mean MI loss.
        param_mae_rotation_deg: mean absolute error on rotation (degrees).
        param_mae_translation: MAE on translation (normalized coords).
        param_mae_scale: MAE on scale offset (if affine).
        det_mean, det_std, det_min, det_max: determinant statistics.
    """
    from training.losses import get_similarity_loss

    sim_fn = get_similarity_loss("mi")
    model.eval()

    all_mi = []
    all_rot_err = []
    all_trans_err = []
    all_scale_err = []
    all_dets = []

    with torch.no_grad():
        for batch in val_loader:
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)
            has_gt = batch["has_gt"]
            gt_params = batch["gt_params"]

            x = torch.cat([moving, fixed], dim=1)
            params, affine = model(x)
            warped = warp_volume_torch(moving, affine)

            mi = sim_fn(warped, fixed)
            all_mi.append(mi.item())

            # Determinant
            R = affine[:, :3, :3]
            dets = torch.det(R).cpu().numpy()
            all_dets.extend(dets.tolist())

            # Param error (synthetic pairs only)
            if has_gt.any():
                mask = has_gt
                pred = params[mask].cpu().numpy()
                gt = gt_params[mask].numpy()

                rot_err = np.abs(np.degrees(pred[:, :3]) - np.degrees(gt[:, :3]))
                all_rot_err.extend(rot_err.mean(axis=1).tolist())

                trans_err = np.abs(pred[:, 3:6] - gt[:, 3:6])
                all_trans_err.extend(trans_err.mean(axis=1).tolist())

                if pred.shape[1] >= 9:
                    scale_err = np.abs(pred[:, 6:9] - gt[:, 6:9])
                    all_scale_err.extend(scale_err.mean(axis=1).tolist())

    dets_arr = np.array(all_dets)

    metrics = {
        "val_loss_mi": np.mean(all_mi) if all_mi else float("nan"),
        "param_mae_rotation_deg": np.mean(all_rot_err) if all_rot_err else float("nan"),
        "param_mae_translation": np.mean(all_trans_err) if all_trans_err else float("nan"),
        "param_mae_scale": np.mean(all_scale_err) if all_scale_err else float("nan"),
        "det_mean": dets_arr.mean() if len(dets_arr) > 0 else float("nan"),
        "det_std": dets_arr.std() if len(dets_arr) > 0 else float("nan"),
        "det_min": dets_arr.min() if len(dets_arr) > 0 else float("nan"),
        "det_max": dets_arr.max() if len(dets_arr) > 0 else float("nan"),
    }

    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    return metrics
