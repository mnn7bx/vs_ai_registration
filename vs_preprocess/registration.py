"""
GlobalNet inference and affine application for longitudinal alignment.
"""

from __future__ import annotations

import json
import logging
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple

from affine_utils import (
    compose_affine,
    decompose_affine,
    normalized_affine_to_world,
    apply_affine_to_volume,
)
from preprocessing import preprocess_for_network

logger = logging.getLogger(__name__)

NET_INPUT_SHAPE = (96, 96, 96)


def load_model(model_path: Path, device: str = "cuda"):
    """
    Load a trained AffineRegNet from a checkpoint.

    Checkpoint must contain:
        'model_state_dict': OrderedDict (saved from AffineRegNet.state_dict())
        'config': dict with image_size, transform_type, num_params, etc.

    Returns:
        (model, config) tuple.  model is an AffineRegNet in eval mode.
    """
    # Import here to avoid circular import at module level
    from training.model import AffineRegNet

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Rebuild the SAME AffineRegNet class used during training.
    # This is critical: train_intra.py saves AffineRegNet.state_dict() which
    # has keys like "backbone.final.weight". A raw GlobalNet would expect
    # "final.weight" and fail with missing key errors.
    model = AffineRegNet(
        image_size=tuple(config["image_size"]),
        transform_type=config["transform_type"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded model from %s (type=%s, params=%d)",
        model_path,
        config["transform_type"],
        config["num_params"],
    )
    return model, config


def predict_affine(
    model: nn.Module,
    moving_tensor: torch.Tensor,
    fixed_tensor: torch.Tensor,
    transform_type: str = "affine",
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Predict affine aligning moving to fixed.

    Args:
        model: trained AffineRegNet (not raw GlobalNet).
        moving_tensor: (1, 1, 96, 96, 96) preprocessed moving.
        fixed_tensor: (1, 1, 96, 96, 96) preprocessed fixed.
        transform_type: "affine" or "rigid".
        device: "cuda" or "cpu".

    Returns:
        affine_3x4: (3, 4) numpy — predicted affine in normalized coords.
        params_raw: (9,) or (6,) numpy — raw network output.
        params_dict: dict from decompose_affine — human-readable.
    """
    x = torch.cat([moving_tensor, fixed_tensor], dim=1).to(device)

    with torch.no_grad():
        # AffineRegNet.forward() returns (params, affine)
        result = model(x)

        if isinstance(result, tuple) and len(result) == 2:
            raw_params, affine = result  # (B, num_params), (B, 3, 4)
        else:
            # Fallback for raw model output
            raw_params = result
            if raw_params.ndim == 3:
                raw_params = raw_params.reshape(raw_params.shape[0], -1)
            num_params = 9 if transform_type == "affine" else 6
            raw_params = raw_params[:, :num_params]
            affine = compose_affine(raw_params, transform_type=transform_type)

    affine_np = affine[0].cpu().numpy()
    params_np = raw_params[0].cpu().numpy()
    params_dict = decompose_affine(affine_np)

    return affine_np, params_np, params_dict


def align_patient_longitudinal(
    patient_records: List,
    model: nn.Module,
    output_dir: Path,
    transform_type: str = "affine",
    keep_intermediates: bool = True,
    device: str = "cuda",
) -> Dict[str, dict]:
    """
    Align all timepoints of one patient to the baseline.

    Args:
        patient_records: sorted by timepoint, first is baseline. List of ScanRecord.
        model: trained intra-patient GlobalNet.
        output_dir: root output directory.
        transform_type: "affine" or "rigid".
        keep_intermediates: save intermediate NIfTI files.
        device: "cuda" or "cpu".

    Returns:
        dict mapping timepoint -> {
            'affine_3x4': np.ndarray (3, 4) in normalized coords,
            'world_affine': np.ndarray (4, 4) in world coords (None for baseline),
            'params_dict': dict,
            'output_dir': Path,
            'warped_nii': nib.Nifti1Image or None (None for baseline),
        }
    """
    baseline = patient_records[0]

    # Cache baseline tensor (used as fixed for all follow-ups)
    baseline_tensor = preprocess_for_network(baseline.volume_path, NET_INPUT_SHAPE)

    results: Dict[str, dict] = {}

    # ── Baseline: identity ──
    bl_out_dir = (
        output_dir / baseline.modality / baseline.patient_id / baseline.patient_timepoint
    )
    bl_out_dir.mkdir(parents=True, exist_ok=True)

    identity_3x4 = np.eye(3, 4, dtype=np.float64)
    identity_3x4_flat = identity_3x4.copy()
    np.save(bl_out_dir / f"{baseline.patient_timepoint}_affine.npy", identity_3x4_flat)

    params_dict_bl = decompose_affine(identity_3x4)
    with open(bl_out_dir / f"{baseline.patient_timepoint}_params.json", "w") as f:
        json.dump(params_dict_bl, f, indent=2)

    results[baseline.timepoint] = {
        "affine_3x4": identity_3x4,
        "world_affine": None,
        "params_dict": params_dict_bl,
        "output_dir": bl_out_dir,
        "warped_nii": None,
    }

    logger.info("  %s = baseline (identity)", baseline.patient_timepoint)

    # ── Follow-ups ──
    baseline_nii = nib.load(str(baseline.volume_path))

    for record in patient_records[1:]:
        out_dir = (
            output_dir / record.modality / record.patient_id / record.patient_timepoint
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # Predict affine
        moving_tensor = preprocess_for_network(record.volume_path, NET_INPUT_SHAPE)
        affine_3x4, params_raw, params_dict = predict_affine(
            model, moving_tensor, baseline_tensor, transform_type, device
        )

        # Convert to world-space
        moving_nii = nib.load(str(record.volume_path))
        world_affine = normalized_affine_to_world(
            affine_3x4, moving_nii, baseline_nii, NET_INPUT_SHAPE
        )

        # Apply to full-resolution volume
        warped_nii = apply_affine_to_volume(
            moving_nii, world_affine, baseline_nii, interpolation="trilinear"
        )

        # Save affine and params
        np.save(out_dir / f"{record.patient_timepoint}_affine.npy", affine_3x4)
        with open(out_dir / f"{record.patient_timepoint}_params.json", "w") as f:
            json.dump(params_dict, f, indent=2)

        if keep_intermediates:
            inter_dir = out_dir / "intermediates"
            inter_dir.mkdir(exist_ok=True)
            nib.save(
                warped_nii,
                str(inter_dir / f"{record.patient_timepoint}_affine_aligned.nii.gz"),
            )

        results[record.timepoint] = {
            "affine_3x4": affine_3x4,
            "world_affine": world_affine,
            "params_dict": params_dict,
            "output_dir": out_dir,
            "warped_nii": warped_nii,
        }

        logger.info(
            "  %s -> baseline: rot=(%.1f, %.1f, %.1f)deg trans=(%.1f, %.1f, %.1f)",
            record.patient_timepoint,
            *params_dict["rotation_deg"],
            *params_dict["translation"],
        )

    return results
