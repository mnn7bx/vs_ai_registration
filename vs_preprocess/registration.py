"""
GlobalNet inference and affine application for longitudinal alignment.

IMPORTANT: Inference preprocessing MUST match training preprocessing exactly.
Training uses v2 preprocessing: isotropic resample → overlap-aware crop → normalize.
See training/dataset.py:LongitudinalPairDataset.__getitem__() for the reference.
"""

from __future__ import annotations

import json
import logging
import sys
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure vs_preprocess/ is importable as a flat package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from affine_utils import (
    compose_affine,
    decompose_affine,
    normalized_affine_to_world_v2,
    apply_affine_to_volume,
)
from preprocessing import (
    load_and_resample_isotropic,
    center_crop_or_pad,
    compute_overlap_center,
)

logger = logging.getLogger(__name__)

NET_INPUT_SHAPE = (96, 96, 96)
MODEL_TARGET_SPACING = (1.0, 1.0, 1.0)
MODEL_OUTPUT_SPACING = (1.0, 1.0, 1.0)


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

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    config = checkpoint["config"]

    # Rebuild the SAME AffineRegNet class used during training.
    model = AffineRegNet(
        image_size=tuple(config["image_size"]),
        transform_type=config["transform_type"],
        head_hidden=config.get("head_hidden", 0),
        head_dropout=config.get("head_dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Update module-level defaults to match the trained model
    global NET_INPUT_SHAPE, MODEL_TARGET_SPACING, MODEL_OUTPUT_SPACING
    NET_INPUT_SHAPE = tuple(config["image_size"])
    MODEL_TARGET_SPACING = tuple(config.get("target_spacing", [1.0, 1.0, 1.0]))
    MODEL_OUTPUT_SPACING = tuple(config.get("output_spacing", config.get("target_spacing", [1.0, 1.0, 1.0])))

    logger.info(
        "Loaded model from %s (type=%s, params=%d, input=%s, spacing=%s)",
        model_path,
        config["transform_type"],
        config["num_params"],
        NET_INPUT_SHAPE,
        MODEL_TARGET_SPACING,
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
        moving_tensor: (1, 1, D, H, W) preprocessed moving.
        fixed_tensor: (1, 1, D, H, W) preprocessed fixed.
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

    affine_np = affine[0].cpu().numpy().astype(np.float64)
    params_np = raw_params[0].cpu().numpy()
    params_dict = decompose_affine(affine_np)

    return affine_np, params_np, params_dict


def _crop_and_normalize(
    data: np.ndarray,
    target_size: Tuple[int, int, int],
    center=None,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Center crop/pad a volume and normalize to [0, 1].

    Returns:
        tensor: (1, 1, D, H, W) float32 tensor.
        crop_origin: (i, j, k) offset of crop in input coordinates.
    """
    cropped, crop_origin = center_crop_or_pad(data, target_size, center=center)
    vmin, vmax = cropped.min(), cropped.max()
    if vmax - vmin > 1e-8:
        cropped = (cropped - vmin) / (vmax - vmin)
    else:
        cropped = np.zeros_like(cropped)
    tensor = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
    return tensor, crop_origin


def _compute_crop_affine(
    resampled_affine: np.ndarray,
    crop_origin: Tuple[int, int, int],
) -> np.ndarray:
    """
    Compute voxel-to-world affine for a cropped sub-volume.

    The rotation/scale part is unchanged (crop doesn't change voxel size).
    Only the origin shifts by crop_origin voxels in world coordinates.
    """
    crop_affine = resampled_affine.copy()
    crop_offset = resampled_affine[:3, :3] @ np.array(crop_origin, dtype=np.float64)
    crop_affine[:3, 3] = resampled_affine[:3, 3] + crop_offset
    return crop_affine


def align_patient_longitudinal(
    patient_records: List,
    model: nn.Module,
    output_dir: Path,
    transform_type: str = "affine",
    keep_intermediates: bool = True,
    device: str = "cuda",
    target_spacing: Tuple[float, float, float] = None,
    output_spacing: Tuple[float, float, float] = None,
) -> Dict[str, dict]:
    """
    Align all timepoints of one patient to the baseline.

    Preprocessing matches training exactly:
        1. Resample both volumes to isotropic spacing
        2. Compute overlap center between fixed and moving
        3. Center crop/pad both to NET_INPUT_SHAPE at the overlap center
        4. Intensity normalize to [0, 1]

    Args:
        patient_records: sorted by timepoint, first is baseline. List of ScanRecord.
        model: trained intra-patient AffineRegNet.
        output_dir: root output directory.
        transform_type: "affine" or "rigid".
        keep_intermediates: save intermediate NIfTI files.
        device: "cuda" or "cpu".
        target_spacing: isotropic spacing used during training. Defaults to
            MODEL_TARGET_SPACING (set by load_model from checkpoint config).
        output_spacing: if provided, resample moving volume to this spacing
            before applying the affine warp. Useful for producing higher-res
            outputs (e.g. 0.5mm) than the training spacing. Defaults to None
            (warp original-resolution moving volume).

    Returns:
        dict mapping timepoint -> {
            'affine_3x4': np.ndarray (3, 4) in normalized coords,
            'world_affine': np.ndarray (4, 4) in world coords (None for baseline),
            'params_dict': dict,
            'output_dir': Path,
            'warped_nii': nib.Nifti1Image or None (None for baseline),
        }
    """
    spacing = target_spacing or MODEL_TARGET_SPACING
    baseline = patient_records[0]

    # ── Load and resample baseline to isotropic (cached for all follow-ups) ──
    fix_data, fix_affine = load_and_resample_isotropic(baseline.volume_path, spacing)

    results: Dict[str, dict] = {}

    # ── Baseline: identity ──
    bl_out_dir = (
        output_dir / baseline.modality / baseline.patient_id / baseline.patient_timepoint
    )
    bl_out_dir.mkdir(parents=True, exist_ok=True)

    identity_3x4 = np.eye(3, 4, dtype=np.float64)
    np.save(bl_out_dir / f"{baseline.patient_timepoint}_affine.npy", identity_3x4)

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
    # Load full-resolution volumes for the final warping step
    baseline_nii = nib.load(str(baseline.volume_path))

    for record in patient_records[1:]:
        out_dir = (
            output_dir / record.modality / record.patient_id / record.patient_timepoint
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── Step 1: Load and resample moving to isotropic ──
        mov_data, mov_affine = load_and_resample_isotropic(record.volume_path, spacing)

        # ── Step 2: Compute overlap center (matching training exactly) ──
        # See training/dataset.py:LongitudinalPairDataset.__getitem__()
        overlap_center_fix = compute_overlap_center(
            fix_affine, fix_data.shape, mov_affine, mov_data.shape
        )

        # Convert overlap center to moving-volume voxel coords
        center_world = (
            fix_affine[:3, :3] @ np.array(overlap_center_fix, dtype=np.float64)
            + fix_affine[:3, 3]
        )
        inv_mov = np.linalg.inv(mov_affine)
        overlap_center_mov = inv_mov[:3, :3] @ center_world + inv_mov[:3, 3]
        overlap_center_mov = tuple(int(round(v)) for v in overlap_center_mov)
        overlap_center_mov = tuple(
            max(0, min(c, s - 1))
            for c, s in zip(overlap_center_mov, mov_data.shape)
        )

        # ── Step 3: Crop both centered on overlap, normalize ──
        fix_tensor, fix_crop_origin = _crop_and_normalize(
            fix_data, NET_INPUT_SHAPE, center=overlap_center_fix
        )
        mov_tensor, mov_crop_origin = _crop_and_normalize(
            mov_data, NET_INPUT_SHAPE, center=overlap_center_mov
        )

        # ── Step 4: Predict affine ──
        affine_3x4, params_raw, params_dict = predict_affine(
            model, mov_tensor, fix_tensor, transform_type, device
        )

        # ── Step 5: Convert to world-space using crop-aware v2 conversion ──
        fix_crop_affine = _compute_crop_affine(fix_affine, fix_crop_origin)
        mov_crop_affine = _compute_crop_affine(mov_affine, mov_crop_origin)

        world_affine = normalized_affine_to_world_v2(
            affine_3x4, mov_crop_affine, fix_crop_affine, NET_INPUT_SHAPE
        )

        # ── Step 6: Apply to full-resolution volume ──
        moving_nii = nib.load(str(record.volume_path))
        if output_spacing is not None:
            # Resample moving to output spacing, then apply affine
            mov_out_data, mov_out_affine = load_and_resample_isotropic(
                record.volume_path, output_spacing
            )
            mov_out_nii = nib.Nifti1Image(mov_out_data, mov_out_affine)
            warped_nii = apply_affine_to_volume(
                mov_out_nii, world_affine, baseline_nii, interpolation="trilinear"
            )
        else:
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
