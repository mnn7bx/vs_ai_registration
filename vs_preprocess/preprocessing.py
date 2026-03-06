"""
N4 bias correction, intensity normalization, resampling.
"""

from __future__ import annotations

import logging
import shutil
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# N4 BIAS FIELD CORRECTION
# ═══════════════════════════════════════════════════════════════════════════


def run_n4_correction(
    input_path: Path,
    output_path: Path,
    bias_field_path: Optional[Path] = None,
    shrink_factor: int = 3,
    convergence_threshold: float = 1e-6,
    max_iterations: Tuple[int, ...] = (50, 50, 30, 20),
) -> None:
    """
    Run N4 bias field correction using SimpleITK.

    Args:
        input_path: input NIfTI path.
        output_path: corrected volume output path.
        bias_field_path: optional path to save the bias field.
        shrink_factor: spatial downsampling for faster computation.
        convergence_threshold: stopping criterion.
        max_iterations: iterations per resolution level.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(str(input_path), sitk.sitkFloat32)

    # Create a foreground mask using Otsu thresholding
    mask_filter = sitk.OtsuThresholdImageFilter()
    mask_filter.SetInsideValue(0)
    mask_filter.SetOutsideValue(1)
    mask = mask_filter.Execute(image)

    # N4 correction
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(x) for x in max_iterations])
    corrector.SetConvergenceThreshold(convergence_threshold)
    corrector.SetShrinkFactor(shrink_factor)

    try:
        corrected = corrector.Execute(image, mask)
    except RuntimeError as e:
        logger.error("N4 failed on %s: %s", input_path, e)
        raise

    sitk.WriteImage(corrected, str(output_path))
    logger.debug("N4 corrected: %s -> %s", input_path, output_path)

    if bias_field_path is not None:
        bias_field_path.parent.mkdir(parents=True, exist_ok=True)
        log_bias = corrector.GetLogBiasFieldAsImage(image)
        sitk.WriteImage(log_bias, str(bias_field_path))


# ═══════════════════════════════════════════════════════════════════════════
# INTENSITY NORMALIZATION
# ═══════════════════════════════════════════════════════════════════════════


def normalize_intensity(
    image_data: np.ndarray,
    brain_mask: np.ndarray,
    method: str = "minmax",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    Normalize intensities. Stats from brain mask, applied to full volume.

    Args:
        image_data: (D, H, W) float array.
        brain_mask: (D, H, W) binary array.
        method: "minmax" (-> [0,1]) or "zscore".
        percentile_low: lower clipping percentile.
        percentile_high: upper clipping percentile.

    Returns:
        (D, H, W) float32 normalized volume.
    """
    brain_voxels = image_data[brain_mask > 0].astype(np.float64)

    if brain_voxels.size == 0:
        logger.warning("Empty brain mask — returning unmodified volume")
        return image_data.astype(np.float32)

    p_low = np.percentile(brain_voxels, percentile_low)
    p_high = np.percentile(brain_voxels, percentile_high)
    clipped = np.clip(image_data, p_low, p_high)

    if method == "minmax":
        denom = p_high - p_low
        if denom < 1e-8:
            logger.warning("Near-zero intensity range — returning zeros")
            return np.zeros_like(image_data, dtype=np.float32)
        normalized = (clipped - p_low) / denom
    elif method == "zscore":
        mean = np.mean(brain_voxels)
        std = np.std(brain_voxels)
        if std < 1e-8:
            logger.warning("Near-zero std — returning zeros")
            return np.zeros_like(image_data, dtype=np.float32)
        normalized = (clipped - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# RESAMPLING
# ═══════════════════════════════════════════════════════════════════════════


def _nib_to_sitk(nii: nib.Nifti1Image) -> sitk.Image:
    """
    Convert nibabel image to SimpleITK image, preserving spatial metadata.

    nibabel stores data as (i, j, k) with an affine.
    SimpleITK uses (x, y, z) physical coordinates.
    We extract origin, spacing, and direction from the nibabel affine.
    """
    data = nii.get_fdata(dtype=np.float32)
    affine = nii.affine.astype(np.float64)

    # SimpleITK expects data indexed as [z, y, x] when using GetImageFromArray
    # but nibabel data is [i, j, k]. The mapping depends on the affine.
    # Safest: write to temp file and read back. But for performance, do it in memory:

    # Extract spacing (magnitude of each column of the rotation part)
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))

    # Direction cosines (normalize columns)
    direction_matrix = affine[:3, :3] / spacing[np.newaxis, :]
    # SimpleITK direction is a flat tuple, column-major
    direction = direction_matrix.flatten(order="F").tolist()

    # Origin
    origin = affine[:3, 3].tolist()

    # SimpleITK expects (z, y, x) ordering for the array
    sitk_image = sitk.GetImageFromArray(data.transpose(2, 1, 0))
    sitk_image.SetOrigin(origin)
    sitk_image.SetSpacing(spacing.tolist())
    sitk_image.SetDirection(direction)

    return sitk_image


def _sitk_to_nib(sitk_image: sitk.Image) -> nib.Nifti1Image:
    """
    Convert SimpleITK image back to nibabel.
    """
    data = sitk.GetArrayFromImage(sitk_image)  # (z, y, x)
    data = data.transpose(2, 1, 0)  # -> (x, y, z) = (i, j, k)

    spacing = np.array(sitk_image.GetSpacing())
    origin = np.array(sitk_image.GetOrigin())
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3, order="F")

    affine = np.eye(4)
    affine[:3, :3] = direction * spacing[np.newaxis, :]
    affine[:3, 3] = origin

    return nib.Nifti1Image(data.astype(np.float32), affine)


def resample_to_isotropic(
    nii: nib.Nifti1Image,
    target_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    interpolation: str = "bspline",
) -> nib.Nifti1Image:
    """
    Resample a NIfTI image to isotropic spacing.

    Args:
        nii: input image.
        target_spacing: (sx, sy, sz) in mm.
        interpolation: "bspline" (order 3), "linear", or "nearest".

    Returns:
        Resampled nibabel image.
    """
    sitk_image = _nib_to_sitk(nii)

    orig_size = np.array(sitk_image.GetSize(), dtype=np.float64)
    orig_spacing = np.array(sitk_image.GetSpacing(), dtype=np.float64)
    target_sp = np.array(target_spacing, dtype=np.float64)

    new_size = np.round(orig_size * orig_spacing / target_sp).astype(int).tolist()

    interp_map = {
        "bspline": sitk.sitkBSpline,
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
    }

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_sp.tolist())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interp_map[interpolation])
    resampler.SetDefaultPixelValue(0.0)

    resampled = resampler.Execute(sitk_image)
    return _sitk_to_nib(resampled)


def resample_mask_to_isotropic(
    nii: nib.Nifti1Image,
    target_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> nib.Nifti1Image:
    """
    Resample binary mask: linear interpolation -> threshold 0.5 -> binarize.
    """
    result = resample_to_isotropic(nii, target_spacing, interpolation="linear")
    data = result.get_fdata()
    binary = (data >= 0.5).astype(np.uint8)
    return nib.Nifti1Image(binary, result.affine, result.header)


# ═══════════════════════════════════════════════════════════════════════════
# PREPROCESS FOR NETWORK INPUT
# ═══════════════════════════════════════════════════════════════════════════


def preprocess_for_network(
    input_path: Path,
    target_size: Tuple[int, int, int] = (96, 96, 96),
) -> torch.Tensor:
    """
    Load and preprocess a volume for GlobalNet input.

    Steps:
        1. Load with nibabel
        2. Reorient to RAS
        3. Resample to target_size using scipy.ndimage.zoom (order=1)
        4. Intensity normalize to [0, 1]
        5. Return as (1, 1, D, H, W) float32 tensor

    IMPORTANT: This function MUST be identical in training and inference.
    Always use scipy.ndimage.zoom with order=1 — do not substitute.
    """
    import torch
    from scipy.ndimage import zoom

    nii = nib.load(str(input_path))

    # Handle 4D volumes (e.g. multi-echo): take first volume before reorienting
    if nii.ndim == 4:
        data_4d = nii.get_fdata(dtype=np.float32)
        nii = nib.Nifti1Image(data_4d[..., 0], nii.affine[:4, :4], nii.header)

    if nii.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {nii.shape}: {input_path}")

    nii = nib.as_closest_canonical(nii)  # reorient to RAS
    data = nii.get_fdata(dtype=np.float32)

    # Resample to target_size
    zoom_factors = [t / o for t, o in zip(target_size, data.shape[:3])]
    resampled = zoom(data, zoom_factors, order=1, mode="constant", cval=0.0)

    # Ensure exact target size (zoom can be off by 1 due to rounding)
    if resampled.shape != target_size:
        padded = np.zeros(target_size, dtype=np.float32)
        slices = tuple(slice(0, min(r, t)) for r, t in zip(resampled.shape, target_size))
        padded[slices] = resampled[slices]
        resampled = padded

    # Intensity normalize to [0, 1]
    vmin, vmax = resampled.min(), resampled.max()
    if vmax - vmin > 1e-8:
        resampled = (resampled - vmin) / (vmax - vmin)
    else:
        resampled = np.zeros_like(resampled)

    tensor = torch.from_numpy(resampled).float()
    return tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
