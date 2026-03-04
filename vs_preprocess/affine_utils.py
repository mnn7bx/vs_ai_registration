"""
Affine composition, decomposition, coordinate conversions, and volume warping.

THIS MODULE IS THE FOUNDATION OF THE PIPELINE. Bugs here silently corrupt
all downstream results. See tests/test_affine_utils.py for mandatory tests.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import SimpleITK as sitk
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSE / DECOMPOSE
# ═══════════════════════════════════════════════════════════════════════════


def compose_affine(
    params: torch.Tensor,
    transform_type: str = "affine",
) -> torch.Tensor:
    """
    Compose a (B, 3, 4) affine matrix from decomposed parameters.

    Args:
        params: (B, 9) for affine [rx, ry, rz, tx, ty, tz, sx, sy, sz]
                (B, 6) for rigid  [rx, ry, rz, tx, ty, tz]
                Rotations in radians. Translations in normalized [-1,1] coords.
                Scale values are OFFSETS from 1.0 (output 0.0 -> scale 1.0).
        transform_type: "affine" or "rigid".

    Returns:
        (B, 3, 4) affine matrix. Zero params -> identity.

    Composition order: A = T * S * Rz * Ry * Rx
    """
    B = params.shape[0]
    device = params.device

    rx = params[:, 0]
    ry = params[:, 1]
    rz = params[:, 2]
    tx = params[:, 3]
    ty = params[:, 4]
    tz = params[:, 5]

    cos_x, sin_x = torch.cos(rx), torch.sin(rx)
    cos_y, sin_y = torch.cos(ry), torch.sin(ry)
    cos_z, sin_z = torch.cos(rz), torch.sin(rz)

    # Combined rotation R = Rz @ Ry @ Rx
    r00 = cos_y * cos_z
    r01 = sin_x * sin_y * cos_z - cos_x * sin_z
    r02 = cos_x * sin_y * cos_z + sin_x * sin_z
    r10 = cos_y * sin_z
    r11 = sin_x * sin_y * sin_z + cos_x * cos_z
    r12 = cos_x * sin_y * sin_z - sin_x * cos_z
    r20 = -sin_y
    r21 = sin_x * cos_y
    r22 = cos_x * cos_y

    R = torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=1,
    )  # (B, 3, 3)

    if transform_type == "affine" and params.shape[1] >= 9:
        sx = torch.clamp(params[:, 6] + 1.0, 0.7, 1.3)
        sy = torch.clamp(params[:, 7] + 1.0, 0.7, 1.3)
        sz = torch.clamp(params[:, 8] + 1.0, 0.7, 1.3)
        S = torch.zeros(B, 3, 3, device=device)
        S[:, 0, 0] = sx
        S[:, 1, 1] = sy
        S[:, 2, 2] = sz
        RS = R @ S
    else:
        RS = R

    t = torch.stack([tx, ty, tz], dim=-1).unsqueeze(-1)  # (B, 3, 1)
    affine = torch.cat([RS, t], dim=-1)  # (B, 3, 4)
    return affine


def decompose_affine(affine: np.ndarray) -> dict:
    """
    Extract human-readable parameters from a 3x4 affine matrix.

    Args:
        affine: (3, 4) or (B, 3, 4) numpy array.

    Returns:
        dict with keys:
            rotation_deg: [rx, ry, rz] in degrees
            translation: [tx, ty, tz]
            scale: [sx, sy, sz]
            determinant: float
    """
    if isinstance(affine, torch.Tensor):
        affine = affine.detach().cpu().numpy()
    if affine.ndim == 3:
        affine = affine[0]

    M = affine[:3, :3].astype(np.float64)
    t = affine[:3, 3].astype(np.float64)

    # Scale: column norms preserve axis order (sx, sy, sz), unlike SVD singular
    # values which are sorted in descending order.
    scale = np.linalg.norm(M, axis=0)  # shape (3,), axis-ordered

    # Rotation: normalise columns of M to unit length, then extract angles
    scale_safe = np.where(scale > 1e-10, scale, 1.0)
    R = M / scale_safe[np.newaxis, :]  # each column divided by its norm

    # Ensure proper rotation (det == +1)
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    ry = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    cos_ry = np.cos(ry)
    if np.abs(cos_ry) > 1e-6:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0

    return {
        "rotation_deg": [float(np.degrees(rx)), float(np.degrees(ry)), float(np.degrees(rz))],
        "translation": [float(t[0]), float(t[1]), float(t[2])],
        "scale": [float(scale[0]), float(scale[1]), float(scale[2])],
        "determinant": float(np.linalg.det(M)),
    }


def invert_affine_3x4(affine_3x4: np.ndarray) -> np.ndarray:
    """
    Invert a (3, 4) affine. Returns (3, 4).
    """
    full = np.eye(4, dtype=np.float64)
    full[:3, :] = affine_3x4
    return np.linalg.inv(full)[:3, :]


def invert_affine_4x4(affine_4x4: np.ndarray) -> np.ndarray:
    """
    Invert a (4, 4) affine. Returns (4, 4).
    """
    return np.linalg.inv(affine_4x4.astype(np.float64))


def extend_to_4x4(affine_3x4: np.ndarray) -> np.ndarray:
    """
    Extend a (3, 4) affine to (4, 4) by appending [0, 0, 0, 1].
    """
    full = np.eye(4, dtype=np.float64)
    full[:3, :] = affine_3x4
    return full


# ═══════════════════════════════════════════════════════════════════════════
# COORDINATE SPACE CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════


def _voxel_to_normalized(shape: Tuple[int, ...]) -> np.ndarray:
    """
    4x4 matrix mapping voxel indices [0..N-1] to normalized coords [-1..1].

    For dimension of size N:
        x_norm = 2 * i / (N - 1) - 1

    Args:
        shape: (D, H, W) spatial dimensions.

    Returns:
        4x4 matrix.
    """
    D, H, W = shape[:3]
    M = np.eye(4, dtype=np.float64)
    M[0, 0] = 2.0 / max(D - 1, 1)
    M[1, 1] = 2.0 / max(H - 1, 1)
    M[2, 2] = 2.0 / max(W - 1, 1)
    M[0, 3] = -1.0
    M[1, 3] = -1.0
    M[2, 3] = -1.0
    return M


def _normalized_to_voxel(shape: Tuple[int, ...]) -> np.ndarray:
    """Inverse of _voxel_to_normalized."""
    return np.linalg.inv(_voxel_to_normalized(shape))


def _resampled_affine(nii: nib.Nifti1Image, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Compute the voxel-to-world affine for a volume resampled to target_shape,
    covering the same physical FOV as the original.

    Same origin. Voxel size scaled by (orig_shape - 1) / (target_shape - 1).
    """
    orig_affine = nii.affine.astype(np.float64).copy()
    orig_shape = np.array(nii.shape[:3], dtype=np.float64)
    target = np.array(target_shape, dtype=np.float64)

    scale = np.where(target > 1, (orig_shape - 1) / (target - 1), 1.0)

    new_affine = orig_affine.copy()
    new_affine[:3, :3] = orig_affine[:3, :3] @ np.diag(scale)
    # origin unchanged
    return new_affine


def normalized_affine_to_world(
    affine_norm_3x4: np.ndarray,
    moving_nii: nib.Nifti1Image,
    fixed_nii: nib.Nifti1Image,
    net_input_shape: Tuple[int, int, int] = (96, 96, 96),
) -> np.ndarray:
    """
    Convert a normalized-coordinate affine (from the network) to a
    world-coordinate (mm) affine for resampling.

    The network predicts an affine that maps:
        moving_normalized -> fixed_normalized

    For resampling the moving into the fixed grid, we need:
        fixed_world_point -> moving_world_point
    (because resampling asks: for each output voxel, where does it come from?)

    Coordinate chain (fixed_world -> moving_world):
        fixed_world
        -> fixed_96_voxel       (inv of A96_fixed)
        -> fixed_normalized     (V2N at 96^3)
        -> moving_normalized    (inv of affine_norm, because affine maps mov->fix)
        -> moving_96_voxel      (N2V at 96^3)
        -> moving_world         (A96_moving)

    Args:
        affine_norm_3x4: (3, 4) predicted affine in normalized space (moving->fixed).
        moving_nii: full-res moving nibabel image.
        fixed_nii: full-res fixed/baseline nibabel image.
        net_input_shape: spatial dims the network was trained on.

    Returns:
        (4, 4) world-space affine: fixed_world -> moving_world.
    """
    aff_4x4 = extend_to_4x4(affine_norm_3x4)
    inv_aff = np.linalg.inv(aff_4x4)  # fixed_norm -> moving_norm

    V2N = _voxel_to_normalized(net_input_shape)
    N2V = _normalized_to_voxel(net_input_shape)

    A96_fixed = _resampled_affine(fixed_nii, net_input_shape)
    A96_moving = _resampled_affine(moving_nii, net_input_shape)

    # Chain: fixed_world -> ... -> moving_world
    chain = A96_moving @ N2V @ inv_aff @ V2N @ np.linalg.inv(A96_fixed)
    return chain


# ═══════════════════════════════════════════════════════════════════════════
# VOLUME WARPING — INFERENCE (numpy/SimpleITK, non-differentiable)
# ═══════════════════════════════════════════════════════════════════════════


def apply_affine_to_volume(
    moving_nii: nib.Nifti1Image,
    world_affine_4x4: np.ndarray,
    reference_nii: nib.Nifti1Image,
    interpolation: str = "trilinear",
) -> nib.Nifti1Image:
    """
    Resample moving image into reference grid using a world-space affine.

    Uses scipy.ndimage.affine_transform internally.

    Args:
        moving_nii: image to resample.
        world_affine_4x4: (4, 4) maps reference_world -> moving_world.
        reference_nii: defines output grid (shape + affine).
        interpolation: "trilinear" (order=3 bspline) or "nearest" (order=0).

    Returns:
        nibabel image in the reference grid.
    """
    from scipy.ndimage import affine_transform

    moving_data = moving_nii.get_fdata(dtype=np.float32)
    ref_shape = reference_nii.shape[:3]

    # Compute voxel-to-voxel mapping:
    # For each voxel in the reference grid, we need the corresponding
    # voxel in the moving grid.
    #
    # ref_voxel -> ref_world -> moving_world -> moving_voxel
    # = inv(moving_affine) @ world_affine_4x4 @ ref_affine @ ref_voxel
    #
    # scipy.ndimage.affine_transform expects the matrix that maps
    # OUTPUT coordinates to INPUT coordinates, which is exactly:
    # M = inv(moving_affine) @ world_affine_4x4 @ ref_affine

    M_v2v = (
        np.linalg.inv(moving_nii.affine.astype(np.float64))
        @ world_affine_4x4.astype(np.float64)
        @ reference_nii.affine.astype(np.float64)
    )

    order = 0 if interpolation == "nearest" else 3

    resampled = affine_transform(
        moving_data,
        matrix=M_v2v[:3, :3],
        offset=M_v2v[:3, 3],
        output_shape=ref_shape,
        order=order,
        mode="constant",
        cval=0.0,
    )

    return nib.Nifti1Image(
        resampled.astype(np.float32),
        reference_nii.affine,
        reference_nii.header,
    )


# ═══════════════════════════════════════════════════════════════════════════
# VOLUME WARPING — TRAINING (torch, differentiable)
# ═══════════════════════════════════════════════════════════════════════════


def affine_3x4_to_theta(affine_3x4: torch.Tensor) -> torch.Tensor:
    """
    Convert (B, 3, 4) affine (moving->fixed in normalized coords) to theta
    for torch.nn.functional.affine_grid.

    affine_grid expects the INVERSE mapping (fixed->moving) because
    grid_sample pulls values FROM the input (moving) AT the grid locations.

    Args:
        affine_3x4: (B, 3, 4) moving_norm -> fixed_norm.

    Returns:
        (B, 3, 4) theta: fixed_norm -> moving_norm (for grid_sample).
    """
    B = affine_3x4.shape[0]
    bottom = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]], dtype=affine_3x4.dtype, device=affine_3x4.device
    ).expand(B, 1, 4)
    full = torch.cat([affine_3x4, bottom], dim=1)  # (B, 4, 4)
    inv_full = torch.inverse(full)  # (B, 4, 4)
    return inv_full[:, :3, :]  # (B, 3, 4)


def warp_volume_torch(
    moving: torch.Tensor,
    affine_3x4: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Differentiable volume warping for training.

    Args:
        moving: (B, 1, D, H, W) input volume.
        affine_3x4: (B, 3, 4) moving_norm -> fixed_norm.
        mode: "bilinear" or "nearest".

    Returns:
        (B, 1, D, H, W) warped volume (in fixed space).
    """
    theta = affine_3x4_to_theta(affine_3x4)
    grid = F.affine_grid(theta, moving.shape, align_corners=True)
    warped = F.grid_sample(
        moving, grid, mode=mode, align_corners=True, padding_mode="border"
    )
    return warped
