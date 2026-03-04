# VS Preprocessing Pipeline — Complete Implementation Specification (v8)

# Every code block in this document is COMPLETE, TESTED-READY code.
# Sonnet: implement each file by copying the code block and adjusting only
# for import paths if the project structure differs.

---

## SECTION 0: PROJECT STRUCTURE

```
vs_preprocess/
├── requirements.txt
├── run.py
├── pipeline.py
├── dataset.py
├── affine_utils.py
├── preprocessing.py
├── skull_strip.py
├── registration.py
├── qc.py
├── tests/
│   ├── __init__.py
│   ├── test_affine_utils.py
│   ├── test_dataset.py
│   └── test_pipeline.py
└── training/
    ├── __init__.py
    ├── requirements.txt
    ├── model.py
    ├── dataset.py
    ├── augmentations.py
    ├── losses.py
    ├── train_intra.py
    ├── train_template.py
    ├── evaluate.py
    └── configs/
        ├── intra_patient.yaml
        └── template.yaml
```

---

## SECTION 1: DESIGN DECISIONS (FIXED — DO NOT CHANGE)

| # | Decision | Choice |
|---|---|---|
| 1 | Final output skull-stripped? | No — full FOV preserved, IAC must not be cropped |
| 2 | Transform type | Linear only: affine (9 DOF) or rigid (6 DOF) |
| 3 | Longitudinal anchor | First timepoint per (patient, modality) |
| 4 | Template registration | Separate optional model, `--register_to_template` |
| 5 | Affine parameterization | Decomposed: 3 rotation + 3 translation + 3 scale |
| 6 | Network input resolution | 96×96×96 |
| 7 | Modality handling | Single model with MI loss (modality-agnostic) |
| 8 | Brain mask | HD-BET on baseline only, reuse for all aligned follow-ups |
| 9 | Intermediates | Kept by default |
| 10 | Resume | Skip existing outputs by default |
| 11 | Seg labels | Binary (tumor vs background), nearest-neighbor resampling |
| 12 | Target spacing | 0.5mm isotropic |
| 13 | QC | PNG per scan + longitudinal PNG per patient + HTML summary |
| 14 | N4 timing | Before brain mask, no mask input |
| 15 | Cross-modality | Deferred — each modality aligned within itself |

### Pipeline flow per patient

```
1. N4 bias correction on every timepoint (CPU, parallel)
2. HD-BET brain mask on baseline only (GPU)
3. GlobalNet: align each follow-up to baseline (GPU)
   - After alignment, ALL timepoints share baseline's coordinate space
   - So baseline brain mask applies to all aligned follow-ups directly
4. Intensity normalization per timepoint (CPU)
   - Stats from brain mask voxels, applied to full volume
5. Resample all to 0.5mm isotropic (CPU)
6. Resample expert seg with same affine + nearest-neighbor (CPU)
7. QC PNGs per scan + longitudinal comparison per patient (CPU)
```

---

## SECTION 2: requirements.txt

```
torch>=2.0,<3.0
monai>=1.3,<2.0
nibabel>=4.0,<6.0
numpy>=1.22,<2.0
SimpleITK>=2.2,<3.0
scipy>=1.9,<2.0
matplotlib>=3.6,<4.0
tqdm>=4.64
pandas>=1.5,<3.0
tensorboard>=2.12
pyyaml>=6.0
hd-bet
```

---

## SECTION 3: dataset.py — COMPLETE CODE

```python
"""
Dataset discovery and patient grouping for VS preprocessing pipeline.

Walks data_dir/<modality>/<patient_timepoint>/ structure, identifies volumes
and segmentations, groups by patient, marks baselines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Suffixes that indicate a file is NOT the primary MRI volume
_EXCLUDE_SUFFIXES = re.compile(
    r"(_seg|_label|_mask|_brain|_stripped|_bet|_synthstrip|_preprocessed|_n4|_bias)",
    re.IGNORECASE,
)


@dataclass
class ScanRecord:
    """One MRI volume + optional segmentation for a single timepoint."""

    modality: str
    patient_timepoint: str
    patient_id: str
    timepoint: str
    volume_path: Path
    seg_path: Optional[Path]
    is_baseline: bool = False


def _parse_patient_timepoint(dirname: str) -> Tuple[str, str]:
    """
    Split directory name into (patient_id, timepoint).

    Strategy: split on the LAST underscore.

    Examples:
        'VS001_tp2'       -> ('VS001', 'tp2')
        'patient_003_tp1' -> ('patient_003', 'tp1')
        'VS001'           -> ('VS001', 'tp1')
    """
    idx = dirname.rfind("_")
    if idx == -1:
        return dirname, "tp1"
    return dirname[:idx], dirname[idx + 1 :]


def _find_volume(nifti_files: List[Path]) -> Optional[Path]:
    """
    Pick the primary MRI volume from a list of NIfTI files.

    Filters out files matching _EXCLUDE_SUFFIXES, then picks the largest.
    Falls back to the largest file overall if all are excluded.
    """
    if not nifti_files:
        return None

    # Strip .nii.gz or .nii to get the stem for matching
    def stem(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            return name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            return name[: -len(".nii")]
        return name

    candidates = [f for f in nifti_files if not _EXCLUDE_SUFFIXES.search(stem(f))]
    if not candidates:
        candidates = nifti_files  # fallback

    return max(candidates, key=lambda p: p.stat().st_size)


def _find_seg(nifti_files: List[Path], volume_path: Path) -> Optional[Path]:
    """
    Pick the expert segmentation from the same directory.

    Looks for files containing '_seg' or '_label' in the stem,
    excluding the primary volume.
    """
    seg_pattern = re.compile(r"(_seg|_label)", re.IGNORECASE)
    matches = [
        f
        for f in nifti_files
        if f != volume_path and seg_pattern.search(f.name)
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.name)[0]


def discover_scans(
    data_dir: Path,
    modalities: Optional[Sequence[str]] = None,
) -> List[ScanRecord]:
    """
    Walk data_dir and return a list of ScanRecord entries.

    Expected structure:
        data_dir/
            <modality>/          e.g., T1C, T2
                <patient_tp>/    e.g., VS001_tp1
                    *.nii.gz

    Args:
        data_dir: Root data directory.
        modalities: If provided, include only these modality names
                    (case-insensitive). None means all.

    Returns:
        List[ScanRecord] sorted by (modality, patient_id, timepoint).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    allowed_mod = None
    if modalities is not None:
        allowed_mod = {m.lower() for m in modalities}

    records: List[ScanRecord] = []

    for mod_dir in sorted(data_dir.iterdir()):
        if not mod_dir.is_dir():
            continue
        modality = mod_dir.name
        if allowed_mod is not None and modality.lower() not in allowed_mod:
            continue

        for tp_dir in sorted(mod_dir.iterdir()):
            if not tp_dir.is_dir():
                continue

            nifti_files = sorted(
                list(tp_dir.glob("*.nii.gz")) + list(tp_dir.glob("*.nii"))
            )
            if not nifti_files:
                logger.warning("No NIfTI files in %s — skipping", tp_dir)
                continue

            volume = _find_volume(nifti_files)
            if volume is None:
                logger.warning("Could not identify volume in %s — skipping", tp_dir)
                continue

            seg = _find_seg(nifti_files, volume)
            patient_id, timepoint = _parse_patient_timepoint(tp_dir.name)

            records.append(
                ScanRecord(
                    modality=modality,
                    patient_timepoint=tp_dir.name,
                    patient_id=patient_id,
                    timepoint=timepoint,
                    volume_path=volume,
                    seg_path=seg,
                )
            )

    records.sort(key=lambda r: (r.modality, r.patient_id, r.timepoint))
    logger.info("Discovered %d scans across %d directories", len(records), len(records))
    return records


def group_by_patient(
    records: List[ScanRecord],
    baseline_selection: str = "first",
) -> Dict[str, List[ScanRecord]]:
    """
    Group ScanRecords by (modality, patient_id) and mark baselines.

    Args:
        records: flat list from discover_scans.
        baseline_selection: "first" (earliest timepoint) or "latest".

    Returns:
        Dict mapping "modality/patient_id" -> sorted list of ScanRecord.
        The baseline has is_baseline=True.
    """
    groups: Dict[str, List[ScanRecord]] = {}
    for r in records:
        key = f"{r.modality}/{r.patient_id}"
        groups.setdefault(key, []).append(r)

    for key, group in groups.items():
        group.sort(key=lambda r: r.timepoint)
        # Reset all baselines
        for r in group:
            r.is_baseline = False
        # Mark baseline
        if baseline_selection == "latest":
            group[-1].is_baseline = True
        else:  # "first"
            group[0].is_baseline = True

    logger.info(
        "Grouped into %d patient-modality groups (total %d scans)",
        len(groups),
        sum(len(g) for g in groups.values()),
    )
    return groups
```

---

## SECTION 4: affine_utils.py — COMPLETE CODE

This is the most critical module. Every function is fully implemented.

```python
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

    U, sigma, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

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
        "scale": [float(sigma[0]), float(sigma[1]), float(sigma[2])],
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
```

---

## SECTION 5: preprocessing.py — COMPLETE CODE

```python
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
```

---

## SECTION 6: skull_strip.py — COMPLETE CODE

```python
"""
Brain mask extraction via HD-BET. Runs on baseline only, dilated for safety.
"""

from __future__ import annotations

import logging
import tempfile
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy.ndimage import binary_dilation, generate_binary_structure

logger = logging.getLogger(__name__)


def extract_brain_mask(
    input_path: Path,
    output_mask_path: Path,
    device: str = "cuda",
    mode: str = "accurate",
    dilation_voxels: int = 2,
) -> None:
    """
    Run HD-BET and save a dilated binary brain mask.

    Args:
        input_path: NIfTI volume to skull-strip.
        output_mask_path: where to save the binary brain mask.
        device: "cuda" or "cpu".
        mode: "accurate" (5-model ensemble) or "fast" (single model).
        dilation_voxels: voxels to dilate. 2 adds margin for IAC.
    """
    from HD_BET.run import run_hd_bet

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hdbet_") as tmpdir:
        tmp_out = str(Path(tmpdir) / "brain")

        run_hd_bet(
            mri_fnames=[str(input_path)],
            output_fnames=[tmp_out],
            mode=mode,
            device=device if device != "cpu" else "cpu",
            postprocess=True,
            do_tta=(mode == "accurate"),
            keep_mask=True,
            overwrite=True,
        )

        # HD-BET writes brain_mask.nii.gz — find it
        tmpdir_path = Path(tmpdir)
        mask_candidates = list(tmpdir_path.glob("*mask*"))
        if not mask_candidates:
            contents = list(tmpdir_path.iterdir())
            raise FileNotFoundError(
                f"HD-BET did not produce a mask file. "
                f"Temp dir contents: {[p.name for p in contents]}"
            )

        mask_path = mask_candidates[0]
        mask_nii = nib.load(str(mask_path))
        mask_data = mask_nii.get_fdata().astype(bool)

    # Dilate
    if dilation_voxels > 0:
        struct = generate_binary_structure(3, 1)  # 6-connected
        mask_data = binary_dilation(mask_data, structure=struct, iterations=dilation_voxels)

    dilated_nii = nib.Nifti1Image(
        mask_data.astype(np.uint8), mask_nii.affine, mask_nii.header
    )
    nib.save(dilated_nii, str(output_mask_path))
    logger.info("Brain mask saved: %s (dilated %d voxels)", output_mask_path, dilation_voxels)
```
## SECTION 7: registration.py — COMPLETE CODE

```python
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
```

---

## SECTION 8: qc.py — COMPLETE CODE

```python
"""
QC visualization: per-scan 3-panel PNG, longitudinal comparison, HTML summary.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def generate_scan_qc(
    volume_path: Path,
    mask_path: Path,
    output_path: Path,
    seg_path: Optional[Path] = None,
    title: str = "",
) -> None:
    """
    3-panel QC PNG (axial, sagittal, coronal) with mask and seg contour overlays.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vol = nib.load(str(volume_path)).get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32) if seg_path and seg_path.exists() else None

    D, H, W = vol.shape[:3]
    slices_info = [
        ("Axial", vol[:, :, W // 2].T, mask[:, :, W // 2].T, seg[:, :, W // 2].T if seg is not None else None),
        ("Sagittal", vol[D // 2, :, :].T, mask[D // 2, :, :].T, seg[D // 2, :, :].T if seg is not None else None),
        ("Coronal", vol[:, H // 2, :].T, mask[:, H // 2, :].T, seg[:, H // 2, :].T if seg is not None else None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100, facecolor="black")
    for ax, (name, vol_slice, mask_slice, seg_slice) in zip(axes, slices_info):
        ax.imshow(vol_slice, cmap="gray", aspect="equal")
        if mask_slice.max() > 0:
            ax.contour(mask_slice, levels=[0.5], colors=["red"], linewidths=0.8)
        if seg_slice is not None and seg_slice.max() > 0:
            ax.contour(seg_slice, levels=[0.5], colors=["lime"], linewidths=1.0)
        ax.set_title(name, fontsize=10, color="white")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, color="white")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    logger.debug("QC saved: %s", output_path)


def generate_longitudinal_qc(
    patient_id: str,
    timepoint_data: List[dict],
    output_path: Path,
) -> None:
    """
    Per-patient longitudinal comparison image.

    Args:
        patient_id: for the title.
        timepoint_data: list of dicts with keys:
            'timepoint': str
            'volume_path': Path (preprocessed, aligned)
            'seg_path': Optional[Path]
            'params_dict': dict from decompose_affine
        output_path: output PNG path.

    Top row: axial center slices side-by-side.
    Bottom row: difference images (tp_n - tp1) with RdBu_r colormap.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(timepoint_data)
    if N < 2:
        logger.debug("Skipping longitudinal QC for %s (only %d timepoint)", patient_id, N)
        return

    fig, axes = plt.subplots(2, N, figsize=(4 * N, 8), dpi=100, facecolor="black")
    if N == 1:
        axes = axes.reshape(2, 1)

    # Load baseline
    base_vol = nib.load(str(timepoint_data[0]["volume_path"])).get_fdata(dtype=np.float32)
    center_z = base_vol.shape[2] // 2

    for i, td in enumerate(timepoint_data):
        vol = nib.load(str(td["volume_path"])).get_fdata(dtype=np.float32)
        # Use same slice index across all timepoints (they're aligned)
        z = min(center_z, vol.shape[2] - 1)
        axial = vol[:, :, z].T

        # Top row: volume with seg overlay
        axes[0, i].imshow(axial, cmap="gray", aspect="equal")
        if td.get("seg_path") and Path(td["seg_path"]).exists():
            seg = nib.load(str(td["seg_path"])).get_fdata(dtype=np.float32)
            sz = min(z, seg.shape[2] - 1)
            seg_slice = seg[:, :, sz].T
            if seg_slice.max() > 0:
                axes[0, i].contour(seg_slice, levels=[0.5], colors=["lime"], linewidths=1.0)
        axes[0, i].set_title(td["timepoint"], fontsize=10, color="white")
        axes[0, i].axis("off")

        # Bottom row: difference from baseline
        if i == 0:
            axes[1, i].imshow(axial, cmap="gray", aspect="equal")
            axes[1, i].set_title("Baseline (ref)", fontsize=9, color="white")
        else:
            base_z = min(center_z, base_vol.shape[2] - 1)
            diff = axial - base_vol[:, :, base_z].T
            vmax = max(np.percentile(np.abs(diff), 99), 1e-6)
            axes[1, i].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            p = td["params_dict"]
            r = p["rotation_deg"]
            t = p["translation"]
            axes[1, i].set_title(
                f"rot=({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f})\n"
                f"trans=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})",
                fontsize=7,
                color="white",
            )
        axes[1, i].axis("off")

    fig.suptitle(f"Longitudinal QC — {patient_id}", fontsize=12, color="white")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    logger.debug("Longitudinal QC saved: %s", output_path)


def generate_qc_summary_html(
    output_dir: Path,
    patient_results: Dict[str, dict],
) -> None:
    """
    Generate browsable HTML summary of all QC images.

    Args:
        output_dir: pipeline output root.
        patient_results: dict mapping group_key -> {
            'records': List[ScanRecord],
            'results': List[ProcessingResult],
        }
    """
    html_path = output_dir / "qc_summary.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []
    for group_key in sorted(patient_results.keys()):
        info = patient_results[group_key]
        modality, patient_id = group_key.split("/", 1)

        # Longitudinal QC
        long_qc = output_dir / modality / patient_id / f"{patient_id}_longitudinal_qc.png"
        long_img = ""
        if long_qc.exists():
            rel = long_qc.relative_to(output_dir)
            long_img = f'<img src="{rel}" style="max-width:100%;">'

        # Per-scan QC thumbnails
        thumbs = []
        for rec in info.get("records", []):
            qc_png = (
                output_dir / rec.modality / rec.patient_id
                / rec.patient_timepoint / f"{rec.patient_timepoint}_qc.png"
            )
            if qc_png.exists():
                rel = qc_png.relative_to(output_dir)
                thumbs.append(
                    f'<div class="qc-thumb">'
                    f'<img src="{rel}" title="{rec.patient_timepoint}">'
                    f'<br><small>{rec.patient_timepoint}</small></div>'
                )

        thumb_html = "\n".join(thumbs)

        sections.append(f"""
        <div class="patient">
            <h2>{modality} / {patient_id}</h2>
            {long_img}
            <div class="qc-grid">{thumb_html}</div>
        </div>
        """)

    body = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>VS Preprocessing QC Summary</title>
    <style>
        body {{ background: #1a1a1a; color: #eee; font-family: sans-serif; margin: 20px; }}
        h1 {{ border-bottom: 1px solid #555; padding-bottom: 10px; }}
        .patient {{ margin: 20px 0; border: 1px solid #444; padding: 15px; border-radius: 8px; }}
        .patient h2 {{ margin-top: 0; color: #8cf; }}
        .qc-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
        .qc-thumb {{ text-align: center; }}
        .qc-thumb img {{ max-width: 350px; border: 1px solid #555; border-radius: 4px; cursor: pointer; transition: transform 0.2s; }}
        .qc-thumb img:hover {{ transform: scale(2); z-index: 10; position: relative; }}
        small {{ color: #aaa; }}
    </style>
</head>
<body>
    <h1>VS Preprocessing QC Summary</h1>
    <p>Generated: {timestamp}</p>
    <p>Pipeline output: <code>{output_dir}</code></p>
    {body}
</body>
</html>"""

    html_path.write_text(html)
    logger.info("QC summary HTML: %s", html_path)
```

---

## SECTION 9: pipeline.py — COMPLETE CODE

```python
"""
Pipeline orchestration: per-patient processing, failure logging, timing.
"""

from __future__ import annotations

import csv
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nibabel as nib

from dataset import ScanRecord, discover_scans, group_by_patient
from registration import load_model, align_patient_longitudinal
from skull_strip import extract_brain_mask
from preprocessing import (
    run_n4_correction,
    normalize_intensity,
    resample_to_isotropic,
    resample_mask_to_isotropic,
)
from affine_utils import apply_affine_to_volume
from qc import generate_scan_qc, generate_longitudinal_qc, generate_qc_summary_html

logger = logging.getLogger(__name__)


@dataclass
class StageTimings:
    n4: float = 0.0
    brain_mask: float = 0.0
    alignment: float = 0.0
    normalization: float = 0.0
    resampling: float = 0.0
    seg_passthrough: float = 0.0
    qc: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.n4 + self.brain_mask + self.alignment
            + self.normalization + self.resampling
            + self.seg_passthrough + self.qc
        )


@dataclass
class ProcessingResult:
    patient_timepoint: str
    success: bool
    timings: StageTimings = field(default_factory=StageTimings)
    error: Optional[str] = None
    skipped: bool = False


def _output_exists(record: ScanRecord, output_dir: Path) -> bool:
    """Check if final preprocessed output already exists."""
    out_path = (
        output_dir / record.modality / record.patient_id
        / record.patient_timepoint / f"{record.patient_timepoint}_preprocessed.nii.gz"
    )
    return out_path.exists()


def process_patient_group(
    group_key: str,
    records: List[ScanRecord],
    model,
    model_config: dict,
    output_dir: Path,
    config: dict,
) -> List[ProcessingResult]:
    """
    Process all timepoints of one patient for one modality.

    Args:
        group_key: "T2/VS001" (for logging).
        records: sorted list of ScanRecords (first = baseline).
        model: trained GlobalNet on correct device.
        model_config: model configuration dict.
        output_dir: root output directory.
        config: pipeline settings from CLI args.

    Returns:
        list of ProcessingResult, one per timepoint.
    """
    logger.info("Processing %s (%d timepoints)", group_key, len(records))
    all_results: List[ProcessingResult] = []
    timings_map: Dict[str, StageTimings] = {}

    baseline = records[0]
    baseline_patient_dir = output_dir / baseline.modality / baseline.patient_id

    # ── Check skip ──
    if config.get("skip_existing", True) and not config.get("force_reprocess", False):
        all_exist = all(_output_exists(r, output_dir) for r in records)
        if all_exist:
            logger.info("  All outputs exist — skipping %s", group_key)
            for r in records:
                all_results.append(
                    ProcessingResult(patient_timepoint=r.patient_timepoint, success=True, skipped=True)
                )
            return all_results

    # ── Stage: N4 on all timepoints ──
    n4_paths: Dict[str, Path] = {}
    for record in records:
        timings = StageTimings()
        timings_map[record.timepoint] = timings
        out_dir = output_dir / record.modality / record.patient_id / record.patient_timepoint

        inter_dir = out_dir / "intermediates"
        if config.get("keep_intermediates", True):
            inter_dir.mkdir(parents=True, exist_ok=True)
        else:
            inter_dir = out_dir  # save alongside final outputs
            inter_dir.mkdir(parents=True, exist_ok=True)

        n4_path = inter_dir / f"{record.patient_timepoint}_n4.nii.gz"
        bias_path = inter_dir / f"{record.patient_timepoint}_bias_field.nii.gz"

        t0 = time.perf_counter()
        if config.get("no_n4", False):
            shutil.copy2(str(record.volume_path), str(n4_path))
        else:
            try:
                run_n4_correction(
                    record.volume_path,
                    n4_path,
                    bias_field_path=bias_path if config.get("keep_intermediates", True) else None,
                )
            except Exception as e:
                logger.error("N4 failed for %s: %s", record.patient_timepoint, e)
                all_results.append(
                    ProcessingResult(
                        patient_timepoint=record.patient_timepoint,
                        success=False,
                        error=f"N4: {e}",
                    )
                )
                continue
        timings.n4 = time.perf_counter() - t0
        n4_paths[record.timepoint] = n4_path
        logger.debug("  N4 %s: %.1fs", record.patient_timepoint, timings.n4)

    # ── Stage: Brain mask on baseline ──
    baseline_n4 = n4_paths.get(baseline.timepoint)
    if baseline_n4 is None:
        logger.error("Baseline N4 failed — cannot proceed for %s", group_key)
        return all_results

    bl_out_dir = output_dir / baseline.modality / baseline.patient_id / baseline.patient_timepoint
    bl_inter = bl_out_dir / "intermediates" if config.get("keep_intermediates", True) else bl_out_dir
    native_mask_path = bl_inter / f"{baseline.patient_timepoint}_brain_mask_native.nii.gz"

    t0 = time.perf_counter()
    try:
        extract_brain_mask(
            baseline_n4,
            native_mask_path,
            device=config.get("device", "cuda"),
            mode="accurate",
            dilation_voxels=2,
        )
    except Exception as e:
        logger.error("HD-BET failed for %s: %s", baseline.patient_timepoint, e)
        for r in records:
            all_results.append(
                ProcessingResult(patient_timepoint=r.patient_timepoint, success=False, error=f"HD-BET: {e}")
            )
        return all_results
    timings_map[baseline.timepoint].brain_mask = time.perf_counter() - t0
    logger.debug("  HD-BET %s: %.1fs", baseline.patient_timepoint, timings_map[baseline.timepoint].brain_mask)

    brain_mask_nii = nib.load(str(native_mask_path))

    # ── Stage: Intra-patient alignment ──
    alignment_results = {}
    if len(records) > 1:
        t0 = time.perf_counter()
        try:
            # Rebuild records list using N4 paths instead of original paths
            # so the alignment network sees bias-corrected input
            from dataclasses import replace
            n4_records = []
            for r in records:
                if r.timepoint in n4_paths:
                    n4_records.append(replace(r, volume_path=n4_paths[r.timepoint]))
            alignment_results = align_patient_longitudinal(
                n4_records,
                model,
                output_dir,
                transform_type=config.get("transform_type", "affine"),
                keep_intermediates=config.get("keep_intermediates", True),
                device=config.get("device", "cuda"),
            )
        except Exception as e:
            logger.error("Alignment failed for %s: %s", group_key, e)
            for r in records:
                all_results.append(
                    ProcessingResult(patient_timepoint=r.patient_timepoint, success=False, error=f"Align: {e}")
                )
            return all_results
        align_time = time.perf_counter() - t0
        # Distribute alignment time equally across follow-ups (rough)
        n_followup = len(records) - 1
        for r in records[1:]:
            timings_map[r.timepoint].alignment = align_time / n_followup
        logger.debug("  Alignment: %.1fs total", align_time)

    # ── Stage: Per-timepoint finishing (normalize, resample, seg, QC) ──
    for record in records:
        if record.timepoint not in n4_paths:
            continue  # N4 failed for this timepoint

        timings = timings_map[record.timepoint]
        out_dir = output_dir / record.modality / record.patient_id / record.patient_timepoint
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Determine the volume to normalize
            # For baseline: use N4-corrected volume directly
            # For follow-ups: use the affine-aligned volume
            if record.is_baseline or record.timepoint not in alignment_results:
                vol_nii = nib.load(str(n4_paths[record.timepoint]))
            else:
                warped = alignment_results[record.timepoint].get("warped_nii")
                if warped is not None:
                    vol_nii = warped
                else:
                    vol_nii = nib.load(str(n4_paths[record.timepoint]))

            # After alignment, all timepoints are in baseline space,
            # so the baseline brain mask applies to all of them directly.
            mask_data = brain_mask_nii.get_fdata(dtype=np.float32)

            # Handle shape mismatch (aligned volume is in baseline grid already)
            vol_data = vol_nii.get_fdata(dtype=np.float32)
            if vol_data.shape != mask_data.shape:
                # For baseline, shapes should match. For follow-ups aligned to
                # baseline, they should also match. If not, use the volume shape
                # and resize mask.
                logger.warning(
                    "Shape mismatch for %s: vol=%s mask=%s — using vol shape",
                    record.patient_timepoint, vol_data.shape, mask_data.shape
                )
                from scipy.ndimage import zoom
                zoom_factors = [v / m for v, m in zip(vol_data.shape[:3], mask_data.shape[:3])]
                mask_data = zoom(mask_data, zoom_factors, order=0) > 0.5
                mask_data = mask_data.astype(np.float32)

            # ── Intensity normalization ──
            t0 = time.perf_counter()
            normalized = normalize_intensity(
                vol_data,
                mask_data,
                method=config.get("intensity_norm", "minmax"),
                percentile_low=config.get("percentile_low", 1.0),
                percentile_high=config.get("percentile_high", 99.0),
            )
            norm_nii = nib.Nifti1Image(normalized, vol_nii.affine, vol_nii.header)
            timings.normalization = time.perf_counter() - t0

            if config.get("keep_intermediates", True):
                inter_dir = out_dir / "intermediates"
                inter_dir.mkdir(exist_ok=True)
                nib.save(norm_nii, str(inter_dir / f"{record.patient_timepoint}_normalized.nii.gz"))

            # ── Resample to isotropic ──
            t0 = time.perf_counter()
            target_spacing = tuple(config.get("target_spacing", [0.5, 0.5, 0.5]))
            final_vol = resample_to_isotropic(norm_nii, target_spacing, interpolation="bspline")
            # Also resample the mask
            mask_nii_for_resample = nib.Nifti1Image(mask_data.astype(np.float32), vol_nii.affine)
            final_mask = resample_mask_to_isotropic(mask_nii_for_resample, target_spacing)
            timings.resampling = time.perf_counter() - t0

            # ── Save final outputs ──
            nib.save(final_vol, str(out_dir / f"{record.patient_timepoint}_preprocessed.nii.gz"))
            nib.save(final_mask, str(out_dir / f"{record.patient_timepoint}_brain_mask.nii.gz"))

            # ── Seg passthrough ──
            if record.seg_path is not None and record.seg_path.exists():
                t0 = time.perf_counter()
                seg_nii = nib.load(str(record.seg_path))

                # Apply same affine to seg (nearest-neighbor) if not baseline
                if not record.is_baseline and record.timepoint in alignment_results:
                    world_affine = alignment_results[record.timepoint].get("world_affine")
                    if world_affine is not None:
                        baseline_nii = nib.load(str(n4_paths[baseline.timepoint]))
                        seg_nii = apply_affine_to_volume(
                            seg_nii, world_affine, baseline_nii, interpolation="nearest"
                        )

                final_seg = resample_to_isotropic(seg_nii, target_spacing, interpolation="nearest")
                nib.save(final_seg, str(out_dir / f"{record.patient_timepoint}_seg.nii.gz"))
                timings.seg_passthrough = time.perf_counter() - t0

            # ── QC ──
            t0 = time.perf_counter()
            seg_out = out_dir / f"{record.patient_timepoint}_seg.nii.gz"
            generate_scan_qc(
                volume_path=out_dir / f"{record.patient_timepoint}_preprocessed.nii.gz",
                mask_path=out_dir / f"{record.patient_timepoint}_brain_mask.nii.gz",
                output_path=out_dir / f"{record.patient_timepoint}_qc.png",
                seg_path=seg_out if seg_out.exists() else None,
                title=f"{record.patient_timepoint} ({record.modality})",
            )
            timings.qc = time.perf_counter() - t0

            all_results.append(
                ProcessingResult(
                    patient_timepoint=record.patient_timepoint,
                    success=True,
                    timings=timings,
                )
            )
            logger.info(
                "  %s done (N4=%.1fs mask=%.1fs align=%.1fs norm=%.1fs resamp=%.1fs qc=%.1fs total=%.1fs)",
                record.patient_timepoint,
                timings.n4, timings.brain_mask, timings.alignment,
                timings.normalization, timings.resampling, timings.qc, timings.total,
            )

        except Exception as e:
            logger.error("Failed processing %s: %s", record.patient_timepoint, e, exc_info=True)
            all_results.append(
                ProcessingResult(
                    patient_timepoint=record.patient_timepoint,
                    success=False,
                    error=str(e),
                )
            )

    # ── Longitudinal QC ──
    if len(records) > 1:
        try:
            tp_data = []
            for r in records:
                fvol = output_dir / r.modality / r.patient_id / r.patient_timepoint / f"{r.patient_timepoint}_preprocessed.nii.gz"
                fseg = output_dir / r.modality / r.patient_id / r.patient_timepoint / f"{r.patient_timepoint}_seg.nii.gz"
                pjson = output_dir / r.modality / r.patient_id / r.patient_timepoint / f"{r.patient_timepoint}_params.json"

                if not fvol.exists():
                    continue

                import json
                params_dict = {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0]}
                if pjson.exists():
                    with open(pjson) as f:
                        params_dict = json.load(f)

                tp_data.append({
                    "timepoint": r.patient_timepoint,
                    "volume_path": fvol,
                    "seg_path": fseg if fseg.exists() else None,
                    "params_dict": params_dict,
                })

            if len(tp_data) >= 2:
                long_qc_path = baseline_patient_dir / f"{baseline.patient_id}_longitudinal_qc.png"
                generate_longitudinal_qc(baseline.patient_id, tp_data, long_qc_path)
        except Exception as e:
            logger.warning("Longitudinal QC failed for %s: %s", group_key, e)

    return all_results


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    intra_model_path: Path,
    config: dict,
) -> List[ProcessingResult]:
    """
    Main pipeline entry point.

    1. Discover scans and group by patient.
    2. Load model.
    3. Process each patient group sequentially.
    4. Write failures.csv and qc_summary.html.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover
    records = discover_scans(data_dir, config.get("modalities"))
    if not records:
        logger.error("No scans found in %s", data_dir)
        return []

    groups = group_by_patient(records, config.get("baseline_selection", "first"))

    # Dry run
    if config.get("dry_run", False):
        for key, group in groups.items():
            tps = ", ".join(r.patient_timepoint for r in group)
            bl = next(r for r in group if r.is_baseline)
            logger.info("  %s: baseline=%s, timepoints=[%s]", key, bl.patient_timepoint, tps)
        logger.info("Dry run complete. %d groups, %d total scans.", len(groups), len(records))
        return []

    # Load model
    model, model_config = load_model(intra_model_path, config.get("device", "cuda"))

    # Process
    all_results: List[ProcessingResult] = []
    patient_meta: Dict[str, dict] = {}

    for group_key, group_records in groups.items():
        results = process_patient_group(
            group_key, group_records, model, model_config, output_dir, config
        )
        all_results.extend(results)
        patient_meta[group_key] = {"records": group_records, "results": results}

    # Failures CSV
    failures = [r for r in all_results if not r.success and not r.skipped]
    if failures:
        csv_path = output_dir / "failures.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_timepoint", "error"])
            for r in failures:
                writer.writerow([r.patient_timepoint, r.error or "unknown"])
        logger.warning("%d failures written to %s", len(failures), csv_path)

    # QC summary HTML
    try:
        generate_qc_summary_html(output_dir, patient_meta)
    except Exception as e:
        logger.warning("QC summary HTML generation failed: %s", e)

    # Summary
    n_success = sum(r.success and not r.skipped for r in all_results)
    n_skip = sum(r.skipped for r in all_results)
    n_fail = len(failures)
    logger.info(
        "Pipeline complete: %d success, %d skipped, %d failed",
        n_success, n_skip, n_fail,
    )
    return all_results
```

---

## SECTION 10: run.py — COMPLETE CODE

```python
#!/usr/bin/env python3
"""
CLI entry point for the VS preprocessing pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Vestibular Schwannoma MRI Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--data_dir", type=Path, required=True, help="Root input directory")
    p.add_argument("--output_dir", type=Path, required=True, help="Root output directory")
    p.add_argument("--intra_model_path", type=Path, required=True, help="Trained GlobalNet weights")

    # Transform
    p.add_argument("--transform_type", default="affine", choices=["affine", "rigid"],
                    help="Transform DOF: affine (9) or rigid (6)")

    # Template registration (optional)
    p.add_argument("--register_to_template", action="store_true",
                    help="Enable secondary template registration")
    p.add_argument("--template_model_path", type=Path, default=None,
                    help="Template GlobalNet weights")
    p.add_argument("--template_path", type=Path, default=None,
                    help="Template NIfTI image (e.g., MNI152)")
    p.add_argument("--template_mask_path", type=Path, default=None,
                    help="Template brain mask NIfTI")

    # Brain mask
    p.add_argument("--mask_method", default="hdbet", choices=["hdbet", "template_warp"],
                    help="Brain mask method")

    # Data
    p.add_argument("--baseline_selection", default="first", choices=["first", "latest"],
                    help="Which timepoint is baseline")
    p.add_argument("--modalities", nargs="*", default=None,
                    help="Only process these modalities (e.g., T2 T1C)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Compute device")

    # Preprocessing
    p.add_argument("--intensity_norm", default="minmax", choices=["minmax", "zscore"],
                    help="Intensity normalization method")
    p.add_argument("--percentile_low", type=float, default=1.0,
                    help="Lower clipping percentile")
    p.add_argument("--percentile_high", type=float, default=99.0,
                    help="Upper clipping percentile")
    p.add_argument("--target_spacing", nargs=3, type=float, default=[0.5, 0.5, 0.5],
                    help="Target voxel spacing in mm")
    p.add_argument("--no_n4", action="store_true", help="Skip N4 bias correction")

    # Pipeline control
    p.add_argument("--n_workers", type=int, default=4,
                    help="CPU parallel workers (reserved for future use)")
    p.add_argument("--keep_intermediates", action="store_true", default=True,
                    help="Keep intermediate files")
    p.add_argument("--no_keep_intermediates", action="store_true",
                    help="Discard intermediate files")
    p.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip scans with existing outputs")
    p.add_argument("--force_reprocess", action="store_true",
                    help="Reprocess all scans (overrides --skip_existing)")
    p.add_argument("--dry_run", action="store_true",
                    help="List discovered scans without processing")
    p.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity")

    args = p.parse_args(argv)

    # Resolve keep_intermediates
    if args.no_keep_intermediates:
        args.keep_intermediates = False

    return args


def setup_logging(output_dir: Path, level: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level))

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def main():
    args = parse_args()
    setup_logging(args.output_dir, args.log_level)

    from pipeline import run_pipeline

    config = vars(args)
    results = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        intra_model_path=args.intra_model_path,
        config=config,
    )

    n_fail = sum(not r.success and not r.skipped for r in results)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
```
## SECTION 11: training/model.py — COMPLETE CODE

```python
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
    ):
        super().__init__()
        self.transform_type = transform_type
        self.num_params = 9 if transform_type == "affine" else 6
        self.image_size = image_size

        self.backbone = GlobalNet(
            image_size=image_size,
            spatial_dims=3,
            in_channels=in_channels,
            num_channel_initial=num_channel_initial,
            depth=depth,
        )

        # Replace final layer
        in_features = self.backbone.final.in_features
        self.backbone.final = nn.Linear(in_features, self.num_params)
        nn.init.zeros_(self.backbone.final.weight)
        nn.init.zeros_(self.backbone.final.bias)

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
            "in_channels": 2,
            "num_channel_initial": 16,
            "depth": 4,
            "transform_type": self.transform_type,
            "num_params": self.num_params,
        }
```

---

## SECTION 12: training/dataset.py — COMPLETE CODE

```python
"""
Datasets for training the affine registration network.

LongitudinalPairDataset: real (follow-up, baseline) pairs.
SyntheticPairDataset: single scans with known synthetic affines.
CombinedDataset: interleaves real and synthetic at a configurable ratio.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from affine_utils import compose_affine, decompose_affine, warp_volume_torch, extend_to_4x4
from preprocessing import preprocess_for_network
from dataset import ScanRecord


class LongitudinalPairDataset(Dataset):
    """
    Yields (moving, fixed) pairs from real longitudinal data.
    For each patient with N timepoints, generates N-1 pairs:
        (tp2, tp1), (tp3, tp1), ..., (tpN, tp1)
    """

    def __init__(
        self,
        patient_groups: Dict[str, List[ScanRecord]],
        target_size=(96, 96, 96),
    ):
        self.pairs: List[tuple] = []
        self.target_size = target_size

        for group_key, records in patient_groups.items():
            if len(records) < 2:
                continue
            baseline = records[0]
            for followup in records[1:]:
                self.pairs.append((followup.volume_path, baseline.volume_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        moving_path, fixed_path = self.pairs[idx]
        moving = preprocess_for_network(moving_path, self.target_size).squeeze(0)  # (1,D,H,W)
        fixed = preprocess_for_network(fixed_path, self.target_size).squeeze(0)
        return {
            "moving": moving,
            "fixed": fixed,
            "has_gt": torch.tensor(False),
            "gt_params": torch.zeros(9),  # placeholder, unused
        }


class SyntheticPairDataset(Dataset):
    """
    Generates synthetic pairs by applying known random affines to scans.

    For each __getitem__:
    1. Pick a random scan as "fixed"
    2. Apply random affine -> "moving"
    3. Ground truth = inverse of applied affine (what network should predict)
    """

    def __init__(
        self,
        scan_paths: List[Path],
        target_size=(96, 96, 96),
        max_rotation_deg: float = 15.0,
        max_translation: float = 0.1,
        max_scale_offset: float = 0.05,
        length: int = 1000,
        transform_type: str = "affine",
    ):
        self.scan_paths = scan_paths
        self.target_size = target_size
        self.max_rotation = np.radians(max_rotation_deg)
        self.max_translation = max_translation
        self.max_scale_offset = max_scale_offset
        self.length = length
        self.transform_type = transform_type
        self.num_params = 9 if transform_type == "affine" else 6

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        path = random.choice(self.scan_paths)
        fixed = preprocess_for_network(path, self.target_size).squeeze(0)  # (1,D,H,W)

        # Sample random params (the affine we APPLY to fixed -> create moving)
        fwd_params = self._sample_params()
        fwd_tensor = torch.tensor(fwd_params, dtype=torch.float32).unsqueeze(0)
        fwd_affine = compose_affine(fwd_tensor, self.transform_type)  # (1,3,4)

        # Warp fixed to create moving
        moving = warp_volume_torch(
            fixed.unsqueeze(0), fwd_affine, mode="bilinear"
        ).squeeze(0)  # (1,D,H,W)

        # Compute inverse params (ground truth for the network)
        inv_params = self._compute_inverse_params(fwd_params)

        return {
            "moving": moving,
            "fixed": fixed,
            "has_gt": torch.tensor(True),
            "gt_params": torch.tensor(inv_params, dtype=torch.float32),
        }

    def _sample_params(self) -> np.ndarray:
        r = self.max_rotation
        t = self.max_translation
        params = [
            np.random.uniform(-r, r),  # rx
            np.random.uniform(-r, r),  # ry
            np.random.uniform(-r, r),  # rz
            np.random.uniform(-t, t),  # tx
            np.random.uniform(-t, t),  # ty
            np.random.uniform(-t, t),  # tz
        ]
        if self.transform_type == "affine":
            s = self.max_scale_offset
            params.extend([
                np.random.uniform(-s, s),  # sx offset
                np.random.uniform(-s, s),  # sy offset
                np.random.uniform(-s, s),  # sz offset
            ])
        return np.array(params, dtype=np.float32)

    def _compute_inverse_params(self, fwd_params: np.ndarray) -> np.ndarray:
        """
        Compute exact inverse params via matrix inversion + decomposition.
        """
        fwd_tensor = torch.tensor(fwd_params, dtype=torch.float32).unsqueeze(0)
        fwd_affine = compose_affine(fwd_tensor, self.transform_type)[0].numpy()

        # Invert the 4x4
        full = extend_to_4x4(fwd_affine)
        inv_full = np.linalg.inv(full)
        inv_3x4 = inv_full[:3, :]

        # Decompose back to params
        d = decompose_affine(inv_3x4)
        inv_params = [
            np.radians(d["rotation_deg"][0]),
            np.radians(d["rotation_deg"][1]),
            np.radians(d["rotation_deg"][2]),
            d["translation"][0],
            d["translation"][1],
            d["translation"][2],
        ]
        if self.transform_type == "affine":
            inv_params.extend([
                d["scale"][0] - 1.0,
                d["scale"][1] - 1.0,
                d["scale"][2] - 1.0,
            ])
        return np.array(inv_params, dtype=np.float32)


class CombinedDataset(Dataset):
    """
    Interleaves real and synthetic pairs at a given ratio.
    """

    def __init__(
        self,
        real_dataset: LongitudinalPairDataset,
        synthetic_dataset: SyntheticPairDataset,
        real_ratio: float = 0.5,
    ):
        self.real = real_dataset
        self.synthetic = synthetic_dataset
        n_real = len(self.real)

        if n_real == 0:
            self.n_real = 0
            self.n_synthetic = len(self.synthetic)
        elif real_ratio <= 0:
            self.n_real = 0
            self.n_synthetic = len(self.synthetic)
        else:
            self.n_real = n_real
            total = int(n_real / real_ratio)
            self.n_synthetic = max(total - n_real, 0)

        self.synthetic.length = max(self.n_synthetic, 1)

    def __len__(self):
        return self.n_real + self.n_synthetic

    def __getitem__(self, idx):
        if idx < self.n_real:
            return self.real[idx]
        else:
            return self.synthetic[idx - self.n_real]
```

---

## SECTION 13: training/augmentations.py — COMPLETE CODE

```python
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
```

---

## SECTION 14: training/losses.py — COMPLETE CODE

```python
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
```

---

## SECTION 15: training/train_intra.py — COMPLETE CODE

```python
"""
Training script for the intra-patient longitudinal alignment model.

Usage:
    python training/train_intra.py \
        --config training/configs/intra_patient.yaml \
        --data_dir /data/vs_cohort \
        --output_dir /training/intra
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import discover_scans, group_by_patient
from affine_utils import warp_volume_torch
from training.model import AffineRegNet
from training.dataset import LongitudinalPairDataset, SyntheticPairDataset, CombinedDataset
from training.augmentations import apply_intensity_augmentation
from training.losses import (
    get_similarity_loss,
    identity_regularization,
    determinant_regularization,
    supervised_param_loss,
)

logger = logging.getLogger(__name__)


def split_by_patient(groups, val_fraction=0.2, seed=42):
    """Split patient groups into train/val, stratified by patient_id."""
    rng = np.random.RandomState(seed)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_fraction))
    val_keys = set(keys[:n_val])
    train_groups = {k: v for k, v in groups.items() if k not in val_keys}
    val_groups = {k: v for k, v in groups.items() if k in val_keys}
    return train_groups, val_groups


def get_all_scan_paths(groups):
    """Extract all volume paths from patient groups."""
    paths = []
    for records in groups.values():
        for r in records:
            paths.append(r.volume_path)
    return paths


def compute_lambda_mi(epoch, warmup_start, warmup_mi_start, warmup_mi_end, lambda_sim):
    """Compute MI loss weight with warm-up schedule."""
    if epoch < warmup_mi_start:
        return 0.0
    elif epoch < warmup_mi_end:
        frac = (epoch - warmup_mi_start) / max(warmup_mi_end - warmup_mi_start, 1)
        return lambda_sim * frac
    else:
        return lambda_sim


def train(args):
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Discover and group
    records = discover_scans(Path(args.data_dir))
    groups = group_by_patient(records)
    train_groups, val_groups = split_by_patient(
        groups,
        val_fraction=cfg.get("val_fraction", 0.2),
        seed=cfg.get("split_seed", 42),
    )
    logger.info(
        "Train: %d groups (%d scans), Val: %d groups (%d scans)",
        len(train_groups), sum(len(v) for v in train_groups.values()),
        len(val_groups), sum(len(v) for v in val_groups.values()),
    )

    # Datasets
    target_size = tuple(cfg.get("image_size", [96, 96, 96]))
    transform_type = cfg.get("transform_type", "affine")

    train_real = LongitudinalPairDataset(train_groups, target_size)
    train_synthetic = SyntheticPairDataset(
        scan_paths=get_all_scan_paths(train_groups),
        target_size=target_size,
        max_rotation_deg=cfg.get("max_rotation_deg", 15.0),
        max_translation=cfg.get("max_translation", 0.1),
        max_scale_offset=cfg.get("max_scale_offset", 0.05),
        length=cfg.get("synthetic_length_per_epoch", 1000),
        transform_type=transform_type,
    )
    train_dataset = CombinedDataset(
        train_real, train_synthetic, real_ratio=cfg.get("real_ratio", 0.5)
    )

    val_real = LongitudinalPairDataset(val_groups, target_size)
    val_synthetic = SyntheticPairDataset(
        scan_paths=get_all_scan_paths(val_groups),
        target_size=target_size,
        max_rotation_deg=cfg.get("max_rotation_deg", 15.0),
        max_translation=cfg.get("max_translation", 0.1),
        max_scale_offset=cfg.get("max_scale_offset", 0.05),
        length=200,
        transform_type=transform_type,
    )
    val_dataset = CombinedDataset(val_real, val_synthetic, real_ratio=0.5)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    model = AffineRegNet(
        image_size=target_size,
        transform_type=transform_type,
    ).to(device)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # Loss
    sim_loss_fn = get_similarity_loss(
        cfg.get("similarity_loss", "mi"),
        num_bins=cfg.get("mi_num_bins", 32),
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.get("scheduler_T0", 50),
        T_mult=cfg.get("scheduler_Tmult", 2),
    )

    # Logging
    writer = SummaryWriter(str(output_dir / "tb_logs"))

    # Training
    best_val_loss = float("inf")
    patience_counter = 0
    max_epochs = cfg.get("max_epochs", 300)
    patience = cfg.get("patience", 30)

    lambda_id = cfg.get("lambda_identity", 0.01)
    lambda_det = cfg.get("lambda_det", 0.1)
    lambda_sup = cfg.get("lambda_supervised", 0.5)
    lambda_sim_max = cfg.get("lambda_sim", 1.0)
    warmup_start = cfg.get("warmup_start", 0)
    warmup_mi_start = cfg.get("warmup_mi_start", 10)
    warmup_mi_end = cfg.get("warmup_mi_end", 30)
    grad_clip = cfg.get("grad_clip_norm", 1.0)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_sim = 0.0
        epoch_sup = 0.0
        n_batches = 0

        lam_mi = compute_lambda_mi(epoch, warmup_start, warmup_mi_start, warmup_mi_end, lambda_sim_max)

        for batch in train_loader:
            moving = batch["moving"].to(device)  # (B, 1, D, H, W)
            fixed = batch["fixed"].to(device)
            has_gt = batch["has_gt"]  # (B,) bool
            gt_params = batch["gt_params"].to(device)  # (B, num_params)

            # Augment moving intensity
            aug_moving = torch.stack([
                apply_intensity_augmentation(moving[i]) for i in range(moving.shape[0])
            ], dim=0)

            x = torch.cat([aug_moving, fixed], dim=1)  # (B, 2, D, H, W)
            params, affine = model(x)

            warped = warp_volume_torch(aug_moving, affine, mode="bilinear")

            # Losses
            loss_sim = lam_mi * sim_loss_fn(warped, fixed) if lam_mi > 0 else torch.tensor(0.0, device=device)
            loss_id = lambda_id * identity_regularization(params)
            loss_det = lambda_det * determinant_regularization(affine)

            loss_sup_val = torch.tensor(0.0, device=device)
            if has_gt.any():
                mask = has_gt.to(device)
                loss_sup_val = lambda_sup * supervised_param_loss(params[mask], gt_params[mask])

            loss = loss_sim + loss_id + loss_det + loss_sup_val

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sim += loss_sim.item()
            epoch_sup += loss_sup_val.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_sup_total = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                moving = batch["moving"].to(device)
                fixed = batch["fixed"].to(device)
                has_gt = batch["has_gt"]
                gt_params = batch["gt_params"].to(device)

                x = torch.cat([moving, fixed], dim=1)
                params, affine = model(x)
                warped = warp_volume_torch(moving, affine)

                v_sim = sim_loss_fn(warped, fixed)
                v_id = lambda_id * identity_regularization(params)
                v_det = lambda_det * determinant_regularization(affine)
                v_loss = v_sim + v_id + v_det

                if has_gt.any():
                    mask = has_gt.to(device)
                    v_sup = supervised_param_loss(params[mask], gt_params[mask])
                    val_sup_total += v_sup.item()

                val_loss_total += v_loss.item()
                val_n += 1

        avg_val_loss = val_loss_total / max(val_n, 1)

        # Logging
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/sim", epoch_sim / max(n_batches, 1), epoch)
        writer.add_scalar("train/sup", epoch_sup / max(n_batches, 1), epoch)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("val/sup", val_sup_total / max(val_n, 1), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("lambda_mi", lam_mi, epoch)

        logger.info(
            "Epoch %d/%d — train=%.4f val=%.4f lr=%.2e lam_mi=%.2f",
            epoch + 1, max_epochs, avg_loss, avg_val_loss,
            optimizer.param_groups[0]["lr"], lam_mi,
        )

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model.get_config(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                },
                str(output_dir / "best_model.pth"),
            )
            logger.info("  New best model saved (val=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Periodic checkpoint
        ckpt_every = cfg.get("checkpoint_every_n_epochs", 10)
        if (epoch + 1) % ckpt_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model.get_config(),
                    "epoch": epoch,
                    "val_loss": avg_val_loss,
                },
                str(output_dir / f"checkpoint_epoch{epoch+1}.pth"),
            )

    # Final save
    final_path = output_dir / "globalnet_intra.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.get_config(),
            "epoch": epoch,
            "val_loss": best_val_loss,
        },
        str(final_path),
    )
    logger.info("Final model saved: %s", final_path)
    writer.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
```

---

## SECTION 16: training/train_template.py — COMPLETE CODE

```python
"""
Training script for template registration model.

Differences from train_intra.py:
1. Fixed image is ALWAYS the MNI152 template (loaded once).
2. All training uses synthetic pairs (patient -> template, with MI only).
3. Looser regularization: lambda_identity=0.001.
4. Wider synthetic ranges: rotation 30deg, translation 0.2, scale 0.15.
5. No supervised warm-up (MI from epoch 0).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import discover_scans, group_by_patient
from preprocessing import preprocess_for_network
from affine_utils import warp_volume_torch
from training.model import AffineRegNet
from training.augmentations import apply_intensity_augmentation
from training.losses import (
    get_similarity_loss,
    identity_regularization,
    determinant_regularization,
)

logger = logging.getLogger(__name__)


class TemplateRegistrationDataset(Dataset):
    """Each sample: (patient_scan, template). Fixed = template always."""

    def __init__(self, scan_paths, template_path, target_size=(96, 96, 96)):
        self.scan_paths = scan_paths
        self.template_tensor = preprocess_for_network(template_path, target_size).squeeze(0)
        self.target_size = target_size

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        moving = preprocess_for_network(self.scan_paths[idx], self.target_size).squeeze(0)
        return {"moving": moving, "fixed": self.template_tensor}


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # All scans as moving images
    records = discover_scans(Path(args.data_dir))
    all_paths = [r.volume_path for r in records]

    # Split
    import numpy as np
    rng = np.random.RandomState(cfg.get("split_seed", 42))
    rng.shuffle(all_paths)
    n_val = max(1, int(len(all_paths) * cfg.get("val_fraction", 0.2)))
    train_paths, val_paths = all_paths[n_val:], all_paths[:n_val]

    template_path = Path(args.template_path)
    target_size = tuple(cfg.get("image_size", [96, 96, 96]))

    train_ds = TemplateRegistrationDataset(train_paths, template_path, target_size)
    val_ds = TemplateRegistrationDataset(val_paths, template_path, target_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8),
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8),
                            shuffle=False, num_workers=4)

    model = AffineRegNet(image_size=target_size, transform_type="affine").to(device)
    sim_loss_fn = get_similarity_loss(cfg.get("similarity_loss", "mi"))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.get("scheduler_T0", 50), T_mult=2)

    writer = SummaryWriter(str(output_dir / "tb_logs"))
    best_val = float("inf")
    patience_counter = 0

    lambda_id = cfg.get("lambda_identity", 0.001)  # looser
    lambda_det = cfg.get("lambda_det", 0.1)

    for epoch in range(cfg.get("max_epochs", 300)):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)
            moving = torch.stack([apply_intensity_augmentation(moving[i]) for i in range(moving.shape[0])])

            x = torch.cat([moving, fixed], dim=1)
            params, affine = model(x)
            warped = warp_volume_torch(moving, affine)

            loss = sim_loss_fn(warped, fixed) + lambda_id * identity_regularization(params) + lambda_det * determinant_regularization(affine)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        scheduler.step()
        avg = epoch_loss / max(n, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                moving = batch["moving"].to(device)
                fixed = batch["fixed"].to(device)
                x = torch.cat([moving, fixed], dim=1)
                params, affine = model(x)
                warped = warp_volume_torch(moving, affine)
                vl = sim_loss_fn(warped, fixed) + lambda_id * identity_regularization(params)
                val_loss += vl.item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        writer.add_scalar("train/loss", avg, epoch)
        writer.add_scalar("val/loss", avg_val, epoch)
        logger.info("Epoch %d — train=%.4f val=%.4f", epoch + 1, avg, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "config": model.get_config(),
                        "epoch": epoch, "val_loss": best_val}, str(output_dir / "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.get("patience", 30):
                break

    torch.save({"model_state_dict": model.state_dict(), "config": model.get_config(),
                "epoch": epoch, "val_loss": best_val}, str(output_dir / "globalnet_template.pth"))
    writer.close()


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--template_path", required=True)
    p.add_argument("--output_dir", required=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()
```

---

## SECTION 17: training/evaluate.py — COMPLETE CODE

```python
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
```

---

## SECTION 18: training/configs/intra_patient.yaml

```yaml
# Intra-patient longitudinal alignment model configuration

image_size: [96, 96, 96]
transform_type: affine

# Data
real_ratio: 0.5
synthetic_length_per_epoch: 1000
max_rotation_deg: 15.0
max_translation: 0.1
max_scale_offset: 0.05

# Training
batch_size: 8
num_workers: 4
max_epochs: 300
lr: 1.0e-4
scheduler_T0: 50
scheduler_Tmult: 2
grad_clip_norm: 1.0

# Loss weights
similarity_loss: mi
mi_num_bins: 32
lambda_sim: 1.0
lambda_identity: 0.01
lambda_det: 0.1
lambda_supervised: 0.5

# Warm-up (epochs)
warmup_start: 0
warmup_mi_start: 10
warmup_mi_end: 30

# Early stopping
patience: 30
val_fraction: 0.2
split_seed: 42

# Logging
checkpoint_every_n_epochs: 10
```

---

## SECTION 19: training/configs/template.yaml

```yaml
# Template registration model configuration

image_size: [96, 96, 96]
transform_type: affine

batch_size: 8
num_workers: 4
max_epochs: 300
lr: 1.0e-4
scheduler_T0: 50

similarity_loss: mi
mi_num_bins: 32
lambda_identity: 0.001
lambda_det: 0.1

patience: 30
val_fraction: 0.2
split_seed: 42
checkpoint_every_n_epochs: 10
```

---

## SECTION 20: tests/test_affine_utils.py — COMPLETE CODE

**THIS TEST MUST PASS BEFORE ANY OTHER WORK PROCEEDS (Day 4 gate).**

```python
"""
Critical tests for affine coordinate math.
Run: python -m pytest tests/test_affine_utils.py -v
"""

import numpy as np
import torch
import nibabel as nib
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from affine_utils import (
    compose_affine,
    decompose_affine,
    invert_affine_3x4,
    extend_to_4x4,
    normalized_affine_to_world,
    apply_affine_to_volume,
    _voxel_to_normalized,
    _normalized_to_voxel,
    warp_volume_torch,
)


def _make_test_nifti(shape=(64, 64, 64), voxel_size=(1.0, 1.0, 1.0)):
    """Create a test NIfTI with a bright sphere in the center."""
    data = np.zeros(shape, dtype=np.float32)
    center = np.array(shape) / 2.0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                dist = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
                if dist < min(shape) / 4:
                    data[i, j, k] = 1.0

    affine = np.diag([*voxel_size, 1.0])
    return nib.Nifti1Image(data, affine)


class TestComposeDecompose:
    def test_identity(self):
        """Zero params -> identity affine."""
        params = torch.zeros(1, 9)
        affine = compose_affine(params, "affine")
        assert affine.shape == (1, 3, 4)
        expected = torch.eye(3, 4).unsqueeze(0)
        assert torch.allclose(affine, expected, atol=1e-6)

    def test_rigid_identity(self):
        params = torch.zeros(1, 6)
        affine = compose_affine(params, "rigid")
        assert affine.shape == (1, 3, 4)
        expected = torch.eye(3, 4).unsqueeze(0)
        assert torch.allclose(affine, expected, atol=1e-6)

    def test_roundtrip(self):
        """Compose -> decompose should recover original params (approximately)."""
        params_np = np.array([0.1, -0.05, 0.15, 0.02, -0.01, 0.03, 0.02, -0.01, 0.01])
        params = torch.tensor(params_np, dtype=torch.float32).unsqueeze(0)
        affine = compose_affine(params, "affine")
        d = decompose_affine(affine[0].numpy())
        # Rotations should be close (in degrees)
        assert abs(d["rotation_deg"][0] - np.degrees(0.1)) < 1.0
        assert abs(d["rotation_deg"][1] - np.degrees(-0.05)) < 1.0
        assert abs(d["rotation_deg"][2] - np.degrees(0.15)) < 1.0
        # Scale should be close to 1 + offset
        assert abs(d["scale"][0] - 1.02) < 0.02
        assert abs(d["scale"][1] - 0.99) < 0.02

    def test_determinant_unity(self):
        """Rigid transform should have det = 1.0."""
        params = torch.tensor([[0.2, -0.1, 0.3, 1.0, -0.5, 2.0]], dtype=torch.float32)
        affine = compose_affine(params, "rigid")
        det = torch.det(affine[0, :3, :3])
        assert abs(det.item() - 1.0) < 1e-5


class TestCoordinateConversion:
    def test_v2n_n2v_roundtrip(self):
        shape = (96, 96, 96)
        V2N = _voxel_to_normalized(shape)
        N2V = _normalized_to_voxel(shape)
        identity = N2V @ V2N
        assert np.allclose(identity, np.eye(4), atol=1e-10)

    def test_v2n_corners(self):
        shape = (96, 96, 96)
        V2N = _voxel_to_normalized(shape)
        # Voxel (0,0,0) -> (-1,-1,-1)
        corner_0 = V2N @ np.array([0, 0, 0, 1])
        assert np.allclose(corner_0[:3], [-1, -1, -1], atol=1e-10)
        # Voxel (95,95,95) -> (1,1,1)
        corner_1 = V2N @ np.array([95, 95, 95, 1])
        assert np.allclose(corner_1[:3], [1, 1, 1], atol=1e-10)


class TestKnownTransformRoundtrip:
    """
    CRITICAL GATE TEST.
    Apply a known world-space translation, recover it through the full pipeline.
    """

    def test_5mm_z_translation(self):
        """
        1. Create a test volume.
        2. Apply a 5mm Z-translation to create a "moved" volume.
        3. Use identity as the network-predicted affine (since the volumes
           differ by exactly 5mm in Z, we test that the coordinate chain
           from normalized -> world is correct by verifying the math).
        """
        nii = _make_test_nifti(shape=(64, 64, 64), voxel_size=(1.0, 1.0, 1.0))

        # Create "moved" volume by shifting 5mm in Z
        data = nii.get_fdata()
        shifted_affine = nii.affine.copy()
        shifted_affine[2, 3] += 5.0  # 5mm Z translation in world
        moved_nii = nib.Nifti1Image(data.copy(), shifted_affine)

        # The world-space affine that maps nii's grid -> moved_nii's world coords
        # should recover the 5mm shift
        # For this test: if we use identity as the normalized affine,
        # the world affine should just reflect the header difference

        identity_norm = np.eye(3, 4)
        world_aff = normalized_affine_to_world(identity_norm, moved_nii, nii, (96, 96, 96))

        # The world affine maps fixed_world -> moving_world
        # Since fixed = nii (origin at 0,0,0) and moving = moved_nii (origin at 0,0,5)
        # A point at (0,0,0) in fixed_world should map to (0,0,5) in moving_world
        test_point = world_aff @ np.array([0, 0, 0, 1])
        assert abs(test_point[2] - 5.0) < 0.5, f"Expected Z≈5.0, got {test_point[2]}"

    def test_apply_and_recover(self):
        """Apply affine to a volume and verify voxel values are in the right place."""
        nii = _make_test_nifti(shape=(64, 64, 64), voxel_size=(1.0, 1.0, 1.0))
        data = nii.get_fdata()

        # The center voxel (32,32,32) should have value 1.0
        assert data[32, 32, 32] == 1.0

        # Apply identity world affine -> output should match input
        identity_world = np.eye(4)
        result = apply_affine_to_volume(nii, identity_world, nii, "trilinear")
        result_data = result.get_fdata()

        # Center should still be ~1.0
        assert abs(result_data[32, 32, 32] - 1.0) < 0.01


class TestWarpVolumeTorch:
    def test_identity_warp(self):
        """Identity affine should not change the volume."""
        vol = torch.randn(1, 1, 32, 32, 32)
        identity = torch.eye(3, 4).unsqueeze(0)
        warped = warp_volume_torch(vol, identity)
        assert torch.allclose(vol, warped, atol=1e-4)
```

---

## SECTION 21: OUTPUT DIRECTORY STRUCTURE

```
output_dir/
├── T2/
│   ├── VS001/
│   │   ├── VS001_tp1/
│   │   │   ├── VS001_tp1_preprocessed.nii.gz
│   │   │   ├── VS001_tp1_brain_mask.nii.gz
│   │   │   ├── VS001_tp1_seg.nii.gz
│   │   │   ├── VS001_tp1_affine.npy
│   │   │   ├── VS001_tp1_params.json
│   │   │   ├── VS001_tp1_qc.png
│   │   │   └── intermediates/
│   │   │       ├── VS001_tp1_n4.nii.gz
│   │   │       ├── VS001_tp1_bias_field.nii.gz
│   │   │       └── VS001_tp1_brain_mask_native.nii.gz
│   │   ├── VS001_tp2/
│   │   │   ├── VS001_tp2_preprocessed.nii.gz
│   │   │   ├── VS001_tp2_brain_mask.nii.gz
│   │   │   ├── VS001_tp2_seg.nii.gz
│   │   │   ├── VS001_tp2_affine.npy
│   │   │   ├── VS001_tp2_params.json
│   │   │   ├── VS001_tp2_qc.png
│   │   │   └── intermediates/
│   │   │       ├── VS001_tp2_n4.nii.gz
│   │   │       ├── VS001_tp2_affine_aligned.nii.gz
│   │   │       └── VS001_tp2_normalized.nii.gz
│   │   └── VS001_longitudinal_qc.png
│   └── VS002/ ...
├── T1C/ ...
├── failures.csv
├── pipeline.log
└── qc_summary.html
```

---

## SECTION 22: ONE-MONTH TIMELINE

### Week 1 — Foundation (Days 1-7)

| Day | Task | Gate |
|---|---|---|
| 1 | Scaffold + requirements + verify all imports | `python -c "import monai, torch, nibabel, SimpleITK"` |
| 2 | `dataset.py` | Tested on real data, patients grouped |
| 3 | `affine_utils.py` | All functions implemented |
| 4 | `tests/test_affine_utils.py` | **ALL TESTS PASS. Do not proceed otherwise.** |
| 5 | `preprocessing.py` N4 | Tested on 3 scans |
| 6 | `preprocessing.py` normalize + resample | Output spacing verified |
| 7 | `skull_strip.py` | Mask covers IAC on 3 test scans |

### Week 2 — Training (Days 8-14)

| Day | Task | Gate |
|---|---|---|
| 8 | `training/model.py` | Forward pass: random input -> (B,9) + near-identity (B,3,4) |
| 9 | `training/dataset.py` + `augmentations.py` | DataLoader yields correct shapes |
| 10 | `training/losses.py` + configs | Losses return finite scalars |
| 11 | `training/train_intra.py` | First 20 epochs: loss decreasing, no NaN |
| 12 | Train to convergence | ~200 epochs, 2-4 hours |
| 13 | `training/evaluate.py` | Metrics computed |
| 14 | Analyze | Rotation MAE < 2°, translation MAE < 2mm, seg Dice > 0.8 |

### Week 3 — Pipeline (Days 15-21)

| Day | Task | Gate |
|---|---|---|
| 15 | `registration.py` | 1 patient aligned correctly |
| 16 | `pipeline.py` | 3 patients end-to-end |
| 17 | `run.py` + integration test | CLI works |
| 18 | `qc.py` per-scan | Overlays look correct |
| 19 | `qc.py` longitudinal + HTML | HTML in browser |
| 20 | Full cohort run | All patients, failures < 5% |
| 21 | Bug fixes | Clean run |

### Week 4 — Polish (Days 22-28)

| Day | Task | Gate |
|---|---|---|
| 22 | `training/train_template.py` | Template model converged |
| 23 | Template integration | `--register_to_template` works |
| 24 | Edge cases | 1-tp patient, 7-tp patient, missing segs |
| 25 | Parallelism (stretch) | `--n_workers 4` speedup |
| 26 | README.md | Complete |
| 27 | Cleanup: types, docstrings | Clean ruff/mypy |
| 28 | Final run + archive | Done |
