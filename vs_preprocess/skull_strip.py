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
    import torch
    from HD_BET.hd_bet_prediction import get_hdbet_predictor, hdbet_predict

    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    torch_device = torch.device("cpu" if device == "cpu" else "cuda")
    use_tta = mode == "accurate"

    predictor = get_hdbet_predictor(use_tta=use_tta, device=torch_device)

    with tempfile.TemporaryDirectory(prefix="hdbet_") as tmpdir:
        # hdbet_predict requires a full .nii.gz output filename when given a
        # single file as input; the mask is written as <stem>_bet.nii.gz
        tmp_out = str(Path(tmpdir) / "brain.nii.gz")

        hdbet_predict(
            input_file_or_folder=str(input_path),
            output_file_or_folder=tmp_out,
            predictor=predictor,
            keep_brain_mask=True,
            compute_brain_extracted_image=False,
        )

        # HD-BET 2.x writes the mask as <output_stem>_bet.nii.gz
        mask_path = Path(tmpdir) / "brain_bet.nii.gz"
        if not mask_path.exists():
            # Fall back to glob in case naming changes in a future patch
            mask_candidates = list(Path(tmpdir).glob("*bet*.nii.gz"))
            if not mask_candidates:
                contents = list(Path(tmpdir).iterdir())
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
