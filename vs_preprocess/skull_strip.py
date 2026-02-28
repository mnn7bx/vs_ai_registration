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
