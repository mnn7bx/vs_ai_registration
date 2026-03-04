"""
Smoke test — verifies all modules import and core CPU operations run correctly.
No GPU, no trained model, no HD-BET required.

Run:
    python -m pytest tests/smoke_test.py -v
    # or directly:
    python tests/smoke_test.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Module imports
# ─────────────────────────────────────────────────────────────────────────────

class TestImports:
    def test_import_dataset(self):
        import dataset  # noqa: F401

    def test_import_affine_utils(self):
        import affine_utils  # noqa: F401

    def test_import_preprocessing(self):
        import preprocessing  # noqa: F401

    def test_import_qc(self):
        import qc  # noqa: F401

    def test_import_pipeline(self):
        import pipeline  # noqa: F401

    def test_import_run(self):
        import run  # noqa: F401

    def test_import_training_model(self):
        from training import model  # noqa: F401

    def test_import_training_losses(self):
        from training import losses  # noqa: F401

    def test_import_training_augmentations(self):
        from training import augmentations  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_nifti(shape=(32, 32, 32), voxel_mm=1.0) -> nib.Nifti1Image:
    data = (np.random.rand(*shape) * 1000).astype(np.float32)
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    return nib.Nifti1Image(data, affine)


def _make_mask(shape=(32, 32, 32)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float32)
    s = slice(shape[0] // 4, 3 * shape[0] // 4)
    mask[s, s, s] = 1.0
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Dataset discovery
# ─────────────────────────────────────────────────────────────────────────────

class TestDatasetSmoke:
    def test_discover_scans_returns_list(self, tmp_path):
        from dataset import discover_scans

        vol = tmp_path / "T2" / "VS001_tp1" / "VS001_tp1.nii.gz"
        vol.parent.mkdir(parents=True)
        nib.save(_make_nifti(), str(vol))

        records = discover_scans(tmp_path)
        assert len(records) >= 1

    def test_group_by_patient_marks_one_baseline(self, tmp_path):
        from dataset import discover_scans, group_by_patient

        for tp in ["VS001_tp1", "VS001_tp2"]:
            vol = tmp_path / "T2" / tp / f"{tp}.nii.gz"
            vol.parent.mkdir(parents=True)
            nib.save(_make_nifti(), str(vol))

        records = discover_scans(tmp_path)
        groups = group_by_patient(records)

        assert "T2/VS001" in groups
        baselines = [r for r in groups["T2/VS001"] if r.is_baseline]
        assert len(baselines) == 1

    def test_discover_scans_from_csv(self, tmp_path):
        """CSV must use the real column names expected by discover_scans_from_csv."""
        import csv as csv_mod
        from dataset import discover_scans_from_csv

        # Image path in CSV is relative to data_root (tmp_path)
        rel_vol = Path("T2") / "126_tp1" / "126_tp1.nii.gz"
        abs_vol = tmp_path / rel_vol
        abs_vol.parent.mkdir(parents=True)
        nib.save(_make_nifti(), str(abs_vol))

        csv_file = tmp_path / "scans.csv"
        with open(csv_file, "w", newline="") as f:
            w = csv_mod.writer(f)
            w.writerow([
                "Study Type", "Reworked Patient ID", "Timepoint",
                "image path", "segmentation path",
            ])
            w.writerow(["T2", "126", "tp1", str(rel_vol), ""])

        records = discover_scans_from_csv(csv_file, tmp_path)
        assert len(records) == 1
        assert records[0].modality == "T2"
        assert records[0].patient_id == "126"


# ─────────────────────────────────────────────────────────────────────────────
# Intensity normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeIntensity:
    def test_minmax_output_range(self):
        from preprocessing import normalize_intensity

        data = (np.random.rand(32, 32, 32) * 500 + 100).astype(np.float32)
        mask = _make_mask()

        result = normalize_intensity(data, mask, method="minmax")

        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_zscore_output_has_zero_mean_inside_mask(self):
        from preprocessing import normalize_intensity

        data = (np.random.rand(32, 32, 32) * 300 + 50).astype(np.float32)
        mask = _make_mask()

        result = normalize_intensity(data, mask, method="zscore")

        # Mean inside mask should be close to zero
        mean_in_mask = result[mask > 0].mean()
        assert abs(mean_in_mask) < 0.1

    def test_output_shape_preserved(self):
        from preprocessing import normalize_intensity

        shape = (24, 30, 36)
        data = np.random.rand(*shape).astype(np.float32)
        mask = np.ones(shape, dtype=np.float32)

        result = normalize_intensity(data, mask)
        assert result.shape == shape


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

class TestResampling:
    def test_resample_to_isotropic_changes_spacing(self):
        from preprocessing import resample_to_isotropic

        # Input at 2mm isotropic, target 1mm → voxel count doubles each axis
        nii = _make_nifti(shape=(16, 16, 16), voxel_mm=2.0)
        result = resample_to_isotropic(nii, target_spacing=(1.0, 1.0, 1.0))

        assert result.shape[0] > nii.shape[0]

    def test_resample_mask_uses_nearest_neighbor(self):
        from preprocessing import resample_mask_to_isotropic

        mask_data = np.zeros((16, 16, 16), dtype=np.float32)
        mask_data[4:12, 4:12, 4:12] = 1.0
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        nii = nib.Nifti1Image(mask_data, affine)

        result = resample_mask_to_isotropic(nii, target_spacing=(1.0, 1.0, 1.0))

        # Nearest-neighbor: output values should be binary
        unique = np.unique(result.get_fdata())
        assert set(unique.tolist()).issubset({0.0, 1.0})


# ─────────────────────────────────────────────────────────────────────────────
# Affine utilities
# ─────────────────────────────────────────────────────────────────────────────

class TestAffineUtilsSmoke:
    def test_identity_compose_is_identity(self):
        """Zero params (B,9) → identity 3×4 matrix."""
        import torch
        from affine_utils import compose_affine

        params = torch.zeros(1, 9)
        mat = compose_affine(params)  # (1, 3, 4)
        assert mat.shape == (1, 3, 4)
        np.testing.assert_allclose(mat[0, :, :3].numpy(), np.eye(3), atol=1e-5)

    def test_compose_returns_correct_shape_rigid(self):
        """Rigid (B,6) params → (B,3,4) matrix."""
        import torch
        from affine_utils import compose_affine

        params = torch.zeros(2, 6)
        mat = compose_affine(params, transform_type="rigid")
        assert mat.shape == (2, 3, 4)

    def test_decompose_affine_returns_dict(self):
        """decompose_affine returns a dict with rotation_deg and translation."""
        import torch
        from affine_utils import compose_affine, decompose_affine

        params = torch.zeros(1, 9)
        mat = compose_affine(params)  # (1, 3, 4) tensor
        result = decompose_affine(mat[0].numpy())  # takes (3, 4) numpy

        assert "rotation_deg" in result
        assert "translation" in result

    def test_apply_affine_identity_preserves_data(self):
        """Identity 4×4 world affine → resampled output matches input."""
        from affine_utils import apply_affine_to_volume

        moving = _make_nifti(shape=(24, 24, 24), voxel_mm=1.0)
        reference = _make_nifti(shape=(24, 24, 24), voxel_mm=1.0)
        identity_4x4 = np.eye(4)  # 4×4 world affine

        result = apply_affine_to_volume(moving, identity_4x4, reference)
        orig_data = moving.get_fdata(dtype=np.float32)
        res_data = result.get_fdata(dtype=np.float32)

        np.testing.assert_allclose(res_data, orig_data, atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# Training model forward pass (CPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestModelForwardPass:
    """
    Runs the full model forward pass at tiny spatial resolution to avoid
    heavy memory/time requirements. img_size must match the input tensor.
    """

    def _make_model(self, transform_type="affine", img_size=(16, 16, 16)):
        import torch  # noqa: F401 — import here avoids top-level torch dep
        from training.model import AffineRegNet
        return AffineRegNet(image_size=img_size, in_channels=2,
                            transform_type=transform_type)

    def test_affineregnet_forward_returns_correct_shape(self):
        import torch
        model = self._make_model(transform_type="affine")
        model.eval()

        x = torch.zeros(1, 2, 16, 16, 16)
        with torch.no_grad():
            params, affine_mat = model(x)

        assert params.shape == (1, 9)
        assert affine_mat.shape == (1, 3, 4)

    def test_affineregnet_rigid_mode(self):
        import torch
        model = self._make_model(transform_type="rigid")
        model.eval()

        x = torch.zeros(1, 2, 16, 16, 16)
        with torch.no_grad():
            params, affine_mat = model(x)

        assert params.shape == (1, 6)
        assert affine_mat.shape == (1, 3, 4)

    def test_model_outputs_near_identity_at_init(self):
        """At initialisation, decomposed head outputs ≈ 0 → affine ≈ identity."""
        import torch
        model = self._make_model()
        model.eval()

        x = torch.zeros(1, 2, 16, 16, 16)
        with torch.no_grad():
            _, affine_mat = model(x)

        mat33 = affine_mat[0, :, :3].numpy()
        np.testing.assert_allclose(mat33, np.eye(3), atol=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIArgParsing:
    def test_required_args_parse(self, tmp_path):
        from run import parse_args

        args = parse_args([
            "--data_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
            "--intra_model_path", str(tmp_path / "model.pt"),
        ])

        assert args.data_dir == tmp_path
        assert args.transform_type == "affine"
        assert args.device == "cuda"

    def test_no_keep_intermediates_flag(self, tmp_path):
        from run import parse_args

        args = parse_args([
            "--data_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
            "--intra_model_path", str(tmp_path / "model.pt"),
            "--no_keep_intermediates",
        ])

        assert args.keep_intermediates is False

    def test_csv_path_arg(self, tmp_path):
        from run import parse_args

        csv = tmp_path / "scans.csv"
        csv.touch()
        args = parse_args([
            "--data_dir", str(tmp_path),
            "--output_dir", str(tmp_path / "out"),
            "--intra_model_path", str(tmp_path / "model.pt"),
            "--csv_path", str(csv),
        ])

        assert args.csv_path == csv


# ─────────────────────────────────────────────────────────────────────────────
# Entry point for direct execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
