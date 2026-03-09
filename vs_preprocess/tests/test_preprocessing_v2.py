"""Tests for v2 preprocessing functions (spacing-aware)."""

import random
import numpy as np
import nibabel as nib
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing import center_crop_or_pad, load_and_resample_isotropic, preprocess_for_network_v2, compute_overlap_center
from vs_preprocess.training.dataset import LongitudinalPairDataset, SyntheticPairDataset
from vs_preprocess.dataset import ScanRecord


class TestCenterCropOrPad:
    def test_pad_smaller_volume(self):
        data = np.ones((64, 64, 64), dtype=np.float32)
        result, origin = center_crop_or_pad(data, (128, 128, 128))
        assert result.shape == (128, 128, 128)
        assert origin == (-32, -32, -32)
        assert result[32, 32, 32] == 1.0
        assert result[0, 0, 0] == 0.0

    def test_crop_larger_volume(self):
        data = np.arange(256 * 256 * 256, dtype=np.float32).reshape(256, 256, 256)
        result, origin = center_crop_or_pad(data, (128, 128, 128))
        assert result.shape == (128, 128, 128)
        assert origin == (64, 64, 64)
        assert result[0, 0, 0] == data[64, 64, 64]

    def test_exact_size_noop(self):
        data = np.ones((128, 128, 128), dtype=np.float32) * 5.0
        result, origin = center_crop_or_pad(data, (128, 128, 128))
        assert result.shape == (128, 128, 128)
        assert origin == (0, 0, 0)
        np.testing.assert_array_equal(result, data)

    def test_anisotropic_volume(self):
        data = np.ones((256, 256, 30), dtype=np.float32)
        result, origin = center_crop_or_pad(data, (128, 128, 128))
        assert result.shape == (128, 128, 128)
        assert origin[0] == 64
        assert origin[1] == 64
        assert origin[2] == -49

    def test_custom_center(self):
        data = np.zeros((256, 256, 256), dtype=np.float32)
        data[100, 100, 100] = 1.0
        result, origin = center_crop_or_pad(data, (128, 128, 128), center=(100, 100, 100))
        assert result.shape == (128, 128, 128)
        assert origin == (36, 36, 36)
        assert result[64, 64, 64] == 1.0


class TestLoadAndResampleIsotropic:
    def _make_nifti(self, shape, spacing, tmp_path, name="test.nii.gz"):
        data = np.random.rand(*shape).astype(np.float32)
        affine = np.diag([*spacing, 1.0])
        nii = nib.Nifti1Image(data, affine)
        path = tmp_path / name
        nib.save(nii, str(path))
        return path

    def test_isotropic_volume_unchanged_shape(self, tmp_path):
        path = self._make_nifti((128, 128, 128), (1.0, 1.0, 1.0), tmp_path)
        data, affine = load_and_resample_isotropic(path, target_spacing=(1.0, 1.0, 1.0))
        assert data.shape == (128, 128, 128)
        np.testing.assert_allclose(np.abs(np.diag(affine[:3, :3])), [1.0, 1.0, 1.0], atol=0.01)

    def test_thick_slice_resampled_shape(self, tmp_path):
        path = self._make_nifti((256, 256, 30), (0.5, 0.5, 5.0), tmp_path)
        data, affine = load_and_resample_isotropic(path, target_spacing=(1.0, 1.0, 1.0))
        assert data.shape[0] == 128
        assert data.shape[1] == 128
        assert data.shape[2] == 150

    def test_4d_volume_handled(self, tmp_path):
        data_4d = np.random.rand(64, 64, 64, 3).astype(np.float32)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        nii = nib.Nifti1Image(data_4d, affine)
        path = tmp_path / "test4d.nii.gz"
        nib.save(nii, str(path))
        data, aff = load_and_resample_isotropic(path, target_spacing=(1.0, 1.0, 1.0))
        assert data.ndim == 3


class TestPreprocessForNetworkV2:
    def _make_nifti(self, shape, spacing, tmp_path, name="test.nii.gz"):
        data = np.random.rand(*shape).astype(np.float32) * 100
        affine = np.diag([*spacing, 1.0])
        nii = nib.Nifti1Image(data, affine)
        path = tmp_path / name
        nib.save(nii, str(path))
        return path

    def test_output_shape(self, tmp_path):
        path = self._make_nifti((256, 256, 30), (0.5, 0.5, 5.0), tmp_path)
        tensor, meta = preprocess_for_network_v2(path, target_size=(128, 128, 128))
        assert tensor.shape == (1, 1, 128, 128, 128)

    def test_normalized_range(self, tmp_path):
        path = self._make_nifti((128, 128, 128), (1.0, 1.0, 1.0), tmp_path)
        tensor, meta = preprocess_for_network_v2(path)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0 + 1e-6

    def test_metadata_returned(self, tmp_path):
        path = self._make_nifti((200, 200, 200), (0.8, 0.8, 0.8), tmp_path)
        tensor, meta = preprocess_for_network_v2(path)
        assert "resampled_affine" in meta
        assert "resampled_shape" in meta
        assert "crop_origin" in meta
        assert "crop_affine" in meta
        assert meta["resampled_affine"].shape == (4, 4)


class TestComputeOverlapCenter:
    def test_identical_volumes(self):
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
        shape = (128, 128, 128)
        center = compute_overlap_center(affine, shape, affine, shape)
        assert center == (64, 64, 64)

    def test_shifted_volumes(self):
        affine_a = np.diag([1.0, 1.0, 1.0, 1.0])
        affine_b = np.diag([1.0, 1.0, 1.0, 1.0])
        affine_b[0, 3] = 20.0
        shape = (128, 128, 128)
        center = compute_overlap_center(affine_a, shape, affine_b, shape)
        assert center[0] == 74
        assert center[1] == 64
        assert center[2] == 64

    def test_no_overlap_fallback(self):
        affine_a = np.diag([1.0, 1.0, 1.0, 1.0])
        affine_b = np.diag([1.0, 1.0, 1.0, 1.0])
        affine_b[0, 3] = 500.0
        shape = (128, 128, 128)
        center = compute_overlap_center(affine_a, shape, affine_b, shape)
        assert center == (64, 64, 64)

    def test_different_shapes(self):
        affine_a = np.diag([1.0, 1.0, 1.0, 1.0])
        shape_a = (200, 200, 200)
        affine_b = np.diag([1.0, 1.0, 1.0, 1.0])
        affine_b[:3, 3] = [50, 50, 50]
        shape_b = (80, 80, 80)
        center = compute_overlap_center(affine_a, shape_a, affine_b, shape_b)
        assert center == (90, 90, 90)


class TestLongitudinalPairDatasetV2:
    def test_output_shape_128(self, tmp_path):
        # Create two NIfTIs
        for name in ["baseline", "followup"]:
            data = np.random.rand(64, 64, 64).astype(np.float32) * 100
            nib.save(nib.Nifti1Image(data, np.diag([1.0, 1.0, 1.0, 1.0])), str(tmp_path / f"{name}.nii.gz"))

        baseline = ScanRecord(
            modality="t1", patient_timepoint="P1_tp1", patient_id="P1",
            timepoint="tp1", volume_path=tmp_path / "baseline.nii.gz",
            seg_path=None, is_baseline=True,
        )
        followup = ScanRecord(
            modality="t1", patient_timepoint="P1_tp2", patient_id="P1",
            timepoint="tp2", volume_path=tmp_path / "followup.nii.gz",
            seg_path=None,
        )
        groups = {"t1/P1": [baseline, followup]}
        ds = LongitudinalPairDataset(groups, target_size=(128, 128, 128))
        sample = ds[0]
        assert sample["moving"].shape == (1, 128, 128, 128)
        assert sample["fixed"].shape == (1, 128, 128, 128)

    def test_mixed_geometry(self, tmp_path):
        # Thick axial: 256x256x30 at 0.5x0.5x5mm
        data_thick = np.random.rand(256, 256, 30).astype(np.float32) * 100
        nib.save(nib.Nifti1Image(data_thick, np.diag([0.5, 0.5, 5.0, 1.0])), str(tmp_path / "thick.nii.gz"))

        # Thin: 256x256x256 at 0.5x0.5x0.5mm
        data_thin = np.random.rand(256, 256, 256).astype(np.float32) * 100
        nib.save(nib.Nifti1Image(data_thin, np.diag([0.5, 0.5, 0.5, 1.0])), str(tmp_path / "thin.nii.gz"))

        baseline = ScanRecord(
            modality="t1", patient_timepoint="P1_tp1", patient_id="P1",
            timepoint="tp1", volume_path=tmp_path / "thin.nii.gz",
            seg_path=None, is_baseline=True,
        )
        followup = ScanRecord(
            modality="t1", patient_timepoint="P1_tp2", patient_id="P1",
            timepoint="tp2", volume_path=tmp_path / "thick.nii.gz",
            seg_path=None,
        )
        groups = {"t1/P1": [baseline, followup]}
        ds = LongitudinalPairDataset(groups, target_size=(128, 128, 128))
        sample = ds[0]
        assert sample["moving"].shape == (1, 128, 128, 128)
        assert sample["fixed"].shape == (1, 128, 128, 128)
        assert sample["moving"].min() >= 0.0
        assert sample["moving"].max() <= 1.0 + 1e-6


class TestSyntheticPairDatasetV2:
    def test_output_shape(self, tmp_path):
        data = np.random.rand(200, 200, 200).astype(np.float32) * 100
        nib.save(nib.Nifti1Image(data, np.diag([0.8, 0.8, 0.8, 1.0])), str(tmp_path / "scan.nii.gz"))
        ds = SyntheticPairDataset(
            scan_paths=[tmp_path / "scan.nii.gz"],
            target_size=(128, 128, 128),
            length=2,
        )
        sample = ds[0]
        assert sample["moving"].shape == (1, 128, 128, 128)
        assert sample["fixed"].shape == (1, 128, 128, 128)
        assert sample["has_gt"].item() is True


class TestTrainingIntegrationV2:
    """Smoke test: full forward pass with v2 preprocessing."""

    def test_training_forward_pass(self, tmp_path):
        """Model forward pass works with v2-preprocessed data from different geometries."""
        from vs_preprocess.training.model import AffineRegNet
        from vs_preprocess.preprocessing import preprocess_for_network_v2

        # Create two fake NIfTIs with different geometry
        for name, shape, spacing in [
            ("fixed", (200, 200, 200), (0.8, 0.8, 0.8)),
            ("moving", (256, 256, 30), (0.5, 0.5, 5.0)),
        ]:
            data = np.random.rand(*shape).astype(np.float32) * 100
            nib.save(nib.Nifti1Image(data, np.diag([*spacing, 1.0])), str(tmp_path / f"{name}.nii.gz"))

        fixed, _ = preprocess_for_network_v2(tmp_path / "fixed.nii.gz")
        moving, _ = preprocess_for_network_v2(tmp_path / "moving.nii.gz")

        x = torch.cat([moving, fixed], dim=1)  # (1, 2, 128, 128, 128)
        model = AffineRegNet(image_size=(128, 128, 128))
        model.eval()
        with torch.no_grad():
            params, affine = model(x)
        assert params.shape == (1, 9)
        assert affine.shape == (1, 3, 4)
