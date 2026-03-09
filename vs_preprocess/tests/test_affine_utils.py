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


class TestNormalizedAffineToWorldV2:
    def test_identity_roundtrip(self):
        from vs_preprocess.affine_utils import normalized_affine_to_world_v2
        identity_3x4 = np.eye(3, 4)
        crop_affine = np.diag([1.0, 1.0, 1.0, 1.0])
        result = normalized_affine_to_world_v2(
            identity_3x4, crop_affine, crop_affine, net_input_shape=(128, 128, 128)
        )
        np.testing.assert_allclose(result, np.eye(4), atol=1e-6)

    def test_shifted_crop(self):
        from vs_preprocess.affine_utils import normalized_affine_to_world_v2
        identity_3x4 = np.eye(3, 4)
        crop_moving = np.diag([1.0, 1.0, 1.0, 1.0])
        crop_moving[:3, 3] = [10.0, 0.0, 0.0]
        crop_fixed = np.diag([1.0, 1.0, 1.0, 1.0])
        result = normalized_affine_to_world_v2(
            identity_3x4, crop_moving, crop_fixed, net_input_shape=(128, 128, 128)
        )
        assert abs(result[0, 3] - 10.0) < 1e-4
