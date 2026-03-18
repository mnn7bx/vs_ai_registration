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
from preprocessing import (
    preprocess_for_network,
    preprocess_for_network_v2,
    load_and_resample_isotropic,
    center_crop_or_pad,
    compute_overlap_center,
)
from dataset import ScanRecord


class LongitudinalPairDataset(Dataset):
    """
    Yields (moving, fixed) pairs from real longitudinal data.
    Uses FOV-aware cropping: both volumes are cropped centered on their
    overlapping physical region so the 128^3 windows contain the same anatomy.
    """

    def __init__(
        self,
        patient_groups: Dict[str, List[ScanRecord]],
        target_size=(128, 128, 128),
        target_spacing=(1.0, 1.0, 1.0),
    ):
        self.pairs: List[tuple] = []
        self.target_size = target_size
        self.target_spacing = target_spacing

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
        try:
            # Step 1: Load and resample both to isotropic (no crop yet)
            mov_data, mov_affine = load_and_resample_isotropic(moving_path, self.target_spacing)
            fix_data, fix_affine = load_and_resample_isotropic(fixed_path, self.target_spacing)

            # Step 2: Compute overlapping region center (in fixed-volume voxel coords)
            overlap_center_fix = compute_overlap_center(
                fix_affine, fix_data.shape, mov_affine, mov_data.shape
            )

            # Convert overlap center to moving-volume voxel coords
            center_world = fix_affine[:3, :3] @ np.array(overlap_center_fix, dtype=np.float64) + fix_affine[:3, 3]
            inv_mov = np.linalg.inv(mov_affine)
            overlap_center_mov = inv_mov[:3, :3] @ center_world + inv_mov[:3, 3]
            overlap_center_mov = tuple(int(round(v)) for v in overlap_center_mov)
            overlap_center_mov = tuple(max(0, min(c, s - 1)) for c, s in zip(overlap_center_mov, mov_data.shape))

            # Step 3: Crop both centered on the overlap
            fix_cropped, _ = center_crop_or_pad(fix_data, self.target_size, center=overlap_center_fix)
            mov_cropped, _ = center_crop_or_pad(mov_data, self.target_size, center=overlap_center_mov)

            # Step 4: Intensity normalize each to [0, 1]
            def _normalize(arr):
                vmin, vmax = arr.min(), arr.max()
                if vmax - vmin > 1e-8:
                    return (arr - vmin) / (vmax - vmin)
                return np.zeros_like(arr)

            mov_cropped = _normalize(mov_cropped)
            fix_cropped = _normalize(fix_cropped)

            moving = torch.from_numpy(mov_cropped).float().unsqueeze(0)  # (1,D,H,W)
            fixed = torch.from_numpy(fix_cropped).float().unsqueeze(0)

        except (ValueError, Exception) as e:
            alt = random.randint(0, max(len(self.pairs) - 1, 0))
            if alt == idx:
                alt = (idx + 1) % len(self.pairs)
            return self.__getitem__(alt)

        return {
            "moving": moving,
            "fixed": fixed,
            "has_gt": torch.tensor(False),
            "gt_params": torch.zeros(9),
            "pair_type": torch.tensor(0),  # default: same-sub (NCC)
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
        target_size=(128, 128, 128),
        target_spacing=(1.0, 1.0, 1.0),
        max_rotation_deg: float = 15.0,
        max_translation: float = 0.1,
        max_scale_offset: float = 0.05,
        length: int = 1000,
        transform_type: str = "affine",
    ):
        self.scan_paths = scan_paths
        self.target_size = target_size
        self.target_spacing = target_spacing
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
        try:
            fixed, _ = preprocess_for_network_v2(path, self.target_size, self.target_spacing)
            fixed = fixed.squeeze(0)  # (1,D,H,W)
        except (ValueError, Exception):
            return self.__getitem__(idx)  # retry with another random scan

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
            "pair_type": torch.tensor(0),  # synthetic treated as same-sub (NCC)
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
