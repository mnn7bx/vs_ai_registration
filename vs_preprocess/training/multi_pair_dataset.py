"""
Multi-pair dataset for cross-modality and longitudinal registration.

Enumerates three pair types from a patient-timepoint index:
  SAME_SUB:    same sub-modality, different timepoints (NCC)
  CROSS_SUB:   different sub-modality same super-modality, different timepoints (MI)
  CROSS_SUPER: different super-modality, same timepoint (MI)
"""

from __future__ import annotations

import enum
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import ScanRecord, get_super_modality
from preprocessing import load_and_resample_isotropic, center_crop_or_pad, compute_overlap_center


class PairType(enum.IntEnum):
    SAME_SUB = 0     # NCC
    CROSS_SUB = 1    # MI
    CROSS_SUPER = 2  # MI


# Type alias for the patient-timepoint index
PatientIndex = Dict[str, Dict[str, Dict[str, ScanRecord]]]


def enumerate_pairs(
    index: PatientIndex,
) -> List[Tuple[Path, Path, PairType]]:
    """
    Enumerate all training pairs from the patient-timepoint index.

    Returns list of (moving_path, fixed_path, pair_type).

    Longitudinal pairs: non-baseline -> baseline (earliest timepoint).
    Cross-super pairs: both directions (A->B and B->A).
    """
    pairs: List[Tuple[Path, Path, PairType]] = []

    for pid, timepoints in index.items():
        sorted_tps = sorted(timepoints.keys())
        if len(sorted_tps) < 1:
            continue

        baseline_tp = sorted_tps[0]

        # --- Longitudinal pairs (across timepoints) ---
        for tp in sorted_tps[1:]:
            tp_mods = timepoints[tp]
            bl_mods = timepoints[baseline_tp]

            for sub_mod, record in tp_mods.items():
                super_mod = get_super_modality(sub_mod)

                # Check baseline for same sub-modality
                if sub_mod in bl_mods:
                    # Same sub-modality longitudinal
                    pairs.append((
                        record.volume_path,
                        bl_mods[sub_mod].volume_path,
                        PairType.SAME_SUB,
                    ))
                else:
                    # Look for any sub-modality in baseline with same super-modality
                    for bl_sub, bl_rec in bl_mods.items():
                        if get_super_modality(bl_sub) == super_mod:
                            pairs.append((
                                record.volume_path,
                                bl_rec.volume_path,
                                PairType.CROSS_SUB,
                            ))

        # --- Cross-super-modality pairs (intra-timepoint) ---
        for tp, mods in timepoints.items():
            sub_list = list(mods.items())
            for i, (sub_a, rec_a) in enumerate(sub_list):
                for sub_b, rec_b in sub_list[i + 1:]:
                    if get_super_modality(sub_a) != get_super_modality(sub_b):
                        # Both directions
                        pairs.append((rec_a.volume_path, rec_b.volume_path, PairType.CROSS_SUPER))
                        pairs.append((rec_b.volume_path, rec_a.volume_path, PairType.CROSS_SUPER))

    return pairs


class MultiPairDataset(Dataset):
    """
    Dataset that loads pairs tagged with PairType for three-tier loss.

    Each __getitem__ returns:
        moving: (1, D, H, W) float32
        fixed:  (1, D, H, W) float32
        pair_type: int (PairType enum value)
        has_gt: False (real pairs have no ground truth)
        gt_params: zeros(9)
    """

    def __init__(
        self,
        patient_index: PatientIndex,
        target_size: Tuple[int, ...] = (128, 128, 128),
        target_spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    ):
        self.target_size = target_size
        self.target_spacing = target_spacing
        self.pairs = enumerate_pairs(patient_index)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        moving_path, fixed_path, pair_type = self.pairs[idx]
        try:
            mov_data, mov_affine = load_and_resample_isotropic(moving_path, self.target_spacing)
            fix_data, fix_affine = load_and_resample_isotropic(fixed_path, self.target_spacing)

            overlap_center_fix = compute_overlap_center(
                fix_affine, fix_data.shape, mov_affine, mov_data.shape
            )
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

            fix_cropped, _ = center_crop_or_pad(fix_data, self.target_size, center=overlap_center_fix)
            mov_cropped, _ = center_crop_or_pad(mov_data, self.target_size, center=overlap_center_mov)

            def _normalize(arr):
                vmin, vmax = arr.min(), arr.max()
                if vmax - vmin > 1e-8:
                    return (arr - vmin) / (vmax - vmin)
                return np.zeros_like(arr)

            mov_cropped = _normalize(mov_cropped)
            fix_cropped = _normalize(fix_cropped)

            moving = torch.from_numpy(mov_cropped).float().unsqueeze(0)
            fixed = torch.from_numpy(fix_cropped).float().unsqueeze(0)

        except Exception:
            alt = random.randint(0, max(len(self.pairs) - 1, 0))
            if alt == idx:
                alt = (idx + 1) % len(self.pairs)
            return self.__getitem__(alt)

        return {
            "moving": moving,
            "fixed": fixed,
            "pair_type": torch.tensor(int(pair_type)),
            "has_gt": torch.tensor(False),
            "gt_params": torch.zeros(9),
        }


class CombinedMultiPairDataset(Dataset):
    """
    Interleaves real multi-pair dataset with synthetic pairs.

    Balances pair types according to configurable ratios:
        ratios[0]: same-sub longitudinal
        ratios[1]: cross-sub longitudinal
        ratios[2]: cross-super intra-timepoint
        ratios[3]: synthetic
    """

    def __init__(
        self,
        real_dataset: MultiPairDataset,
        synthetic_dataset,
        ratios: List[float] = None,
    ):
        if ratios is None:
            ratios = [0.25, 0.25, 0.25, 0.25]
        assert len(ratios) == 4
        total_ratio = sum(ratios)
        ratios = [r / total_ratio for r in ratios]

        # Separate real pairs by type
        self.same_sub_indices = []
        self.cross_sub_indices = []
        self.cross_super_indices = []
        for i, (_, _, pt) in enumerate(real_dataset.pairs):
            if pt == PairType.SAME_SUB:
                self.same_sub_indices.append(i)
            elif pt == PairType.CROSS_SUB:
                self.cross_sub_indices.append(i)
            elif pt == PairType.CROSS_SUPER:
                self.cross_super_indices.append(i)

        self.real_dataset = real_dataset
        self.synthetic_dataset = synthetic_dataset

        # Compute counts: use the largest real bucket to set epoch size
        real_counts = [
            len(self.same_sub_indices),
            len(self.cross_sub_indices),
            len(self.cross_super_indices),
        ]
        max_bucket = 0
        for i, count in enumerate(real_counts):
            if ratios[i] > 0 and count > 0:
                epoch_from_bucket = int(count / ratios[i])
                max_bucket = max(max_bucket, epoch_from_bucket)

        if max_bucket == 0:
            max_bucket = len(synthetic_dataset)

        self.n_same_sub = int(max_bucket * ratios[0])
        self.n_cross_sub = int(max_bucket * ratios[1])
        self.n_cross_super = int(max_bucket * ratios[2])
        self.n_synthetic = int(max_bucket * ratios[3])

        self.synthetic_dataset.length = max(self.n_synthetic, 1)

    def __len__(self):
        return self.n_same_sub + self.n_cross_sub + self.n_cross_super + self.n_synthetic

    def __getitem__(self, idx):
        if idx < self.n_same_sub:
            real_idx = self.same_sub_indices[idx % len(self.same_sub_indices)] if self.same_sub_indices else 0
            return self.real_dataset[real_idx]
        idx -= self.n_same_sub

        if idx < self.n_cross_sub:
            real_idx = self.cross_sub_indices[idx % len(self.cross_sub_indices)] if self.cross_sub_indices else 0
            return self.real_dataset[real_idx]
        idx -= self.n_cross_sub

        if idx < self.n_cross_super:
            real_idx = self.cross_super_indices[idx % len(self.cross_super_indices)] if self.cross_super_indices else 0
            return self.real_dataset[real_idx]
        idx -= self.n_cross_super

        # Synthetic
        sample = self.synthetic_dataset[idx % len(self.synthetic_dataset)]
        sample["pair_type"] = torch.tensor(int(PairType.SAME_SUB))  # synthetic = NCC
        return sample
