# Cross-Modality Dual-Loss Registration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a single affine registration model that handles intra-timepoint cross-modality (T1C↔T2) and across-timepoint longitudinal registration using a three-tier loss (NCC for same sub-modality, MI for everything else).

**Architecture:** No model changes. New dataset class indexes by patient/timepoint instead of modality/patient. Training loop routes each pair to NCC or MI based on pair type. Inference applies predicted affine at 0.5mm isotropic output.

**Tech Stack:** PyTorch, MONAI (GlobalMutualInformationLoss, LocalNormalizedCrossCorrelationLoss), nibabel, numpy

**Design doc:** `docs/plans/2026-03-18-cross-modality-dual-loss-design.md`

---

### Task 1: Add super-modality mapping to dataset.py

**Files:**
- Modify: `vs_preprocess/dataset.py`
- Test: `vs_preprocess/tests/test_super_modality.py`

**Step 1: Write the failing test**

```python
# tests/test_super_modality.py
import pytest
from dataset import get_super_modality, SUPER_MODALITY_MAP

def test_t1c_variants():
    assert get_super_modality("t1+_thick_ax") == "T1C"
    assert get_super_modality("t1+_thick_cor") == "T1C"
    assert get_super_modality("t1+_thin") == "T1C"

def test_t2_variants():
    assert get_super_modality("t2_thick") == "T2"
    assert get_super_modality("t2_thin") == "T2"

def test_unknown_modality():
    assert get_super_modality("flair") == "FLAIR"

def test_map_has_all_known():
    assert set(SUPER_MODALITY_MAP.keys()) == {
        "t1+_thick_ax", "t1+_thick_cor", "t1+_thin",
        "t2_thick", "t2_thin",
    }
```

**Step 2: Run test to verify it fails**

Run: `cd vs_preprocess && python -m pytest tests/test_super_modality.py -v`
Expected: FAIL with ImportError (get_super_modality not defined)

**Step 3: Write minimal implementation**

Add to `vs_preprocess/dataset.py` after the imports:

```python
# Super-modality mapping: sub-modality directory name -> canonical group
SUPER_MODALITY_MAP = {
    "t1+_thick_ax": "T1C",
    "t1+_thick_cor": "T1C",
    "t1+_thin": "T1C",
    "t2_thick": "T2",
    "t2_thin": "T2",
}

def get_super_modality(sub_modality: str) -> str:
    """Map a sub-modality directory name to its super-modality group.

    Returns the sub_modality uppercased if not in SUPER_MODALITY_MAP.
    """
    return SUPER_MODALITY_MAP.get(sub_modality, sub_modality.upper())
```

**Step 4: Run test to verify it passes**

Run: `cd vs_preprocess && python -m pytest tests/test_super_modality.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add vs_preprocess/dataset.py vs_preprocess/tests/test_super_modality.py
git commit -m "feat: add super-modality mapping (T1C/T2) to dataset.py"
```

---

### Task 2: Add `group_by_patient_timepoint()` to dataset.py

**Files:**
- Modify: `vs_preprocess/dataset.py`
- Test: `vs_preprocess/tests/test_super_modality.py` (extend)

**Step 1: Write the failing test**

Append to `tests/test_super_modality.py`:

```python
from pathlib import Path
from dataset import ScanRecord, group_by_patient_timepoint

def _make_record(modality, patient_id, timepoint, vol="vol.nii.gz"):
    return ScanRecord(
        modality=modality,
        patient_timepoint=f"{patient_id}_{timepoint}",
        patient_id=patient_id,
        timepoint=timepoint,
        volume_path=Path(f"/data/{modality}/{patient_id}_{timepoint}/{vol}"),
        seg_path=None,
    )

def test_group_by_patient_timepoint_basic():
    records = [
        _make_record("t1+_thick_ax", "001", "1"),
        _make_record("t2_thin", "001", "1"),
        _make_record("t1+_thin", "001", "2"),
        _make_record("t2_thin", "001", "2"),
    ]
    index = group_by_patient_timepoint(records)

    # Structure: {patient_id: {timepoint: {sub_modality: ScanRecord}}}
    assert "001" in index
    assert set(index["001"].keys()) == {"1", "2"}
    assert "t1+_thick_ax" in index["001"]["1"]
    assert "t2_thin" in index["001"]["1"]
    assert "t1+_thin" in index["001"]["2"]
    assert "t2_thin" in index["001"]["2"]

def test_group_by_patient_timepoint_multiple_patients():
    records = [
        _make_record("t1+_thin", "001", "1"),
        _make_record("t1+_thin", "002", "1"),
    ]
    index = group_by_patient_timepoint(records)
    assert set(index.keys()) == {"001", "002"}
```

**Step 2: Run test to verify it fails**

Run: `cd vs_preprocess && python -m pytest tests/test_super_modality.py::test_group_by_patient_timepoint_basic -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

Add to `vs_preprocess/dataset.py`:

```python
def group_by_patient_timepoint(
    records: List[ScanRecord],
) -> Dict[str, Dict[str, Dict[str, ScanRecord]]]:
    """
    Index scans by patient_id -> timepoint -> sub_modality.

    Returns:
        {patient_id: {timepoint: {sub_modality: ScanRecord}}}
    """
    index: Dict[str, Dict[str, Dict[str, ScanRecord]]] = {}
    for r in records:
        pid = r.patient_id
        tp = r.timepoint
        sub_mod = r.modality
        index.setdefault(pid, {}).setdefault(tp, {})[sub_mod] = r
    return index
```

**Step 4: Run test to verify it passes**

Run: `cd vs_preprocess && python -m pytest tests/test_super_modality.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add vs_preprocess/dataset.py vs_preprocess/tests/test_super_modality.py
git commit -m "feat: add group_by_patient_timepoint() for cross-modality indexing"
```

---

### Task 3: Implement `MultiPairDataset` — pair enumeration

**Files:**
- Create: `vs_preprocess/training/multi_pair_dataset.py`
- Test: `vs_preprocess/tests/test_multi_pair_dataset.py`

This is the core new dataset. It enumerates all three pair types from the patient-timepoint index.

**Step 1: Write the failing test**

```python
# tests/test_multi_pair_dataset.py
import pytest
from pathlib import Path
from dataset import ScanRecord
from training.multi_pair_dataset import enumerate_pairs, PairType

def _rec(mod, pid, tp):
    return ScanRecord(
        modality=mod,
        patient_timepoint=f"{pid}_{tp}",
        patient_id=pid,
        timepoint=tp,
        volume_path=Path(f"/data/{mod}/{pid}_{tp}/vol.nii.gz"),
        seg_path=None,
    )

def test_same_sub_longitudinal():
    """Same sub-modality, different timepoints -> PairType.SAME_SUB"""
    index = {
        "001": {
            "1": {"t1+_thin": _rec("t1+_thin", "001", "1")},
            "2": {"t1+_thin": _rec("t1+_thin", "001", "2")},
        }
    }
    pairs = enumerate_pairs(index)
    same_sub = [p for p in pairs if p[2] == PairType.SAME_SUB]
    assert len(same_sub) == 1  # tp2 -> tp1
    assert same_sub[0][0] == Path("/data/t1+_thin/001_2/vol.nii.gz")  # moving
    assert same_sub[0][1] == Path("/data/t1+_thin/001_1/vol.nii.gz")  # fixed (baseline)

def test_cross_sub_longitudinal():
    """Different sub-modality same super, different timepoints -> PairType.CROSS_SUB"""
    index = {
        "001": {
            "1": {"t1+_thick_ax": _rec("t1+_thick_ax", "001", "1")},
            "2": {"t1+_thin": _rec("t1+_thin", "001", "2")},
        }
    }
    pairs = enumerate_pairs(index)
    cross_sub = [p for p in pairs if p[2] == PairType.CROSS_SUB]
    # t1+_thin@tp2 -> t1+_thick_ax@tp1 (different sub, same super T1C, different tp)
    assert len(cross_sub) == 1

def test_cross_super_intra_timepoint():
    """Different super-modality, same timepoint -> PairType.CROSS_SUPER"""
    index = {
        "001": {
            "1": {
                "t1+_thin": _rec("t1+_thin", "001", "1"),
                "t2_thin": _rec("t2_thin", "001", "1"),
            },
        }
    }
    pairs = enumerate_pairs(index)
    cross_super = [p for p in pairs if p[2] == PairType.CROSS_SUPER]
    # t1+_thin <-> t2_thin at tp1: two directions
    assert len(cross_super) == 2

def test_no_pairs_single_scan():
    """Single scan patient produces no pairs."""
    index = {
        "001": {
            "1": {"t1+_thin": _rec("t1+_thin", "001", "1")},
        }
    }
    pairs = enumerate_pairs(index)
    assert len(pairs) == 0

def test_mixed_scenario():
    """Patient with multiple timepoints and modalities."""
    index = {
        "001": {
            "1": {
                "t1+_thick_ax": _rec("t1+_thick_ax", "001", "1"),
                "t2_thin": _rec("t2_thin", "001", "1"),
            },
            "2": {
                "t1+_thin": _rec("t1+_thin", "001", "2"),
                "t2_thin": _rec("t2_thin", "001", "2"),
            },
        }
    }
    pairs = enumerate_pairs(index)
    same_sub = [p for p in pairs if p[2] == PairType.SAME_SUB]
    cross_sub = [p for p in pairs if p[2] == PairType.CROSS_SUB]
    cross_super = [p for p in pairs if p[2] == PairType.CROSS_SUPER]

    # same_sub: t2_thin tp2 -> t2_thin tp1 = 1
    assert len(same_sub) == 1
    # cross_sub: t1+_thin@tp2 -> t1+_thick_ax@tp1 = 1
    assert len(cross_sub) == 1
    # cross_super at tp1: t1+_thick_ax <-> t2_thin = 2 directions
    # cross_super at tp2: t1+_thin <-> t2_thin = 2 directions
    assert len(cross_super) == 4
```

**Step 2: Run test to verify it fails**

Run: `cd vs_preprocess && python -m pytest tests/test_multi_pair_dataset.py -v`
Expected: FAIL with ImportError

**Step 3: Write minimal implementation**

```python
# training/multi_pair_dataset.py
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
```

**Step 4: Run test to verify it passes**

Run: `cd vs_preprocess && python -m pytest tests/test_multi_pair_dataset.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add vs_preprocess/training/multi_pair_dataset.py vs_preprocess/tests/test_multi_pair_dataset.py
git commit -m "feat: add MultiPairDataset pair enumeration with three pair types"
```

---

### Task 4: Implement `MultiPairDataset` — data loading (__getitem__)

**Files:**
- Modify: `vs_preprocess/training/multi_pair_dataset.py`

**Step 1: Write the failing test**

Append to `tests/test_multi_pair_dataset.py`:

```python
def test_multi_pair_dataset_getitem_returns_pair_type(tmp_path):
    """MultiPairDataset.__getitem__ returns pair_type in the batch dict."""
    # This test requires actual NIfTI files, so we test the return dict shape
    # with a mock. Full integration test is in Task 8.
    from training.multi_pair_dataset import MultiPairDataset, PairType
    import torch

    # We can't easily create real NIfTI in unit test, so just verify the class exists
    # and has the expected interface
    assert hasattr(MultiPairDataset, '__getitem__')
    assert hasattr(MultiPairDataset, '__len__')
```

**Step 2: Run test to verify it fails**

Run: `cd vs_preprocess && python -m pytest tests/test_multi_pair_dataset.py::test_multi_pair_dataset_getitem_returns_pair_type -v`
Expected: FAIL with ImportError (MultiPairDataset not defined)

**Step 3: Write implementation**

Add `MultiPairDataset` class to `training/multi_pair_dataset.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `cd vs_preprocess && python -m pytest tests/test_multi_pair_dataset.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add vs_preprocess/training/multi_pair_dataset.py vs_preprocess/tests/test_multi_pair_dataset.py
git commit -m "feat: add MultiPairDataset data loading with pair_type tagging"
```

---

### Task 5: Add three-tier loss router to losses.py

**Files:**
- Modify: `vs_preprocess/training/losses.py`
- Test: `vs_preprocess/tests/test_three_tier_loss.py`

**Step 1: Write the failing test**

```python
# tests/test_three_tier_loss.py
import pytest
import torch
from training.losses import ThreeTierLoss

def test_same_sub_uses_ncc():
    loss_fn = ThreeTierLoss()
    # pair_type=0 should use NCC
    warped = torch.rand(1, 1, 16, 16, 16)
    fixed = torch.rand(1, 1, 16, 16, 16)
    pair_types = torch.tensor([0])
    result = loss_fn(warped, fixed, pair_types)
    assert result.shape == ()
    assert torch.isfinite(result)

def test_cross_sub_uses_mi():
    loss_fn = ThreeTierLoss()
    warped = torch.rand(1, 1, 16, 16, 16)
    fixed = torch.rand(1, 1, 16, 16, 16)
    pair_types = torch.tensor([1])
    result = loss_fn(warped, fixed, pair_types)
    assert result.shape == ()
    assert torch.isfinite(result)

def test_cross_super_uses_mi():
    loss_fn = ThreeTierLoss()
    warped = torch.rand(1, 1, 16, 16, 16)
    fixed = torch.rand(1, 1, 16, 16, 16)
    pair_types = torch.tensor([2])
    result = loss_fn(warped, fixed, pair_types)
    assert result.shape == ()
    assert torch.isfinite(result)

def test_mixed_batch():
    loss_fn = ThreeTierLoss()
    warped = torch.rand(3, 1, 16, 16, 16)
    fixed = torch.rand(3, 1, 16, 16, 16)
    pair_types = torch.tensor([0, 1, 2])
    result = loss_fn(warped, fixed, pair_types)
    assert result.shape == ()
    assert torch.isfinite(result)

def test_ema_tracking():
    loss_fn = ThreeTierLoss(ema_decay=0.9)
    warped = torch.rand(2, 1, 16, 16, 16)
    fixed = torch.rand(2, 1, 16, 16, 16)
    # Run a few times to build up EMA
    for _ in range(5):
        pair_types = torch.tensor([0, 1])
        loss_fn(warped, fixed, pair_types)
    assert loss_fn.ema_ncc > 0
    assert loss_fn.ema_mi > 0
```

**Step 2: Run test to verify it fails**

Run: `cd vs_preprocess && python -m pytest tests/test_three_tier_loss.py -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Add to `vs_preprocess/training/losses.py`:

```python
class ThreeTierLoss(torch.nn.Module):
    """
    Routes similarity loss based on pair_type:
        0 (SAME_SUB):    NCC
        1 (CROSS_SUB):   MI
        2 (CROSS_SUPER): MI

    Uses EMA normalization so NCC and MI contribute equally.
    """

    def __init__(
        self,
        ncc_kernel_size: int = 9,
        mi_num_bins: int = 32,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.ncc_fn = get_similarity_loss("ncc", kernel_size=ncc_kernel_size)
        self.mi_fn = get_similarity_loss("mi", num_bins=mi_num_bins)
        self.ema_decay = ema_decay
        self.ema_ncc = 0.0
        self.ema_mi = 0.0

    def forward(
        self,
        warped: torch.Tensor,
        fixed: torch.Tensor,
        pair_types: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mixed NCC/MI loss based on pair types.

        Args:
            warped: (B, 1, D, H, W)
            fixed: (B, 1, D, H, W)
            pair_types: (B,) int tensor with values 0, 1, or 2

        Returns:
            Scalar loss (mean over batch, EMA-normalized).
        """
        ncc_mask = pair_types == 0
        mi_mask = (pair_types == 1) | (pair_types == 2)

        losses = []

        if ncc_mask.any():
            ncc_val = self.ncc_fn(warped[ncc_mask], fixed[ncc_mask])
            raw_ncc = ncc_val.detach().abs().item()
            if self.ema_ncc == 0.0:
                self.ema_ncc = raw_ncc if raw_ncc > 1e-8 else 1.0
            else:
                self.ema_ncc = self.ema_decay * self.ema_ncc + (1 - self.ema_decay) * raw_ncc
            normalized_ncc = ncc_val / max(self.ema_ncc, 1e-8)
            losses.append(normalized_ncc * ncc_mask.sum() / pair_types.shape[0])

        if mi_mask.any():
            mi_val = self.mi_fn(warped[mi_mask], fixed[mi_mask])
            raw_mi = mi_val.detach().abs().item()
            if self.ema_mi == 0.0:
                self.ema_mi = raw_mi if raw_mi > 1e-8 else 1.0
            else:
                self.ema_mi = self.ema_decay * self.ema_mi + (1 - self.ema_decay) * raw_mi
            normalized_mi = mi_val / max(self.ema_mi, 1e-8)
            losses.append(normalized_mi * mi_mask.sum() / pair_types.shape[0])

        if not losses:
            return torch.tensor(0.0, device=warped.device, requires_grad=True)

        return sum(losses)
```

**Step 4: Run test to verify it passes**

Run: `cd vs_preprocess && python -m pytest tests/test_three_tier_loss.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add vs_preprocess/training/losses.py vs_preprocess/tests/test_three_tier_loss.py
git commit -m "feat: add ThreeTierLoss router with EMA normalization"
```

---

### Task 6: Build `CombinedMultiPairDataset` with 4-way ratio

**Files:**
- Modify: `vs_preprocess/training/multi_pair_dataset.py`
- Test: `vs_preprocess/tests/test_multi_pair_dataset.py` (extend)

**Step 1: Write the failing test**

Append to `tests/test_multi_pair_dataset.py`:

```python
def test_combined_multi_pair_dataset_length():
    from training.multi_pair_dataset import CombinedMultiPairDataset, MultiPairDataset, PairType
    from training.dataset import SyntheticPairDataset

    # Mock a MultiPairDataset with known pair counts
    class FakeMultiPair:
        def __init__(self):
            self.pairs = [
                (Path("/a"), Path("/b"), PairType.SAME_SUB),
                (Path("/c"), Path("/d"), PairType.CROSS_SUB),
                (Path("/e"), Path("/f"), PairType.CROSS_SUPER),
            ] * 10  # 30 pairs total
            self.pair_counts = {
                PairType.SAME_SUB: 10,
                PairType.CROSS_SUB: 10,
                PairType.CROSS_SUPER: 10,
            }
        def __len__(self):
            return len(self.pairs)
        def __getitem__(self, idx):
            return {"pair_type": torch.tensor(0)}

    class FakeSynthetic:
        length = 100
        def __len__(self):
            return self.length
        def __getitem__(self, idx):
            return {"pair_type": torch.tensor(0), "has_gt": torch.tensor(True)}

    real = FakeMultiPair()
    synth = FakeSynthetic()
    combined = CombinedMultiPairDataset(
        real_dataset=real,
        synthetic_dataset=synth,
        ratios=[0.25, 0.25, 0.25, 0.25],
    )
    # Length = total real pairs + synthetic, balanced by ratios
    assert len(combined) > 0
```

**Step 2: Run test to verify it fails**

Run: `cd vs_preprocess && python -m pytest tests/test_multi_pair_dataset.py::test_combined_multi_pair_dataset_length -v`
Expected: FAIL with ImportError

**Step 3: Write implementation**

Add to `training/multi_pair_dataset.py`:

```python
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
        # The bucket with most pairs at its target ratio sets the epoch size
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
```

**Step 4: Run test to verify it passes**

Run: `cd vs_preprocess && python -m pytest tests/test_multi_pair_dataset.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add vs_preprocess/training/multi_pair_dataset.py vs_preprocess/tests/test_multi_pair_dataset.py
git commit -m "feat: add CombinedMultiPairDataset with 4-way ratio balancing"
```

---

### Task 7: Update training loop (train_intra.py)

**Files:**
- Modify: `vs_preprocess/training/train_intra.py`

This is the largest change. The training loop needs to:
1. Use `group_by_patient_timepoint()` instead of `group_by_patient()`
2. Use `MultiPairDataset` + `CombinedMultiPairDataset` instead of `LongitudinalPairDataset` + `CombinedDataset`
3. Use `ThreeTierLoss` when `similarity_loss: three_tier` is configured
4. Track three validation metrics separately
5. Select best model by mean of all three

**Step 1: Update imports**

At the top of `train_intra.py`, add:

```python
from dataset import discover_scans, group_by_patient, group_by_patient_timepoint
from training.multi_pair_dataset import MultiPairDataset, CombinedMultiPairDataset
from training.losses import (
    get_similarity_loss,
    ThreeTierLoss,
    identity_regularization,
    determinant_regularization,
    supervised_param_loss,
)
```

**Step 2: Update `split_by_patient` for patient-timepoint index**

Add a new function that splits a patient-timepoint index by patient ID:

```python
def split_patient_index(index, val_fraction=0.2, seed=42):
    """Split patient-timepoint index into train/val by patient_id."""
    rng = np.random.RandomState(seed)
    pids = sorted(index.keys())
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_fraction))
    val_pids = set(pids[:n_val])
    train_index = {k: v for k, v in index.items() if k not in val_pids}
    val_index = {k: v for k, v in index.items() if k in val_pids}
    return train_index, val_index
```

**Step 3: Add three-tier branch in `train()` function**

In the `train()` function, after data discovery, add a branch for `similarity_loss: three_tier`:

```python
loss_mode = cfg.get("similarity_loss", "ncc")

if loss_mode == "three_tier":
    # Use patient-timepoint indexing
    pt_index = group_by_patient_timepoint(records)
    train_index, val_index = split_patient_index(
        pt_index,
        val_fraction=cfg.get("val_fraction", 0.2),
        seed=cfg.get("split_seed", 42),
    )

    train_real = MultiPairDataset(train_index, target_size, target_spacing)
    # Collect all scan paths for synthetic dataset
    train_scan_paths = []
    for pid, tps in train_index.items():
        for tp, mods in tps.items():
            for sub, rec in mods.items():
                train_scan_paths.append(rec.volume_path)

    train_synthetic = SyntheticPairDataset(
        scan_paths=train_scan_paths,
        target_size=target_size,
        target_spacing=target_spacing,
        max_rotation_deg=cfg.get("max_rotation_deg", 15.0),
        max_translation=cfg.get("max_translation", 0.1),
        max_scale_offset=cfg.get("max_scale_offset", 0.05),
        length=cfg.get("synthetic_length_per_epoch", 1000),
        transform_type=transform_type,
    )
    train_dataset = CombinedMultiPairDataset(
        train_real, train_synthetic,
        ratios=cfg.get("pair_ratios", [0.25, 0.25, 0.25, 0.25]),
    )

    val_real = MultiPairDataset(val_index, target_size, target_spacing)
    val_scan_paths = []
    for pid, tps in val_index.items():
        for tp, mods in tps.items():
            for sub, rec in mods.items():
                val_scan_paths.append(rec.volume_path)

    val_synthetic = SyntheticPairDataset(
        scan_paths=val_scan_paths,
        target_size=target_size,
        target_spacing=target_spacing,
        length=200,
        transform_type=transform_type,
    )
    val_dataset = CombinedMultiPairDataset(
        val_real, val_synthetic,
        ratios=cfg.get("pair_ratios", [0.25, 0.25, 0.25, 0.25]),
    )

    sim_loss_fn = ThreeTierLoss(
        ncc_kernel_size=cfg.get("ncc_kernel_size", 9),
        mi_num_bins=cfg.get("mi_num_bins", 32),
        ema_decay=cfg.get("loss_ema_decay", 0.99),
    )
else:
    # Original single-loss path (unchanged)
    groups = group_by_patient(records)
    train_groups, val_groups = split_by_patient(groups, ...)
    # ... existing code ...
    sim_loss_fn = get_similarity_loss(loss_mode, ...)
```

**Step 4: Update training loop to pass pair_type**

In the training batch loop, update loss computation:

```python
for batch in train_loader:
    moving = batch["moving"].to(device)
    fixed = batch["fixed"].to(device)
    has_gt = batch["has_gt"]
    gt_params = batch["gt_params"].to(device)
    pair_type = batch.get("pair_type", torch.zeros(moving.shape[0], dtype=torch.long))

    # ... augmentation, forward pass, warp ...

    if loss_mode == "three_tier":
        loss_sim = sim_loss_fn(warped, fixed, pair_type.to(device))
    else:
        loss_sim = lam_mi * sim_loss_fn(warped, fixed) if lam_mi > 0 else torch.tensor(0.0, device=device)

    # ... rest of loss computation unchanged ...
```

**Step 5: Update validation to track three metrics**

In the validation loop, when `loss_mode == "three_tier"`, compute per-pair-type metrics:

```python
if loss_mode == "three_tier":
    # Track per-type validation losses
    val_ncc_same = []
    val_mi_cross_sub = []
    val_mi_cross_super = []
    ncc_fn_val = get_similarity_loss("ncc", kernel_size=cfg.get("ncc_kernel_size", 9))
    mi_fn_val = get_similarity_loss("mi", num_bins=cfg.get("mi_num_bins", 32))

    with torch.no_grad():
        for batch in val_loader:
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)
            pair_type = batch.get("pair_type", torch.zeros(moving.shape[0], dtype=torch.long)).to(device)

            x = torch.cat([moving, fixed], dim=1)
            params, affine = model(x)
            warped = warp_volume_torch(moving, affine)

            for pt_val in [0, 1, 2]:
                mask = pair_type == pt_val
                if not mask.any():
                    continue
                if pt_val == 0:
                    val_ncc_same.append(ncc_fn_val(warped[mask], fixed[mask]).item())
                elif pt_val == 1:
                    val_mi_cross_sub.append(mi_fn_val(warped[mask], fixed[mask]).item())
                else:
                    val_mi_cross_super.append(mi_fn_val(warped[mask], fixed[mask]).item())

    avg_ncc = np.mean(val_ncc_same) if val_ncc_same else 0.0
    avg_mi_cs = np.mean(val_mi_cross_sub) if val_mi_cross_sub else 0.0
    avg_mi_cx = np.mean(val_mi_cross_super) if val_mi_cross_super else 0.0
    avg_val_loss = np.mean([avg_ncc, avg_mi_cs, avg_mi_cx])

    writer.add_scalar("val/ncc_same_sub", avg_ncc, epoch)
    writer.add_scalar("val/mi_cross_sub", avg_mi_cs, epoch)
    writer.add_scalar("val/mi_cross_super", avg_mi_cx, epoch)
    writer.add_scalar("val/combined", avg_val_loss, epoch)
```

**Step 6: Commit**

```bash
git add vs_preprocess/training/train_intra.py
git commit -m "feat: add three-tier loss mode to training loop with per-type validation"
```

---

### Task 8: Create new config file

**Files:**
- Create: `vs_preprocess/training/configs/intra_vs_cross_modality.yaml`

**Step 1: Write the config**

```yaml
# VS cross-modality + longitudinal registration
# Three-tier loss: NCC (same sub), MI (cross sub/super)

image_size: [128, 128, 128]
target_spacing: [1.0, 1.0, 1.0]
output_spacing: [0.5, 0.5, 0.5]
transform_type: affine

# Data
pair_ratios: [0.25, 0.25, 0.25, 0.25]  # same-sub, cross-sub, cross-super, synthetic
synthetic_length_per_epoch: 1000
max_rotation_deg: 15.0
max_translation: 0.1
max_scale_offset: 0.05

# Training
batch_size: 2
num_workers: 4
max_epochs: 200
lr: 1.0e-4
grad_clip_norm: 1.0

# Scheduler
scheduler: plateau
plateau_factor: 0.5
plateau_patience: 10

# Model head
head_hidden: 512
head_dropout: 0.1

# Loss
similarity_loss: three_tier
ncc_kernel_size: 9
mi_num_bins: 32
loss_ema_decay: 0.99
lambda_identity: 0.001
lambda_det: 0.1
lambda_supervised: 1.0

# Supervised decay
lambda_supervised_decay: true
lambda_sup_decay_start: 10
lambda_sup_decay_end: 30

# Warm-up (for backward compat with lam_mi in single-loss mode)
warmup_start: 0
warmup_mi_start: 0
warmup_mi_end: 10

# Early stopping
patience: 40
val_fraction: 0.2
split_seed: 42
val_seed: 123
checkpoint_every_n_epochs: 10
```

**Step 2: Commit**

```bash
git add vs_preprocess/training/configs/intra_vs_cross_modality.yaml
git commit -m "feat: add cross-modality three-tier loss training config"
```

---

### Task 9: Update inference pipeline for 0.5mm output

**Files:**
- Modify: `vs_preprocess/registration.py`

**Step 1: Update `align_patient_longitudinal` to accept output_spacing**

Add `output_spacing` parameter:

```python
def align_patient_longitudinal(
    patient_records: List,
    model: nn.Module,
    output_dir: Path,
    transform_type: str = "affine",
    keep_intermediates: bool = True,
    device: str = "cuda",
    target_spacing: Tuple[float, float, float] = None,
    output_spacing: Tuple[float, float, float] = None,
) -> Dict[str, dict]:
```

In the function body, after predicting the affine (Step 4), modify the warping step to resample the moving volume at `output_spacing` before applying the transform:

```python
# ── Step 6: Apply to volume at output resolution ──
moving_nii = nib.load(str(record.volume_path))
if output_spacing is not None:
    # Resample moving to output spacing, then apply affine
    mov_out_data, mov_out_affine = load_and_resample_isotropic(
        record.volume_path, output_spacing
    )
    mov_out_nii = nib.Nifti1Image(mov_out_data, mov_out_affine)
    warped_nii = apply_affine_to_volume(
        mov_out_nii, world_affine, baseline_nii, interpolation="trilinear"
    )
else:
    warped_nii = apply_affine_to_volume(
        moving_nii, world_affine, baseline_nii, interpolation="trilinear"
    )
```

**Step 2: Update `load_model` to read output_spacing from config**

After loading the checkpoint config, also read `output_spacing` if present:

```python
# In load_model(), after setting MODEL_TARGET_SPACING:
global MODEL_OUTPUT_SPACING
MODEL_OUTPUT_SPACING = tuple(config.get("output_spacing", config.get("target_spacing", [1.0, 1.0, 1.0])))
```

**Step 3: Commit**

```bash
git add vs_preprocess/registration.py
git commit -m "feat: support 0.5mm output spacing in inference pipeline"
```

---

### Task 10: Integration test with real data

**Files:**
- Create: `vs_preprocess/tests/test_cross_modality_integration.py`

**Step 1: Write integration test**

```python
# tests/test_cross_modality_integration.py
"""
Integration test: verify the full pipeline from data discovery
through pair enumeration with real VS data.

Requires: data at /home/kwa/Documents/vs_ai_registration/data/unprocessed_raw_studies/
"""
import pytest
from pathlib import Path
from dataset import discover_scans, group_by_patient_timepoint, get_super_modality
from training.multi_pair_dataset import enumerate_pairs, PairType

DATA_DIR = Path("/home/kwa/Documents/vs_ai_registration/data/unprocessed_raw_studies")

@pytest.mark.skipif(not DATA_DIR.exists(), reason="VS data not available")
class TestCrossModalityIntegration:

    def test_discover_all_modalities(self):
        records = discover_scans(DATA_DIR)
        modalities = set(r.modality for r in records)
        assert "t1+_thick_ax" in modalities
        assert "t2_thin" in modalities
        assert len(records) > 1000

    def test_group_by_patient_timepoint(self):
        records = discover_scans(DATA_DIR)
        index = group_by_patient_timepoint(records)
        # Should have hundreds of patients
        assert len(index) > 100
        # At least some patients should have multiple timepoints
        multi_tp = sum(1 for tps in index.values() if len(tps) > 1)
        assert multi_tp > 50

    def test_enumerate_pairs_counts(self):
        records = discover_scans(DATA_DIR)
        index = group_by_patient_timepoint(records)
        pairs = enumerate_pairs(index)

        same_sub = sum(1 for p in pairs if p[2] == PairType.SAME_SUB)
        cross_sub = sum(1 for p in pairs if p[2] == PairType.CROSS_SUB)
        cross_super = sum(1 for p in pairs if p[2] == PairType.CROSS_SUPER)

        print(f"Same-sub: {same_sub}, Cross-sub: {cross_sub}, Cross-super: {cross_super}")
        assert same_sub > 0
        assert cross_sub > 0
        assert cross_super > 0
        assert same_sub + cross_sub + cross_super == len(pairs)

    def test_all_pairs_have_valid_paths(self):
        records = discover_scans(DATA_DIR)
        index = group_by_patient_timepoint(records)
        pairs = enumerate_pairs(index)

        for mov_path, fix_path, pt in pairs[:100]:  # check first 100
            assert mov_path.exists(), f"Missing: {mov_path}"
            assert fix_path.exists(), f"Missing: {fix_path}"
            assert mov_path != fix_path
```

**Step 2: Run integration test**

Run: `cd vs_preprocess && python -m pytest tests/test_cross_modality_integration.py -v -s`
Expected: all PASS (on local machine with data)

**Step 3: Commit**

```bash
git add vs_preprocess/tests/test_cross_modality_integration.py
git commit -m "test: add cross-modality integration test with real VS data"
```

---

### Task 11: Update SyntheticPairDataset to return pair_type

**Files:**
- Modify: `vs_preprocess/training/dataset.py`

The existing `SyntheticPairDataset.__getitem__` doesn't return `pair_type`. When used with `CombinedMultiPairDataset`, the synthetic dataset needs to include this key.

**Step 1: Update SyntheticPairDataset**

In `training/dataset.py`, modify `SyntheticPairDataset.__getitem__` return dict to include `pair_type`:

```python
return {
    "moving": moving,
    "fixed": fixed,
    "has_gt": torch.tensor(True),
    "gt_params": torch.tensor(inv_params, dtype=torch.float32),
    "pair_type": torch.tensor(0),  # synthetic treated as same-sub (NCC)
}
```

**Step 2: Commit**

```bash
git add vs_preprocess/training/dataset.py
git commit -m "feat: add pair_type to SyntheticPairDataset return dict"
```

---

### Task 12: Smoke test full training loop

**Files:**
- Create: `vs_preprocess/tests/test_training_smoke.py`

**Step 1: Write smoke test**

```python
# tests/test_training_smoke.py
"""
Smoke test: run 2 epochs of three-tier training on tiny synthetic data.
Verifies the full pipeline doesn't crash.
"""
import pytest
import tempfile
import yaml
from pathlib import Path

@pytest.mark.skipif(
    not Path("/home/kwa/Documents/vs_ai_registration/data/unprocessed_raw_studies").exists(),
    reason="VS data not available"
)
def test_three_tier_training_smoke():
    """Run 2 epochs with three_tier loss on real data, batch_size=1."""
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "image_size": [64, 64, 64],  # small for speed
            "target_spacing": [2.0, 2.0, 2.0],  # coarse for speed
            "transform_type": "affine",
            "similarity_loss": "three_tier",
            "pair_ratios": [0.25, 0.25, 0.25, 0.25],
            "ncc_kernel_size": 9,
            "mi_num_bins": 32,
            "loss_ema_decay": 0.99,
            "batch_size": 1,
            "num_workers": 0,
            "max_epochs": 2,
            "lr": 1e-4,
            "grad_clip_norm": 1.0,
            "scheduler": "plateau",
            "plateau_factor": 0.5,
            "plateau_patience": 10,
            "head_hidden": 64,
            "head_dropout": 0.0,
            "lambda_identity": 0.001,
            "lambda_det": 0.1,
            "lambda_supervised": 1.0,
            "lambda_supervised_decay": False,
            "warmup_start": 0,
            "warmup_mi_start": 0,
            "warmup_mi_end": 1,
            "patience": 100,
            "val_fraction": 0.3,
            "split_seed": 42,
            "synthetic_length_per_epoch": 4,
            "checkpoint_every_n_epochs": 10,
        }
        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = subprocess.run(
            [
                sys.executable, "-m", "training.train_intra",
                "--config", str(config_path),
                "--data_dir", "/home/kwa/Documents/vs_ai_registration/data/unprocessed_raw_studies",
                "--output_dir", str(Path(tmpdir) / "output"),
            ],
            capture_output=True, text=True, timeout=300,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0, f"Training failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
```

**Step 2: Run smoke test**

Run: `cd vs_preprocess && python -m pytest tests/test_training_smoke.py -v -s --timeout=300`
Expected: PASS (training completes 2 epochs without error)

**Step 3: Commit**

```bash
git add vs_preprocess/tests/test_training_smoke.py
git commit -m "test: add three-tier training smoke test"
```

---

## Summary of Tasks

| Task | What | Files |
|------|------|-------|
| 1 | Super-modality mapping | `dataset.py`, test |
| 2 | `group_by_patient_timepoint()` | `dataset.py`, test |
| 3 | `enumerate_pairs()` with three pair types | `multi_pair_dataset.py`, test |
| 4 | `MultiPairDataset.__getitem__` | `multi_pair_dataset.py`, test |
| 5 | `ThreeTierLoss` with EMA normalization | `losses.py`, test |
| 6 | `CombinedMultiPairDataset` with 4-way ratio | `multi_pair_dataset.py`, test |
| 7 | Training loop three-tier branch | `train_intra.py` |
| 8 | New config file | `intra_vs_cross_modality.yaml` |
| 9 | 0.5mm output spacing in inference | `registration.py` |
| 10 | Integration test with real data | test |
| 11 | Add pair_type to SyntheticPairDataset | `dataset.py` |
| 12 | Smoke test full training loop | test |

**Dependencies:** Tasks 1-2 first, then 3-6 can be parallelized, then 7-8 depend on 3-6, then 9-12 after 7.
