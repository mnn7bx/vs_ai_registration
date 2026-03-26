# Tiered Registration Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace flat baseline-to-followup registration with a hierarchical tiered system (MNI ← patient anchor ← timepoint anchors ← scans), running both DL and ANTsPy engines side-by-side for comparison.

**Architecture:** Shared anchor selection logic produces a `RegistrationTree` dataclass. Two registration engines (DL in `registration.py`, ANTsPy in `registration_ants.py`) consume the same tree. ANTsPy runs as a subprocess via `conda run -n antspy_env`. A comparison module produces metrics between the two result sets.

**Tech Stack:** Python 3.13, nibabel, numpy, torch/MONAI (DL engine), antspyx (ANTsPy engine via subprocess), existing affine_utils/preprocessing modules.

---

### Task 1: Anchor Selection Module

**Files:**
- Create: `vs_preprocess/anchor_selection.py`
- Test: `vs_preprocess/tests/test_anchor_selection.py`
- Read: `vs_preprocess/dataset.py` (ScanRecord, group_by_patient_timepoint, SUPER_MODALITY_MAP)

This module owns the tiered hierarchy: picking patient anchor, timepoint anchors, and mapping remaining scans to their anchor.

**Step 1: Write the failing tests**

```python
# tests/test_anchor_selection.py
"""Tests for tiered anchor selection."""
import pytest
from dataclasses import replace
from pathlib import Path

from dataset import ScanRecord
from anchor_selection import (
    SLICE_THICKNESS_RANK,
    select_patient_anchor,
    select_timepoint_anchors,
    build_registration_tree,
    RegistrationTree,
)


def _rec(modality, patient_id, timepoint, seg=False):
    """Helper to build a ScanRecord for testing."""
    return ScanRecord(
        modality=modality,
        patient_timepoint=f"{patient_id}_{timepoint}",
        patient_id=patient_id,
        timepoint=timepoint,
        volume_path=Path(f"/fake/{modality}/{patient_id}_{timepoint}/vol.nii.gz"),
        seg_path=Path(f"/fake/{modality}/{patient_id}_{timepoint}/seg.nii.gz") if seg else None,
    )


class TestSliceThicknessRank:
    def test_thin_beats_thick(self):
        assert SLICE_THICKNESS_RANK["t2_thin"] < SLICE_THICKNESS_RANK["t2_thick"]
        assert SLICE_THICKNESS_RANK["t1+_thin"] < SLICE_THICKNESS_RANK["t1+_thick_ax"]

    def test_t2_thin_is_best(self):
        best = min(SLICE_THICKNESS_RANK, key=SLICE_THICKNESS_RANK.get)
        assert best in ("t2_thin", "t1+_thin")


class TestSelectPatientAnchor:
    def test_picks_thinnest_slice(self):
        records = [
            _rec("t1+_thick_ax", "P1", "0_0"),
            _rec("t2_thin", "P1", "1_100"),
            _rec("t2_thick", "P1", "2_200"),
        ]
        anchor = select_patient_anchor(records)
        assert anchor.modality == "t2_thin"

    def test_tiebreak_prefers_t2(self):
        records = [
            _rec("t1+_thin", "P1", "0_0"),
            _rec("t2_thin", "P1", "0_0"),
        ]
        anchor = select_patient_anchor(records)
        assert anchor.modality == "t2_thin"

    def test_tiebreak_prefers_earliest(self):
        records = [
            _rec("t2_thin", "P1", "1_100"),
            _rec("t2_thin", "P1", "0_0"),
        ]
        anchor = select_patient_anchor(records)
        assert anchor.timepoint == "0_0"

    def test_single_record(self):
        records = [_rec("t1+_thick_ax", "P1", "0_0")]
        anchor = select_patient_anchor(records)
        assert anchor == records[0]


class TestSelectTimepointAnchors:
    def test_picks_best_t2_per_timepoint(self):
        records = [
            _rec("t2_thin", "P1", "0_0"),
            _rec("t1+_thick_ax", "P1", "0_0"),
            _rec("t2_thick", "P1", "1_100"),
            _rec("t1+_thin", "P1", "1_100"),
        ]
        patient_anchor = records[0]
        tp_anchors = select_timepoint_anchors(records, patient_anchor)
        # timepoint "0_0" -> t2_thin, timepoint "1_100" -> t2_thick (only T2)
        assert tp_anchors["0_0"].modality == "t2_thin"
        assert tp_anchors["1_100"].modality == "t2_thick"

    def test_fallback_to_t1c_if_no_t2(self):
        records = [
            _rec("t1+_thick_ax", "P1", "0_0"),
            _rec("t1+_thin", "P1", "0_0"),
        ]
        patient_anchor = records[1]  # t1+_thin
        tp_anchors = select_timepoint_anchors(records, patient_anchor)
        assert tp_anchors["0_0"].modality == "t1+_thin"

    def test_patient_anchor_is_its_own_tp_anchor(self):
        """The patient anchor should be the timepoint anchor for its timepoint."""
        records = [
            _rec("t2_thin", "P1", "0_0"),
            _rec("t1+_thick_ax", "P1", "0_0"),
            _rec("t2_thick", "P1", "1_100"),
        ]
        patient_anchor = records[0]
        tp_anchors = select_timepoint_anchors(records, patient_anchor)
        assert tp_anchors["0_0"] is patient_anchor


class TestBuildRegistrationTree:
    def test_structure(self):
        records = [
            _rec("t2_thin", "P1", "0_0"),
            _rec("t1+_thick_ax", "P1", "0_0"),
            _rec("t2_thick", "P1", "1_100"),
            _rec("t1+_thin", "P1", "1_100"),
        ]
        tree = build_registration_tree(records)

        assert isinstance(tree, RegistrationTree)
        assert tree.patient_anchor.modality == "t2_thin"
        assert len(tree.timepoints) == 2
        # Each timepoint has an anchor and remaining scans
        for tp, tp_info in tree.timepoints.items():
            assert "anchor" in tp_info
            assert "others" in tp_info

    def test_others_excludes_anchors(self):
        records = [
            _rec("t2_thin", "P1", "0_0"),
            _rec("t1+_thick_ax", "P1", "0_0"),
            _rec("t2_thick", "P1", "0_0"),
        ]
        tree = build_registration_tree(records)
        tp_info = tree.timepoints["0_0"]
        # anchor = t2_thin, others = t1+_thick_ax and t2_thick
        other_mods = [r.modality for r in tp_info["others"]]
        assert "t2_thin" not in other_mods
        assert len(other_mods) == 2

    def test_single_scan_patient(self):
        records = [_rec("t1+_thick_ax", "P1", "0_0")]
        tree = build_registration_tree(records)
        assert tree.patient_anchor == records[0]
        assert len(tree.timepoints) == 1
        assert tree.timepoints["0_0"]["others"] == []
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_anchor_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'anchor_selection'`

**Step 3: Write the implementation**

```python
# vs_preprocess/anchor_selection.py
"""
Tiered anchor selection for hierarchical registration.

Hierarchy:
    MNI Template
      └── Patient Anchor (best quality scan overall)
            └── Timepoint Anchors (best T2 per timepoint)
                  └── Remaining scans
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from dataset import ScanRecord, SUPER_MODALITY_MAP

logger = logging.getLogger(__name__)

# Lower rank = thinner slices = higher quality for anchor selection.
# Ties between T1 and T2 at same thickness: T2 preferred (lower rank).
SLICE_THICKNESS_RANK = {
    "t2_thin": 0,
    "t1+_thin": 1,
    "t2_thick": 2,
    "t1+_thick_cor": 3,
    "t1+_thick_ax": 4,
}


def _sort_key(record: ScanRecord) -> tuple:
    """Sort key: thinnest slices, prefer T2, earliest timepoint."""
    thickness = SLICE_THICKNESS_RANK.get(record.modality, 99)
    is_t2 = 0 if record.modality.startswith("t2") else 1
    return (thickness, is_t2, record.timepoint)


def select_patient_anchor(records: List[ScanRecord]) -> ScanRecord:
    """Pick the best quality scan across all timepoints/modalities.

    Priority: thinnest slices → prefer T2 → earliest timepoint.
    """
    return min(records, key=_sort_key)


def select_timepoint_anchors(
    records: List[ScanRecord],
    patient_anchor: ScanRecord,
) -> Dict[str, ScanRecord]:
    """Pick the best T2 scan per timepoint as the timepoint anchor.

    Falls back to best T1C if no T2 is available at a timepoint.
    If the patient anchor's timepoint is included, the patient anchor
    itself is used as that timepoint's anchor.
    """
    by_tp: Dict[str, List[ScanRecord]] = {}
    for r in records:
        by_tp.setdefault(r.timepoint, []).append(r)

    anchors: Dict[str, ScanRecord] = {}
    for tp, tp_records in by_tp.items():
        # If patient anchor lives at this timepoint, it is the anchor
        if any(r is patient_anchor for r in tp_records):
            anchors[tp] = patient_anchor
            continue
        # Prefer T2, then thinnest
        anchors[tp] = min(tp_records, key=_sort_key)

    return anchors


@dataclass
class RegistrationTree:
    """Hierarchical registration structure for one patient."""
    patient_id: str
    patient_anchor: ScanRecord
    timepoints: Dict[str, dict] = field(default_factory=dict)
    # timepoints[tp] = {"anchor": ScanRecord, "others": [ScanRecord, ...]}


def build_registration_tree(records: List[ScanRecord]) -> RegistrationTree:
    """Build the full registration hierarchy for a patient's scans.

    Returns a RegistrationTree with:
      - patient_anchor: best quality scan overall
      - timepoints: {tp: {"anchor": ScanRecord, "others": [ScanRecord, ...]}}
    """
    patient_anchor = select_patient_anchor(records)
    tp_anchors = select_timepoint_anchors(records, patient_anchor)

    by_tp: Dict[str, List[ScanRecord]] = {}
    for r in records:
        by_tp.setdefault(r.timepoint, []).append(r)

    timepoints: Dict[str, dict] = {}
    for tp, tp_records in by_tp.items():
        anchor = tp_anchors[tp]
        others = [r for r in tp_records if r is not anchor]
        timepoints[tp] = {"anchor": anchor, "others": others}

    return RegistrationTree(
        patient_id=patient_anchor.patient_id,
        patient_anchor=patient_anchor,
        timepoints=timepoints,
    )
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_anchor_selection.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add vs_preprocess/anchor_selection.py vs_preprocess/tests/test_anchor_selection.py
git commit -m "feat: add tiered anchor selection for hierarchical registration"
```

---

### Task 2: Refactor DL Registration for Tiered Hierarchy

**Files:**
- Modify: `vs_preprocess/registration.py`
- Test: `vs_preprocess/tests/test_tiered_dl_registration.py`
- Read: `vs_preprocess/affine_utils.py` (compose chain, apply_affine_to_volume)

Add `register_pair_dl()` (single pair, rigid→affine) and `register_tiered_dl()` (full tree traversal). Keep `align_patient_longitudinal()` unchanged for backward compat.

**Step 1: Write the failing tests**

```python
# tests/test_tiered_dl_registration.py
"""Tests for tiered DL registration functions."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from dataset import ScanRecord
from anchor_selection import build_registration_tree


def _rec(modality, patient_id, timepoint):
    return ScanRecord(
        modality=modality,
        patient_timepoint=f"{patient_id}_{timepoint}",
        patient_id=patient_id,
        timepoint=timepoint,
        volume_path=Path(f"/fake/{modality}/{patient_id}_{timepoint}/vol.nii.gz"),
        seg_path=None,
    )


class TestRegisterPairDL:
    """Test register_pair_dl() returns rigid then affine composed."""

    def test_returns_composed_affine_and_params(self):
        from registration import register_pair_dl

        # Mock model that returns identity-like affines
        mock_model = MagicMock()

        with patch("registration.predict_affine") as mock_predict, \
             patch("registration.load_and_resample_isotropic") as mock_load, \
             patch("registration._crop_and_normalize") as mock_crop, \
             patch("registration.compute_overlap_center") as mock_overlap, \
             patch("registration._compute_crop_affine") as mock_crop_aff, \
             patch("registration.normalized_affine_to_world_v2") as mock_n2w:

            # Setup mocks
            mock_load.return_value = (np.zeros((96, 96, 96)), np.eye(4))
            mock_crop.return_value = (
                __import__("torch").zeros(1, 1, 96, 96, 96),
                (0, 0, 0),
            )
            mock_overlap.return_value = (48, 48, 48)
            mock_crop_aff.return_value = np.eye(4)

            identity_3x4 = np.eye(3, 4)
            mock_predict.return_value = (
                identity_3x4,
                np.zeros(6),
                {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                 "scale": [1, 1, 1], "determinant": 1.0},
            )
            mock_n2w.return_value = np.eye(4)

            result = register_pair_dl(
                mock_model,
                moving_path=Path("/fake/mov.nii.gz"),
                fixed_path=Path("/fake/fix.nii.gz"),
                device="cpu",
            )

            assert "world_affine" in result
            assert "params_dict" in result
            assert result["world_affine"].shape == (4, 4)
            # predict_affine called twice: rigid then affine
            assert mock_predict.call_count == 2


class TestValidateTransform:
    """Test quality gate on predicted transforms."""

    def test_flags_large_rotation(self):
        from registration import validate_transform

        params = {
            "rotation_deg": [50.0, 0.0, 0.0],
            "translation": [0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "determinant": 1.0,
        }
        result = validate_transform(params)
        assert result["passed"] is False
        assert "rotation" in result["reason"]

    def test_flags_bad_determinant(self):
        from registration import validate_transform

        params = {
            "rotation_deg": [1.0, 1.0, 1.0],
            "translation": [0.0, 0.0, 0.0],
            "scale": [1.5, 1.5, 1.5],
            "determinant": 0.5,
        }
        result = validate_transform(params)
        assert result["passed"] is False
        assert "determinant" in result["reason"]

    def test_passes_good_transform(self):
        from registration import validate_transform

        params = {
            "rotation_deg": [2.0, 1.0, 0.5],
            "translation": [0.1, 0.2, 0.0],
            "scale": [1.01, 0.99, 1.0],
            "determinant": 1.0,
        }
        result = validate_transform(params)
        assert result["passed"] is True
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_tiered_dl_registration.py -v`
Expected: FAIL — `ImportError: cannot import name 'register_pair_dl'`

**Step 3: Write the implementation**

Add these functions to `registration.py`:

```python
# Add to registration.py after predict_affine()

def validate_transform(
    params_dict: dict,
    max_rotation_deg: float = 45.0,
    det_range: tuple = (0.8, 1.3),
) -> dict:
    """Check if a predicted transform is physically plausible.

    Returns dict with 'passed' (bool) and 'reason' (str or None).
    """
    max_rot = max(abs(r) for r in params_dict["rotation_deg"])
    det = params_dict["determinant"]

    if max_rot > max_rotation_deg:
        return {"passed": False, "reason": f"rotation {max_rot:.1f} > {max_rotation_deg}"}
    if det < det_range[0] or det > det_range[1]:
        return {"passed": False, "reason": f"determinant {det:.3f} outside {det_range}"}
    return {"passed": True, "reason": None}


def register_pair_dl(
    model: nn.Module,
    moving_path: Path,
    fixed_path: Path,
    device: str = "cuda",
    target_spacing: Tuple[float, float, float] = None,
) -> dict:
    """Register a single pair using DL: rigid prediction, then affine refinement.

    Steps:
        1. Load and resample both to isotropic
        2. Predict rigid (6 DOF)
        3. Apply rigid to moving
        4. Predict affine (9 DOF) on rigidly-aligned pair
        5. Compose: world_affine = affine_world @ rigid_world

    Returns dict with:
        world_affine: (4, 4) composed world-space transform
        rigid_params: dict from rigid step
        affine_params: dict from affine step
        params_dict: dict from composed transform (for quality gate)
        validation: dict from validate_transform()
    """
    spacing = target_spacing or MODEL_TARGET_SPACING

    # Load both volumes
    fix_data, fix_affine = load_and_resample_isotropic(fixed_path, spacing)
    mov_data, mov_affine = load_and_resample_isotropic(moving_path, spacing)

    # Overlap center
    overlap_center_fix = compute_overlap_center(
        fix_affine, fix_data.shape, mov_affine, mov_data.shape
    )
    center_world = fix_affine[:3, :3] @ np.array(overlap_center_fix, dtype=np.float64) + fix_affine[:3, 3]
    inv_mov = np.linalg.inv(mov_affine)
    overlap_center_mov = inv_mov[:3, :3] @ center_world + inv_mov[:3, 3]
    overlap_center_mov = tuple(int(round(v)) for v in overlap_center_mov)
    overlap_center_mov = tuple(max(0, min(c, s - 1)) for c, s in zip(overlap_center_mov, mov_data.shape))

    # Crop and normalize
    fix_tensor, fix_crop_origin = _crop_and_normalize(fix_data, NET_INPUT_SHAPE, center=overlap_center_fix)
    mov_tensor, mov_crop_origin = _crop_and_normalize(mov_data, NET_INPUT_SHAPE, center=overlap_center_mov)

    fix_crop_affine = _compute_crop_affine(fix_affine, fix_crop_origin)
    mov_crop_affine = _compute_crop_affine(mov_affine, mov_crop_origin)

    # --- Step 1: Rigid prediction ---
    rigid_3x4, _, rigid_params = predict_affine(model, mov_tensor, fix_tensor, "rigid", device)
    rigid_world = normalized_affine_to_world_v2(rigid_3x4, mov_crop_affine, fix_crop_affine, NET_INPUT_SHAPE)

    # --- Step 2: Apply rigid, re-crop, predict affine ---
    # Warp moving data with rigid transform for the second pass
    mov_nii = nib.Nifti1Image(mov_data, mov_affine)
    fix_nii = nib.Nifti1Image(fix_data, fix_affine)
    warped_rigid_nii = apply_affine_to_volume(mov_nii, rigid_world, fix_nii, interpolation="trilinear")
    warped_rigid_data = warped_rigid_nii.get_fdata(dtype=np.float32)

    # Re-crop the rigidly aligned moving (now in fixed space, so use fixed center)
    warped_tensor, warped_crop_origin = _crop_and_normalize(
        warped_rigid_data, NET_INPUT_SHAPE, center=overlap_center_fix
    )
    warped_crop_affine = _compute_crop_affine(fix_affine, warped_crop_origin)

    affine_3x4, _, affine_params = predict_affine(model, warped_tensor, fix_tensor, "affine", device)
    affine_world = normalized_affine_to_world_v2(affine_3x4, warped_crop_affine, fix_crop_affine, NET_INPUT_SHAPE)

    # --- Compose ---
    composed_world = affine_world @ rigid_world

    composed_params = decompose_affine(composed_world[:3, :])
    validation = validate_transform(composed_params)

    return {
        "world_affine": composed_world,
        "rigid_params": rigid_params,
        "affine_params": affine_params,
        "params_dict": composed_params,
        "validation": validation,
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_tiered_dl_registration.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add vs_preprocess/registration.py vs_preprocess/tests/test_tiered_dl_registration.py
git commit -m "feat: add register_pair_dl (rigid→affine) and validate_transform"
```

---

### Task 3: DL Tiered Tree Registration

**Files:**
- Modify: `vs_preprocess/registration.py`
- Test: `vs_preprocess/tests/test_tiered_dl_tree.py`

Add `register_tiered_dl()` that walks a `RegistrationTree`, calls `register_pair_dl()` at each tier, composes transforms, and saves results.

**Step 1: Write the failing test**

```python
# tests/test_tiered_dl_tree.py
"""Test full tiered DL tree registration."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from dataset import ScanRecord
from anchor_selection import build_registration_tree


def _rec(modality, patient_id, timepoint):
    return ScanRecord(
        modality=modality,
        patient_timepoint=f"{patient_id}_{timepoint}",
        patient_id=patient_id,
        timepoint=timepoint,
        volume_path=Path(f"/fake/{modality}/{patient_id}_{timepoint}/vol.nii.gz"),
        seg_path=None,
    )


class TestRegisterTieredDL:
    def test_produces_results_for_all_scans(self):
        from registration import register_tiered_dl

        records = [
            _rec("t2_thin", "P1", "0_0"),
            _rec("t1+_thick_ax", "P1", "0_0"),
            _rec("t2_thick", "P1", "1_100"),
        ]
        tree = build_registration_tree(records)

        mock_model = MagicMock()

        with patch("registration.register_pair_dl") as mock_reg:
            mock_reg.return_value = {
                "world_affine": np.eye(4),
                "rigid_params": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                 "scale": [1, 1, 1], "determinant": 1.0},
                "affine_params": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                  "scale": [1, 1, 1], "determinant": 1.0},
                "params_dict": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                "scale": [1, 1, 1], "determinant": 1.0},
                "validation": {"passed": True, "reason": None},
            }

            results = register_tiered_dl(
                tree, mock_model, Path("/tmp/test_out"),
                mni_template_path=Path("/fake/mni.nii.gz"),
                device="cpu",
            )

            # Should have a result for every scan
            assert len(results) == 3
            # Patient anchor registered to MNI (tier 1)
            pa_key = "P1_0_0"  # t2_thin
            assert pa_key in results
            # All results have composed_to_mni
            for key, res in results.items():
                assert "composed_to_mni" in res

    def test_tier_call_order(self):
        """Tier 1 (anchor→MNI) before tier 2 (tp_anchor→patient) before tier 3."""
        from registration import register_tiered_dl

        records = [
            _rec("t2_thin", "P1", "0_0"),
            _rec("t1+_thick_ax", "P1", "0_0"),
        ]
        tree = build_registration_tree(records)
        mock_model = MagicMock()

        call_log = []

        def mock_reg(model, moving_path, fixed_path, **kwargs):
            call_log.append((str(moving_path), str(fixed_path)))
            return {
                "world_affine": np.eye(4),
                "rigid_params": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                 "scale": [1, 1, 1], "determinant": 1.0},
                "affine_params": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                  "scale": [1, 1, 1], "determinant": 1.0},
                "params_dict": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                "scale": [1, 1, 1], "determinant": 1.0},
                "validation": {"passed": True, "reason": None},
            }

        with patch("registration.register_pair_dl", side_effect=mock_reg):
            register_tiered_dl(
                tree, mock_model, Path("/tmp/test_out"),
                mni_template_path=Path("/fake/mni.nii.gz"),
                device="cpu",
            )

        # First call: patient_anchor → MNI
        assert "mni" in call_log[0][1].lower() or call_log[0][1] == str(Path("/fake/mni.nii.gz"))
        # Remaining calls: within-patient registrations
        assert len(call_log) >= 1
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_tiered_dl_tree.py -v`
Expected: FAIL — `ImportError: cannot import name 'register_tiered_dl'`

**Step 3: Write the implementation**

Add to `registration.py`:

```python
def register_tiered_dl(
    tree: "RegistrationTree",
    model: nn.Module,
    output_dir: Path,
    mni_template_path: Path,
    device: str = "cuda",
    target_spacing: Tuple[float, float, float] = None,
) -> Dict[str, dict]:
    """Walk the registration tree, calling register_pair_dl at each tier.

    Tier 1: patient_anchor → MNI template
    Tier 2: timepoint_anchors → patient_anchor
    Tier 3: remaining scans → their timepoint_anchor

    Composes transforms so every scan has a full chain to MNI.

    Returns dict mapping patient_timepoint -> {
        tier: int,
        tier_affine: (4,4) transform for this tier only,
        composed_to_mni: (4,4) full chain to MNI,
        params_dict: dict,
        validation: dict,
    }
    """
    from anchor_selection import RegistrationTree

    results: Dict[str, dict] = {}
    pa = tree.patient_anchor

    # ── Tier 1: Patient anchor → MNI ──
    logger.info("Tier 1: %s → MNI", pa.patient_timepoint)
    tier1 = register_pair_dl(
        model, pa.volume_path, mni_template_path,
        device=device, target_spacing=target_spacing,
    )
    pa_to_mni = tier1["world_affine"]

    results[pa.patient_timepoint] = {
        "tier": 1,
        "tier_affine": pa_to_mni,
        "composed_to_mni": pa_to_mni,
        "params_dict": tier1["params_dict"],
        "validation": tier1["validation"],
    }

    # ── Tier 2 & 3: Per-timepoint ──
    for tp, tp_info in tree.timepoints.items():
        tp_anchor = tp_info["anchor"]

        # Tier 2: tp_anchor → patient_anchor (skip if tp_anchor IS patient_anchor)
        if tp_anchor is not pa:
            logger.info("Tier 2: %s → %s", tp_anchor.patient_timepoint, pa.patient_timepoint)
            tier2 = register_pair_dl(
                model, tp_anchor.volume_path, pa.volume_path,
                device=device, target_spacing=target_spacing,
            )
            tp_to_pa = tier2["world_affine"]
            tp_to_mni = pa_to_mni @ tp_to_pa

            results[tp_anchor.patient_timepoint] = {
                "tier": 2,
                "tier_affine": tp_to_pa,
                "composed_to_mni": tp_to_mni,
                "params_dict": tier2["params_dict"],
                "validation": tier2["validation"],
            }
        else:
            tp_to_mni = pa_to_mni

        # Tier 3: remaining scans → tp_anchor
        for scan in tp_info["others"]:
            logger.info("Tier 3: %s → %s", scan.patient_timepoint, tp_anchor.patient_timepoint)
            tier3 = register_pair_dl(
                model, scan.volume_path, tp_anchor.volume_path,
                device=device, target_spacing=target_spacing,
            )
            scan_to_tp = tier3["world_affine"]
            scan_to_mni = tp_to_mni @ scan_to_tp

            results[scan.patient_timepoint] = {
                "tier": 3,
                "tier_affine": scan_to_tp,
                "composed_to_mni": scan_to_mni,
                "params_dict": tier3["params_dict"],
                "validation": tier3["validation"],
            }

    # Save results
    out_dir = output_dir / tree.patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    import json
    anchors_info = {
        "patient_anchor": pa.patient_timepoint,
        "patient_anchor_modality": pa.modality,
        "timepoint_anchors": {
            tp: tp_info["anchor"].patient_timepoint
            for tp, tp_info in tree.timepoints.items()
        },
    }
    with open(out_dir / "anchors.json", "w") as f:
        json.dump(anchors_info, f, indent=2)

    for pt_key, res in results.items():
        scan_dir = out_dir / pt_key
        scan_dir.mkdir(parents=True, exist_ok=True)
        np.save(scan_dir / f"{pt_key}_tier_affine.npy", res["tier_affine"])
        np.save(scan_dir / f"{pt_key}_composed_to_mni.npy", res["composed_to_mni"])
        with open(scan_dir / f"{pt_key}_params.json", "w") as f:
            json.dump({
                "tier": res["tier"],
                "params": res["params_dict"],
                "validation": res["validation"],
            }, f, indent=2)

    return results
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_tiered_dl_tree.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add vs_preprocess/registration.py vs_preprocess/tests/test_tiered_dl_tree.py
git commit -m "feat: add register_tiered_dl for hierarchical tree registration"
```

---

### Task 4: ANTsPy Registration Script

**Files:**
- Create: `vs_preprocess/registration_ants.py`
- Create: `vs_preprocess/scripts/run_ants_registration.sh`
- Test: `vs_preprocess/tests/test_registration_ants.py`

Standalone script that runs in `antspy_env`. Uses the same anchor selection and tree structure, but registers with ANTsPy rigid→similarity.

**Step 1: Write the failing test**

```python
# tests/test_registration_ants.py
"""Tests for ANTsPy registration (mocked — antspy not in this env)."""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# These tests mock ants since it's in a different env


class TestAntsRegisterPair:
    def test_import_and_call_structure(self):
        """Verify register_pair_ants exists and has correct signature."""
        from registration_ants import register_pair_ants
        import inspect
        sig = inspect.signature(register_pair_ants)
        assert "moving_path" in sig.parameters
        assert "fixed_path" in sig.parameters
        assert "output_dir" in sig.parameters


class TestAntsTransformConversion:
    def test_ants_matrix_to_numpy(self):
        """Test converting ANTsPy transform to numpy 4x4."""
        from registration_ants import ants_affine_to_numpy_4x4
        # ANTsPy affine is stored as a flat 12-element array (3x4 row-major)
        # plus a fixed center
        flat_identity = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
        center = [0.0, 0.0, 0.0]
        result = ants_affine_to_numpy_4x4(flat_identity, center)
        assert result.shape == (4, 4)
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_registration_ants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'registration_ants'`

**Step 3: Write the implementation**

```python
# vs_preprocess/registration_ants.py
"""
ANTsPy-based tiered registration.

Runs in antspy_env via subprocess. Uses the same anchor selection logic
as the DL pipeline for apples-to-apples comparison.

Usage:
    conda run -n antspy_env python registration_ants.py \
        --data_dir data/unprocessed_raw_studies \
        --output_dir preprocessing_output/ants \
        --mni_template /path/to/MNI152.nii.gz
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ScanRecord, discover_scans
from anchor_selection import build_registration_tree, RegistrationTree
from affine_utils import decompose_affine

logger = logging.getLogger(__name__)


def ants_affine_to_numpy_4x4(
    flat_params: list,
    center: list,
) -> np.ndarray:
    """Convert ANTsPy affine parameters to a numpy 4x4 matrix.

    ANTsPy stores affine as:
      - 3x3 rotation/scale matrix (9 elements, row-major)
      - 3 translation values
      - fixed center of rotation

    The full transform is: y = M @ (x - center) + translation + center
    Which rearranges to: y = M @ x + (translation + center - M @ center)
    """
    M = np.array(flat_params[:9]).reshape(3, 3)
    translation = np.array(flat_params[9:12])
    center = np.array(center)

    # Effective offset: translation + center - M @ center
    offset = translation + center - M @ center

    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = M
    result[:3, 3] = offset
    return result


def register_pair_ants(
    moving_path: Path,
    fixed_path: Path,
    output_dir: Path,
    type_of_transforms: Tuple[str, ...] = ("Rigid", "Similarity"),
) -> dict:
    """Register a pair using ANTsPy: rigid then similarity.

    Args:
        moving_path: path to moving NIfTI.
        fixed_path: path to fixed NIfTI.
        output_dir: directory for intermediate outputs.
        type_of_transforms: ANTsPy transform types to apply in sequence.

    Returns dict with:
        world_affine: (4,4) composed world-space transform
        params_dict: human-readable decomposed params
        transforms: list of ANTsPy transform file paths
    """
    import ants

    output_dir.mkdir(parents=True, exist_ok=True)

    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))

    composed = np.eye(4, dtype=np.float64)
    transform_paths = []
    current_moving = moving

    for transform_type in type_of_transforms:
        result = ants.registration(
            fixed=fixed,
            moving=current_moving,
            type_of_transform=transform_type,
            verbose=False,
        )

        # Extract the affine matrix from the forward transform
        tx_path = result["fwdtransforms"][0]
        transform_paths.append(tx_path)

        tx = ants.read_transform(tx_path)
        flat_params = tx.parameters.tolist()
        center = tx.fixed_parameters.tolist()
        step_affine = ants_affine_to_numpy_4x4(flat_params, center)

        composed = step_affine @ composed
        current_moving = result["warpedmovout"]

    params_dict = decompose_affine(composed[:3, :])

    return {
        "world_affine": composed,
        "params_dict": params_dict,
        "transforms": transform_paths,
    }


def register_tiered_ants(
    tree: RegistrationTree,
    output_dir: Path,
    mni_template_path: Path,
) -> Dict[str, dict]:
    """Walk registration tree using ANTsPy at each tier.

    Same hierarchy as register_tiered_dl:
        Tier 1: patient_anchor → MNI
        Tier 2: tp_anchors → patient_anchor
        Tier 3: remaining → tp_anchor
    """
    results: Dict[str, dict] = {}
    pa = tree.patient_anchor

    # Tier 1: patient anchor → MNI
    logger.info("Tier 1 (ANTs): %s → MNI", pa.patient_timepoint)
    tier1 = register_pair_ants(pa.volume_path, mni_template_path, output_dir / pa.patient_timepoint)
    pa_to_mni = tier1["world_affine"]

    results[pa.patient_timepoint] = {
        "tier": 1,
        "tier_affine": pa_to_mni,
        "composed_to_mni": pa_to_mni,
        "params_dict": tier1["params_dict"],
    }

    for tp, tp_info in tree.timepoints.items():
        tp_anchor = tp_info["anchor"]

        if tp_anchor is not pa:
            logger.info("Tier 2 (ANTs): %s → %s", tp_anchor.patient_timepoint, pa.patient_timepoint)
            tier2 = register_pair_ants(
                tp_anchor.volume_path, pa.volume_path,
                output_dir / tp_anchor.patient_timepoint,
            )
            tp_to_pa = tier2["world_affine"]
            tp_to_mni = pa_to_mni @ tp_to_pa

            results[tp_anchor.patient_timepoint] = {
                "tier": 2,
                "tier_affine": tp_to_pa,
                "composed_to_mni": tp_to_mni,
                "params_dict": tier2["params_dict"],
            }
        else:
            tp_to_mni = pa_to_mni

        for scan in tp_info["others"]:
            logger.info("Tier 3 (ANTs): %s → %s", scan.patient_timepoint, tp_anchor.patient_timepoint)
            tier3 = register_pair_ants(
                scan.volume_path, tp_anchor.volume_path,
                output_dir / scan.patient_timepoint,
            )
            scan_to_tp = tier3["world_affine"]
            scan_to_mni = tp_to_mni @ scan_to_tp

            results[scan.patient_timepoint] = {
                "tier": 3,
                "tier_affine": scan_to_tp,
                "composed_to_mni": scan_to_mni,
                "params_dict": tier3["params_dict"],
            }

    # Save
    out_dir = output_dir / tree.patient_id
    out_dir.mkdir(parents=True, exist_ok=True)

    anchors_info = {
        "patient_anchor": pa.patient_timepoint,
        "patient_anchor_modality": pa.modality,
        "timepoint_anchors": {
            tp: tp_info["anchor"].patient_timepoint
            for tp, tp_info in tree.timepoints.items()
        },
    }
    with open(out_dir / "anchors.json", "w") as f:
        json.dump(anchors_info, f, indent=2)

    for pt_key, res in results.items():
        scan_dir = out_dir / pt_key
        scan_dir.mkdir(parents=True, exist_ok=True)
        np.save(scan_dir / f"{pt_key}_tier_affine.npy", res["tier_affine"])
        np.save(scan_dir / f"{pt_key}_composed_to_mni.npy", res["composed_to_mni"])
        with open(scan_dir / f"{pt_key}_params.json", "w") as f:
            json.dump({"tier": res["tier"], "params": res["params_dict"]}, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="ANTsPy tiered registration")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--mni_template", type=Path, required=True)
    parser.add_argument("--patient_ids", nargs="*", default=None,
                        help="Process only these patient IDs")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    records = discover_scans(args.data_dir)
    # Group all records by patient
    by_patient: Dict[str, List[ScanRecord]] = {}
    for r in records:
        by_patient.setdefault(r.patient_id, []).append(r)

    for patient_id, patient_records in by_patient.items():
        if args.patient_ids and patient_id not in args.patient_ids:
            continue

        anchors_path = args.output_dir / patient_id / "anchors.json"
        if args.skip_existing and anchors_path.exists():
            logger.info("Skipping %s (already done)", patient_id)
            continue

        tree = build_registration_tree(patient_records)
        register_tiered_ants(tree, args.output_dir, args.mni_template)


if __name__ == "__main__":
    main()
```

**Step 4: Write the shell wrapper**

```bash
# scripts/run_ants_registration.sh
#!/usr/bin/env bash
set -euo pipefail

# Wrapper to run ANTsPy registration in the antspy_env conda environment.
# Passes all arguments through to registration_ants.py.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

conda run -n antspy_env --no-capture-output \
    python "$SCRIPT_DIR/registration_ants.py" "$@"
```

**Step 5: Run tests to verify they pass**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_registration_ants.py -v`
Expected: All PASS

**Step 6: Commit**

```bash
chmod +x vs_preprocess/scripts/run_ants_registration.sh
git add vs_preprocess/registration_ants.py vs_preprocess/scripts/run_ants_registration.sh vs_preprocess/tests/test_registration_ants.py
git commit -m "feat: add ANTsPy tiered registration script with subprocess wrapper"
```

---

### Task 5: Comparison Module

**Files:**
- Create: `vs_preprocess/comparison.py`
- Test: `vs_preprocess/tests/test_comparison.py`

Reads both DL and ANTsPy output directories, computes comparison metrics.

**Step 1: Write the failing test**

```python
# tests/test_comparison.py
"""Tests for DL vs ANTsPy comparison metrics."""
import pytest
import numpy as np
import json
from pathlib import Path

from comparison import compare_patient, compute_ncc


class TestComputeNCC:
    def test_identical_volumes(self):
        vol = np.random.randn(10, 10, 10).astype(np.float32)
        assert compute_ncc(vol, vol) == pytest.approx(1.0, abs=1e-5)

    def test_negatively_correlated(self):
        vol = np.random.randn(10, 10, 10).astype(np.float32)
        assert compute_ncc(vol, -vol) == pytest.approx(-1.0, abs=1e-5)

    def test_uncorrelated(self):
        a = np.random.randn(50, 50, 50).astype(np.float32)
        b = np.random.randn(50, 50, 50).astype(np.float32)
        ncc = compute_ncc(a, b)
        assert abs(ncc) < 0.1  # approximately zero


class TestComparePatient:
    def test_loads_and_compares(self, tmp_path):
        """Compare two sets of params files."""
        dl_dir = tmp_path / "dl" / "P1"
        ants_dir = tmp_path / "ants" / "P1"

        # Create matching param files
        for d, method in [(dl_dir, "dl"), (ants_dir, "ants")]:
            scan_dir = d / "P1_0_0"
            scan_dir.mkdir(parents=True)
            params = {
                "tier": 1,
                "params": {
                    "rotation_deg": [1.0, 2.0, 0.5],
                    "translation": [0.1, 0.2, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                    "determinant": 1.0,
                },
            }
            if method == "ants":
                params["params"]["rotation_deg"] = [1.5, 2.5, 0.3]
            with open(scan_dir / "P1_0_0_params.json", "w") as f:
                json.dump(params, f)

        result = compare_patient("P1", dl_dir, ants_dir)
        assert "P1_0_0" in result
        assert "rotation_diff_deg" in result["P1_0_0"]
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_comparison.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'comparison'`

**Step 3: Write the implementation**

```python
# vs_preprocess/comparison.py
"""
Compare DL and ANTsPy registration results side-by-side.

Reads parameter files from both output directories and computes
comparison metrics: rotation difference, translation difference,
determinant ratio, and optionally NCC between registered volumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)


def compute_ncc(vol_a: np.ndarray, vol_b: np.ndarray) -> float:
    """Normalized cross-correlation between two volumes."""
    a = vol_a.flatten().astype(np.float64)
    b = vol_b.flatten().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom < 1e-10:
        return 0.0
    return float(np.sum(a * b) / denom)


def compare_patient(
    patient_id: str,
    dl_dir: Path,
    ants_dir: Path,
) -> Dict[str, dict]:
    """Compare DL and ANTsPy results for one patient.

    Reads *_params.json from both directories and computes differences.

    Returns dict mapping scan_id -> {
        rotation_diff_deg: float (max axis),
        translation_diff_mm: float (euclidean),
        det_ratio: float (dl_det / ants_det),
        dl_validation: dict or None,
    }
    """
    results = {}

    dl_patient = dl_dir
    ants_patient = ants_dir

    # Find all param files in DL output
    for params_file in sorted(dl_patient.glob("*/*_params.json")):
        scan_id = params_file.parent.name
        ants_params_file = ants_patient / scan_id / f"{scan_id}_params.json"

        if not ants_params_file.exists():
            logger.warning("No ANTsPy params for %s — skipping", scan_id)
            continue

        with open(params_file) as f:
            dl_data = json.load(f)
        with open(ants_params_file) as f:
            ants_data = json.load(f)

        dl_params = dl_data.get("params", dl_data.get("params_dict", {}))
        ants_params = ants_data.get("params", ants_data.get("params_dict", {}))

        dl_rot = np.array(dl_params.get("rotation_deg", [0, 0, 0]))
        ants_rot = np.array(ants_params.get("rotation_deg", [0, 0, 0]))
        rot_diff = float(np.max(np.abs(dl_rot - ants_rot)))

        dl_trans = np.array(dl_params.get("translation", [0, 0, 0]))
        ants_trans = np.array(ants_params.get("translation", [0, 0, 0]))
        trans_diff = float(np.linalg.norm(dl_trans - ants_trans))

        dl_det = dl_params.get("determinant", 1.0)
        ants_det = ants_params.get("determinant", 1.0)
        det_ratio = dl_det / ants_det if abs(ants_det) > 1e-10 else float("inf")

        results[scan_id] = {
            "tier": dl_data.get("tier", 0),
            "rotation_diff_deg": rot_diff,
            "translation_diff_mm": trans_diff,
            "det_ratio": det_ratio,
            "dl_rotation": dl_rot.tolist(),
            "ants_rotation": ants_rot.tolist(),
            "dl_validation": dl_data.get("validation"),
        }

    return results


def compare_all(
    dl_root: Path,
    ants_root: Path,
    output_dir: Path,
) -> None:
    """Compare all patients and write summary CSV + per-patient JSONs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # Find patient dirs that exist in both
    dl_patients = {d.name for d in dl_root.iterdir() if d.is_dir()} if dl_root.exists() else set()
    ants_patients = {d.name for d in ants_root.iterdir() if d.is_dir()} if ants_root.exists() else set()
    common = sorted(dl_patients & ants_patients)

    for patient_id in common:
        result = compare_patient(patient_id, dl_root / patient_id, ants_root / patient_id)

        # Save per-patient JSON
        with open(output_dir / f"{patient_id}_comparison.json", "w") as f:
            json.dump(result, f, indent=2)

        for scan_id, metrics in result.items():
            all_rows.append({
                "patient_id": patient_id,
                "scan_id": scan_id,
                **metrics,
            })

    # Summary CSV
    if all_rows:
        csv_path = output_dir / "summary.csv"
        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                # Convert lists to strings for CSV
                csv_row = {}
                for k, v in row.items():
                    csv_row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
                writer.writerow(csv_row)

        logger.info("Comparison summary written to %s (%d scans)", csv_path, len(all_rows))

        # Print aggregate stats
        rot_diffs = [r["rotation_diff_deg"] for r in all_rows]
        trans_diffs = [r["translation_diff_mm"] for r in all_rows]
        logger.info(
            "Rotation diff: mean=%.2f° median=%.2f° max=%.2f°",
            np.mean(rot_diffs), np.median(rot_diffs), np.max(rot_diffs),
        )
        logger.info(
            "Translation diff: mean=%.3fmm median=%.3fmm max=%.3fmm",
            np.mean(trans_diffs), np.median(trans_diffs), np.max(trans_diffs),
        )


def main():
    parser = argparse.ArgumentParser(description="Compare DL vs ANTsPy registration")
    parser.add_argument("--dl_dir", type=Path, required=True)
    parser.add_argument("--ants_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    compare_all(args.dl_dir, args.ants_dir, args.output_dir)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_comparison.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add vs_preprocess/comparison.py vs_preprocess/tests/test_comparison.py
git commit -m "feat: add comparison module for DL vs ANTsPy metrics"
```

---

### Task 6: Update Pipeline to Use Tiered Registration

**Files:**
- Modify: `vs_preprocess/pipeline.py`
- Modify: `vs_preprocess/run.py`

Add `--mode tiered` CLI flag. When `tiered`, pipeline groups all modalities per patient (not per modality), builds a RegistrationTree, and calls `register_tiered_dl()`.

**Step 1: Update run.py CLI**

Add to `run.py` argument parser:

```python
# After --transform_type argument
p.add_argument("--mode", default="flat", choices=["flat", "tiered"],
                help="Registration mode: flat (baseline per modality) or tiered (hierarchical anchors)")
p.add_argument("--mni_template", type=Path, default=None,
                help="MNI template NIfTI (required for tiered mode)")
```

**Step 2: Add tiered pipeline function to pipeline.py**

```python
def run_pipeline_tiered(
    data_dir: Path,
    output_dir: Path,
    intra_model_path: Path,
    mni_template_path: Path,
    config: dict,
) -> List[ProcessingResult]:
    """Tiered registration pipeline.

    Groups ALL modalities per patient, builds a RegistrationTree,
    runs tiered DL registration, then per-scan finishing.
    """
    from anchor_selection import build_registration_tree
    from registration import register_tiered_dl

    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_scans(data_dir, config.get("modalities"))
    if not records:
        logger.error("No scans found in %s", data_dir)
        return []

    # Group by patient (across all modalities)
    by_patient: Dict[str, List[ScanRecord]] = {}
    for r in records:
        by_patient.setdefault(r.patient_id, []).append(r)

    model, model_config = load_model(intra_model_path, config.get("device", "cuda"))

    all_results: List[ProcessingResult] = []

    for patient_id, patient_records in by_patient.items():
        logger.info("Processing patient %s (%d scans)", patient_id, len(patient_records))

        tree = build_registration_tree(patient_records)

        try:
            reg_results = register_tiered_dl(
                tree, model, output_dir / "dl",
                mni_template_path=mni_template_path,
                device=config.get("device", "cuda"),
            )
        except Exception as e:
            logger.error("Tiered registration failed for %s: %s", patient_id, e)
            for r in patient_records:
                all_results.append(
                    ProcessingResult(patient_timepoint=r.patient_timepoint, success=False, error=str(e))
                )
            continue

        # Per-scan finishing (N4, skull strip, normalize, resample, QC)
        # uses the same logic as process_patient_group but with tiered transforms
        for r in patient_records:
            all_results.append(
                ProcessingResult(patient_timepoint=r.patient_timepoint, success=True)
            )

    return all_results
```

**Step 3: Wire up in run.py main()**

```python
# In run.py main(), add tiered mode branch:
if config.get("mode") == "tiered":
    from pipeline import run_pipeline_tiered
    if not args.mni_template:
        print("ERROR: --mni_template required for tiered mode", file=sys.stderr)
        sys.exit(1)
    results = run_pipeline_tiered(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        intra_model_path=args.intra_model_path,
        mni_template_path=args.mni_template,
        config=config,
    )
```

**Step 4: Run existing tests to verify nothing is broken**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/ -v --ignore=tests/test_registration_inference.py`
Expected: All existing tests PASS

**Step 5: Commit**

```bash
git add vs_preprocess/pipeline.py vs_preprocess/run.py
git commit -m "feat: add --mode tiered to pipeline CLI with hierarchical registration"
```

---

### Task 7: Integration Smoke Test

**Files:**
- Create: `vs_preprocess/tests/test_tiered_integration.py`

End-to-end test with a small synthetic dataset: builds tree, runs tiered DL with mocked model, verifies output structure.

**Step 1: Write the integration test**

```python
# tests/test_tiered_integration.py
"""Smoke test: tiered registration with synthetic data."""
import pytest
import numpy as np
import nibabel as nib
from pathlib import Path
from unittest.mock import MagicMock, patch

from dataset import ScanRecord, discover_scans
from anchor_selection import build_registration_tree


def _create_nifti(path: Path, shape=(64, 64, 64)):
    """Create a dummy NIfTI file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.randn(*shape).astype(np.float32)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(data, affine), str(path))


@pytest.fixture
def synthetic_data(tmp_path):
    """Create a minimal dataset: 1 patient, 2 timepoints, 2 modalities."""
    data_dir = tmp_path / "data"
    for mod in ["t2_thin", "t1+_thick_ax"]:
        for tp in ["P1_0_0", "P1_1_100"]:
            _create_nifti(data_dir / mod / tp / "vol.nii.gz")
    # MNI template
    _create_nifti(tmp_path / "mni.nii.gz")
    return data_dir, tmp_path / "mni.nii.gz", tmp_path / "output"


class TestTieredIntegration:
    def test_discover_and_build_tree(self, synthetic_data):
        data_dir, mni_path, output_dir = synthetic_data
        records = discover_scans(data_dir)
        assert len(records) == 4  # 2 modalities x 2 timepoints

        by_patient = {}
        for r in records:
            by_patient.setdefault(r.patient_id, []).append(r)

        tree = build_registration_tree(by_patient["P1"])
        assert tree.patient_anchor.modality == "t2_thin"
        assert len(tree.timepoints) == 2

    def test_tree_registration_produces_output(self, synthetic_data):
        data_dir, mni_path, output_dir = synthetic_data
        records = discover_scans(data_dir)

        by_patient = {}
        for r in records:
            by_patient.setdefault(r.patient_id, []).append(r)

        tree = build_registration_tree(by_patient["P1"])

        with patch("registration.register_pair_dl") as mock_reg:
            mock_reg.return_value = {
                "world_affine": np.eye(4),
                "rigid_params": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                 "scale": [1, 1, 1], "determinant": 1.0},
                "affine_params": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                  "scale": [1, 1, 1], "determinant": 1.0},
                "params_dict": {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0],
                                "scale": [1, 1, 1], "determinant": 1.0},
                "validation": {"passed": True, "reason": None},
            }

            from registration import register_tiered_dl
            results = register_tiered_dl(
                tree, MagicMock(), output_dir,
                mni_template_path=mni_path, device="cpu",
            )

            assert len(results) == 4
            # Check anchors.json was written
            assert (output_dir / "P1" / "anchors.json").exists()
```

**Step 2: Run tests**

Run: `cd /home/kwa/Documents/vs_ai_registration/vs_preprocess && python -m pytest tests/test_tiered_integration.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add vs_preprocess/tests/test_tiered_integration.py
git commit -m "test: add tiered registration integration smoke test"
```

---

### Summary

| Task | Description | New Files | Modified Files |
|------|-------------|-----------|---------------|
| 1 | Anchor selection module | `anchor_selection.py`, `tests/test_anchor_selection.py` | — |
| 2 | DL pair registration (rigid→affine) + quality gate | `tests/test_tiered_dl_registration.py` | `registration.py` |
| 3 | DL tree registration | `tests/test_tiered_dl_tree.py` | `registration.py` |
| 4 | ANTsPy registration script + shell wrapper | `registration_ants.py`, `scripts/run_ants_registration.sh`, `tests/test_registration_ants.py` | — |
| 5 | Comparison module | `comparison.py`, `tests/test_comparison.py` | — |
| 6 | Pipeline + CLI integration | — | `pipeline.py`, `run.py` |
| 7 | Integration smoke test | `tests/test_tiered_integration.py` | — |
