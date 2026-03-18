# tests/test_super_modality.py
import pytest
from pathlib import Path

from dataset import get_super_modality, SUPER_MODALITY_MAP, ScanRecord, group_by_patient_timepoint


# --- Task 1: Super-modality mapping tests ---

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


# --- Task 2: group_by_patient_timepoint tests ---

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
