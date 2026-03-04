import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from dataset import _parse_patient_timepoint, group_by_patient, ScanRecord


def test_parse_patient_timepoint_first_underscore():
    """First-underscore split: 126_0_0 -> (126, 0_0)."""
    pid, tp = _parse_patient_timepoint("126_0_0")
    assert pid == "126"
    assert tp == "0_0"


def test_parse_no_underscore():
    pid, tp = _parse_patient_timepoint("VS001")
    assert pid == "VS001"
    assert tp == "tp1"


def test_parse_single_underscore():
    pid, tp = _parse_patient_timepoint("VS001_baseline")
    assert pid == "VS001"
    assert tp == "baseline"


def test_group_by_patient_marks_baseline():
    """First record in sorted timepoints should be baseline."""
    records = [
        ScanRecord("T2", "p1_tp1", "p1", "tp1", Path("/a.nii.gz"), None),
        ScanRecord("T2", "p1_tp2", "p1", "tp2", Path("/b.nii.gz"), None),
    ]
    groups = group_by_patient(records)
    assert "T2/p1" in groups
    group = groups["T2/p1"]
    baseline = next(r for r in group if r.is_baseline)
    assert baseline.timepoint == "tp1"


def test_group_by_patient_single_record():
    """Single-timepoint patient: baseline flag set, no crash."""
    records = [
        ScanRecord("T1", "p2_tp1", "p2", "tp1", Path("/c.nii.gz"), None),
    ]
    groups = group_by_patient(records)
    assert "T1/p2" in groups
    assert len(groups["T1/p2"]) == 1
    assert groups["T1/p2"][0].is_baseline is True


def test_group_by_patient_multiple_modalities():
    """Records from different modalities go into different groups."""
    records = [
        ScanRecord("T1", "p1_tp1", "p1", "tp1", Path("/a.nii.gz"), None),
        ScanRecord("T2", "p1_tp1", "p1", "tp1", Path("/b.nii.gz"), None),
    ]
    groups = group_by_patient(records)
    assert "T1/p1" in groups
    assert "T2/p1" in groups
    assert groups["T1/p1"][0].modality == "T1"
    assert groups["T2/p1"][0].modality == "T2"
