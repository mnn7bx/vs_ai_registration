"""Tests for organize_vs.py — VS dataset organizer."""
from __future__ import annotations

from pathlib import Path

import pytest
from vs_preprocess.scripts.organize_vs import (
    classify_vs_files,
    organize_scan,
    parse_vs_folder_name,
)


class TestParseVsFolderName:
    def test_baseline(self):
        result = parse_vs_folder_name("126_0_0")
        assert result == {"patient_id": "126", "timepoint": "0", "days": "0"}

    def test_followup(self):
        result = parse_vs_folder_name("126_1_312")
        assert result == {"patient_id": "126", "timepoint": "1", "days": "312"}

    def test_large_timepoint(self):
        result = parse_vs_folder_name("415_15_4200")
        assert result == {"patient_id": "415", "timepoint": "15", "days": "4200"}

    def test_invalid_format(self):
        with pytest.raises(ValueError):
            parse_vs_folder_name("badname")

    def test_too_few_parts(self):
        with pytest.raises(ValueError):
            parse_vs_folder_name("126_0")


class TestClassifyVsFiles:
    def test_image_only(self):
        files = [Path("126_0_0_t2_thin_image.nii.gz")]
        result = classify_vs_files(files)
        assert result["image"] == files[0]
        assert result["seg"] is None

    def test_image_and_seg(self):
        files = [
            Path("439_0_0_t1+_thick_ax_image.nii.gz"),
            Path("439_0_0_t1+_thick_ax_RVS_uvauser2.nii.gz"),
            Path("439_0_0_t1+_thick_ax_R_VS__uvauser2_.nii.gz"),
        ]
        result = classify_vs_files(files)
        assert result["image"] == files[0]
        assert result["seg"] is not None
        assert result["seg"] != result["image"]

    def test_no_image_raises(self):
        files = [Path("439_0_0_t1+_thick_ax_RVS_uvauser2.nii.gz")]
        with pytest.raises(ValueError, match="No image file"):
            classify_vs_files(files)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No files"):
            classify_vs_files([])

    def test_picks_first_seg_alphabetically(self):
        files = [
            Path("126_0_0_t2_thin_image.nii.gz"),
            Path("126_0_0_t2_thin_LVS_uvauser2.nii.gz"),
            Path("126_0_0_t2_thin_AS_uvauser3.nii.gz"),
        ]
        result = classify_vs_files(files)
        assert result["seg"] == files[2]


class TestOrganizeScan:
    def _make_scan_dir(self, tmp: Path, folder: str, files: list[str]) -> Path:
        scan_dir = tmp / folder
        scan_dir.mkdir(parents=True)
        for f in files:
            (scan_dir / f).write_bytes(b"\x00" * 100)
        return scan_dir

    def test_creates_symlinks(self, tmp_path):
        raw = self._make_scan_dir(
            tmp_path / "raw", "126_1_312",
            ["126_1_312_t2_thin_image.nii.gz", "126_1_312_t2_thin_LVS_uvauser2.nii.gz"],
        )
        out = tmp_path / "organized"
        result = organize_scan(raw, "t2_thin", out)

        assert result is not None
        expected_dir = out / "t2_thin" / "126_1"
        assert expected_dir.is_dir()

        image_link = expected_dir / "126_1.nii.gz"
        assert image_link.is_symlink()
        assert image_link.resolve() == (raw / "126_1_312_t2_thin_image.nii.gz").resolve()

        seg_link = expected_dir / "126_1_seg.nii.gz"
        assert seg_link.is_symlink()

    def test_skips_empty_dir(self, tmp_path):
        raw = tmp_path / "raw" / "126_0_0"
        raw.mkdir(parents=True)
        out = tmp_path / "organized"
        result = organize_scan(raw, "t2_thin", out)
        assert result is None

    def test_image_only_no_seg(self, tmp_path):
        raw = self._make_scan_dir(
            tmp_path / "raw", "200_0_0",
            ["200_0_0_t2_thin_image.nii.gz"],
        )
        out = tmp_path / "organized"
        result = organize_scan(raw, "t2_thin", out)
        expected_dir = out / "t2_thin" / "200_0"
        assert (expected_dir / "200_0.nii.gz").is_symlink()
        assert not (expected_dir / "200_0_seg.nii.gz").exists()


class TestOrganizeScanCollision:
    def test_same_patient_timepoint_different_days_skips(self, tmp_path):
        raw = tmp_path / "raw"
        for folder, fname in [
            ("126_1_312", "126_1_312_t2_thin_image.nii.gz"),
            ("126_1_400", "126_1_400_t2_thin_image.nii.gz"),
        ]:
            d = raw / folder
            d.mkdir(parents=True)
            (d / fname).write_bytes(b"\x00" * 100)

        out = tmp_path / "organized"
        r1 = organize_scan(raw / "126_1_312", "t2_thin", out)
        assert r1 is not None
        r2 = organize_scan(raw / "126_1_400", "t2_thin", out)
        assert r2 is not None
        link = out / "t2_thin" / "126_1" / "126_1.nii.gz"
        assert link.resolve() == (raw / "126_1_312" / "126_1_312_t2_thin_image.nii.gz").resolve()
