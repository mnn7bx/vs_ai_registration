"""
Pipeline unit tests.
Run: python -m pytest tests/test_pipeline.py -v

Heavy dependencies (HD-BET, GlobalNet, QC renderers) are patched out so these
tests run on CPU without any trained model or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np
import pytest

# Make vs_preprocess importable from the tests/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline import (
    ProcessingResult,
    StageTimings,
    _output_exists,
    process_patient_group,
)
from dataset import ScanRecord


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_nifti(path: Path, shape=(20, 20, 20), voxel_mm=1.0) -> Path:
    """Write a small float32 NIfTI to *path* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(*shape).astype(np.float32)
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def _make_mask(path: Path, shape=(20, 20, 20)) -> Path:
    """Write a binary brain mask NIfTI to *path* and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _scan_record(tmp_path: Path, modality="T2", patient_id="VS001", timepoint="tp1"):
    """Create a ScanRecord pointing at a real NIfTI file on disk."""
    vol = tmp_path / modality / f"{patient_id}_{timepoint}" / f"{patient_id}_{timepoint}.nii.gz"
    _make_nifti(vol)
    return ScanRecord(
        modality=modality,
        patient_timepoint=f"{patient_id}_{timepoint}",
        patient_id=patient_id,
        timepoint=timepoint,
        volume_path=vol,
        seg_path=None,
        is_baseline=True,
    )


def _default_config(**overrides) -> dict:
    cfg = {
        "skip_existing": True,
        "force_reprocess": False,
        "keep_intermediates": True,
        "no_n4": False,
        "device": "cpu",
        "transform_type": "affine",
        "intensity_norm": "minmax",
        "percentile_low": 1.0,
        "percentile_high": 99.0,
        "target_spacing": [1.0, 1.0, 1.0],  # coarser spacing keeps tests fast
    }
    cfg.update(overrides)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# StageTimings
# ─────────────────────────────────────────────────────────────────────────────

class TestStageTimings:
    def test_total_sums_all_fields(self):
        t = StageTimings(n4=1.0, brain_mask=2.0, alignment=3.0,
                         normalization=0.5, resampling=0.5,
                         seg_passthrough=0.1, qc=0.1)
        assert t.total == pytest.approx(7.2)

    def test_default_total_is_zero(self):
        assert StageTimings().total == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# _output_exists
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputExists:
    def test_returns_false_when_file_absent(self, tmp_path):
        record = ScanRecord("T2", "VS001_tp1", "VS001", "tp1",
                            Path("/data/vol.nii.gz"), None, is_baseline=True)
        assert _output_exists(record, tmp_path) is False

    def test_returns_true_when_preprocessed_file_present(self, tmp_path):
        record = ScanRecord("T2", "VS001_tp1", "VS001", "tp1",
                            Path("/data/vol.nii.gz"), None, is_baseline=True)
        out = tmp_path / "T2" / "VS001" / "VS001_tp1" / "VS001_tp1_preprocessed.nii.gz"
        _make_nifti(out)
        assert _output_exists(record, tmp_path) is True


# ─────────────────────────────────────────────────────────────────────────────
# ProcessingResult
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessingResult:
    def test_success_result_fields(self):
        r = ProcessingResult(patient_timepoint="VS001_tp1", success=True)
        assert r.success is True
        assert r.skipped is False
        assert r.error is None

    def test_failure_result_stores_error(self):
        r = ProcessingResult(patient_timepoint="VS001_tp1", success=False, error="N4: boom")
        assert r.success is False
        assert "N4" in r.error

    def test_skipped_result(self):
        r = ProcessingResult(patient_timepoint="VS001_tp1", success=True, skipped=True)
        assert r.skipped is True


# ─────────────────────────────────────────────────────────────────────────────
# process_patient_group — skip logic
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessPatientGroupSkip:
    def test_skips_group_when_all_outputs_exist(self, tmp_path):
        """All outputs present → returns skipped results without calling N4."""
        record = _scan_record(tmp_path)
        # Pre-create the output file that _output_exists checks for
        out = tmp_path / "T2" / "VS001" / "VS001_tp1" / "VS001_tp1_preprocessed.nii.gz"
        _make_nifti(out)

        with patch("pipeline.run_n4_correction") as mock_n4:
            results = process_patient_group(
                "T2/VS001", [record], model=None, model_config={},
                output_dir=tmp_path, config=_default_config(),
            )

        assert len(results) == 1
        assert results[0].skipped is True
        assert results[0].success is True
        mock_n4.assert_not_called()

    def test_force_reprocess_overrides_skip(self, tmp_path):
        """force_reprocess=True must bypass the skip-existing check."""
        record = _scan_record(tmp_path)
        out = tmp_path / "T2" / "VS001" / "VS001_tp1" / "VS001_tp1_preprocessed.nii.gz"
        _make_nifti(out)

        cfg = _default_config(force_reprocess=True)

        # N4 is called when we don't skip; patch heavy deps to avoid real work
        with (
            patch("pipeline.run_n4_correction", side_effect=RuntimeError("forced")),
        ):
            results = process_patient_group(
                "T2/VS001", [record], model=None, model_config={},
                output_dir=tmp_path, config=cfg,
            )

        # N4 raised, so result should be a failure — not a skip
        assert any(not r.skipped for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# process_patient_group — N4 failure
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessPatientGroupN4Failure:
    def test_n4_failure_recorded_as_failure_result(self, tmp_path):
        record = _scan_record(tmp_path)

        with patch("pipeline.run_n4_correction", side_effect=RuntimeError("N4 exploded")):
            results = process_patient_group(
                "T2/VS001", [record], model=None, model_config={},
                output_dir=tmp_path, config=_default_config(),
            )

        assert len(results) == 1
        assert results[0].success is False
        assert "N4" in results[0].error

    def test_baseline_n4_failure_aborts_whole_group(self, tmp_path):
        """If the baseline N4 fails, follow-ups cannot proceed either."""
        from dataclasses import replace

        baseline = _scan_record(tmp_path, timepoint="tp1")
        baseline = replace(baseline, is_baseline=True)
        followup = _scan_record(tmp_path, timepoint="tp2")
        followup = replace(followup, is_baseline=False)

        with patch("pipeline.run_n4_correction", side_effect=RuntimeError("disk full")):
            results = process_patient_group(
                "T2/VS001", [baseline, followup], model=None, model_config={},
                output_dir=tmp_path, config=_default_config(),
            )

        # Both timepoints reported as failures (baseline fails, group aborts)
        assert all(not r.success for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# process_patient_group — single-baseline, end-to-end with mocked GPU stages
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessPatientGroupSingleBaseline:
    """
    End-to-end for a single-timepoint patient.
    N4 and HD-BET are patched; normalization + resampling run for real.
    """

    def _run_single_baseline(self, tmp_path):
        record = _scan_record(tmp_path)

        def fake_n4(input_path, output_path, **kwargs):
            """Copy input to output so downstream stages have a real file."""
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(input_path), str(output_path))

        def fake_hdbet(input_path, output_path, **kwargs):
            """Write a small binary mask that matches the input volume."""
            vol = nib.load(str(input_path))
            mask = np.ones(vol.shape[:3], dtype=np.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(mask, vol.affine), str(output_path))

        with (
            patch("pipeline.run_n4_correction", side_effect=fake_n4),
            patch("pipeline.extract_brain_mask", side_effect=fake_hdbet),
            patch("pipeline.generate_scan_qc"),
            patch("pipeline.generate_longitudinal_qc"),
        ):
            results = process_patient_group(
                "T2/VS001", [record], model=None, model_config={},
                output_dir=tmp_path, config=_default_config(),
            )

        return results

    def test_single_baseline_succeeds(self, tmp_path):
        results = self._run_single_baseline(tmp_path)
        assert len(results) == 1
        assert results[0].success is True
        assert not results[0].skipped

    def test_single_baseline_writes_preprocessed_file(self, tmp_path):
        self._run_single_baseline(tmp_path)
        out = tmp_path / "T2" / "VS001" / "VS001_tp1" / "VS001_tp1_preprocessed.nii.gz"
        assert out.exists(), f"Expected output not found: {out}"

    def test_single_baseline_writes_brain_mask(self, tmp_path):
        self._run_single_baseline(tmp_path)
        mask = tmp_path / "T2" / "VS001" / "VS001_tp1" / "VS001_tp1_brain_mask.nii.gz"
        assert mask.exists(), f"Expected brain mask not found: {mask}"

    def test_single_baseline_no_alignment_called(self, tmp_path):
        """Single timepoint → align_patient_longitudinal must not be called."""
        record = _scan_record(tmp_path)

        def fake_n4(input_path, output_path, **kwargs):
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(input_path), str(output_path))

        def fake_hdbet(input_path, output_path, **kwargs):
            vol = nib.load(str(input_path))
            mask = np.ones(vol.shape[:3], dtype=np.float32)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(mask, vol.affine), str(output_path))

        with (
            patch("pipeline.run_n4_correction", side_effect=fake_n4),
            patch("pipeline.extract_brain_mask", side_effect=fake_hdbet),
            patch("pipeline.generate_scan_qc"),
            patch("pipeline.align_patient_longitudinal") as mock_align,
        ):
            process_patient_group(
                "T2/VS001", [record], model=None, model_config={},
                output_dir=tmp_path, config=_default_config(),
            )

        mock_align.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# process_patient_group — HD-BET failure
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessPatientGroupHDBETFailure:
    def test_hdbet_failure_aborts_group(self, tmp_path):
        record = _scan_record(tmp_path)

        def fake_n4(input_path, output_path, **kwargs):
            import shutil
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(input_path), str(output_path))

        with (
            patch("pipeline.run_n4_correction", side_effect=fake_n4),
            patch("pipeline.extract_brain_mask", side_effect=RuntimeError("CUDA OOM")),
        ):
            results = process_patient_group(
                "T2/VS001", [record], model=None, model_config={},
                output_dir=tmp_path, config=_default_config(),
            )

        assert all(not r.success for r in results)
        assert any("HD-BET" in (r.error or "") for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline — dry run
# ─────────────────────────────────────────────────────────────────────────────

class TestRunPipelineDryRun:
    def test_dry_run_returns_empty_list(self, tmp_path):
        """Dry run must not process anything and return []."""
        from pipeline import run_pipeline

        data_dir = tmp_path / "data"
        # Build a minimal scan directory: data/T2/VS001_tp1/vol.nii.gz
        vol = data_dir / "T2" / "VS001_tp1" / "VS001_tp1.nii.gz"
        _make_nifti(vol)

        cfg = _default_config(dry_run=True)

        with patch("pipeline.load_model") as mock_load:
            results = run_pipeline(
                data_dir=data_dir,
                output_dir=tmp_path / "out",
                intra_model_path=Path("/fake/model.pt"),
                config=cfg,
            )

        assert results == []
        mock_load.assert_not_called()

    def test_dry_run_from_csv_returns_empty_list(self, tmp_path):
        """CSV-based dry run must also return []."""
        import csv as csv_mod
        from pipeline import run_pipeline_from_csv

        data_root = tmp_path / "data"
        vol = data_root / "T2" / "VS001_tp1" / "VS001_tp1.nii.gz"
        _make_nifti(vol)

        csv_file = tmp_path / "scans.csv"
        with open(csv_file, "w", newline="") as f:
            w = csv_mod.writer(f)
            w.writerow(["modality", "patient_timepoint", "volume_path", "seg_path"])
            w.writerow(["T2", "VS001_tp1", str(vol), ""])

        cfg = _default_config(dry_run=True)

        with patch("pipeline.load_model") as mock_load:
            results = run_pipeline_from_csv(
                csv_path=csv_file,
                data_root=data_root,
                output_dir=tmp_path / "out",
                intra_model_path=Path("/fake/model.pt"),
                config=cfg,
            )

        assert results == []
        mock_load.assert_not_called()
