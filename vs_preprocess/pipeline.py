"""
Pipeline orchestration: per-patient processing, failure logging, timing.
"""

from __future__ import annotations

import csv
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import nibabel as nib

from dataset import ScanRecord, discover_scans, group_by_patient
from registration import load_model, align_patient_longitudinal
from skull_strip import extract_brain_mask
from preprocessing import (
    run_n4_correction,
    normalize_intensity,
    resample_to_isotropic,
    resample_mask_to_isotropic,
)
from affine_utils import apply_affine_to_volume
from qc import generate_scan_qc, generate_longitudinal_qc, generate_qc_summary_html

logger = logging.getLogger(__name__)


@dataclass
class StageTimings:
    n4: float = 0.0
    brain_mask: float = 0.0
    alignment: float = 0.0
    normalization: float = 0.0
    resampling: float = 0.0
    seg_passthrough: float = 0.0
    qc: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.n4 + self.brain_mask + self.alignment
            + self.normalization + self.resampling
            + self.seg_passthrough + self.qc
        )


@dataclass
class ProcessingResult:
    patient_timepoint: str
    success: bool
    timings: StageTimings = field(default_factory=StageTimings)
    error: Optional[str] = None
    skipped: bool = False


def _output_exists(record: ScanRecord, output_dir: Path) -> bool:
    """Check if final preprocessed output already exists."""
    out_path = (
        output_dir / record.modality / record.patient_id
        / record.patient_timepoint / f"{record.patient_timepoint}_preprocessed.nii.gz"
    )
    return out_path.exists()


def process_patient_group(
    group_key: str,
    records: List[ScanRecord],
    model,
    model_config: dict,
    output_dir: Path,
    config: dict,
) -> List[ProcessingResult]:
    """
    Process all timepoints of one patient for one modality.

    Args:
        group_key: "T2/VS001" (for logging).
        records: sorted list of ScanRecords (first = baseline).
        model: trained GlobalNet on correct device.
        model_config: model configuration dict.
        output_dir: root output directory.
        config: pipeline settings from CLI args.

    Returns:
        list of ProcessingResult, one per timepoint.
    """
    logger.info("Processing %s (%d timepoints)", group_key, len(records))
    all_results: List[ProcessingResult] = []
    timings_map: Dict[str, StageTimings] = {}

    baseline = records[0]
    baseline_patient_dir = output_dir / baseline.modality / baseline.patient_id

    # ── Check skip ──
    if config.get("skip_existing", True) and not config.get("force_reprocess", False):
        all_exist = all(_output_exists(r, output_dir) for r in records)
        if all_exist:
            logger.info("  All outputs exist — skipping %s", group_key)
            for r in records:
                all_results.append(
                    ProcessingResult(patient_timepoint=r.patient_timepoint, success=True, skipped=True)
                )
            return all_results

    # ── Stage: N4 on all timepoints ──
    n4_paths: Dict[str, Path] = {}
    for record in records:
        timings = StageTimings()
        timings_map[record.timepoint] = timings
        out_dir = output_dir / record.modality / record.patient_id / record.patient_timepoint

        inter_dir = out_dir / "intermediates"
        if config.get("keep_intermediates", True):
            inter_dir.mkdir(parents=True, exist_ok=True)
        else:
            inter_dir = out_dir  # save alongside final outputs
            inter_dir.mkdir(parents=True, exist_ok=True)

        n4_path = inter_dir / f"{record.patient_timepoint}_n4.nii.gz"
        bias_path = inter_dir / f"{record.patient_timepoint}_bias_field.nii.gz"

        t0 = time.perf_counter()
        try:
            if config.get("no_n4", False):
                shutil.copy2(str(record.volume_path), str(n4_path))
            else:
                run_n4_correction(
                    record.volume_path,
                    n4_path,
                    bias_field_path=bias_path if config.get("keep_intermediates", True) else None,
                )
        except Exception as e:
            logger.error("N4 failed for %s: %s", record.patient_timepoint, e)
            all_results.append(
                ProcessingResult(
                    patient_timepoint=record.patient_timepoint,
                    success=False,
                    error=f"N4: {e}",
                )
            )
            continue
        timings.n4 = time.perf_counter() - t0
        n4_paths[record.timepoint] = n4_path
        logger.debug("  N4 %s: %.1fs", record.patient_timepoint, timings.n4)

    # ── Stage: Brain mask on baseline ──
    baseline_n4 = n4_paths.get(baseline.timepoint)
    if baseline_n4 is None:
        logger.error("Baseline N4 failed — cannot proceed for %s", group_key)
        return all_results

    bl_out_dir = output_dir / baseline.modality / baseline.patient_id / baseline.patient_timepoint
    bl_inter = bl_out_dir / "intermediates" if config.get("keep_intermediates", True) else bl_out_dir
    native_mask_path = bl_inter / f"{baseline.patient_timepoint}_brain_mask_native.nii.gz"

    t0 = time.perf_counter()
    try:
        extract_brain_mask(
            baseline_n4,
            native_mask_path,
            device=config.get("device", "cuda"),
            mode="accurate",
            dilation_voxels=2,
        )
    except Exception as e:
        logger.error("HD-BET failed for %s: %s", baseline.patient_timepoint, e)
        for r in records:
            all_results.append(
                ProcessingResult(patient_timepoint=r.patient_timepoint, success=False, error=f"HD-BET: {e}")
            )
        return all_results
    timings_map[baseline.timepoint].brain_mask = time.perf_counter() - t0
    logger.debug("  HD-BET %s: %.1fs", baseline.patient_timepoint, timings_map[baseline.timepoint].brain_mask)

    brain_mask_nii = nib.load(str(native_mask_path))

    # ── Stage: Intra-patient alignment ──
    alignment_results = {}
    if len(records) > 1:
        t0 = time.perf_counter()
        try:
            # Rebuild records list using N4 paths instead of original paths
            # so the alignment network sees bias-corrected input
            from dataclasses import replace
            n4_records = []
            for r in records:
                if r.timepoint in n4_paths:
                    n4_records.append(replace(r, volume_path=n4_paths[r.timepoint]))
            alignment_results = align_patient_longitudinal(
                n4_records,
                model,
                output_dir,
                transform_type=config.get("transform_type", "affine"),
                keep_intermediates=config.get("keep_intermediates", True),
                device=config.get("device", "cuda"),
            )
        except Exception as e:
            logger.error("Alignment failed for %s: %s", group_key, e)
            for r in records:
                all_results.append(
                    ProcessingResult(patient_timepoint=r.patient_timepoint, success=False, error=f"Align: {e}")
                )
            return all_results
        align_time = time.perf_counter() - t0
        # Distribute alignment time equally across follow-ups (rough)
        n_followup = len(records) - 1
        for r in records[1:]:
            timings_map[r.timepoint].alignment = align_time / n_followup
        logger.debug("  Alignment: %.1fs total", align_time)

    # ── Stage: Per-timepoint finishing (normalize, resample, seg, QC) ──
    for record in records:
        if record.timepoint not in n4_paths:
            continue  # N4 failed for this timepoint

        timings = timings_map[record.timepoint]
        out_dir = output_dir / record.modality / record.patient_id / record.patient_timepoint
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Determine the volume to normalize
            # For baseline: use N4-corrected volume directly
            # For follow-ups: use the affine-aligned volume
            if record.is_baseline or record.timepoint not in alignment_results:
                vol_nii = nib.load(str(n4_paths[record.timepoint]))
            else:
                warped = alignment_results[record.timepoint].get("warped_nii")
                if warped is not None:
                    vol_nii = warped
                else:
                    vol_nii = nib.load(str(n4_paths[record.timepoint]))

            # After alignment, all timepoints are in baseline space,
            # so the baseline brain mask applies to all of them directly.
            mask_data = brain_mask_nii.get_fdata(dtype=np.float32)

            # Handle shape mismatch (aligned volume is in baseline grid already)
            vol_data = vol_nii.get_fdata(dtype=np.float32)
            if vol_data.shape != mask_data.shape:
                # For baseline, shapes should match. For follow-ups aligned to
                # baseline, they should also match. If not, use the volume shape
                # and resize mask.
                logger.warning(
                    "Shape mismatch for %s: vol=%s mask=%s — using vol shape",
                    record.patient_timepoint, vol_data.shape, mask_data.shape
                )
                from scipy.ndimage import zoom
                zoom_factors = [v / m for v, m in zip(vol_data.shape[:3], mask_data.shape[:3])]
                mask_data = zoom(mask_data, zoom_factors, order=0) > 0.5
                mask_data = mask_data.astype(np.float32)

            # ── Intensity normalization ──
            t0 = time.perf_counter()
            normalized = normalize_intensity(
                vol_data,
                mask_data,
                method=config.get("intensity_norm", "minmax"),
                percentile_low=config.get("percentile_low", 1.0),
                percentile_high=config.get("percentile_high", 99.0),
            )
            norm_nii = nib.Nifti1Image(normalized, vol_nii.affine, vol_nii.header)
            timings.normalization = time.perf_counter() - t0

            if config.get("keep_intermediates", True):
                inter_dir = out_dir / "intermediates"
                inter_dir.mkdir(exist_ok=True)
                nib.save(norm_nii, str(inter_dir / f"{record.patient_timepoint}_normalized.nii.gz"))

            # ── Resample to isotropic ──
            t0 = time.perf_counter()
            target_spacing = tuple(config.get("target_spacing", [0.5, 0.5, 0.5]))
            final_vol = resample_to_isotropic(norm_nii, target_spacing, interpolation="bspline")
            # Also resample the mask
            mask_nii_for_resample = nib.Nifti1Image(mask_data.astype(np.float32), vol_nii.affine)
            final_mask = resample_mask_to_isotropic(mask_nii_for_resample, target_spacing)
            timings.resampling = time.perf_counter() - t0

            # ── Save final outputs ──
            nib.save(final_vol, str(out_dir / f"{record.patient_timepoint}_preprocessed.nii.gz"))
            nib.save(final_mask, str(out_dir / f"{record.patient_timepoint}_brain_mask.nii.gz"))

            # ── Seg passthrough ──
            if record.seg_path is not None and record.seg_path.exists():
                t0 = time.perf_counter()
                seg_nii = nib.load(str(record.seg_path))

                # Apply same affine to seg (nearest-neighbor) if not baseline
                if not record.is_baseline and record.timepoint in alignment_results:
                    world_affine = alignment_results[record.timepoint].get("world_affine")
                    if world_affine is not None:
                        baseline_nii = nib.load(str(n4_paths[baseline.timepoint]))
                        seg_nii = apply_affine_to_volume(
                            seg_nii, world_affine, baseline_nii, interpolation="nearest"
                        )

                final_seg = resample_to_isotropic(seg_nii, target_spacing, interpolation="nearest")
                nib.save(final_seg, str(out_dir / f"{record.patient_timepoint}_seg.nii.gz"))
                timings.seg_passthrough = time.perf_counter() - t0

            # ── QC ──
            t0 = time.perf_counter()
            seg_out = out_dir / f"{record.patient_timepoint}_seg.nii.gz"
            generate_scan_qc(
                volume_path=out_dir / f"{record.patient_timepoint}_preprocessed.nii.gz",
                mask_path=out_dir / f"{record.patient_timepoint}_brain_mask.nii.gz",
                output_path=out_dir / f"{record.patient_timepoint}_qc.png",
                seg_path=seg_out if seg_out.exists() else None,
                title=f"{record.patient_timepoint} ({record.modality})",
            )
            timings.qc = time.perf_counter() - t0

            all_results.append(
                ProcessingResult(
                    patient_timepoint=record.patient_timepoint,
                    success=True,
                    timings=timings,
                )
            )
            logger.info(
                "  %s done (N4=%.1fs mask=%.1fs align=%.1fs norm=%.1fs resamp=%.1fs qc=%.1fs total=%.1fs)",
                record.patient_timepoint,
                timings.n4, timings.brain_mask, timings.alignment,
                timings.normalization, timings.resampling, timings.qc, timings.total,
            )

        except Exception as e:
            logger.error("Failed processing %s: %s", record.patient_timepoint, e, exc_info=True)
            all_results.append(
                ProcessingResult(
                    patient_timepoint=record.patient_timepoint,
                    success=False,
                    error=str(e),
                )
            )

    # ── Longitudinal QC ──
    if len(records) > 1:
        try:
            tp_data = []
            for r in records:
                fvol = output_dir / r.modality / r.patient_id / r.patient_timepoint / f"{r.patient_timepoint}_preprocessed.nii.gz"
                fseg = output_dir / r.modality / r.patient_id / r.patient_timepoint / f"{r.patient_timepoint}_seg.nii.gz"
                pjson = output_dir / r.modality / r.patient_id / r.patient_timepoint / f"{r.patient_timepoint}_params.json"

                if not fvol.exists():
                    continue

                import json
                params_dict = {"rotation_deg": [0, 0, 0], "translation": [0, 0, 0]}
                if pjson.exists():
                    with open(pjson) as f:
                        params_dict = json.load(f)

                tp_data.append({
                    "timepoint": r.patient_timepoint,
                    "volume_path": fvol,
                    "seg_path": fseg if fseg.exists() else None,
                    "params_dict": params_dict,
                })

            if len(tp_data) >= 2:
                long_qc_path = baseline_patient_dir / f"{baseline.patient_id}_longitudinal_qc.png"
                generate_longitudinal_qc(baseline.patient_id, tp_data, long_qc_path)
        except Exception as e:
            logger.warning("Longitudinal QC failed for %s: %s", group_key, e)

    return all_results


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    intra_model_path: Path,
    config: dict,
) -> List[ProcessingResult]:
    """
    Main pipeline entry point.

    1. Discover scans and group by patient.
    2. Load model.
    3. Process each patient group sequentially.
    4. Write failures.csv and qc_summary.html.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover
    records = discover_scans(data_dir, config.get("modalities"))
    if not records:
        logger.error("No scans found in %s", data_dir)
        return []

    groups = group_by_patient(records, config.get("baseline_selection", "first"))

    # Dry run
    if config.get("dry_run", False):
        for key, group in groups.items():
            tps = ", ".join(r.patient_timepoint for r in group)
            bl = next(r for r in group if r.is_baseline)
            logger.info("  %s: baseline=%s, timepoints=[%s]", key, bl.patient_timepoint, tps)
        logger.info("Dry run complete. %d groups, %d total scans.", len(groups), len(records))
        return []

    # Load model
    model, model_config = load_model(intra_model_path, config.get("device", "cuda"))

    # Process
    all_results: List[ProcessingResult] = []
    patient_meta: Dict[str, dict] = {}

    for group_key, group_records in groups.items():
        results = process_patient_group(
            group_key, group_records, model, model_config, output_dir, config
        )
        all_results.extend(results)
        patient_meta[group_key] = {"records": group_records, "results": results}

    # Failures CSV
    failures = [r for r in all_results if not r.success and not r.skipped]
    if failures:
        csv_path = output_dir / "failures.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_timepoint", "error"])
            for r in failures:
                writer.writerow([r.patient_timepoint, r.error or "unknown"])
        logger.warning("%d failures written to %s", len(failures), csv_path)

    # QC summary HTML
    try:
        generate_qc_summary_html(output_dir, patient_meta)
    except Exception as e:
        logger.warning("QC summary HTML generation failed: %s", e)

    # Summary
    n_success = sum(r.success and not r.skipped for r in all_results)
    n_skip = sum(r.skipped for r in all_results)
    n_fail = len(failures)
    logger.info(
        "Pipeline complete: %d success, %d skipped, %d failed",
        n_success, n_skip, n_fail,
    )
    return all_results


def run_pipeline_from_csv(
    csv_path: Path,
    data_root: Path,
    output_dir: Path,
    intra_model_path: Path,
    config: dict,
) -> List[ProcessingResult]:
    """
    run_pipeline variant that loads data from the coregistered CSV.

    All other arguments and behavior identical to run_pipeline().
    """
    from dataset import discover_scans_from_csv, group_by_patient

    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_scans_from_csv(csv_path, data_root, config.get("modalities"))
    if not records:
        logger.error("No scans found from CSV: %s", csv_path)
        return []

    groups = group_by_patient(records, config.get("baseline_selection", "first"))

    if config.get("dry_run", False):
        for key, group in groups.items():
            tps = ", ".join(r.patient_timepoint for r in group)
            bl = next(r for r in group if r.is_baseline)
            logger.info("  %s: baseline=%s, timepoints=[%s]", key, bl.patient_timepoint, tps)
        logger.info("Dry run: %d groups, %d scans.", len(groups), len(records))
        return []

    model, model_config = load_model(intra_model_path, config.get("device", "cuda"))

    all_results: List[ProcessingResult] = []
    patient_meta: Dict[str, dict] = {}

    for group_key, group_records in groups.items():
        results = process_patient_group(
            group_key, group_records, model, model_config, output_dir, config
        )
        all_results.extend(results)
        patient_meta[group_key] = {"records": group_records, "results": results}

    failures = [r for r in all_results if not r.success and not r.skipped]
    if failures:
        csv_path_out = output_dir / "failures.csv"
        with open(csv_path_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_timepoint", "error"])
            for r in failures:
                writer.writerow([r.patient_timepoint, r.error or "unknown"])
        logger.warning("%d failures written to %s", len(failures), csv_path_out)

    try:
        generate_qc_summary_html(output_dir, patient_meta)
    except Exception as e:
        logger.warning("QC summary failed: %s", e)

    n_success = sum(r.success and not r.skipped for r in all_results)
    n_skip = sum(r.skipped for r in all_results)
    logger.info("Pipeline complete: %d success, %d skipped, %d failed",
                n_success, n_skip, len(failures))
    return all_results
