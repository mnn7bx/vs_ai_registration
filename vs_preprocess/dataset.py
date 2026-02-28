"""
Dataset discovery and patient grouping for VS preprocessing pipeline.

Walks data_dir/<modality>/<patient_timepoint>/ structure, identifies volumes
and segmentations, groups by patient, marks baselines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Suffixes that indicate a file is NOT the primary MRI volume
_EXCLUDE_SUFFIXES = re.compile(
    r"(_seg|_label|_mask|_brain|_stripped|_bet|_synthstrip|_preprocessed|_n4|_bias)",
    re.IGNORECASE,
)


@dataclass
class ScanRecord:
    """One MRI volume + optional segmentation for a single timepoint."""

    modality: str
    patient_timepoint: str
    patient_id: str
    timepoint: str
    volume_path: Path
    seg_path: Optional[Path]
    is_baseline: bool = False


def _parse_patient_timepoint(dirname: str) -> Tuple[str, str]:
    """
    Split directory name into (patient_id, timepoint).

    Strategy: split on the FIRST underscore.

    Examples:
        'VS001_tp2'       -> ('VS001', 'tp2')
        '126_0_0'         -> ('126', '0_0')
        '126_1_312'       -> ('126', '1_312')
        'VS001'           -> ('VS001', 'tp1')
    """
    idx = dirname.find("_")
    if idx == -1:
        return dirname, "tp1"
    return dirname[:idx], dirname[idx + 1:]


def _find_volume(nifti_files: List[Path]) -> Optional[Path]:
    """
    Pick the primary MRI volume from a list of NIfTI files.

    Filters out files matching _EXCLUDE_SUFFIXES, then picks the largest.
    Falls back to the largest file overall if all are excluded.
    """
    if not nifti_files:
        return None

    # Strip .nii.gz or .nii to get the stem for matching
    def stem(p: Path) -> str:
        name = p.name
        if name.endswith(".nii.gz"):
            return name[: -len(".nii.gz")]
        elif name.endswith(".nii"):
            return name[: -len(".nii")]
        return name

    candidates = [f for f in nifti_files if not _EXCLUDE_SUFFIXES.search(stem(f))]
    if not candidates:
        candidates = nifti_files  # fallback

    return max(candidates, key=lambda p: p.stat().st_size)


def _find_seg(nifti_files: List[Path], volume_path: Path) -> Optional[Path]:
    """
    Pick the expert segmentation from the same directory.

    Looks for files containing '_seg' or '_label' in the stem,
    excluding the primary volume.
    """
    seg_pattern = re.compile(r"(_seg|_label)", re.IGNORECASE)
    matches = [
        f
        for f in nifti_files
        if f != volume_path and seg_pattern.search(f.name)
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda p: p.name)[0]


def discover_scans(
    data_dir: Path,
    modalities: Optional[Sequence[str]] = None,
) -> List[ScanRecord]:
    """
    Walk data_dir and return a list of ScanRecord entries.

    Expected structure:
        data_dir/
            <modality>/          e.g., T1C, T2
                <patient_tp>/    e.g., VS001_tp1
                    *.nii.gz

    Args:
        data_dir: Root data directory.
        modalities: If provided, include only these modality names
                    (case-insensitive). None means all.

    Returns:
        List[ScanRecord] sorted by (modality, patient_id, timepoint).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    allowed_mod = None
    if modalities is not None:
        allowed_mod = {m.lower() for m in modalities}

    records: List[ScanRecord] = []

    for mod_dir in sorted(data_dir.iterdir()):
        if not mod_dir.is_dir():
            continue
        modality = mod_dir.name
        if allowed_mod is not None and modality.lower() not in allowed_mod:
            continue

        for tp_dir in sorted(mod_dir.iterdir()):
            if not tp_dir.is_dir():
                continue

            nifti_files = sorted(
                list(tp_dir.glob("*.nii.gz")) + list(tp_dir.glob("*.nii"))
            )
            if not nifti_files:
                logger.warning("No NIfTI files in %s — skipping", tp_dir)
                continue

            volume = _find_volume(nifti_files)
            if volume is None:
                logger.warning("Could not identify volume in %s — skipping", tp_dir)
                continue

            seg = _find_seg(nifti_files, volume)
            patient_id, timepoint = _parse_patient_timepoint(tp_dir.name)

            records.append(
                ScanRecord(
                    modality=modality,
                    patient_timepoint=tp_dir.name,
                    patient_id=patient_id,
                    timepoint=timepoint,
                    volume_path=volume,
                    seg_path=seg,
                )
            )

    records.sort(key=lambda r: (r.modality, r.patient_id, r.timepoint))
    logger.info("Discovered %d scans across %d directories", len(records), len(records))
    return records


def group_by_patient(
    records: List[ScanRecord],
    baseline_selection: str = "first",
) -> Dict[str, List[ScanRecord]]:
    """
    Group ScanRecords by (modality, patient_id) and mark baselines.

    Args:
        records: flat list from discover_scans.
        baseline_selection: "first" (earliest timepoint) or "latest".

    Returns:
        Dict mapping "modality/patient_id" -> sorted list of ScanRecord.
        The baseline has is_baseline=True.
    """
    groups: Dict[str, List[ScanRecord]] = {}
    for r in records:
        key = f"{r.modality}/{r.patient_id}"
        groups.setdefault(key, []).append(r)

    for key, group in groups.items():
        group.sort(key=lambda r: r.timepoint)
        # Reset all baselines
        for r in group:
            r.is_baseline = False
        # Mark baseline
        if baseline_selection == "latest":
            group[-1].is_baseline = True
        else:  # "first"
            group[0].is_baseline = True

    logger.info(
        "Grouped into %d patient-modality groups (total %d scans)",
        len(groups),
        sum(len(g) for g in groups.values()),
    )
    return groups


def discover_scans_from_csv(
    csv_path: Path,
    data_root: Path,
    modalities: Optional[Sequence[str]] = None,
) -> List[ScanRecord]:
    """
    Build ScanRecord list from the coregistered dataset CSV.

    Uses CSV columns to resolve exact image and seg paths, bypassing
    the _find_seg() pattern match (which doesn't work for R_VS_ naming).

    CSV columns used:
        'Reworked Patient ID' -> patient_id
        'Timepoint'           -> timepoint
        'Study Type'          -> modality
        'coregistered_image'  -> volume path (fallback: 'image path')
        'coregistered_segmentation' -> seg path (fallback: 'segmentation path')

    Args:
        csv_path: path to df_coregistered_*.csv
        data_root: root dir that csv relative paths are anchored to
                   (the directory containing raw_studies/)
        modalities: if provided, only include these modality names.

    Returns:
        List[ScanRecord] sorted by (modality, patient_id, timepoint).
    """
    import pandas as pd

    csv_path = Path(csv_path)
    data_root = Path(data_root)

    df = pd.read_csv(csv_path)

    allowed_mod = None
    if modalities is not None:
        allowed_mod = {m.lower() for m in modalities}

    records: List[ScanRecord] = []

    for _, row in df.iterrows():
        # Resolve modality from Study Type column (e.g. t1+_thick_ax)
        modality = str(row.get("Study Type", "")).strip()
        if not modality:
            modality = str(row.get("Series Description", "")).strip()
        if not modality:
            logger.debug("Skipping row with empty modality: %s", row.get("Patient_MRI_Days Tracker"))
            continue

        if allowed_mod is not None and modality.lower() not in allowed_mod:
            continue

        patient_id = str(int(row["Reworked Patient ID"]))
        timepoint = str(row["Timepoint"]).strip()
        patient_timepoint = f"{patient_id}_{timepoint}"

        # Resolve image path: prefer coregistered_image (what's on disk)
        img_col = str(row.get("coregistered_image", "")).strip()
        if not img_col:
            img_col = str(row.get("image path", "")).strip()
        if not img_col:
            logger.warning("No image path for %s — skipping", patient_timepoint)
            continue

        volume_path = data_root / img_col
        if not volume_path.exists():
            # Try raw_studies relative path from CSV
            alt = str(row.get("image path", "")).strip()
            if alt:
                volume_path = data_root / alt
        if not volume_path.exists():
            logger.warning("Image not found: %s — skipping", volume_path)
            continue

        # Resolve seg path
        seg_col = str(row.get("coregistered_segmentation", "")).strip()
        if not seg_col:
            seg_col = str(row.get("segmentation path", "")).strip()

        seg_path: Optional[Path] = None
        if seg_col:
            candidate = data_root / seg_col
            if candidate.exists():
                seg_path = candidate

        records.append(
            ScanRecord(
                modality=modality,
                patient_timepoint=patient_timepoint,
                patient_id=patient_id,
                timepoint=timepoint,
                volume_path=volume_path,
                seg_path=seg_path,
            )
        )

    records.sort(key=lambda r: (r.modality, r.patient_id, r.timepoint))
    logger.info(
        "CSV: discovered %d scans for %d patient-timepoints",
        len(records),
        len(set(r.patient_id for r in records)),
    )
    return records
