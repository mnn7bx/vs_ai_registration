"""Organize raw VS dataset into discover_scans()-compatible structure."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def parse_vs_folder_name(folder_name: str) -> dict:
    """Parse '{patient}_{timepoint}_{days}' folder name.

    Returns dict with keys: patient_id, timepoint, days.
    """
    parts = folder_name.split("_")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(
            f"Expected '{{patient}}_{{timepoint}}_{{days}}', got '{folder_name}'"
        )
    return {"patient_id": parts[0], "timepoint": parts[1], "days": parts[2]}


def classify_vs_files(file_paths: list[Path]) -> dict:
    """Classify VS scan files into image and segmentation.

    Image file contains '_image' in stem. Segmentation is the first
    non-image file alphabetically (deterministic selection).

    Returns dict with keys: image (Path), seg (Path | None).
    """
    if not file_paths:
        raise ValueError("No files provided")

    image = None
    segs = []
    for f in file_paths:
        if "_image" in f.stem:
            image = f
        else:
            segs.append(f)

    if image is None:
        raise ValueError(f"No image file found in {file_paths}")

    seg = sorted(segs, key=lambda p: p.name)[0] if segs else None
    return {"image": image, "seg": seg}


def organize_scan(raw_scan_dir: Path, modality: str, output_dir: Path) -> Path | None:
    """Create discovery-compatible symlinks for a single VS scan.

    Output structure:
        output_dir/<modality>/<patient>_<timepoint>/<patient>_<timepoint>.nii.gz
        output_dir/<modality>/<patient>_<timepoint>/<patient>_<timepoint>_seg.nii.gz

    Returns output directory path, or None if source is empty.
    """
    nii_files = list(raw_scan_dir.glob("*.nii.gz"))
    if not nii_files:
        return None

    folder_name = raw_scan_dir.name
    parsed = parse_vs_folder_name(folder_name)
    classified = classify_vs_files(nii_files)

    out_name = f"{parsed['patient_id']}_{parsed['timepoint']}"
    out_dir = output_dir / modality / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_link = out_dir / f"{out_name}.nii.gz"
    if not image_link.exists():
        image_link.symlink_to(classified["image"].resolve())

    if classified["seg"] is not None:
        seg_link = out_dir / f"{out_name}_seg.nii.gz"
        if not seg_link.exists():
            seg_link.symlink_to(classified["seg"].resolve())

    return out_dir


def organize_all(raw_root: Path, output_dir: Path, modalities: list[str] | None = None):
    """Organize all scans under raw_root into discovery-compatible layout.

    raw_root structure:
        raw_root/<modality>/<patient>_<timepoint>_<days>/*.nii.gz

    Args:
        raw_root: Path to unprocessed_raw_studies/
        output_dir: Where to create organized symlink tree
        modalities: If set, only process these modality dirs
    """
    available = sorted(p.name for p in raw_root.iterdir() if p.is_dir())
    to_process = modalities if modalities else available
    log.info("Modalities to process: %s", to_process)

    stats = {"organized": 0, "skipped_empty": 0, "errors": 0}
    for modality in to_process:
        mod_dir = raw_root / modality
        if not mod_dir.is_dir():
            log.warning("Modality dir not found: %s", mod_dir)
            continue
        for scan_dir in sorted(mod_dir.iterdir()):
            if not scan_dir.is_dir():
                continue
            try:
                result = organize_scan(scan_dir, modality, output_dir)
                if result is None:
                    stats["skipped_empty"] += 1
                else:
                    stats["organized"] += 1
            except (ValueError, OSError) as e:
                log.warning("Skipping %s/%s: %s", modality, scan_dir.name, e)
                stats["errors"] += 1

    log.info(
        "Done: %d organized, %d skipped (empty), %d errors",
        stats["organized"], stats["skipped_empty"], stats["errors"],
    )
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Organize raw VS dataset for discover_scans()"
    )
    parser.add_argument(
        "--raw_dir", type=Path, required=True,
        help="Path to unprocessed_raw_studies/",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output directory for organized symlink tree",
    )
    parser.add_argument(
        "--modalities", nargs="+", default=None,
        help="Only process these modalities (default: all)",
    )
    args = parser.parse_args()
    organize_all(args.raw_dir, args.output_dir, args.modalities)


if __name__ == "__main__":
    main()
