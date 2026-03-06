"""Organize raw VS dataset into discover_scans()-compatible structure."""
from __future__ import annotations

from pathlib import Path


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
