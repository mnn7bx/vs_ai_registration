"""
QC visualization: per-scan 3-panel PNG, longitudinal comparison, HTML summary.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import nibabel as nib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def generate_scan_qc(
    volume_path: Path,
    mask_path: Path,
    output_path: Path,
    seg_path: Optional[Path] = None,
    title: str = "",
) -> None:
    """
    3-panel QC PNG (axial, sagittal, coronal) with mask and seg contour overlays.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vol = nib.load(str(volume_path)).get_fdata(dtype=np.float32)
    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32) if seg_path and seg_path.exists() else None

    D, H, W = vol.shape[:3]
    slices_info = [
        ("Axial", vol[:, :, W // 2].T, mask[:, :, W // 2].T, seg[:, :, W // 2].T if seg is not None else None),
        ("Sagittal", vol[D // 2, :, :].T, mask[D // 2, :, :].T, seg[D // 2, :, :].T if seg is not None else None),
        ("Coronal", vol[:, H // 2, :].T, mask[:, H // 2, :].T, seg[:, H // 2, :].T if seg is not None else None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100, facecolor="black")
    for ax, (name, vol_slice, mask_slice, seg_slice) in zip(axes, slices_info):
        ax.imshow(vol_slice, cmap="gray", aspect="equal")
        if mask_slice.size > 0 and mask_slice.max() > 0:
            ax.contour(mask_slice, levels=[0.5], colors=["red"], linewidths=0.8)
        if seg_slice is not None and seg_slice.size > 0 and seg_slice.max() > 0:
            ax.contour(seg_slice, levels=[0.5], colors=["lime"], linewidths=1.0)
        ax.set_title(name, fontsize=10, color="white")
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=11, color="white")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    logger.debug("QC saved: %s", output_path)


def generate_longitudinal_qc(
    patient_id: str,
    timepoint_data: List[dict],
    output_path: Path,
) -> None:
    """
    Per-patient longitudinal comparison image.

    Args:
        patient_id: for the title.
        timepoint_data: list of dicts with keys:
            'timepoint': str
            'volume_path': Path (preprocessed, aligned)
            'seg_path': Optional[Path]
            'params_dict': dict from decompose_affine
        output_path: output PNG path.

    Top row: axial center slices side-by-side.
    Bottom row: difference images (tp_n - tp1) with RdBu_r colormap.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    N = len(timepoint_data)
    if N < 2:
        logger.debug("Skipping longitudinal QC for %s (only %d timepoint)", patient_id, N)
        return

    fig, axes = plt.subplots(2, N, figsize=(4 * N, 8), dpi=100, facecolor="black")
    if N == 1:
        axes = axes.reshape(2, 1)

    # Load baseline
    base_vol = nib.load(str(timepoint_data[0]["volume_path"])).get_fdata(dtype=np.float32)
    center_z = base_vol.shape[2] // 2

    for i, td in enumerate(timepoint_data):
        vol = nib.load(str(td["volume_path"])).get_fdata(dtype=np.float32)
        # Use same slice index across all timepoints (they're aligned)
        z = min(center_z, vol.shape[2] - 1)
        axial = vol[:, :, z].T

        # Top row: volume with seg overlay
        axes[0, i].imshow(axial, cmap="gray", aspect="equal")
        if td.get("seg_path") and Path(td["seg_path"]).exists():
            seg = nib.load(str(td["seg_path"])).get_fdata(dtype=np.float32)
            sz = min(z, seg.shape[2] - 1)
            seg_slice = seg[:, :, sz].T
            if seg_slice.max() > 0:
                axes[0, i].contour(seg_slice, levels=[0.5], colors=["lime"], linewidths=1.0)
        axes[0, i].set_title(td["timepoint"], fontsize=10, color="white")
        axes[0, i].axis("off")

        # Bottom row: difference from baseline
        if i == 0:
            axes[1, i].imshow(axial, cmap="gray", aspect="equal")
            axes[1, i].set_title("Baseline (ref)", fontsize=9, color="white")
        else:
            base_z = min(center_z, base_vol.shape[2] - 1)
            diff = axial - base_vol[:, :, base_z].T
            vmax = max(np.percentile(np.abs(diff), 99), 1e-6)
            axes[1, i].imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            p = td.get("params_dict") or {}
            r = p.get("rotation_deg", [0.0, 0.0, 0.0])
            t = p.get("translation", [0.0, 0.0, 0.0])
            axes[1, i].set_title(
                f"rot=({r[0]:.1f}, {r[1]:.1f}, {r[2]:.1f})\n"
                f"trans=({t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f})",
                fontsize=7,
                color="white",
            )
        axes[1, i].axis("off")

    fig.suptitle(f"Longitudinal QC — {patient_id}", fontsize=12, color="white")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=100, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    logger.debug("Longitudinal QC saved: %s", output_path)


def generate_qc_summary_html(
    output_dir: Path,
    patient_results: Dict[str, dict],
) -> None:
    """
    Generate browsable HTML summary of all QC images.

    Args:
        output_dir: pipeline output root.
        patient_results: dict mapping group_key -> {
            'records': List[ScanRecord],
            'results': List[ProcessingResult],
        }
    """
    html_path = output_dir / "qc_summary.html"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections = []
    for group_key in sorted(patient_results.keys()):
        info = patient_results[group_key]
        parts = group_key.split("/", 1)
        if len(parts) != 2:
            continue
        modality, patient_id = parts

        # Longitudinal QC
        long_qc = output_dir / modality / patient_id / f"{patient_id}_longitudinal_qc.png"
        long_img = ""
        if long_qc.exists():
            rel = long_qc.relative_to(output_dir)
            long_img = f'<img src="{rel}" style="max-width:100%;">'

        # Per-scan QC thumbnails
        thumbs = []
        for rec in info.get("records", []):
            qc_png = (
                output_dir / rec.modality / rec.patient_id
                / rec.patient_timepoint / f"{rec.patient_timepoint}_qc.png"
            )
            if qc_png.exists():
                rel = qc_png.relative_to(output_dir)
                thumbs.append(
                    f'<div class="qc-thumb">'
                    f'<img src="{rel}" title="{rec.patient_timepoint}">'
                    f'<br><small>{rec.patient_timepoint}</small></div>'
                )

        thumb_html = "\n".join(thumbs)

        sections.append(f"""
        <div class="patient">
            <h2>{modality} / {patient_id}</h2>
            {long_img}
            <div class="qc-grid">{thumb_html}</div>
        </div>
        """)

    body = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>VS Preprocessing QC Summary</title>
    <style>
        body {{ background: #1a1a1a; color: #eee; font-family: sans-serif; margin: 20px; }}
        h1 {{ border-bottom: 1px solid #555; padding-bottom: 10px; }}
        .patient {{ margin: 20px 0; border: 1px solid #444; padding: 15px; border-radius: 8px; }}
        .patient h2 {{ margin-top: 0; color: #8cf; }}
        .qc-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px; }}
        .qc-thumb {{ text-align: center; }}
        .qc-thumb img {{ max-width: 350px; border: 1px solid #555; border-radius: 4px; cursor: pointer; transition: transform 0.2s; }}
        .qc-thumb img:hover {{ transform: scale(2); z-index: 10; position: relative; }}
        small {{ color: #aaa; }}
    </style>
</head>
<body>
    <h1>VS Preprocessing QC Summary</h1>
    <p>Generated: {timestamp}</p>
    <p>Pipeline output: <code>{output_dir}</code></p>
    {body}
</body>
</html>"""

    output_dir.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html)
    logger.info("QC summary HTML: %s", html_path)
