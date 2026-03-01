#!/usr/bin/env python3
"""
CLI entry point for the VS preprocessing pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Vestibular Schwannoma MRI Preprocessing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--data_dir", type=Path, required=True, help="Root input directory")
    p.add_argument("--output_dir", type=Path, required=True, help="Root output directory")
    p.add_argument("--intra_model_path", type=Path, required=True, help="Trained GlobalNet weights")

    # Transform
    p.add_argument("--transform_type", default="affine", choices=["affine", "rigid"],
                    help="Transform DOF: affine (9) or rigid (6)")

    # Template registration (optional)
    p.add_argument("--register_to_template", action="store_true",
                    help="Enable secondary template registration")
    p.add_argument("--template_model_path", type=Path, default=None,
                    help="Template GlobalNet weights")
    p.add_argument("--template_path", type=Path, default=None,
                    help="Template NIfTI image (e.g., MNI152)")
    p.add_argument("--template_mask_path", type=Path, default=None,
                    help="Template brain mask NIfTI")

    # Brain mask
    p.add_argument("--mask_method", default="hdbet", choices=["hdbet", "template_warp"],
                    help="Brain mask method")

    # Data
    p.add_argument("--baseline_selection", default="first", choices=["first", "latest"],
                    help="Which timepoint is baseline")
    p.add_argument("--modalities", nargs="*", default=None,
                    help="Only process these modalities (e.g., T2 T1C)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="Compute device")

    # Preprocessing
    p.add_argument("--intensity_norm", default="minmax", choices=["minmax", "zscore"],
                    help="Intensity normalization method")
    p.add_argument("--percentile_low", type=float, default=1.0,
                    help="Lower clipping percentile")
    p.add_argument("--percentile_high", type=float, default=99.0,
                    help="Upper clipping percentile")
    p.add_argument("--target_spacing", nargs=3, type=float, default=[0.5, 0.5, 0.5],
                    help="Target voxel spacing in mm")
    p.add_argument("--no_n4", action="store_true", help="Skip N4 bias correction")

    # Pipeline control
    p.add_argument("--n_workers", type=int, default=4,
                    help="CPU parallel workers (reserved for future use)")
    p.add_argument("--keep_intermediates", action="store_true", default=True,
                    help="Keep intermediate files")
    p.add_argument("--no_keep_intermediates", action="store_true",
                    help="Discard intermediate files")
    p.add_argument("--skip_existing", action="store_true", default=True,
                    help="Skip scans with existing outputs")
    p.add_argument("--force_reprocess", action="store_true",
                    help="Reprocess all scans (overrides --skip_existing)")
    p.add_argument("--dry_run", action="store_true",
                    help="List discovered scans without processing")
    p.add_argument("--log_level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="Logging verbosity")

    # CSV-based data loading
    p.add_argument("--csv_path", type=Path, default=None,
                    help="CSV file for data loading (uses discover_scans_from_csv)")
    p.add_argument("--data_root", type=Path, default=None,
                    help="Root dir for resolving CSV relative paths")

    args = p.parse_args(argv)

    # Resolve keep_intermediates
    if args.no_keep_intermediates:
        args.keep_intermediates = False

    return args


def setup_logging(output_dir: Path, level: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level))

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File
    fh = logging.FileHandler(str(log_path), mode="a")
    fh.setFormatter(fmt)
    root.addHandler(fh)


def main():
    args = parse_args()
    setup_logging(args.output_dir, args.log_level)

    config = vars(args)

    if args.csv_path is not None:
        from pipeline import run_pipeline_from_csv
        results = run_pipeline_from_csv(
            csv_path=args.csv_path,
            data_root=args.data_root or args.data_dir,
            output_dir=args.output_dir,
            intra_model_path=args.intra_model_path,
            config=config,
        )
    else:
        from pipeline import run_pipeline
        results = run_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            intra_model_path=args.intra_model_path,
            config=config,
        )

    n_fail = sum(not r.success and not r.skipped for r in results)
    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
