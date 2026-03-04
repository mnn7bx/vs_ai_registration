# VS Preprocessing Pipeline

Longitudinal MRI preprocessing and intra-patient registration for Vestibular Schwannoma (VS) studies.

## What it does

For each patient (across one or more timepoints and MRI modalities):

1. **N4 bias field correction** (SimpleITK, CPU)
2. **Brain masking** on the baseline timepoint (HD-BET, GPU)
3. **Intra-patient alignment** — follow-ups registered to baseline via a trained GlobalNet (GPU)
4. **Intensity normalisation** within the brain mask (min-max or z-score)
5. **Isotropic resampling** to 0.5 mm target spacing
6. **Expert segmentation passthrough** with nearest-neighbour resampling
7. **QC outputs** — per-scan PNGs, longitudinal comparison PNGs, HTML summary

All outputs are in the baseline coordinate space.  Full FOV is preserved (no skull-stripping in the final output) so the IAC is never cropped.

---

## Project structure

```
vs_preprocess/
├── run.py                  # CLI entry point
├── pipeline.py             # Orchestration
├── dataset.py              # Scan discovery and patient grouping
├── affine_utils.py         # Affine math (compose / decompose / warp)
├── preprocessing.py        # N4, normalisation, resampling
├── skull_strip.py          # HD-BET integration
├── registration.py         # GlobalNet inference
├── qc.py                   # QC PNG + HTML generation
├── requirements.txt
├── tests/
│   ├── test_affine_utils.py
│   ├── test_dataset.py
│   ├── test_pipeline.py    # Pipeline unit tests (mocked GPU)
│   └── smoke_test.py       # Import + CPU-only integration checks
└── training/
    ├── model.py            # AffineRegNet (GlobalNet + decomposed head)
    ├── dataset.py          # Training data loader
    ├── augmentations.py    # Random affine augmentations
    ├── losses.py           # MI, identity-reg, det-reg losses
    ├── train_intra.py      # Intra-patient training loop
    ├── train_template.py   # Template registration training
    ├── evaluate.py         # Evaluation metrics
    └── configs/
        ├── intra_patient.yaml
        └── template.yaml
```

---

## Prerequisites

- Python 3.10+
- CUDA GPU (for HD-BET and registration inference)
- Conda or virtualenv recommended

### Install

```bash
pip install -r vs_preprocess/requirements.txt
```

HD-BET is installed via the `hd-bet` package above and downloads model weights on first use.

---

## Data layout

The pipeline supports two input formats.

### Directory-based

```
data_dir/
├── T2/
│   ├── VS001_tp1/
│   │   └── VS001_tp1.nii.gz        # primary MRI volume
│   ├── VS001_tp2/
│   │   ├── VS001_tp2.nii.gz
│   │   └── VS001_tp2_seg.nii.gz    # optional expert segmentation
│   └── VS002_tp1/
│       └── VS002_tp1.nii.gz
└── T1C/
    └── ...
```

Patient ID and timepoint are derived from the directory name by splitting on the **first** underscore:
`126_0_0` → patient `126`, timepoint `0_0`.

### CSV-based

A CSV with the following columns (as produced by the coregistration pipeline):

| Column | Description |
|--------|-------------|
| `Study Type` | Modality (e.g. `T2`) |
| `Reworked Patient ID` | Numeric patient ID |
| `Timepoint` | Timepoint label |
| `image path` or `coregistered_image` | Path to volume (relative to `--data_root`) |
| `segmentation path` or `coregistered_segmentation` | Path to seg (optional) |

---

## Usage

### Run the pipeline (directory-based)

```bash
python vs_preprocess/run.py \
  --data_dir /path/to/data \
  --output_dir /path/to/output \
  --intra_model_path /path/to/model.pt
```

### Run from CSV

```bash
python vs_preprocess/run.py \
  --data_dir /path/to/data \
  --output_dir /path/to/output \
  --intra_model_path /path/to/model.pt \
  --csv_path /path/to/df_coregistered.csv \
  --data_root /path/to/data
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--transform_type` | `affine` | `affine` (9-DOF) or `rigid` (6-DOF) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--intensity_norm` | `minmax` | `minmax` or `zscore` |
| `--target_spacing` | `0.5 0.5 0.5` | Output voxel spacing in mm |
| `--modalities` | all | Restrict to specific modalities, e.g. `T2 T1C` |
| `--no_n4` | off | Skip N4 bias correction |
| `--no_keep_intermediates` | off | Discard intermediate files |
| `--force_reprocess` | off | Ignore existing outputs and redo |
| `--dry_run` | off | Discover and list scans without processing |
| `--log_level` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Output structure

```
output_dir/
├── T2/
│   └── VS001/
│       ├── VS001_tp1/
│       │   ├── VS001_tp1_preprocessed.nii.gz   # final output
│       │   ├── VS001_tp1_brain_mask.nii.gz
│       │   ├── VS001_tp1_seg.nii.gz             # if seg provided
│       │   ├── VS001_tp1_qc.png
│       │   └── intermediates/
│       │       ├── VS001_tp1_n4.nii.gz
│       │       ├── VS001_tp1_bias_field.nii.gz
│       │       └── VS001_tp1_brain_mask_native.nii.gz
│       ├── VS001_tp2/
│       │   ├── VS001_tp2_preprocessed.nii.gz
│       │   ├── VS001_tp2_brain_mask.nii.gz
│       │   ├── VS001_tp2_qc.png
│       │   └── intermediates/
│       │       ├── VS001_tp2_n4.nii.gz
│       │       └── VS001_tp2_normalized.nii.gz
│       └── VS001_longitudinal_qc.png
├── failures.csv
├── pipeline.log
└── qc_summary.html
```

---

## Training

Train the intra-patient registration model from scratch:

```bash
python vs_preprocess/training/train_intra.py \
  --config vs_preprocess/training/configs/intra_patient.yaml
```

Key config options (`intra_patient.yaml`):

```yaml
data_dir: /path/to/preprocessed
output_dir: /path/to/training_output
transform_type: affine   # or rigid
epochs: 200
batch_size: 2
lr: 1e-4
real_ratio: 0.5          # fraction of real pairs vs synthetic
```

Training logs to TensorBoard (`tensorboard --logdir <output_dir>/runs`).

### Evaluate

```bash
python vs_preprocess/training/evaluate.py \
  --model_path /path/to/best_model.pt \
  --data_dir /path/to/test_data
```

Target metrics: rotation MAE < 2°, translation MAE < 2 mm, seg Dice > 0.8.

---

## Tests

```bash
# All tests
cd vs_preprocess
python -m pytest tests/ -v

# Only pipeline unit tests (no GPU needed)
python -m pytest tests/test_pipeline.py -v

# Smoke test (imports + CPU integration, no GPU needed)
python -m pytest tests/smoke_test.py -v

# Core affine math tests
python -m pytest tests/test_affine_utils.py -v
```

The pipeline and smoke tests mock HD-BET and GlobalNet inference, so they run on CPU without a trained model.

---

## Design decisions (fixed)

| # | Decision |
|---|----------|
| 1 | Full FOV preserved — no skull-stripping in final output |
| 2 | Linear transforms only: affine (9-DOF) or rigid (6-DOF) |
| 3 | First timepoint is baseline per (patient, modality) |
| 4 | Template registration optional (`--register_to_template`, reserved) |
| 5 | Decomposed affine: 3 rotation + 3 translation + 3 scale |
| 6 | Network input resolution: 96×96×96 |
| 7 | Single model with MI loss (modality-agnostic) |
| 8 | HD-BET on baseline only; mask reused for aligned follow-ups |
| 9 | Intermediate files kept by default |
| 10 | Resumable: existing outputs skipped by default |
| 11 | Binary seg (tumor vs background), nearest-neighbour resampling |
| 12 | Target spacing: 0.5 mm isotropic |
| 13 | QC: per-scan PNG + longitudinal PNG + HTML summary |
| 14 | N4 before brain mask, no mask input to N4 |
| 15 | Cross-modality alignment deferred — each modality aligned within itself |
