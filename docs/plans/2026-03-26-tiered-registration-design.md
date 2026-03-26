# Tiered Registration Pipeline Design

**Date:** 2026-03-26
**Branch:** `feat/cross-modality-registration`
**Status:** Approved

## Motivation

The current single-shot DL registration produces catastrophic failures in ~12% of cases (rotations up to 177°, determinants 0.6–2.2) and even "OK" cases show unacceptable quality. Replace with a hierarchical tiered approach that maximizes longitudinal registration consistency, mirroring the proven ANTsPy pipeline structure.

## Goals

1. Build an ANTsPy tiered registration pipeline as the benchmark/comparison
2. Augment the existing DL pipeline with the same tiered hierarchy
3. Run both side-by-side to compare registration quality
4. Long term: improve DL model until it replaces ANTsPy entirely

## Registration Hierarchy

Three-tier anchor structure per patient:

```
MNI Template (standard space)
  └── Patient Anchor (best quality scan overall — thinnest slices)
        ├── Timepoint Anchor T=0 (best T2 at timepoint 0)
        │     ├── T1C scan at T=0
        │     └── other T2 scans at T=0
        ├── Timepoint Anchor T=1 (best T2 at timepoint 1)
        │     ├── T1C scan at T=1
        │     └── ...
        └── Timepoint Anchor T=N
              └── ...
```

### Anchor Selection Rules

1. **Patient anchor**: Across all timepoints and modalities, pick the scan with thinnest slices. Tiebreaker: prefer T2, then earliest timepoint.
2. **Timepoint anchors**: Per timepoint, pick the best T2 (thinnest slices). If no T2 at that timepoint, pick best T1C.
3. **Remaining scans**: Register to their timepoint anchor.

### Slice Thickness Ranking (thinnest first)

1. `t2_thin` / `t1+_thin`
2. `t2_thick` / `t1+_thick_cor`
3. `t1+_thick_ax`

Derived from sub-modality name, not NIFTI headers (faster, deterministic).

### Transform Chain Composition

```
scan_to_mni = mni_from_patient @ patient_from_tp_anchor @ tp_anchor_from_scan
```

Each intermediate affine is saved, plus the composed final.

## Registration Engines

### DL Engine (existing model, augmented)

At each tier:
1. Rigid prediction (6 DOF) — existing model with `transform_type="rigid"`
2. Apply rigid transform to moving image
3. Affine prediction (9 DOF) — run model again on rigidly-aligned pair
4. Compose: `tier_transform = affine @ rigid`

### ANTsPy Engine (new, runs in `antspy_env`)

At each tier:
1. `ants.registration(fixed, moving, type_of_transform="Rigid")`
2. `ants.registration(fixed, moving, type_of_transform="Similarity", initial_transform=rigid_result)`
3. Compose: `tier_transform = similarity @ rigid`

### MNI Registration

- ANTsPy path: `ants.registration()` rigid→similarity to MNI152
- DL path: model prediction against MNI152 (model wasn't trained for this — comparison data)

### Quality Gate (both engines)

After each tier, validate the output transform:
- Rotation > 45° → flag as failed
- Determinant outside [0.8, 1.3] → flag
- Log warnings but don't block (need all results for comparison)

## Architecture & File Layout

```
vs_preprocess/
  registration.py          ← refactor: add tiered hierarchy logic, anchor selection
  registration_ants.py     ← NEW: ANTsPy registration (standalone, runs in antspy_env)
  comparison.py            ← NEW: metrics comparing DL vs ANTsPy outputs
  pipeline.py              ← update: call tiered registration instead of flat

scripts/
  run_ants_registration.sh ← NEW: wrapper calling conda run -n antspy_env
```

### Shared Between Both Paths

- Anchor selection logic (lives in `registration.py`, importable)
- Output format: affine matrices (.npy), parameter JSONs, QC PNGs
- Input: same `discover_scans()` → `group_by_patient_timepoint()` grouping

## Output Structure

```
preprocessing_output/
  dl/
    t1+_thin/131/
      anchors.json                    ← anchor selection (shared)
      131_patient_anchor_to_mni.npy   ← tier 1 transform
      131_tp0_anchor_to_patient.npy   ← tier 2 transform
      131_1_196/
        131_1_196_to_tp_anchor.npy    ← tier 3 transform
        131_1_196_composed.npy        ← full chain to MNI
        131_1_196_preprocessed.nii.gz
        131_1_196_params.json
        131_1_196_qc.png
  ants/
    t1+_thin/131/
      (same structure)
  comparison/
    131_comparison.json               ← per-patient metrics
    summary.csv                       ← aggregate comparison
```

## Comparison Metrics (`comparison.py`)

- NCC between DL-registered and ANTsPy-registered volumes
- MI between registered volume and fixed target at each tier
- Transform parameter differences (rotation, translation, scale)
- Failure rate: % of tiers where quality gate triggered

## Execution Flow

**Step 1 — DL pipeline** (monai_env):
```bash
conda run -n monai_env python -m run \
  --data_dir data/unprocessed_raw_studies \
  --output_dir preprocessing_output/dl \
  --intra_model_path training_output/vs_cross_modality/best_model.pth \
  --mode tiered --device cuda
```

**Step 2 — ANTsPy pipeline** (antspy_env, separate process):
```bash
scripts/run_ants_registration.sh \
  --data_dir data/unprocessed_raw_studies \
  --output_dir preprocessing_output/ants \
  --mni_template /path/to/MNI152.nii.gz
```

**Step 3 — Comparison** (either env):
```bash
python -m comparison \
  --dl_dir preprocessing_output/dl \
  --ants_dir preprocessing_output/ants \
  --output_dir preprocessing_output/comparison
```

Steps 1 and 2 can run in parallel. Step 3 runs after both complete.
Both pipelines respect `--skip_existing`. Anchor selection is deterministic.

## Environment

- **DL pipeline**: `monai_env` (torch + MONAI + HD-BET)
- **ANTsPy pipeline**: `antspy_env` (antspyx), invoked via `conda run -n antspy_env`
- **Comparison**: no special deps (numpy, nibabel)
