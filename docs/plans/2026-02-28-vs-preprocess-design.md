# VS Preprocessing Pipeline — Design Document
**Date:** 2026-02-28
**Spec reference:** vs_preprocess_IMPLEMENTATION_SPEC.md (v8)

---

## Summary

Implement the complete VS MRI preprocessing and longitudinal registration pipeline
as described in the implementation spec, with minimal adaptations for the actual
dataset structure.

---

## 1. Architecture

```
vs_preprocess/           ← inside /home/kwa/Documents/vs_ai_registration/
├── requirements.txt
├── run.py
├── pipeline.py
├── dataset.py
├── affine_utils.py
├── preprocessing.py
├── skull_strip.py
├── registration.py
├── qc.py
├── tests/
│   ├── __init__.py
│   ├── test_affine_utils.py  ← GATE: must pass 100% before pipeline work
│   ├── test_dataset.py
│   └── test_pipeline.py
└── training/
    ├── __init__.py
    ├── requirements.txt
    ├── model.py
    ├── dataset.py
    ├── augmentations.py
    ├── losses.py
    ├── train_intra.py
    ├── train_template.py
    ├── evaluate.py
    └── configs/
        ├── intra_patient.yaml
        └── template.yaml
```

---

## 2. Data

- **Input data:** `/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies/`
- **Modalities:** `t1+_thick_ax`, `t1+_thick_cor`, `t1+_thin`, `t2_thin`
- **CSV:** `/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv`
- **Scale:** 187 patients, ~8.6 timepoints avg, 1603 total scan rows
- **Current state:** All files have `_coregistered` suffix (previously registered with ANTsPy)
- **Future:** Unregistered originals available later on `/mnt/d/`

### File naming
- Images: `{patientID}_{timepoint}_{modality}_image_coregistered.nii.gz`
- Segs:   `{patientID}_{timepoint}_{modality}_R_VS__uvauser2__coregistered.nii.gz`

---

## 3. Design Decisions (Fixed)

All decisions from spec Section 1 are adopted verbatim:

| # | Decision | Choice |
|---|---|---|
| 1 | Final output skull-stripped? | No — full FOV preserved |
| 2 | Transform type | Linear only: affine (9 DOF) or rigid (6 DOF) |
| 3 | Longitudinal anchor | First timepoint per (patient, modality) |
| 4 | Template registration | Optional `--register_to_template` |
| 5 | Affine parameterization | Decomposed: 3 rotation + 3 translation + 3 scale |
| 6 | Network input resolution | 96×96×96 |
| 7 | Modality handling | Single model with MI loss |
| 8 | Brain mask | HD-BET on baseline only |
| 9 | Intermediates | Kept by default |
| 10 | Resume | Skip existing outputs |
| 11 | Seg labels | Binary, nearest-neighbor resampling |
| 12 | Target spacing | 0.5mm isotropic |
| 13 | QC | PNG per scan + longitudinal PNG + HTML |
| 14 | N4 timing | Before brain mask, no mask input |

---

## 4. Adaptations from Spec (Only 2)

### Adaptation 1 — Patient ID Parsing (`dataset.py`)

**Problem:** Spec splits `_parse_patient_timepoint` on the LAST underscore.
`126_0_0` → patient=`126_0`, timepoint=`0` (wrong).

**Fix:** Change to split on the FIRST underscore.
`126_0_0` → patient=`126`, timepoint=`0_0` (matches CSV).

```python
# Spec version (wrong for this data):
idx = dirname.rfind("_")

# Adapted version:
idx = dirname.find("_")
```

### Adaptation 2 — Segmentation Discovery (`dataset.py`)

**Problem:** Spec's `_find_seg` looks for `_seg|_label` in filename.
Actual seg files are named `*_R_VS__uvauser2__coregistered.nii.gz`.

**Fix:** Add `discover_scans_from_csv(csv_path, data_root)` factory function that:
- Reads CSV columns: `image path` (primary) or `coregistered_image` (fallback)
- Reads CSV columns: `segmentation path` (primary) or `coregistered_segmentation` (fallback)
- Parses `Reworked Patient ID` and `Timepoint` directly from CSV
- Returns `List[ScanRecord]` identical to `discover_scans()` output

Original `discover_scans()` is kept unchanged for future unregistered data.

---

## 5. Pipeline Flow

```
Per patient-modality group:
1. N4 bias correction on every timepoint (CPU)
2. HD-BET brain mask on baseline only (GPU)
3. GlobalNet: align each follow-up to baseline (GPU)
   → All timepoints now share baseline coordinate space
   → Baseline brain mask applies to all aligned follow-ups
4. Intensity normalization per timepoint (CPU)
   → Stats from brain mask voxels, applied to full volume
5. Resample all to 0.5mm isotropic (CPU)
6. Resample expert seg with same affine + nearest-neighbor (CPU)
7. QC PNGs per scan + longitudinal comparison (CPU)
```

---

## 6. Training Strategy

Training data: existing `_coregistered` files.
- **Real pairs:** (follow-up, baseline) — already aligned → network learns near-identity
- **Synthetic pairs:** known random affines applied to baseline scans → network learns to correct transforms
- **Ratio:** 50% real : 50% synthetic

Loss = MI + identity regularization + determinant regularization + supervised MSE (synthetic only)

MI warm-up: epoch 0-9 no MI, epoch 10-29 linear ramp, epoch 30+ full MI weight.

---

## 7. Output Structure

```
output_dir/
├── {modality}/
│   └── {patient_id}/
│       ├── {patient_timepoint}_preprocessed.nii.gz
│       ├── {patient_timepoint}_brain_mask.nii.gz
│       ├── {patient_timepoint}_seg.nii.gz
│       ├── {patient_timepoint}_affine.npy
│       ├── {patient_timepoint}_params.json
│       ├── {patient_timepoint}_qc.png
│       └── intermediates/
│           ├── {patient_timepoint}_n4.nii.gz
│           ├── {patient_timepoint}_bias_field.nii.gz
│           └── {patient_timepoint}_brain_mask_native.nii.gz
├── failures.csv
├── pipeline.log
└── qc_summary.html
```
