# Cross-Modality & Longitudinal Registration: Dual-Loss Design

**Date:** 2026-03-18
**Status:** Approved

## Goal

Train a single affine registration model that handles:
1. **Intra-timepoint cross-modality** registration (T1C <-> T2)
2. **Across-timepoint same-modality** longitudinal registration
3. Thick/thin and acquisition plane variants treated as sub-modalities within T1C or T2

## Data

### Super-Modality Mapping

| Raw directory | Super-modality |
|---|---|
| `t1+_thick_ax`, `t1+_thick_cor`, `t1+_thin` | T1C |
| `t2_thick`, `t2_thin` | T2 |

### Available Pairs (from 1689 scans, 618 patient-timepoints)

| Pair type | Description | Count | Loss |
|---|---|---|---|
| Same sub-modality longitudinal | e.g., `t1+_thin` tp1 -> `t1+_thin` tp2 | ~927 | NCC |
| Cross sub-modality longitudinal | e.g., `t1+_thick_ax` tp1 -> `t1+_thin` tp2 | ~1676 | MI |
| Cross super-modality intra-timepoint | e.g., T1C tp1 -> T2 tp1 | ~1139 | MI |
| Synthetic | Random affine of any scan | configurable | NCC + supervised |

**Total real pairs: ~3742**

570/618 patient-timepoints (92%) have both T1C and T2, enabling cross-modality pairs.

## Three-Tier Loss

Each pair is tagged with a `pair_type`:
- `0` — same sub-modality (NCC)
- `1` — cross sub-modality, same super-modality (MI)
- `2` — cross super-modality (MI)
- Synthetic pairs use type `0` (NCC + supervised param loss)

**Rationale for three tiers:** Same sub-modality pairs (identical tissue contrast, same resolution profile) benefit from NCC's sharper gradient. Cross sub-modality pairs have partial volume and resolution mismatches (especially `t1+_thick_ax` <-> `t1+_thick_cor` with orthogonal blur axes) that make NCC unreliable. Cross super-modality pairs have inverted tissue contrast (CSF dark on T1, bright on T2) where NCC fails entirely.

### Loss Computation

```
For each sample in batch:
    if pair_type == 0 or synthetic:
        sim_loss = NCC(warped_moving, fixed)
    else:
        sim_loss = MI(warped_moving, fixed)

    reg_loss = identity_reg(params) + det_reg(affine)

    if has_gt (synthetic only):
        sup_loss = supervised_param_loss(params, gt_params)

total = mean(sim_losses) + lambda_reg * reg_loss + lambda_sup * sup_loss
```

### Loss Scale Balancing

NCC and MI return values on different scales. Use running EMA normalization: track exponential moving average of NCC and MI losses separately, divide each by its EMA so they contribute equally to the total loss.

## Batch Construction

Mixed batches with configurable ratios (default 25% each):
- 25% same-sub longitudinal (NCC)
- 25% cross-sub longitudinal (MI)
- 25% cross-super intra-timepoint (MI)
- 25% synthetic (NCC + supervised)

## Dataset Implementation

### New Patient-Timepoint Index

Replace current `modality/patient_id` grouping with a patient-timepoint index:

```
patient_index = {
    "VS001": {
        "tp1": {"t1+_thick_ax": path, "t2_thin": path},
        "tp2": {"t1+_thin": path, "t2_thin": path},
    },
    ...
}
```

### Pair Enumeration

Pairs are enumerated once at dataset construction:

1. **Same-sub longitudinal:** For each patient, for each sub-modality present at 2+ timepoints, pair each non-baseline with the baseline.
2. **Cross-sub longitudinal:** For each patient, for each pair of sub-modalities in the same super-modality, pair scans across different timepoints (each direction).
3. **Cross-super intra-timepoint:** For each patient-timepoint with both T1C and T2 variants, pair every T1C variant with every T2 variant.

Each pair stored as `(moving_path, fixed_path, pair_type)`.

### CombinedDataset

Wraps the new real-pair dataset + existing `SyntheticPairDataset` with the 4-way ratio. Returns `(moving, fixed, pair_type, has_gt, gt_params)`.

## Resolution Strategy

### Training
- Input: 128^3 at ~1.0mm isotropic (current setting)
- Sufficient for predicting 9 affine parameters — affine transforms are global, fine detail unnecessary

### Inference
- Predict affine from downsampled input (128^3 at ~1.0mm)
- Apply predicted affine to original moving volume resampled at **0.5mm isotropic**
- Output: registered volume at 0.5mm isotropic

The prediction resolution and output resolution are decoupled. This avoids the VRAM cost of processing 340^3 volumes while achieving 0.5mm output quality.

## Model Architecture

**No changes.** AffineRegNet with `(B, 2, D, H, W)` input, 9-param affine output via decomposed head (LeakyReLU, 512 hidden units). No modality conditioning needed — the model learns structural alignment, and the loss function handles contrast differences.

## Validation

Three metrics tracked separately:

| Metric | Pairs | Loss |
|---|---|---|
| `val_ncc_same_sub` | Same sub-modality longitudinal | NCC |
| `val_mi_cross_sub` | Cross sub-modality longitudinal | MI |
| `val_mi_cross_super` | Cross super-modality intra-timepoint | MI |

**Best model selection:** `best = min(mean(val_ncc_same_sub, val_mi_cross_sub, val_mi_cross_super))`. All three are negative (lower = better). If scale imbalance causes one metric to dominate, add fixed normalization.

## Training Configuration

Based on current `intra_vs_local.yaml`, with additions:

- `similarity_loss: three_tier` (new mode)
- `pair_ratios: [0.25, 0.25, 0.25, 0.25]` (same-sub, cross-sub, cross-super, synthetic)
- `loss_ema_decay: 0.99` (for NCC/MI scale normalization)
- `supervised_decay_start: 10`
- `supervised_decay_end: 30`
- `image_size: [128, 128, 128]`
- `target_spacing: [1.0, 1.0, 1.0]` (training input)
- `output_spacing: [0.5, 0.5, 0.5]` (inference output)
- Scheduler, LR, batch size, etc. unchanged

## Supervised Warm-Up

Unchanged from current design. Synthetic pairs provide supervised param loss with linear decay (epochs 10-30). Only synthetic pairs have ground-truth affine parameters.

## Summary of Changes Required

1. **`organize_vs.py` or data loading**: Add super-modality mapping
2. **`dataset.py`**: New patient-timepoint indexing, pair enumeration with `pair_type`
3. **`training/dataset.py`**: New `MultiPairDataset` class replacing `LongitudinalPairDataset`
4. **`training/train_intra.py`**: Three-tier loss selection per sample, EMA normalization, three validation metrics
5. **`training/losses.py`**: Loss router that selects NCC or MI based on `pair_type`
6. **`registration.py`**: Decouple prediction resolution from output resolution (0.5mm)
7. **Config**: New YAML with three-tier loss settings
