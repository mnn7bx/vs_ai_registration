# Methodology: Automated Longitudinal MRI Preprocessing Pipeline for Vestibular Schwannoma Monitoring

## 1. Overview

We present an automated preprocessing pipeline for longitudinal MRI studies of vestibular schwannoma (VS). The pipeline standardizes heterogeneous clinical MRI acquisitions into a spatially aligned, intensity-normalized, isotropically resampled representation suitable for volumetric analysis and downstream machine learning. The core component is a learned affine registration model that aligns follow-up scans to each patient's baseline acquisition, eliminating the need for manual landmark-based registration.

The pipeline operates in two phases: (1) training an affine registration network on the target dataset, and (2) applying the trained model within a fully automated preprocessing pipeline that chains bias correction, skull stripping, registration, intensity normalization, resampling, and quality control.

## 2. Data

### 2.1. Dataset

The dataset comprises 1,689 MRI scans from clinical VS monitoring studies, spanning 762 patient-modality groups across five acquisition types:

| Modality | Description |
|----------|-------------|
| `t1+_thick_ax` | Post-contrast T1-weighted, thick-slice axial |
| `t1+_thick_cor` | Post-contrast T1-weighted, thick-slice coronal |
| `t1+_thin` | Post-contrast T1-weighted, thin-slice (volumetric) |
| `t2_thick` | T2-weighted, thick-slice |
| `t2_thin` | T2-weighted, thin-slice (volumetric) |

Scans are organized in a standardized directory hierarchy: `data_dir/<modality>/<patient_id>_<timepoint>/`, where each subdirectory contains one NIfTI volume and an optional expert segmentation. Patient identity is extracted from the directory name by splitting on the first underscore. Volumes are identified as the largest NIfTI file not matching exclusion patterns (e.g., `_seg`, `_mask`, `_brain`). Expert segmentations are identified by the presence of `_seg` or `_label` in the filename.

### 2.2. Patient Grouping

Scans are grouped by the tuple (modality, patient_id), producing patient-modality groups. Within each group, timepoints are sorted chronologically, and the earliest timepoint is designated as the baseline. All subsequent timepoints are treated as follow-ups to be registered to the baseline. An 80/20 patient-level split (stratified by patient ID, seed=42) separates training and validation sets.

## 3. Registration Model

### 3.1. Architecture

The registration model (AffineRegNet) is built on MONAI's GlobalNet encoder-decoder backbone, a 3D convolutional architecture that processes concatenated image pairs and outputs spatial transformation parameters. We replace MONAI's default output head (which produces 12 raw affine matrix entries) with a decomposed affine head that outputs 9 interpretable parameters for affine registration or 6 for rigid registration.

**Backbone.** The GlobalNet encoder-decoder uses 3D convolutions with an initial channel count of 16 and a depth of 4 levels. The input is a two-channel volume formed by concatenating the moving and fixed images along the channel dimension, each of spatial size 128 x 128 x 128. Total parameter count: 73,006,857.

**Decomposed Affine Head.** The output head is a two-layer MLP (512 hidden units, ReLU activation, 10% dropout) that maps the flattened bottom-level feature map to 9 decomposed affine parameters:

- **Rotations** (3 parameters): Euler angles r_x, r_y, r_z in radians, composed as R = R_z R_y R_x.
- **Translations** (3 parameters): t_x, t_y, t_z in normalized [-1, 1] coordinates.
- **Scales** (3 parameters): s_x, s_y, s_z as offsets from 1.0 (i.e., network output of 0.0 corresponds to unit scale), clamped to [0.7, 1.3].

The final layer is zero-initialized so that the network predicts the identity transform at initialization. The 9 parameters are composed into a 3x4 affine matrix as A = T * S * R_z * R_y * R_x, where T is the translation matrix and S = diag(1 + s_x, 1 + s_y, 1 + s_z).

### 3.2. Training Data Construction

Each training epoch draws from two data sources, interleaved at a 1:1 ratio:

**Real longitudinal pairs.** For each patient-modality group with at least two timepoints, all (follow-up, baseline) pairs are enumerated. Both volumes are loaded, reoriented to RAS, and resampled to 1.0 mm isotropic spacing. A shared crop center is computed from the physical overlap region of the two volumes' fields of view, and both are center-cropped (or zero-padded) to 128 x 128 x 128 voxels. Intensities are normalized to [0, 1] via min-max scaling. Real pairs have no ground-truth transform available.

**Synthetic pairs.** A single scan is randomly selected, preprocessed identically (RAS reorientation, 1.0 mm resampling, center crop to 128^3, min-max normalization), and used as the fixed image. A random affine transform is sampled and applied via differentiable grid sampling to generate the moving image. The ground-truth parameters are the exact inverse of the applied transform, computed by matrix inversion and decomposition. Synthetic transform ranges:

| Parameter | Range |
|-----------|-------|
| Rotation (each axis) | [-15, +15] degrees |
| Translation (each axis) | [-0.1, +0.1] (normalized coords) |
| Scale offset (each axis) | [-0.05, +0.05] |

### 3.3. Augmentation

Intensity augmentation is applied to the moving image during training (each with probability 0.5):

1. **Intensity shift**: uniform offset in [-0.1, +0.1]
2. **Intensity scale**: multiplicative factor in [0.9, 1.1]
3. **Gaussian noise**: additive noise with standard deviation in [0.01, 0.05]
4. **Simulated bias field**: low-frequency multiplicative field generated by trilinearly upsampling an 8x8x8 random grid (values in [0.9, 1.1]) to volume size

No intensity augmentation is applied during validation.

### 3.4. Loss Function

The total loss is a weighted combination of four terms:

**L = lambda_sim * L_sim + lambda_sup * L_sup + lambda_id * L_id + lambda_det * L_det**

**Similarity loss (L_sim).** Local normalized cross-correlation (NCC) with a kernel size of 9 voxels, computed between the warped moving image and the fixed image. NCC is preferred over mutual information for this application as it provides a smoother optimization landscape for within-modality registration.

**Supervised parameter loss (L_sup).** Mean squared error between predicted parameters and ground-truth parameters, applied only to synthetic pairs (where the ground truth is known). Rotation and translation components are weighted equally; scale components are weighted at 0.1x to prevent scale from dominating the loss.

**Identity regularization (L_id).** L2 penalty on the raw parameter vector, encouraging predictions near the identity transform: L_id = mean(params^2). Weight: lambda_id = 0.001.

**Determinant regularization (L_det).** Penalizes the determinant of the 3x3 rotation-scale submatrix deviating from 1.0: L_det = mean((det(RS) - 1)^2). This discourages volume-changing transforms. Weight: lambda_det = 0.1.

### 3.5. Training Schedule

Training uses a phased loss weighting strategy over three stages:

**Phase 1 — Supervised warm-up (epochs 1-10).** The similarity loss weight (lambda_sim) ramps linearly from 0.0 to 1.0. The supervised loss weight (lambda_sup) remains at 1.0. This phase teaches the model the structure of reasonable affine transforms using ground-truth supervision from synthetic pairs, before the potentially noisy NCC gradient dominates.

**Phase 2 — Supervised decay (epochs 10-30).** lambda_sim remains at 1.0. lambda_sup decays linearly from 1.0 to 0.0. The model transitions from supervised learning to self-supervised image similarity optimization.

**Phase 3 — Pure similarity (epochs 30+).** lambda_sup = 0.0, lambda_sim = 1.0. The model optimizes purely for image alignment quality via NCC.

### 3.6. Optimization

- **Optimizer**: Adam with initial learning rate 1 x 10^-4
- **Learning rate schedule**: Cosine annealing with warm restarts (T_0 = 50, T_mult = 2)
- **Gradient clipping**: Max norm of 1.0
- **Batch size**: 4
- **Early stopping**: Patience of 40 epochs without validation improvement
- **Maximum epochs**: 200
- **Checkpointing**: Best model saved on validation improvement; periodic checkpoints every 10 epochs

### 3.7. Differentiable Warping

During training, volumes are warped using PyTorch's `affine_grid` and `grid_sample` with bilinear interpolation and border padding, enabling end-to-end gradient flow through the spatial transformer. The predicted 3x4 affine matrix (moving-to-fixed in normalized coordinates) is inverted to obtain the fixed-to-moving mapping required by `grid_sample`, which samples input values at grid-specified locations.

### 3.8. Training Results

The model was trained on the full VS dataset (610 training groups, 1,344 scans; 152 validation groups, 345 scans) on a single NVIDIA A40 GPU at approximately 30 minutes per epoch.

| Metric | Value |
|--------|-------|
| Best validation loss | -0.1657 (epoch 45) |
| Final training loss | -0.1584 (epoch 45) |
| NCC ramp completion | Epoch 11 |
| Supervised decay completion | Epoch 31 |
| Approximate training time | 22.5 hours (45 epochs) |

The model showed consistent improvement throughout training with no signs of overfitting. Validation loss tracked closely with training loss, and new best models were saved nearly every epoch from epoch 31 onward.

## 4. Preprocessing Pipeline

After training, the learned registration model is deployed within an automated preprocessing pipeline that processes each patient-modality group independently. The pipeline consists of seven sequential stages.

### 4.1. N4 Bias Field Correction

Each volume undergoes N4 bias field correction using SimpleITK. An Otsu threshold-based foreground mask is computed automatically. The N4 algorithm runs with a shrink factor of 3, convergence threshold of 1 x 10^-6, and a multi-resolution iteration scheme of [50, 50, 30, 20]. The corrected volume and optionally the estimated log bias field are saved.

### 4.2. Brain Mask Extraction

A brain mask is extracted from the bias-corrected baseline volume using HD-BET in accurate mode (5-model ensemble). The resulting mask is dilated by 2 voxels using a 6-connected structuring element to provide a safety margin around the internal auditory canal (IAC), ensuring the tumor region is fully included. Brain masking is performed only on the baseline; after alignment, the baseline mask applies to all timepoints.

### 4.3. Intra-Patient Affine Registration

For each patient-modality group with multiple timepoints, all follow-up volumes are registered to the baseline using the trained AffineRegNet.

**Network preprocessing.** Each volume is loaded, reoriented to RAS, resampled to 1.0 mm isotropic spacing via scipy `zoom` (order 1), and min-max normalized to [0, 1]. The moving and fixed tensors (each 1 x 1 x 128 x 128 x 128) are concatenated along the channel dimension and passed through the network.

**Coordinate conversion.** The network predicts a 3x4 affine matrix in normalized [-1, 1] coordinates. This is converted to a 4x4 world-space (mm) affine through the following coordinate chain:

```
fixed_world -> fixed_voxel (128^3) -> fixed_normalized -> moving_normalized -> moving_voxel (128^3) -> moving_world
```

Specifically: M_world = A_moving * N2V * inv(A_net) * V2N * inv(A_fixed), where A_moving and A_fixed are the voxel-to-world affines of the resampled 128^3 volumes, V2N and N2V convert between voxel indices and normalized coordinates, and A_net is the network's predicted affine.

**Volume resampling.** The world-space affine is applied to resample the full-resolution moving volume into the baseline grid using `scipy.ndimage.affine_transform` with cubic B-spline interpolation (order 3). The output voxel-to-voxel mapping is: M_v2v = inv(moving_affine) * M_world * reference_affine. The predicted affine parameters (rotation, translation, scale) and the affine matrix are saved per timepoint.

### 4.4. Intensity Normalization

Intensities are normalized within the brain mask region. Statistics (percentiles, mean, standard deviation) are computed from brain-masked voxels only, and the normalization is applied to the full volume. Two methods are supported:

- **Min-max normalization** (default): Clips intensities to the [1st, 99th] percentile range computed within the brain mask, then linearly scales to [0, 1].
- **Z-score normalization**: Clips to the same percentile range, then standardizes using the brain-masked mean and standard deviation.

### 4.5. Isotropic Resampling

All volumes (and masks) are resampled to 0.5 mm isotropic spacing. Volumes are converted from nibabel to SimpleITK format, resampled using B-spline interpolation (for volumes) or linear interpolation with threshold-based binarization at 0.5 (for masks), then converted back to nibabel format. The output grid size is computed to preserve the physical field of view.

### 4.6. Segmentation Passthrough

If an expert segmentation exists for a timepoint, it is transformed through the same spatial pipeline: the world-space affine from registration is applied with nearest-neighbor interpolation to preserve label integrity, followed by isotropic resampling with nearest-neighbor interpolation.

### 4.7. Quality Control

Two levels of automated QC are generated:

**Per-scan QC.** A 3-panel PNG image showing axial, sagittal, and coronal center slices of the preprocessed volume, with brain mask contours (red) and tumor segmentation contours (green, if available) overlaid.

**Longitudinal QC.** For patients with multiple timepoints, a comparison image shows all timepoints side by side (top row: aligned axial slices with segmentation overlays; bottom row: difference images relative to baseline using a diverging RdBu colormap) with predicted registration parameters annotated.

A browsable HTML summary is generated aggregating all QC images across the dataset, with hover-to-zoom functionality for efficient visual review.

## 5. Pipeline Outputs

For each scan, the pipeline produces:

| File | Description |
|------|-------------|
| `<timepoint>_preprocessed.nii.gz` | Final bias-corrected, aligned, normalized, isotropically resampled volume |
| `<timepoint>_brain_mask.nii.gz` | Brain mask (dilated, resampled to output spacing) |
| `<timepoint>_seg.nii.gz` | Expert segmentation (if available), spatially aligned and resampled |
| `<timepoint>_affine.npy` | Predicted 3x4 affine matrix (normalized coordinates) |
| `<timepoint>_params.json` | Human-readable registration parameters (rotation in degrees, translation, scale, determinant) |
| `<timepoint>_qc.png` | Per-scan QC visualization |
| `intermediates/` | N4-corrected volume, bias field, aligned volume, normalized volume (optional) |

Additionally, per-patient longitudinal QC images and a dataset-wide HTML summary are generated. Any processing failures are logged to `failures.csv` with human-readable error messages.

## 6. Implementation

The pipeline is implemented in Python using the following libraries:

| Component | Library |
|-----------|---------|
| Neural network | PyTorch, MONAI (GlobalNet backbone) |
| Bias correction, resampling | SimpleITK |
| Brain extraction | HD-BET (v2.x) |
| NIfTI I/O | nibabel |
| Spatial transforms | scipy.ndimage |
| Visualization | matplotlib |

All processing is performed on a single GPU (NVIDIA A40) for neural network inference and HD-BET, with CPU-based processing for N4 correction, intensity normalization, and resampling. The pipeline supports incremental processing via skip-existing logic and records per-stage timing for performance profiling.

## 7. Software Availability

The complete pipeline source code, training configurations, and test suite are available at the project repository. The pipeline can be invoked via:

```bash
python vs_preprocess/run.py \
  --data_dir /path/to/organized/data \
  --output_dir /path/to/output \
  --intra_model_path /path/to/best_model.pth \
  --target_spacing 0.5 0.5 0.5
```
