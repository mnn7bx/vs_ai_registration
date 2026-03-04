# VS Preprocessing Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a complete deep-learning-based MRI preprocessing and longitudinal registration pipeline for Vestibular Schwannoma (VS) studies, following the spec in `vs_preprocess_IMPLEMENTATION_SPEC.md`.

**Architecture:** A MONAI GlobalNet encoder predicts a decomposed 9-DOF affine transform aligning each follow-up MRI timepoint to the patient's baseline. The preprocessing pipeline applies N4 bias correction → HD-BET skull-strip → GlobalNet alignment → intensity normalization → 0.5mm isotropic resampling, with QC PNGs and HTML output. Data loading is driven from the existing CSV to correctly resolve patient IDs, timepoints, and segmentation paths.

**Tech Stack:** Python 3.10+, PyTorch ≥2.0, MONAI ≥1.3, nibabel, SimpleITK, scipy, HD-BET, matplotlib, pandas, tqdm, tensorboard, pyyaml

---

## Key Paths

| Name | Path |
|---|---|
| Project root | `/home/kwa/Documents/vs_ai_registration/vs_preprocess/` |
| Spec file | `/home/kwa/Documents/vs_ai_registration/vs_preprocess_IMPLEMENTATION_SPEC.md` |
| Data directory | `/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies/` |
| CSV | `/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv` |
| Design doc | `docs/plans/2026-02-28-vs-preprocess-design.md` |

---

## WEEK 1 — Foundation

---

### Task 1: Scaffold project directory + requirements.txt

**Files:**
- Create: `vs_preprocess/` directory tree
- Create: `vs_preprocess/requirements.txt`
- Create: `vs_preprocess/tests/__init__.py`
- Create: `vs_preprocess/training/__init__.py`

**Step 1: Create the directory structure**

```bash
cd /home/kwa/Documents/vs_ai_registration
mkdir -p vs_preprocess/tests
mkdir -p vs_preprocess/training/configs
touch vs_preprocess/tests/__init__.py
touch vs_preprocess/training/__init__.py
```

**Step 2: Create requirements.txt**

Copy exactly from spec Section 2:

```
torch>=2.0,<3.0
monai>=1.3,<2.0
nibabel>=4.0,<6.0
numpy>=1.22,<2.0
SimpleITK>=2.2,<3.0
scipy>=1.9,<2.0
matplotlib>=3.6,<4.0
tqdm>=4.64
pandas>=1.5,<3.0
tensorboard>=2.12
pyyaml>=6.0
hd-bet
```

Save to `vs_preprocess/requirements.txt`.

**Step 3: Install dependencies**

```bash
cd vs_preprocess
pip install -r requirements.txt
```

**Step 4: Verify imports gate**

```bash
python -c "import monai, torch, nibabel, SimpleITK, scipy, matplotlib, pandas, yaml; print('All imports OK')"
```
Expected output: `All imports OK`

**Step 5: Commit**

```bash
git init   # (only if not already a git repo)
git add vs_preprocess/
git commit -m "feat: scaffold vs_preprocess project with requirements"
```

---

### Task 2: dataset.py — data discovery with CSV-based loading

**Files:**
- Create: `vs_preprocess/dataset.py`

**Context — 2 adaptations from spec:**

1. **`_parse_patient_timepoint` uses FIRST underscore**, not last.
   - Spec: `dirname.rfind("_")` → `126_0_0` gives patient=`126_0`, tp=`0` ❌
   - Correct: `dirname.find("_")` → `126_0_0` gives patient=`126`, tp=`0_0` ✓

2. **Add `discover_scans_from_csv()`** — the seg files are named `*_R_VS__uvauser2__coregistered.nii.gz` which don't match the spec's `_seg|_label` pattern. The CSV has the correct paths in columns `image path` / `coregistered_image` and `segmentation path` / `coregistered_segmentation`.

**Step 1: Copy spec Section 3 into `vs_preprocess/dataset.py`**

Copy the full code block from spec Section 3 verbatim.

**Step 2: Apply Adaptation 1 — fix `_parse_patient_timepoint`**

Find and replace in the function:

```python
# BEFORE (spec version — wrong for this data):
idx = dirname.rfind("_")

# AFTER (correct for 126_0_0 naming):
idx = dirname.find("_")
```

**Step 3: Add `discover_scans_from_csv()` at the end of `dataset.py`**

```python
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
        'Study_Labels'        -> modality (via Series Description mapping)
        'image path'          -> volume (fallback: 'coregistered_image')
        'segmentation path'   -> seg (fallback: 'coregistered_segmentation')

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
        # Resolve modality from Study_Labels column (e.g. t1+_thick_ax)
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
```

**Step 4: Smoke-test dataset loading**

```bash
cd vs_preprocess
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from dataset import discover_scans_from_csv, group_by_patient

CSV = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv")
DATA_ROOT = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7")

records = discover_scans_from_csv(CSV, DATA_ROOT)
print(f"Records loaded: {len(records)}")

groups = group_by_patient(records)
print(f"Patient-modality groups: {len(groups)}")

# Show a sample group
key = list(groups.keys())[0]
grp = groups[key]
print(f"\nSample group: {key}")
for r in grp:
    print(f"  {r.patient_timepoint} vol={r.volume_path.name} seg={r.seg_path.name if r.seg_path else 'None'} baseline={r.is_baseline}")
EOF
```

Expected: prints record count (~1603), group count (~187×4 modalities but will vary), sample group with correct patient=`126` and timepoints like `0_0`, `1_312`, etc.

**Step 5: Commit**

```bash
git add vs_preprocess/dataset.py
git commit -m "feat: dataset.py with CSV-based discovery and first-underscore patient parsing"
```

---

### Task 3: affine_utils.py — coordinate math (CRITICAL)

**Files:**
- Create: `vs_preprocess/affine_utils.py`

> ⚠️ This is the most critical module. Bugs silently corrupt all downstream results.

**Step 1: Copy spec Section 4 verbatim into `vs_preprocess/affine_utils.py`**

No changes needed. Copy the entire code block from spec Section 4 exactly.

**Step 2: Quick import check**

```bash
cd vs_preprocess
python -c "from affine_utils import compose_affine, decompose_affine, normalized_affine_to_world, apply_affine_to_volume, warp_volume_torch; print('affine_utils imports OK')"
```

Expected: `affine_utils imports OK`

**Step 3: Commit**

```bash
git add vs_preprocess/affine_utils.py
git commit -m "feat: affine_utils.py - coordinate math foundation"
```

---

### Task 4: tests/test_affine_utils.py — GATE TEST (must pass before Week 2)

**Files:**
- Create: `vs_preprocess/tests/test_affine_utils.py`

> 🚨 ALL TESTS IN THIS FILE MUST PASS BEFORE ANY OTHER PIPELINE WORK PROCEEDS.

**Step 1: Copy spec Section 20 verbatim into `vs_preprocess/tests/test_affine_utils.py`**

No changes needed. Copy the entire code block from spec Section 20.

**Step 2: Run the tests**

```bash
cd vs_preprocess
python -m pytest tests/test_affine_utils.py -v
```

Expected output — all must PASS:
```
PASSED tests/test_affine_utils.py::TestComposeDecompose::test_identity
PASSED tests/test_affine_utils.py::TestComposeDecompose::test_rigid_identity
PASSED tests/test_affine_utils.py::TestComposeDecompose::test_roundtrip
PASSED tests/test_affine_utils.py::TestComposeDecompose::test_determinant_unity
PASSED tests/test_affine_utils.py::TestCoordinateConversion::test_v2n_n2v_roundtrip
PASSED tests/test_affine_utils.py::TestCoordinateConversion::test_v2n_corners
PASSED tests/test_affine_utils.py::TestKnownTransformRoundtrip::test_5mm_z_translation
PASSED tests/test_affine_utils.py::TestKnownTransformRoundtrip::test_apply_and_recover
PASSED tests/test_affine_utils.py::TestWarpVolumeTorch::test_identity_warp
```

If any test fails: debug `affine_utils.py` before proceeding. Do NOT move to Task 5 until all pass.

**Step 3: Commit**

```bash
git add vs_preprocess/tests/test_affine_utils.py
git commit -m "test: affine_utils gate tests all passing"
```

---

### Task 5: preprocessing.py — N4, normalize, resample

**Files:**
- Create: `vs_preprocess/preprocessing.py`

**Step 1: Copy spec Section 5 verbatim into `vs_preprocess/preprocessing.py`**

No changes needed.

**Step 2: Test N4 on one real scan**

```bash
cd vs_preprocess
python - <<'EOF'
from pathlib import Path
from preprocessing import run_n4_correction
import tempfile, time

scan = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies/t1+_thick_ax/126_0_0/126_0_0_t1+_thick_ax_image_coregistered.nii.gz")
out = Path(tempfile.mktemp(suffix="_n4.nii.gz"))

t = time.perf_counter()
run_n4_correction(scan, out)
print(f"N4 done in {time.perf_counter()-t:.1f}s -> {out}")
print(f"Output exists: {out.exists()}")
EOF
```

Expected: completes in 30-120s, output file exists.

**Step 3: Test resampling**

```bash
cd vs_preprocess
python - <<'EOF'
import nibabel as nib
from pathlib import Path
from preprocessing import resample_to_isotropic

scan = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies/t1+_thick_ax/126_0_0/126_0_0_t1+_thick_ax_image_coregistered.nii.gz")
nii = nib.load(str(scan))
print(f"Input spacing: {[round(float(x),3) for x in nii.header.get_zooms()]}")

resampled = resample_to_isotropic(nii, (0.5, 0.5, 0.5))
print(f"Output spacing: {[round(float(x),3) for x in resampled.header.get_zooms()]}")
print(f"Output shape: {resampled.shape}")
EOF
```

Expected: output spacing = `[0.5, 0.5, 0.5]`.

**Step 4: Commit**

```bash
git add vs_preprocess/preprocessing.py
git commit -m "feat: preprocessing.py - N4, normalize, resample"
```

---

### Task 6: skull_strip.py — HD-BET wrapper

**Files:**
- Create: `vs_preprocess/skull_strip.py`

**Step 1: Copy spec Section 6 verbatim into `vs_preprocess/skull_strip.py`**

No changes needed.

**Step 2: Verify HD-BET is installed**

```bash
python -c "from HD_BET.run import run_hd_bet; print('HD-BET OK')"
```

If not found: `pip install hd-bet` (already in requirements.txt — check it ran).

**Step 3: Test skull strip on one baseline scan**

```bash
cd vs_preprocess
python - <<'EOF'
from pathlib import Path
import tempfile
from skull_strip import extract_brain_mask

scan = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies/t1+_thick_ax/126_0_0/126_0_0_t1+_thick_ax_image_coregistered.nii.gz")
out = Path(tempfile.mktemp(suffix="_mask.nii.gz"))

# Use cpu if no GPU available
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

extract_brain_mask(scan, out, device=device)
print(f"Mask saved: {out.exists()}")

import nibabel as nib, numpy as np
mask = nib.load(str(out)).get_fdata()
print(f"Mask shape: {mask.shape}, nonzero voxels: {np.sum(mask > 0)}")
EOF
```

Expected: mask exists, nonzero voxels > 0 (should be in tens of thousands for a brain).

**Step 4: Commit**

```bash
git add vs_preprocess/skull_strip.py
git commit -m "feat: skull_strip.py - HD-BET wrapper with dilation"
```

---

## WEEK 2 — Training

---

### Task 7: training/model.py — AffineRegNet

**Files:**
- Create: `vs_preprocess/training/model.py`

**Step 1: Copy spec Section 11 verbatim into `vs_preprocess/training/model.py`**

No changes needed.

**Step 2: Gate test — forward pass produces near-identity at init**

```bash
cd vs_preprocess
python - <<'EOF'
import torch
from training.model import AffineRegNet

model = AffineRegNet(image_size=(96,96,96), transform_type="affine")
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

x = torch.randn(2, 2, 96, 96, 96)
params, affine = model(x)
print(f"params shape: {params.shape}")  # (2, 9)
print(f"affine shape: {affine.shape}")  # (2, 3, 4)

# Near-identity check: with zero-initialized head, affine should be close to eye(3,4)
identity = torch.eye(3, 4).unsqueeze(0).expand(2, -1, -1)
max_dev = (affine - identity).abs().max().item()
print(f"Max deviation from identity: {max_dev:.6f}")
assert max_dev < 0.01, f"Expected near-identity at init, got max_dev={max_dev}"
print("PASS: forward pass produces near-identity at init")
EOF
```

Expected: params shape `(2, 9)`, affine shape `(2, 3, 4)`, max deviation < 0.01.

**Step 3: Commit**

```bash
git add vs_preprocess/training/model.py
git commit -m "feat: training/model.py - AffineRegNet with zero-init decomposed head"
```

---

### Task 8: training/augmentations.py

**Files:**
- Create: `vs_preprocess/training/augmentations.py`

**Step 1: Copy spec Section 13 verbatim into `vs_preprocess/training/augmentations.py`**

**Step 2: Quick test**

```bash
cd vs_preprocess
python - <<'EOF'
import torch
from training.augmentations import apply_intensity_augmentation

vol = torch.rand(1, 32, 32, 32)
aug = apply_intensity_augmentation(vol, p=1.0)
print(f"Input range: [{vol.min():.3f}, {vol.max():.3f}]")
print(f"Output range: [{aug.min():.3f}, {aug.max():.3f}]")
print(f"Shape preserved: {aug.shape == vol.shape}")
EOF
```

**Step 3: Commit**

```bash
git add vs_preprocess/training/augmentations.py
git commit -m "feat: training/augmentations.py - intensity augmentations"
```

---

### Task 9: training/losses.py

**Files:**
- Create: `vs_preprocess/training/losses.py`

**Step 1: Copy spec Section 14 verbatim into `vs_preprocess/training/losses.py`**

**Step 2: Verify losses return finite scalars**

```bash
cd vs_preprocess
python - <<'EOF'
import torch
from training.losses import (
    get_similarity_loss, identity_regularization,
    determinant_regularization, supervised_param_loss,
)

# MI loss
mi_fn = get_similarity_loss("mi")
vol = torch.rand(2, 1, 32, 32, 32)
mi_val = mi_fn(vol, vol * 0.95 + 0.025)
print(f"MI loss: {mi_val.item():.4f} (finite: {mi_val.isfinite().item()})")

# Regularization
params = torch.randn(2, 9) * 0.1
affine = torch.eye(3, 4).unsqueeze(0).expand(2, -1, -1)
id_loss = identity_regularization(params)
det_loss = determinant_regularization(affine)
print(f"Identity reg: {id_loss.item():.4f}, Det reg: {det_loss.item():.4f}")

# Supervised
gt = torch.zeros(2, 9)
sup = supervised_param_loss(params, gt)
print(f"Supervised loss: {sup.item():.4f}")
print("All losses finite:", all(x.isfinite().item() for x in [mi_val, id_loss, det_loss, sup]))
EOF
```

Expected: all values finite, no NaN/Inf.

**Step 3: Commit**

```bash
git add vs_preprocess/training/losses.py
git commit -m "feat: training/losses.py - MI, regularization, supervised param loss"
```

---

### Task 10: training/dataset.py — LongitudinalPairDataset + SyntheticPairDataset

**Files:**
- Create: `vs_preprocess/training/dataset.py`

**Step 1: Copy spec Section 12 verbatim into `vs_preprocess/training/dataset.py`**

No changes needed.

**Step 2: Test dataset loading yields correct shapes**

```bash
cd vs_preprocess
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from pathlib import Path
from dataset import discover_scans_from_csv, group_by_patient
from training.dataset import LongitudinalPairDataset, SyntheticPairDataset, CombinedDataset

CSV = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv")
DATA_ROOT = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7")

records = discover_scans_from_csv(CSV, DATA_ROOT)
groups = group_by_patient(records)

# Use a small subset for testing
sample_groups = dict(list(groups.items())[:5])
real_ds = LongitudinalPairDataset(sample_groups, target_size=(96, 96, 96))
print(f"Real pairs: {len(real_ds)}")

if len(real_ds) > 0:
    sample = real_ds[0]
    print(f"  moving shape: {sample['moving'].shape}")  # (1, 96, 96, 96)
    print(f"  fixed shape: {sample['fixed'].shape}")
    print(f"  has_gt: {sample['has_gt']}")

# Synthetic dataset
all_paths = [r.volume_path for rs in sample_groups.values() for r in rs]
syn_ds = SyntheticPairDataset(all_paths, target_size=(96,96,96), length=10)
sample_syn = syn_ds[0]
print(f"Synthetic sample - moving: {sample_syn['moving'].shape}, has_gt: {sample_syn['has_gt']}")
EOF
```

Expected: shapes `(1, 96, 96, 96)`, `has_gt=True` for synthetic, `has_gt=False` for real.

**Step 3: Commit**

```bash
git add vs_preprocess/training/dataset.py
git commit -m "feat: training/dataset.py - longitudinal pairs + synthetic pairs"
```

---

### Task 11: training configs

**Files:**
- Create: `vs_preprocess/training/configs/intra_patient.yaml`
- Create: `vs_preprocess/training/configs/template.yaml`
- Create: `vs_preprocess/training/requirements.txt`

**Step 1: Copy spec Section 18 into `vs_preprocess/training/configs/intra_patient.yaml`**

**Step 2: Copy spec Section 19 into `vs_preprocess/training/configs/template.yaml`**

**Step 3: Create `vs_preprocess/training/requirements.txt`** (same as root requirements.txt)

```
torch>=2.0,<3.0
monai>=1.3,<2.0
nibabel>=4.0,<6.0
numpy>=1.22,<2.0
SimpleITK>=2.2,<3.0
scipy>=1.9,<2.0
matplotlib>=3.6,<4.0
tqdm>=4.64
pandas>=1.5,<3.0
tensorboard>=2.12
pyyaml>=6.0
hd-bet
```

**Step 4: Commit**

```bash
git add vs_preprocess/training/configs/ vs_preprocess/training/requirements.txt
git commit -m "feat: training configs and requirements"
```

---

### Task 12: training/train_intra.py — intra-patient alignment training

**Files:**
- Create: `vs_preprocess/training/train_intra.py`

**Step 1: Copy spec Section 15 verbatim into `vs_preprocess/training/train_intra.py`**

No changes needed.

**Step 2: Smoke test — first 2 epochs (verifies no crash)**

```bash
cd vs_preprocess
python training/train_intra.py \
    --config training/configs/intra_patient.yaml \
    --data_dir /home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies \
    --output_dir /tmp/vs_train_test
```

Wait for 2 epochs. Expected: loss prints, no NaN, no crash.

**Step 3: Commit**

```bash
git add vs_preprocess/training/train_intra.py
git commit -m "feat: training/train_intra.py - intra-patient GlobalNet training"
```

**Step 4: Full training run (background)**

Run to convergence (hours):

```bash
cd vs_preprocess
nohup python training/train_intra.py \
    --config training/configs/intra_patient.yaml \
    --data_dir /home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies \
    --output_dir /home/kwa/Documents/vs_ai_registration/training_output/intra \
    > /tmp/vs_train.log 2>&1 &
echo "Training PID: $!"
```

Monitor: `tail -f /tmp/vs_train.log`
Checkpoint: `training_output/intra/best_model.pth`

---

### Task 13: training/evaluate.py

**Files:**
- Create: `vs_preprocess/training/evaluate.py`

**Step 1: Copy spec Section 17 verbatim into `vs_preprocess/training/evaluate.py`**

**Step 2: Commit**

```bash
git add vs_preprocess/training/evaluate.py
git commit -m "feat: training/evaluate.py - evaluation metrics"
```

**Step 3: Evaluate once training is complete (run after Task 12 finishes)**

```bash
cd vs_preprocess
python - <<'EOF'
import sys; sys.path.insert(0, ".")
import torch
from pathlib import Path
from dataset import discover_scans_from_csv, group_by_patient
from training.model import AffineRegNet
from training.dataset import LongitudinalPairDataset, SyntheticPairDataset, CombinedDataset
from training.evaluate import evaluate_model
from torch.utils.data import DataLoader

MODEL = Path("/home/kwa/Documents/vs_ai_registration/training_output/intra/best_model.pth")
CSV = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv")
DATA_ROOT = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7")

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(str(MODEL), map_location=device)
model = AffineRegNet(**{k: ckpt["config"][k] for k in ["image_size","transform_type"]})
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()

records = discover_scans_from_csv(CSV, DATA_ROOT)
groups = group_by_patient(records)
# Use last 20% as val
keys = sorted(groups.keys())
val_keys = keys[:len(keys)//5]
val_groups = {k: groups[k] for k in val_keys}

val_ds = LongitudinalPairDataset(val_groups)
val_syn = SyntheticPairDataset([r.volume_path for rs in val_groups.values() for r in rs], length=200)
from training.dataset import CombinedDataset
val_combined = CombinedDataset(val_ds, val_syn, real_ratio=0.5)
val_loader = DataLoader(val_combined, batch_size=4, num_workers=2)

metrics = evaluate_model(model, val_loader, device=device)
print("\n=== Evaluation Results ===")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
EOF
```

Target: rotation MAE < 2°, translation MAE < 0.02 (normalized), det_mean ≈ 1.0.

---

## WEEK 3 — Inference Pipeline

---

### Task 14: registration.py — GlobalNet inference

**Files:**
- Create: `vs_preprocess/registration.py`

**Step 1: Copy spec Section 7 verbatim into `vs_preprocess/registration.py`**

No changes needed.

**Step 2: Smoke test — align 1 patient (requires trained model)**

```bash
cd vs_preprocess
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from pathlib import Path
import torch
from dataset import discover_scans_from_csv, group_by_patient
from registration import load_model, align_patient_longitudinal

MODEL = Path("/home/kwa/Documents/vs_ai_registration/training_output/intra/best_model.pth")
CSV = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv")
DATA_ROOT = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7")
OUT = Path("/tmp/vs_reg_test")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, config = load_model(MODEL, device)

records = discover_scans_from_csv(CSV, DATA_ROOT)
groups = group_by_patient(records)

# Pick a patient with multiple timepoints
key = next(k for k, v in groups.items() if len(v) >= 2)
patient_records = groups[key]
print(f"Aligning: {key} ({len(patient_records)} timepoints)")

results = align_patient_longitudinal(patient_records, model, OUT, device=device)
for tp, info in results.items():
    p = info["params_dict"]
    print(f"  tp={tp}: rot={[round(x,1) for x in p['rotation_deg']]}° det={p['determinant']:.4f}")
EOF
```

Expected: determinants near 1.0 for baseline, small rotations for follow-ups.

**Step 3: Commit**

```bash
git add vs_preprocess/registration.py
git commit -m "feat: registration.py - GlobalNet inference and affine application"
```

---

### Task 15: qc.py — visualization

**Files:**
- Create: `vs_preprocess/qc.py`

**Step 1: Copy spec Section 8 verbatim into `vs_preprocess/qc.py`**

**Step 2: Test QC PNG generation**

```bash
cd vs_preprocess
python - <<'EOF'
from pathlib import Path
import nibabel as nib, numpy as np, tempfile
from qc import generate_scan_qc

# Use a preprocessed volume if available, else raw input
vol_path = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies/t1+_thick_ax/126_0_0/126_0_0_t1+_thick_ax_image_coregistered.nii.gz")

# Create a dummy mask
nii = nib.load(str(vol_path))
data = nii.get_fdata(dtype="float32")
mask_data = (data > data.mean()).astype("uint8")
mask_nii = nib.Nifti1Image(mask_data, nii.affine)
mask_path = Path(tempfile.mktemp(suffix="_mask.nii.gz"))
nib.save(mask_nii, str(mask_path))

out = Path("/tmp/test_qc.png")
generate_scan_qc(vol_path, mask_path, out, title="Test QC")
print(f"QC PNG saved: {out.exists()}, size: {out.stat().st_size} bytes")
EOF
```

Expected: PNG file exists, size > 10KB.

**Step 3: Commit**

```bash
git add vs_preprocess/qc.py
git commit -m "feat: qc.py - per-scan and longitudinal QC visualization"
```

---

### Task 16: pipeline.py — full orchestration

**Files:**
- Create: `vs_preprocess/pipeline.py`

**Step 1: Copy spec Section 9 verbatim into `vs_preprocess/pipeline.py`**

**Step 2: Add CSV-aware `run_pipeline_from_csv()` wrapper at the end**

This wrapper calls `discover_scans_from_csv()` instead of `discover_scans()` when a CSV path is provided:

```python
def run_pipeline_from_csv(
    csv_path: Path,
    data_root: Path,
    output_dir: Path,
    intra_model_path: Path,
    config: dict,
) -> List[ProcessingResult]:
    """
    run_pipeline variant that loads data from the coregistered CSV.

    All other arguments and behavior identical to run_pipeline().
    """
    from dataset import discover_scans_from_csv, group_by_patient

    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_scans_from_csv(csv_path, data_root, config.get("modalities"))
    if not records:
        logger.error("No scans found from CSV: %s", csv_path)
        return []

    groups = group_by_patient(records, config.get("baseline_selection", "first"))

    if config.get("dry_run", False):
        for key, group in groups.items():
            tps = ", ".join(r.patient_timepoint for r in group)
            bl = next(r for r in group if r.is_baseline)
            logger.info("  %s: baseline=%s, timepoints=[%s]", key, bl.patient_timepoint, tps)
        logger.info("Dry run: %d groups, %d scans.", len(groups), len(records))
        return []

    model, model_config = load_model(intra_model_path, config.get("device", "cuda"))

    all_results: List[ProcessingResult] = []
    patient_meta: Dict[str, dict] = {}

    for group_key, group_records in groups.items():
        results = process_patient_group(
            group_key, group_records, model, model_config, output_dir, config
        )
        all_results.extend(results)
        patient_meta[group_key] = {"records": group_records, "results": results}

    failures = [r for r in all_results if not r.success and not r.skipped]
    if failures:
        csv_path_out = output_dir / "failures.csv"
        with open(csv_path_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["patient_timepoint", "error"])
            for r in failures:
                writer.writerow([r.patient_timepoint, r.error or "unknown"])
        logger.warning("%d failures written to %s", len(failures), csv_path_out)

    try:
        generate_qc_summary_html(output_dir, patient_meta)
    except Exception as e:
        logger.warning("QC summary failed: %s", e)

    n_success = sum(r.success and not r.skipped for r in all_results)
    n_skip = sum(r.skipped for r in all_results)
    logger.info("Pipeline complete: %d success, %d skipped, %d failed",
                n_success, n_skip, len(failures))
    return all_results
```

**Step 3: Integration test — 3 patients end-to-end**

```bash
cd vs_preprocess
python - <<'EOF'
import sys; sys.path.insert(0, ".")
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from pipeline import run_pipeline_from_csv

CSV = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv")
DATA_ROOT = Path("/home/kwa/Documents/df_coregistered_02_23_2026_V6.7")
MODEL = Path("/home/kwa/Documents/vs_ai_registration/training_output/intra/best_model.pth")
OUT = Path("/tmp/vs_pipeline_test")

import torch
config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "transform_type": "affine",
    "intensity_norm": "minmax",
    "target_spacing": [0.5, 0.5, 0.5],
    "keep_intermediates": True,
    "skip_existing": False,
    "dry_run": False,
    # Limit to 3 patients for test — use modalities filter
    "modalities": ["t1+_thick_ax"],
}

# Dry run first
config["dry_run"] = True
run_pipeline_from_csv(CSV, DATA_ROOT, OUT, MODEL, config)
EOF
```

Expected: dry run lists patients without errors.

**Step 4: Commit**

```bash
git add vs_preprocess/pipeline.py
git commit -m "feat: pipeline.py with CSV-aware run_pipeline_from_csv wrapper"
```

---

### Task 17: run.py — CLI entry point

**Files:**
- Create: `vs_preprocess/run.py`

**Step 1: Copy spec Section 10 verbatim into `vs_preprocess/run.py`**

**Step 2: Extend `main()` to support `--csv_path` and `--data_root` flags**

Add two optional arguments to `parse_args()`:

```python
# After the existing required args:
p.add_argument("--csv_path", type=Path, default=None,
                help="CSV file for data loading (uses discover_scans_from_csv)")
p.add_argument("--data_root", type=Path, default=None,
                help="Root dir for resolving CSV relative paths")
```

Add CSV dispatch to `main()`, after `setup_logging`:

```python
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
```

**Step 3: Test CLI dry run**

```bash
cd vs_preprocess
python run.py \
    --data_dir /home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies \
    --csv_path /home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv \
    --data_root /home/kwa/Documents/df_coregistered_02_23_2026_V6.7 \
    --output_dir /tmp/vs_out \
    --intra_model_path /home/kwa/Documents/vs_ai_registration/training_output/intra/best_model.pth \
    --dry_run \
    --modalities t1+_thick_ax \
    --log_level INFO
```

Expected: lists discovered patients, exits 0.

**Step 4: Commit**

```bash
git add vs_preprocess/run.py
git commit -m "feat: run.py CLI with --csv_path support"
```

---

### Task 18: Full cohort run + failure analysis

**Step 1: Run full pipeline on all modalities**

```bash
python run.py \
    --data_dir /home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies \
    --csv_path /home/kwa/Documents/df_coregistered_02_23_2026_V6.7/df_coregistered_02_23_2026_V6.7.csv \
    --data_root /home/kwa/Documents/df_coregistered_02_23_2026_V6.7 \
    --output_dir /home/kwa/Documents/vs_ai_registration/pipeline_output \
    --intra_model_path /home/kwa/Documents/vs_ai_registration/training_output/intra/best_model.pth \
    --device cuda \
    --keep_intermediates \
    --log_level INFO
```

**Step 2: Check failure rate**

```bash
wc -l /home/kwa/Documents/vs_ai_registration/pipeline_output/failures.csv
```

Target: < 5% of patients (< ~10).

**Step 3: Open QC HTML in browser**

```bash
xdg-open /home/kwa/Documents/vs_ai_registration/pipeline_output/qc_summary.html
```

Verify: longitudinal QC images show aligned timepoints, seg contours visible.

**Step 4: Commit results summary**

```bash
git add vs_preprocess/
git commit -m "chore: full cohort run complete, QC verified"
```

---

## WEEK 4 — Polish

---

### Task 19: training/train_template.py — optional template model

**Files:**
- Create: `vs_preprocess/training/train_template.py`

**Step 1: Copy spec Section 16 verbatim into `vs_preprocess/training/train_template.py`**

**Step 2: Commit**

```bash
git add vs_preprocess/training/train_template.py
git commit -m "feat: training/train_template.py - MNI152 template registration"
```

**Step 3: Train template model (requires MNI152 template)**

```bash
python training/train_template.py \
    --config training/configs/template.yaml \
    --data_dir /home/kwa/Documents/df_coregistered_02_23_2026_V6.7/raw_studies \
    --template_path <path_to_MNI152.nii.gz> \
    --output_dir /home/kwa/Documents/vs_ai_registration/training_output/template
```

---

### Task 20: Edge cases + tests

**Step 1: Test 1-timepoint patient (baseline only)**

Verify pipeline doesn't crash when `len(records) == 1`.
Check `registration.py::align_patient_longitudinal` returns just the identity for baseline.

**Step 2: Test single-modality filter**

```bash
python run.py ... --modalities t2_thin --dry_run
```

**Step 3: Test `--force_reprocess`**

Run pipeline on a patient, run again with `--force_reprocess`, verify outputs are recreated.

**Step 4: Add `tests/test_dataset.py`**

```python
# vs_preprocess/tests/test_dataset.py
import pytest
from pathlib import Path
from dataset import _parse_patient_timepoint, discover_scans_from_csv, group_by_patient

def test_parse_patient_timepoint_first_underscore():
    """First-underscore split: 126_0_0 -> (126, 0_0)."""
    pid, tp = _parse_patient_timepoint("126_0_0")
    assert pid == "126"
    assert tp == "0_0"

def test_parse_no_underscore():
    pid, tp = _parse_patient_timepoint("VS001")
    assert pid == "VS001"
    assert tp == "tp1"

def test_group_by_patient_marks_baseline():
    from dataclasses import dataclass
    from dataset import ScanRecord, group_by_patient
    from pathlib import Path
    records = [
        ScanRecord("T2","p1_tp1","p1","tp1",Path("/a.nii.gz"),None),
        ScanRecord("T2","p1_tp2","p1","tp2",Path("/b.nii.gz"),None),
    ]
    groups = group_by_patient(records)
    assert "T2/p1" in groups
    assert groups["T2/p1"][0].is_baseline == True
    assert groups["T2/p1"][1].is_baseline == False
```

Run:
```bash
python -m pytest tests/test_dataset.py -v
```

**Step 5: Commit**

```bash
git add vs_preprocess/tests/test_dataset.py
git commit -m "test: dataset tests - patient parsing and grouping"
```

---

### Task 21: Final cleanup

**Step 1: Run full test suite**

```bash
cd vs_preprocess
python -m pytest tests/ -v
```

All tests must pass.

**Step 2: Lint (optional)**

```bash
pip install ruff
ruff check vs_preprocess/ --ignore E501
```

Fix any critical issues.

**Step 3: Final commit**

```bash
git add vs_preprocess/
git commit -m "feat: complete VS preprocessing pipeline - all tests passing"
```

---

## Summary of Adaptations from Spec

| Location | Spec Version | Adaptation |
|---|---|---|
| `dataset.py::_parse_patient_timepoint` | `rfind("_")` (last) | `find("_")` (first) |
| `dataset.py` | no CSV loading | add `discover_scans_from_csv()` |
| `pipeline.py` | `run_pipeline()` only | add `run_pipeline_from_csv()` |
| `run.py::main()` | always uses `discover_scans` | dispatches on `--csv_path` |

Everything else is verbatim from the spec.

---

## Gate Checklist

- [ ] Task 1: `python -c "import monai, torch, nibabel, SimpleITK"` → OK
- [ ] Task 4: ALL `test_affine_utils.py` tests PASS ← **Do not proceed without this**
- [ ] Task 7: Model forward pass → near-identity at init
- [ ] Task 9: All losses return finite scalars
- [ ] Task 12: Training loss decreasing, no NaN
- [ ] Task 14: 1 patient aligned, determinant ≈ 1.0
- [ ] Task 18: Full cohort < 5% failures, QC HTML renders correctly
