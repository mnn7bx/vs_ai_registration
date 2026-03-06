#!/bin/bash
#SBATCH --job-name=vs_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=vs_train_%j.log
#SBATCH --error=vs_train_%j.err

# ── Paths (EDIT THESE) ──────────────────────────────
PROJECT_DIR=/sfs/ceph/standard/gam_ai_group/people_scratch/matt
RAW_DATA=$PROJECT_DIR/data/unprocessed_raw_studies
ORGANIZED_DATA=$PROJECT_DIR/data/vs_organized
OUTPUT_DIR=$PROJECT_DIR/training_output/vs_intra
CONFIG=$PROJECT_DIR/vs_preprocess/training/configs/intra_vs.yaml
CONDA_ENV=monai_gpu_env_12_8_2025

# ── Environment setup ───────────────────────────────
module load anaconda3 2>/dev/null || source /etc/profile.d/conda.sh 2>/dev/null
conda activate $CONDA_ENV

# Install missing packages (skips already-installed)
pip install --quiet --no-input \
    "monai>=1.3,<2.0" \
    "nibabel>=4.0,<6.0" \
    "numpy>=1.22,<2.0" \
    "SimpleITK>=2.2,<3.0" \
    "scipy>=1.9,<2.0" \
    "matplotlib>=3.6,<4.0" \
    "tqdm>=4.64" \
    "pandas>=1.5,<3.0" \
    "tensorboard>=2.12" \
    "pyyaml>=6.0" \
    "hd-bet" \
    "pytest>=7.0"

echo "Environment ready. Installed packages:"
pip list 2>/dev/null | grep -iE "torch|monai|nibabel|numpy|simpleitk"

cd $PROJECT_DIR

# ── Step 1: Organize data (idempotent, skips existing) ──
echo "Organizing data..."
python vs_preprocess/scripts/organize_vs.py \
    --raw_dir $RAW_DATA \
    --output_dir $ORGANIZED_DATA

# ── Step 2: Train ───────────────────────────────────
echo "Starting training..."
python vs_preprocess/training/train_intra.py \
    --config $CONFIG \
    --data_dir $ORGANIZED_DATA \
    --output_dir $OUTPUT_DIR
