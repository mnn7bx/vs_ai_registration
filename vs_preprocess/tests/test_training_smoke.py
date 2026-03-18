# tests/test_training_smoke.py
"""
Smoke test: run 2 epochs of three-tier training on real data with tiny volumes.
Verifies the full pipeline doesn't crash.
"""
import pytest
import tempfile
import yaml
from pathlib import Path

_DATA_DIR = Path("/home/kwa/Documents/vs_ai_registration/data/unprocessed_raw_studies")


@pytest.mark.skipif(not _DATA_DIR.exists(), reason="VS data not available")
def test_three_tier_training_smoke():
    """Run 2 epochs with three_tier loss on real data, batch_size=1."""
    import subprocess
    import sys

    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "image_size": [64, 64, 64],  # small for speed
            "target_spacing": [2.0, 2.0, 2.0],  # coarse for speed
            "transform_type": "affine",
            "similarity_loss": "three_tier",
            "pair_ratios": [0.25, 0.25, 0.25, 0.25],
            "ncc_kernel_size": 9,
            "mi_num_bins": 32,
            "loss_ema_decay": 0.99,
            "batch_size": 1,
            "num_workers": 0,
            "max_epochs": 2,
            "lr": 1e-4,
            "grad_clip_norm": 1.0,
            "scheduler": "plateau",
            "plateau_factor": 0.5,
            "plateau_patience": 10,
            "head_hidden": 64,
            "head_dropout": 0.0,
            "lambda_identity": 0.001,
            "lambda_det": 0.1,
            "lambda_supervised": 1.0,
            "lambda_supervised_decay": False,
            "warmup_start": 0,
            "warmup_mi_start": 0,
            "warmup_mi_end": 1,
            "patience": 100,
            "val_fraction": 0.3,
            "split_seed": 42,
            "synthetic_length_per_epoch": 4,
            "checkpoint_every_n_epochs": 10,
        }
        config_path = Path(tmpdir) / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = subprocess.run(
            [
                sys.executable, "-m", "training.train_intra",
                "--config", str(config_path),
                "--data_dir", str(_DATA_DIR),
                "--output_dir", str(Path(tmpdir) / "output"),
                "--max_patients", "5",
            ],
            capture_output=True, text=True, timeout=600,
            cwd=str(Path(__file__).parent.parent),
        )
        print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        assert result.returncode == 0, (
            f"Training failed:\n"
            f"STDOUT:\n{result.stdout[-1000:]}\n"
            f"STDERR:\n{result.stderr[-1000:]}"
        )
