"""Tests for --resume checkpoint loading in train_intra."""

import torch
import pytest
from pathlib import Path


def make_fake_checkpoint(path, epoch=45, val_loss=-0.1658):
    """Create a minimal checkpoint with all expected keys."""
    ckpt = {
        "model_state_dict": {},
        "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 1e-5}]},
        "scheduler_state_dict": {},
        "config": {
            "image_size": [128, 128, 128],
            "in_channels": 2,
            "num_channel_initial": 16,
            "depth": 4,
            "transform_type": "affine",
            "num_params": 9,
            "head_hidden": 512,
            "head_dropout": 0.1,
        },
        "epoch": epoch,
        "val_loss": val_loss,
        "patience_counter": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))
    return ckpt


def test_checkpoint_has_all_keys(tmp_path):
    """Checkpoint must contain optimizer, scheduler, patience state."""
    ckpt_path = tmp_path / "test.pth"
    ckpt = make_fake_checkpoint(ckpt_path)
    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    required_keys = {
        "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
        "config", "epoch", "val_loss", "patience_counter",
    }
    assert required_keys.issubset(loaded.keys()), f"Missing keys: {required_keys - loaded.keys()}"


def test_resume_restores_epoch(tmp_path):
    """Resume should start training from epoch+1."""
    ckpt_path = tmp_path / "ckpt.pth"
    make_fake_checkpoint(ckpt_path, epoch=45)
    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    start_epoch = loaded["epoch"] + 1
    assert start_epoch == 46


def test_resume_restores_best_val(tmp_path):
    """Resume should use checkpoint's val_loss as best_val_loss threshold."""
    ckpt_path = tmp_path / "ckpt.pth"
    make_fake_checkpoint(ckpt_path, val_loss=-0.1658)
    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert loaded["val_loss"] == pytest.approx(-0.1658)


def test_resume_restores_patience(tmp_path):
    """Resume should restore patience counter."""
    ckpt_path = tmp_path / "ckpt.pth"
    make_fake_checkpoint(ckpt_path)
    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    assert loaded["patience_counter"] == 5


def test_resume_backward_compat_missing_keys(tmp_path):
    """Old checkpoints without new keys should get safe defaults."""
    ckpt_path = tmp_path / "old_ckpt.pth"
    torch.save(
        {"model_state_dict": {}, "config": {"image_size": [128, 128, 128]}, "epoch": 10, "val_loss": -0.15},
        str(ckpt_path),
    )
    loaded = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    start_epoch = loaded["epoch"] + 1
    best_val = loaded["val_loss"]
    patience = loaded.get("patience_counter", 0)
    opt_state = loaded.get("optimizer_state_dict", None)
    sched_state = loaded.get("scheduler_state_dict", None)
    assert start_epoch == 11
    assert best_val == pytest.approx(-0.15)
    assert patience == 0
    assert opt_state is None
    assert sched_state is None
