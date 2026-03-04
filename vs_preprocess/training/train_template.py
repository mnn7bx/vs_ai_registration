"""
Training script for template registration model.

Differences from train_intra.py:
1. Fixed image is ALWAYS the MNI152 template (loaded once).
2. All training uses synthetic pairs (patient -> template, with MI only).
3. Looser regularization: lambda_identity=0.001.
4. Wider synthetic ranges: rotation 30deg, translation 0.2, scale 0.15.
5. No supervised warm-up (MI from epoch 0).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import discover_scans, group_by_patient
from preprocessing import preprocess_for_network
from affine_utils import warp_volume_torch
from training.model import AffineRegNet
from training.augmentations import apply_intensity_augmentation
from training.losses import (
    get_similarity_loss,
    identity_regularization,
    determinant_regularization,
)

logger = logging.getLogger(__name__)


class TemplateRegistrationDataset(Dataset):
    """Each sample: (patient_scan, template). Fixed = template always."""

    def __init__(self, scan_paths, template_path, target_size=(96, 96, 96)):
        self.scan_paths = scan_paths
        self.template_tensor = preprocess_for_network(template_path, target_size).squeeze(0)
        self.target_size = target_size

    def __len__(self):
        return len(self.scan_paths)

    def __getitem__(self, idx):
        moving = preprocess_for_network(self.scan_paths[idx], self.target_size).squeeze(0)
        return {"moving": moving, "fixed": self.template_tensor}


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # All scans as moving images
    records = discover_scans(Path(args.data_dir))
    all_paths = [r.volume_path for r in records]

    # Split
    import numpy as np
    rng = np.random.RandomState(cfg.get("split_seed", 42))
    rng.shuffle(all_paths)
    n_val = max(1, int(len(all_paths) * cfg.get("val_fraction", 0.2)))
    train_paths, val_paths = all_paths[n_val:], all_paths[:n_val]

    template_path = Path(args.template_path)
    target_size = tuple(cfg.get("image_size", [96, 96, 96]))

    train_ds = TemplateRegistrationDataset(train_paths, template_path, target_size)
    val_ds = TemplateRegistrationDataset(val_paths, template_path, target_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 8),
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 8),
                            shuffle=False, num_workers=4)

    model = AffineRegNet(image_size=target_size, transform_type="affine").to(device)
    sim_loss_fn = get_similarity_loss(cfg.get("similarity_loss", "mi"))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.get("scheduler_T0", 50), T_mult=2)

    writer = SummaryWriter(str(output_dir / "tb_logs"))
    best_val = float("inf")
    patience_counter = 0

    lambda_id = cfg.get("lambda_identity", 0.001)  # looser
    lambda_det = cfg.get("lambda_det", 0.1)

    for epoch in range(cfg.get("max_epochs", 300)):
        model.train()
        epoch_loss = 0.0
        n = 0
        for batch in train_loader:
            moving = batch["moving"].to(device)
            fixed = batch["fixed"].to(device)
            moving = torch.stack([apply_intensity_augmentation(moving[i]) for i in range(moving.shape[0])])

            x = torch.cat([moving, fixed], dim=1)
            params, affine = model(x)
            warped = warp_volume_torch(moving, affine)

            loss = sim_loss_fn(warped, fixed) + lambda_id * identity_regularization(params) + lambda_det * determinant_regularization(affine)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n += 1

        scheduler.step()
        avg = epoch_loss / max(n, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                moving = batch["moving"].to(device)
                fixed = batch["fixed"].to(device)
                x = torch.cat([moving, fixed], dim=1)
                params, affine = model(x)
                warped = warp_volume_torch(moving, affine)
                vl = sim_loss_fn(warped, fixed) + lambda_id * identity_regularization(params)
                val_loss += vl.item()
                vn += 1
        avg_val = val_loss / max(vn, 1)

        writer.add_scalar("train/loss", avg, epoch)
        writer.add_scalar("val/loss", avg_val, epoch)
        logger.info("Epoch %d — train=%.4f val=%.4f", epoch + 1, avg, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "config": model.get_config(),
                        "epoch": epoch, "val_loss": best_val}, str(output_dir / "best_model.pth"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.get("patience", 30):
                break

    torch.save({"model_state_dict": model.state_dict(), "config": model.get_config(),
                "epoch": epoch, "val_loss": best_val}, str(output_dir / "globalnet_template.pth"))
    writer.close()


def main():
    logging.basicConfig(level=logging.INFO)
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--template_path", required=True)
    p.add_argument("--output_dir", required=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()
