"""
Training script for the intra-patient longitudinal alignment model.

Usage:
    python training/train_intra.py \
        --config training/configs/intra_patient.yaml \
        --data_dir /data/vs_cohort \
        --output_dir /training/intra
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import discover_scans, group_by_patient, group_by_patient_timepoint
from affine_utils import warp_volume_torch
from training.model import AffineRegNet
from training.dataset import LongitudinalPairDataset, SyntheticPairDataset, CombinedDataset
from training.multi_pair_dataset import MultiPairDataset, CombinedMultiPairDataset
from training.augmentations import apply_intensity_augmentation
from training.losses import (
    get_similarity_loss,
    ThreeTierLoss,
    identity_regularization,
    determinant_regularization,
    supervised_param_loss,
)

logger = logging.getLogger(__name__)


def split_by_patient(groups, val_fraction=0.2, seed=42):
    """Split patient groups into train/val, stratified by patient_id."""
    rng = np.random.RandomState(seed)
    keys = sorted(groups.keys())
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_fraction))
    val_keys = set(keys[:n_val])
    train_groups = {k: v for k, v in groups.items() if k not in val_keys}
    val_groups = {k: v for k, v in groups.items() if k in val_keys}
    return train_groups, val_groups


def split_patient_index(index, val_fraction=0.2, seed=42):
    """Split patient-timepoint index into train/val by patient_id."""
    rng = np.random.RandomState(seed)
    pids = sorted(index.keys())
    rng.shuffle(pids)
    n_val = max(1, int(len(pids) * val_fraction))
    val_pids = set(pids[:n_val])
    train_index = {k: v for k, v in index.items() if k not in val_pids}
    val_index = {k: v for k, v in index.items() if k in val_pids}
    return train_index, val_index


def get_all_scan_paths(groups):
    """Extract all volume paths from patient groups."""
    paths = []
    for records in groups.values():
        for r in records:
            paths.append(r.volume_path)
    return paths


def compute_lambda_mi(epoch, warmup_start, warmup_mi_start, warmup_mi_end, lambda_sim):
    """Compute MI loss weight with warm-up schedule."""
    if epoch < warmup_mi_start:
        return 0.0
    elif epoch < warmup_mi_end:
        frac = (epoch - warmup_mi_start) / max(warmup_mi_end - warmup_mi_start, 1)
        return lambda_sim * frac
    else:
        return lambda_sim


def compute_lambda_sup(epoch, cfg):
    """Compute supervised loss weight with optional linear decay to 0."""
    base = cfg.get("lambda_supervised", 0.5)
    if not cfg.get("lambda_supervised_decay", False):
        return base
    start = cfg.get("lambda_sup_decay_start", 30)
    end = cfg.get("lambda_sup_decay_end", 60)
    if epoch < start:
        return base
    elif epoch >= end:
        return 0.0
    else:
        frac = (epoch - start) / max(end - start, 1)
        return base * (1.0 - frac)


def train(args):
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Discover scans
    modality_filter = cfg.get("modalities", None)
    if args.csv_path:
        from dataset import discover_scans_from_csv
        records = discover_scans_from_csv(
            Path(args.csv_path),
            Path(args.data_root) if args.data_root else Path(args.data_dir).parent,
            modalities=modality_filter,
        )
    else:
        records = discover_scans(Path(args.data_dir), modalities=modality_filter)

    # Optional patient limit (for debugging / smoke tests)
    max_patients = getattr(args, "max_patients", None)
    if max_patients is not None:
        all_pids = sorted(set(r.patient_id for r in records))
        keep_pids = set(all_pids[:max_patients])
        records = [r for r in records if r.patient_id in keep_pids]
        logger.info("--max_patients=%d: kept %d scans from %d patients",
                     max_patients, len(records), len(keep_pids))

    # Common config
    target_size = tuple(cfg.get("image_size", [96, 96, 96]))
    target_spacing = tuple(cfg.get("target_spacing", [1.0, 1.0, 1.0]))
    transform_type = cfg.get("transform_type", "affine")
    loss_mode = cfg.get("similarity_loss", "ncc")

    if loss_mode == "three_tier":
        # ── Three-tier mode: patient-timepoint indexing ──────────
        pt_index = group_by_patient_timepoint(records)
        train_index, val_index = split_patient_index(
            pt_index,
            val_fraction=cfg.get("val_fraction", 0.2),
            seed=cfg.get("split_seed", 42),
        )

        # Log pair counts
        from training.multi_pair_dataset import PairType, enumerate_pairs
        all_pairs = enumerate_pairs(pt_index)
        logger.info(
            "Three-tier pairs: same_sub=%d, cross_sub=%d, cross_super=%d",
            sum(1 for p in all_pairs if p[2] == PairType.SAME_SUB),
            sum(1 for p in all_pairs if p[2] == PairType.CROSS_SUB),
            sum(1 for p in all_pairs if p[2] == PairType.CROSS_SUPER),
        )

        train_scan_count = sum(
            len(mods) for tps in train_index.values() for mods in tps.values()
        )
        val_scan_count = sum(
            len(mods) for tps in val_index.values() for mods in tps.values()
        )
        logger.info(
            "Train: %d patients (%d scans), Val: %d patients (%d scans)",
            len(train_index), train_scan_count,
            len(val_index), val_scan_count,
        )

        train_real = MultiPairDataset(train_index, target_size, target_spacing)
        train_scan_paths = []
        for pid, tps in train_index.items():
            for tp, mods in tps.items():
                for sub, rec in mods.items():
                    train_scan_paths.append(rec.volume_path)

        train_synthetic = SyntheticPairDataset(
            scan_paths=train_scan_paths,
            target_size=target_size,
            target_spacing=target_spacing,
            max_rotation_deg=cfg.get("max_rotation_deg", 15.0),
            max_translation=cfg.get("max_translation", 0.1),
            max_scale_offset=cfg.get("max_scale_offset", 0.05),
            length=cfg.get("synthetic_length_per_epoch", 1000),
            transform_type=transform_type,
        )
        train_dataset = CombinedMultiPairDataset(
            train_real, train_synthetic,
            ratios=cfg.get("pair_ratios", [0.25, 0.25, 0.25, 0.25]),
        )

        val_real = MultiPairDataset(val_index, target_size, target_spacing)
        val_scan_paths = []
        for pid, tps in val_index.items():
            for tp, mods in tps.items():
                for sub, rec in mods.items():
                    val_scan_paths.append(rec.volume_path)

        val_synthetic = SyntheticPairDataset(
            scan_paths=val_scan_paths,
            target_size=target_size,
            target_spacing=target_spacing,
            length=200,
            transform_type=transform_type,
        )
        val_dataset = CombinedMultiPairDataset(
            val_real, val_synthetic,
            ratios=cfg.get("pair_ratios", [0.25, 0.25, 0.25, 0.25]),
        )

        sim_loss_fn = ThreeTierLoss(
            ncc_kernel_size=cfg.get("ncc_kernel_size", 9),
            mi_num_bins=cfg.get("mi_num_bins", 32),
            ema_decay=cfg.get("loss_ema_decay", 0.99),
        )
    else:
        # ── Original single-loss path ────────────────────────────
        groups = group_by_patient(records)
        train_groups, val_groups = split_by_patient(
            groups,
            val_fraction=cfg.get("val_fraction", 0.2),
            seed=cfg.get("split_seed", 42),
        )
        logger.info(
            "Train: %d groups (%d scans), Val: %d groups (%d scans)",
            len(train_groups), sum(len(v) for v in train_groups.values()),
            len(val_groups), sum(len(v) for v in val_groups.values()),
        )

        train_real = LongitudinalPairDataset(train_groups, target_size, target_spacing)
        train_synthetic = SyntheticPairDataset(
            scan_paths=get_all_scan_paths(train_groups),
            target_size=target_size,
            target_spacing=target_spacing,
            max_rotation_deg=cfg.get("max_rotation_deg", 15.0),
            max_translation=cfg.get("max_translation", 0.1),
            max_scale_offset=cfg.get("max_scale_offset", 0.05),
            length=cfg.get("synthetic_length_per_epoch", 1000),
            transform_type=transform_type,
        )
        train_dataset = CombinedDataset(
            train_real, train_synthetic, real_ratio=cfg.get("real_ratio", 0.5)
        )

        val_real = LongitudinalPairDataset(val_groups, target_size, target_spacing)
        val_synthetic = SyntheticPairDataset(
            scan_paths=get_all_scan_paths(val_groups),
            target_size=target_size,
            target_spacing=target_spacing,
            max_rotation_deg=cfg.get("max_rotation_deg", 15.0),
            max_translation=cfg.get("max_translation", 0.1),
            max_scale_offset=cfg.get("max_scale_offset", 0.05),
            length=200,
            transform_type=transform_type,
        )
        val_dataset = CombinedDataset(val_real, val_synthetic, real_ratio=0.5)

        sim_loss_fn = get_similarity_loss(
            loss_mode,
            num_bins=cfg.get("mi_num_bins", 32),
            kernel_size=cfg.get("ncc_kernel_size", 9),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    model = AffineRegNet(
        image_size=target_size,
        transform_type=transform_type,
        head_hidden=cfg.get("head_hidden", 0),
        head_dropout=cfg.get("head_dropout", 0.0),
    ).to(device)
    logger.info("Model params: %d", sum(p.numel() for p in model.parameters()))

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-4))
    scheduler_type = cfg.get("scheduler", "cosine_restart")
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.get("plateau_factor", 0.5),
            patience=cfg.get("plateau_patience", 10),
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.get("scheduler_T_max", cfg.get("max_epochs", 200)),
            eta_min=cfg.get("scheduler_eta_min", 1e-7),
        )
    else:  # cosine_restart
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.get("scheduler_T0", 50),
            T_mult=cfg.get("scheduler_Tmult", 2),
        )

    # ── Resume from checkpoint ───────────────────────────
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except (ValueError, KeyError):
                logger.warning("Scheduler state incompatible, starting fresh scheduler")
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        patience_counter = ckpt.get("patience_counter", 0)
        logger.info(
            "Resumed from %s (epoch %d, best_val=%.4f, patience=%d)",
            ckpt_path, ckpt["epoch"], best_val_loss, patience_counter,
        )

    # Logging
    writer = SummaryWriter(str(output_dir / "tb_logs"))

    def _ckpt_config():
        """Model config + target_spacing for checkpoint reproducibility."""
        c = model.get_config()
        c["target_spacing"] = list(target_spacing)
        return c

    # Training
    max_epochs = cfg.get("max_epochs", 300)
    patience = cfg.get("patience", 30)

    lambda_id = cfg.get("lambda_identity", 0.01)
    lambda_det = cfg.get("lambda_det", 0.1)
    lambda_sim_max = cfg.get("lambda_sim", 1.0)
    warmup_start = cfg.get("warmup_start", 0)
    warmup_mi_start = cfg.get("warmup_mi_start", 10)
    warmup_mi_end = cfg.get("warmup_mi_end", 30)
    grad_clip = cfg.get("grad_clip_norm", 1.0)

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_sim = 0.0
        epoch_sup = 0.0
        epoch_grad_norm = 0.0
        n_batches = 0

        lam_mi = compute_lambda_mi(epoch, warmup_start, warmup_mi_start, warmup_mi_end, lambda_sim_max)
        lambda_sup = compute_lambda_sup(epoch, cfg)

        for batch in train_loader:
            moving = batch["moving"].to(device)  # (B, 1, D, H, W)
            fixed = batch["fixed"].to(device)
            has_gt = batch["has_gt"]  # (B,) bool
            gt_params = batch["gt_params"].to(device)  # (B, num_params)

            # Augment moving intensity
            aug_moving = torch.stack([
                apply_intensity_augmentation(moving[i]) for i in range(moving.shape[0])
            ], dim=0)

            x = torch.cat([aug_moving, fixed], dim=1)  # (B, 2, D, H, W)
            params, affine = model(x)

            warped = warp_volume_torch(aug_moving, affine, mode="bilinear")

            # Losses
            if loss_mode == "three_tier":
                pair_type = batch.get("pair_type", torch.zeros(moving.shape[0], dtype=torch.long))
                loss_sim = sim_loss_fn(warped, fixed, pair_type.to(device))
            else:
                loss_sim = lam_mi * sim_loss_fn(warped, fixed) if lam_mi > 0 else torch.tensor(0.0, device=device)
            loss_id = lambda_id * identity_regularization(params)
            loss_det = lambda_det * determinant_regularization(affine)

            loss_sup_val = torch.tensor(0.0, device=device)
            if has_gt.any():
                mask = has_gt.to(device)
                loss_sup_val = lambda_sup * supervised_param_loss(params[mask], gt_params[mask])

            loss = loss_sim + loss_id + loss_det + loss_sup_val

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            epoch_grad_norm += grad_norm.item()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_sim += loss_sim.item()
            epoch_sup += loss_sup_val.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation (with optional deterministic seeding)
        val_seed = cfg.get("val_seed")
        if val_seed is not None:
            random.seed(val_seed)
            np.random.seed(val_seed)
            torch.manual_seed(val_seed)

        model.eval()

        if loss_mode == "three_tier":
            # ── Three-tier validation: per-type metrics ──────────
            val_ncc_same = []
            val_mi_cross_sub = []
            val_mi_cross_super = []
            ncc_fn_val = get_similarity_loss("ncc", kernel_size=cfg.get("ncc_kernel_size", 9))
            mi_fn_val = get_similarity_loss("mi", num_bins=cfg.get("mi_num_bins", 32))

            with torch.no_grad():
                for batch in val_loader:
                    moving = batch["moving"].to(device)
                    fixed = batch["fixed"].to(device)
                    pair_type = batch.get(
                        "pair_type", torch.zeros(moving.shape[0], dtype=torch.long)
                    ).to(device)

                    x = torch.cat([moving, fixed], dim=1)
                    params, affine = model(x)
                    warped = warp_volume_torch(moving, affine)

                    for pt_val in [0, 1, 2]:
                        mask = pair_type == pt_val
                        if not mask.any():
                            continue
                        if pt_val == 0:
                            val_ncc_same.append(ncc_fn_val(warped[mask], fixed[mask]).item())
                        elif pt_val == 1:
                            val_mi_cross_sub.append(mi_fn_val(warped[mask], fixed[mask]).item())
                        else:
                            val_mi_cross_super.append(mi_fn_val(warped[mask], fixed[mask]).item())

            avg_ncc = np.mean(val_ncc_same) if val_ncc_same else 0.0
            avg_mi_cs = np.mean(val_mi_cross_sub) if val_mi_cross_sub else 0.0
            avg_mi_cx = np.mean(val_mi_cross_super) if val_mi_cross_super else 0.0
            avg_val_loss = np.mean([avg_ncc, avg_mi_cs, avg_mi_cx])

            writer.add_scalar("val/ncc_same_sub", avg_ncc, epoch)
            writer.add_scalar("val/mi_cross_sub", avg_mi_cs, epoch)
            writer.add_scalar("val/mi_cross_super", avg_mi_cx, epoch)
            writer.add_scalar("val/combined", avg_val_loss, epoch)
            val_sup_total = 0.0
            val_n = 1  # avoid div-by-zero in logging below
        else:
            # ── Original single-loss validation ──────────────────
            val_loss_total = 0.0
            val_sup_total = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    moving = batch["moving"].to(device)
                    fixed = batch["fixed"].to(device)
                    has_gt = batch["has_gt"]
                    gt_params = batch["gt_params"].to(device)

                    x = torch.cat([moving, fixed], dim=1)
                    params, affine = model(x)
                    warped = warp_volume_torch(moving, affine)

                    v_sim = sim_loss_fn(warped, fixed)
                    v_id = lambda_id * identity_regularization(params)
                    v_det = lambda_det * determinant_regularization(affine)
                    v_loss = v_sim + v_id + v_det

                    if has_gt.any():
                        mask = has_gt.to(device)
                        v_sup = supervised_param_loss(params[mask], gt_params[mask])
                        val_sup_total += v_sup.item()

                    val_loss_total += v_loss.item()
                    val_n += 1

            avg_val_loss = val_loss_total / max(val_n, 1)

        # Re-seed randomly after val to not affect training
        if val_seed is not None:
            reseed = int(torch.empty((), dtype=torch.int64).random_().item()) % (2**31)
            random.seed(reseed)
            np.random.seed(reseed)
            torch.manual_seed(reseed)

        # Step scheduler
        if scheduler_type == "plateau":
            scheduler.step(avg_val_loss)
        else:
            scheduler.step()

        # Logging
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/sim", epoch_sim / max(n_batches, 1), epoch)
        writer.add_scalar("train/sup", epoch_sup / max(n_batches, 1), epoch)
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        writer.add_scalar("val/sup", val_sup_total / max(val_n, 1), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("lambda_mi", lam_mi, epoch)
        writer.add_scalar("lambda_sup", lambda_sup, epoch)
        avg_grad_norm = epoch_grad_norm / max(n_batches, 1)
        writer.add_scalar("train/grad_norm", avg_grad_norm, epoch)
        if avg_grad_norm < 1e-7:
            logger.warning(
                "Epoch %d: gradient norm near zero (%.2e) — model may be stuck",
                epoch + 1, avg_grad_norm,
            )

        logger.info(
            "Epoch %d/%d — train=%.4f val=%.4f lr=%.2e lam_mi=%.2f lam_sup=%.2f",
            epoch + 1, max_epochs, avg_loss, avg_val_loss,
            optimizer.param_groups[0]["lr"], lam_mi, lambda_sup,
        )

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": _ckpt_config(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                },
                str(output_dir / "best_model.pth"),
            )
            logger.info("  New best model saved (val=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Periodic checkpoint
        ckpt_every = cfg.get("checkpoint_every_n_epochs", 10)
        if (epoch + 1) % ckpt_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": _ckpt_config(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "patience_counter": patience_counter,
                },
                str(output_dir / f"checkpoint_epoch{epoch+1}.pth"),
            )

    # Final save
    final_path = output_dir / "globalnet_intra.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": model.get_config(),
            "epoch": epoch,
            "val_loss": best_val_loss,
            "patience_counter": patience_counter,
        },
        str(final_path),
    )
    logger.info("Final model saved: %s", final_path)
    writer.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default=None, help="CSV file path for data loading")
    parser.add_argument("--data_root", type=str, default=None, help="Data root for CSV path resolution")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--max_patients", type=int, default=None,
                        help="Limit to N patients (for debugging / smoke tests)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
