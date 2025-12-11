# src/train_medsam_kvasir.py
"""
Training script for MedSam-like model on Kvasir-SEG.

Usage (from project root):
    python -m src.train_medsam_kvasir

You can override defaults with flags, e.g.:
    python -m src.train_medsam_kvasir --epochs 50 --batch_size 2 --device cuda:0
"""

import os
from os.path import join
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset_kvasir import KvasirSegDataset
from src.model_medsam_like import MedSamLikeModel


# ---------------------------------------------------------
# Dice loss and Dice metric
# ---------------------------------------------------------
def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """
    logits: (B,1,H,W) raw outputs from model
    targets: (B,1,H,W) binary masks (0/1)
    """
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def dice_metric_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """
    Compute Dice score using thresholded predictions.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


# ---------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, bce_loss_fn):
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks, bboxes, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        bboxes = bboxes.to(device)

        optimizer.zero_grad()

        logits = model(images, bboxes)  # (B,1,H,W)

        loss_dice = dice_loss_from_logits(logits, masks)
        loss_bce = bce_loss_fn(logits, masks)
        loss = loss_dice + loss_bce

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss /= max(len(loader), 1)
    return epoch_loss


@torch.no_grad()
def validate_one_epoch(model, loader, device, bce_loss_fn):
    model.eval()
    val_loss = 0.0
    val_dice = 0.0
    num_batches = 0

    for images, masks, bboxes, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        bboxes = bboxes.to(device)

        logits = model(images, bboxes)

        loss_dice = dice_loss_from_logits(logits, masks)
        loss_bce = bce_loss_fn(logits, masks)
        loss = loss_dice + loss_bce

        val_loss += loss.item()
        val_dice += dice_metric_from_logits(logits, masks)
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0

    val_loss /= num_batches
    val_dice /= num_batches
    return val_loss, val_dice


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Path to Kvasir-SEG root folder",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="work_dir",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Input image size (img_size x img_size)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size (keep small for 6GB VRAM)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (0 is safest on Windows)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for AdamW",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, e.g. 'cuda:0' or 'cpu'",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint_latest.pth to resume from (optional)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = join(args.work_dir, f"medsam_kvasir_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    log_path = join(run_dir, "train_log.txt")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------- Datasets ----------------
    train_dataset = KvasirSegDataset(
        args.data_root, split="train", img_size=args.img_size, bbox_shift=20
    )
    val_dataset = KvasirSegDataset(
        args.data_root, split="val", img_size=args.img_size, bbox_shift=20
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ---------------- Model ----------------
    model = MedSamLikeModel(img_size=args.img_size).to(device)

    # Optimizer on all trainable parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    bce_loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    best_val_dice = 0.0

    # Optional resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_dice = ckpt.get("best_val_dice", 0.0)
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    # ---------------- Training loop ----------------
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, bce_loss_fn)
        val_loss, val_dice = validate_one_epoch(model, val_loader, device, bce_loss_fn)

        log_line = (
            f"Epoch {epoch+1:03d}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}"
        )
        print(log_line)

        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        # Save latest checkpoint
        latest_ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_dice": best_val_dice,
            "args": vars(args),
        }
        torch.save(latest_ckpt, join(run_dir, "checkpoint_latest.pth"))

        # Save best checkpoint based on val_dice
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_ckpt = latest_ckpt
            torch.save(best_ckpt, join(run_dir, "checkpoint_best.pth"))
            print(f"  -> New best model saved (val_dice={best_val_dice:.4f})")


if __name__ == "__main__":
    main()
