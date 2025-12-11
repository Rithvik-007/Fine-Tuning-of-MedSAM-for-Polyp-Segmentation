# src/train_finetune_medsam.py

import os
from os.path import join
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything import sam_model_registry
from src.dataset_kvasir import KvasirSegDataset


# -------------------------------------------------------------------
# Dice loss & Dice metric
# -------------------------------------------------------------------
def dice_loss_from_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def dice_metric_from_logits(logits, targets, eps=1e-6):
    preds = (torch.sigmoid(logits) > 0.5).float()
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return dice.mean().item()


# -------------------------------------------------------------------
# Training and validation loops
# -------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, device, bce_loss):
    model.train()
    epoch_loss = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, masks, bboxes, _ in pbar:
        images = images.to(device)          # (B,3,512,512)
        masks = masks.to(device)            # (B,1,512,512)
        bboxes = bboxes.to(device)          # (B,4)

        # ------------------------------------------------------------
        # Resize images → 1024 for MedSAM
        # ------------------------------------------------------------
        images_1024 = F.interpolate(
            images, size=(1024, 1024), mode="bilinear", align_corners=False
        )
        scale = 1024.0 / 512.0
        bboxes_1024 = bboxes * scale        # scale boxes too

        optimizer.zero_grad()

        # ------------------------------------------------------------
        # MedSAM forward
        # ------------------------------------------------------------
        img_embed = model.image_encoder(images_1024)     # (B,256,64,64)

        box_input = bboxes_1024[:, None, :]              # (B,1,4)
        sparse, dense = model.prompt_encoder(
            points=None, boxes=box_input, masks=None
        )

        low_res_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        logits_512 = F.interpolate(
            low_res_logits, size=(512, 512), mode="bilinear", align_corners=False
        )

        # ------------------------------------------------------------
        # Loss
        # ------------------------------------------------------------
        l1 = dice_loss_from_logits(logits_512, masks)
        l2 = bce_loss(logits_512, masks)
        loss = l1 + l2

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return epoch_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(model, loader, device, bce_loss):
    model.eval()
    total_loss = 0
    total_dice = 0
    batches = 0

    for images, masks, bboxes, _ in loader:
        images = images.to(device)
        masks = masks.to(device)
        bboxes = bboxes.to(device)

        images_1024 = F.interpolate(
            images, size=(1024, 1024), mode="bilinear", align_corners=False
        )
        scale = 1024.0 / 512.0
        bboxes_1024 = bboxes * scale

        img_embed = model.image_encoder(images_1024)
        sparse, dense = model.prompt_encoder(
            points=None, boxes=bboxes_1024[:, None, :], masks=None
        )
        low_res_logits, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits_512 = F.interpolate(
            low_res_logits, size=(512, 512), mode="bilinear", align_corners=False
        )

        l1 = dice_loss_from_logits(logits_512, masks)
        l2 = bce_loss(logits_512, masks)
        loss = l1 + l2
        dice = dice_metric_from_logits(logits_512, masks)

        total_loss += loss.item()
        total_dice += dice
        batches += 1

    return total_loss / batches, total_dice / batches


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--checkpoint", default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)  # must stay small
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work_dir", default="work_dir/finetuned_medsam")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = join(args.work_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = KvasirSegDataset(args.data_root, split="train", img_size=512)
    val_ds = KvasirSegDataset(args.data_root, split="val", img_size=512)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Load MedSAM
    model = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    model.to(device)

    # Freeze prompt encoder only
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    bce_loss = nn.BCEWithLogitsLoss()

    best_dice = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, bce_loss)
        val_loss, val_dice = validate_one_epoch(model, val_loader, device, bce_loss)

        print(f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_dice={val_dice:.4f}")

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
            },
            join(run_dir, "checkpoint_latest.pth"),
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_dice": best_dice,
                },
                join(run_dir, "checkpoint_best.pth"),
            )
            print(" -> Saved best model")

    print(f"Training completed. Best val Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
