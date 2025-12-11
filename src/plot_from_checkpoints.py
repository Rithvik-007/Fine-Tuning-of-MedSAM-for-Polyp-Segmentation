# src/plot_from_checkpoints.py
import os
from os.path import join
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry
from src.dataset_kvasir import KvasirSegDataset
from src.model_medsam_like import MedSamLikeModel


def dice_metric_from_logits(logits, targets, eps: float = 1e-6):
    """Compute Dice score from raw logits."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * inter + eps) / (union + eps)
    return dice  # (B,)


@torch.no_grad()
def eval_scratch_model(ckpt_path: str, data_root: str, device: torch.device, img_size: int = 512) -> float:
    print(f"[Scratch] Evaluating checkpoint: {ckpt_path}")
    ds = KvasirSegDataset(data_root, split="val", img_size=img_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = MedSamLikeModel().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    all_dice = []
    for images, masks, bboxes, _ in dl:
        images = images.to(device)          # (1,3,H,W)
        masks = masks.to(device)            # (1,1,H,W)
        bboxes = bboxes.to(device)          # (1,4)

        logits = model(images, bboxes)      # (1,1,H,W)
        dice = dice_metric_from_logits(logits, masks)
        all_dice.extend(dice.cpu().tolist())

    mean_dice = sum(all_dice) / max(len(all_dice), 1)
    print(f"[Scratch] Mean Dice: {mean_dice:.4f}")
    return mean_dice


@torch.no_grad()
def eval_medsam(ckpt_path: str, base_medsam_ckpt: str, data_root: str, device: torch.device,
                img_size: int = 512, label: str = "MedSAM") -> float:
    """
    ckpt_path:
        - For general MedSAM: pass base_medsam_ckpt itself and set is_finetuned=False
        - For fine-tuned MedSAM: pass finetuned checkpoint_best.pth and base_medsam_ckpt separately
    """
    print(f"[{label}] Evaluating checkpoint: {ckpt_path}")

    ds = KvasirSegDataset(data_root, split="val", img_size=img_size)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # Build base MedSAM architecture
    medsam = sam_model_registry["vit_b"](checkpoint=base_medsam_ckpt).to(device)

    # If ckpt_path != base_medsam_ckpt, we assume it's a fine-tuned state_dict
    if ckpt_path != base_medsam_ckpt:
        state = torch.load(ckpt_path, map_location=device)
        medsam.load_state_dict(state["model"])

    medsam.eval()

    all_dice = []
    for images, masks, bboxes, _ in dl:
        images = images.to(device)
        masks = masks.to(device)
        bboxes = bboxes.to(device)

        # Resize images to 1024x1024
        H, W = images.shape[2], images.shape[3]
        images_1024 = F.interpolate(
            images, size=(1024, 1024), mode="bilinear", align_corners=False
        )

        # Scale bboxes to 1024 coords
        # bboxes originally in (W,H) space = img_size x img_size
        scale_x = 1024.0 / W
        scale_y = 1024.0 / H
        b = bboxes.clone().float()
        b[:, 0] *= scale_x
        b[:, 2] *= scale_x
        b[:, 1] *= scale_y
        b[:, 3] *= scale_y
        b = b.view(-1, 1, 4)  # (B,1,4)

        # Forward through MedSAM
        img_embed = medsam.image_encoder(images_1024)
        sparse, dense = medsam.prompt_encoder(points=None, boxes=b, masks=None)
        low_res, _ = medsam.mask_decoder(
            image_embeddings=img_embed,
            image_pe=medsam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        logits = F.interpolate(
            low_res, size=(H, W), mode="bilinear", align_corners=False
        )

        dice = dice_metric_from_logits(logits, masks)
        all_dice.extend(dice.cpu().tolist())

    mean_dice = sum(all_dice) / max(len(all_dice), 1)
    print(f"[{label}] Mean Dice: {mean_dice:.4f}")
    return mean_dice


def plot_bar(dice_scratch, dice_general, dice_finetuned, out_path: str):
    import numpy as np
    labels = ["Scratch Model", "MedSAM (General)", "MedSAM (Fine-Tuned)"]
    scores = [dice_scratch, dice_general, dice_finetuned]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, scores)

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{score:.3f}",
                 ha="center", va="bottom", fontsize=11)

    plt.ylim(0.8, 1.0)
    plt.ylabel("Dice Score")
    plt.title("Model Performance Comparison on Kvasir-SEG (Dice)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved bar plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/Kvasir-SEG")
    parser.add_argument("--scratch_ckpt", required=True,
                        help="Path to scratch model checkpoint_best.pth")
    parser.add_argument("--medsam_ckpt", required=True,
                        help="Path to original MedSAM checkpoint (medsam_vit_b.pth)")
    parser.add_argument("--finetuned_ckpt", required=True,
                        help="Path to fine-tuned MedSAM checkpoint_best.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--out_dir", default="work_dir/plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1) Scratch model
    dice_scratch = eval_scratch_model(
        ckpt_path=args.scratch_ckpt,
        data_root=args.data_root,
        device=device,
        img_size=args.img_size,
    )

    # 2) MedSAM general (use base MedSAM checkpoint as both args)
    dice_general = eval_medsam(
        ckpt_path=args.medsam_ckpt,
        base_medsam_ckpt=args.medsam_ckpt,
        data_root=args.data_root,
        device=device,
        img_size=args.img_size,
        label="MedSAM General",
    )

    # 3) MedSAM fine-tuned
    dice_finetuned = eval_medsam(
        ckpt_path=args.finetuned_ckpt,
        base_medsam_ckpt=args.medsam_ckpt,
        data_root=args.data_root,
        device=device,
        img_size=args.img_size,
        label="MedSAM Fine-Tuned",
    )

    # Plot bar chart
    out_path = join(args.out_dir, "model_comparison_dice_bar.png")
    plot_bar(dice_scratch, dice_general, dice_finetuned, out_path)


if __name__ == "__main__":
    main()
