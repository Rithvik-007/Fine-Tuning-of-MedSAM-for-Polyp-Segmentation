# src/eval_original_medsam_kvasir.py
"""
Evaluate ORIGINAL MedSAM (pretrained generalist model) on Kvasir-SEG val set.

Usage (from project root):
    python -m src.eval_original_medsam_kvasir \
        --data_root data/Kvasir-SEG \
        --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
        --device cuda:0
"""

import os
from os.path import join
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry
from src.dataset_kvasir import KvasirSegDataset


def dice_metric_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """
    Compute Dice score using thresholded predictions.
    logits: (B,1,H,W)
    targets: (B,1,H,W)
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice  # (B,)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Path to Kvasir-SEG root folder",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dir/MedSAM/medsam_vit_b.pth",
        help="Path to MedSAM ViT-B checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (e.g. cuda:0 or cpu)",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Image size used by KvasirSegDataset (must match training, e.g. 512)",
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=8,
        help="Number of qualitative examples to save",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="work_dir/medsam_eval_kvasir",
        help="Output directory for metrics and visualizations",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = join(args.out_dir, "vis_medsam")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------- Dataset / loader (val split) ----------------
    val_dataset = KvasirSegDataset(
        args.data_root, split="val", img_size=args.img_size, bbox_shift=20
    )
    # Force batch_size=1 to avoid SAM internal batch shape issues
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # ---------------- Load ORIGINAL MedSAM model ----------------
    print(f"Loading MedSAM checkpoint from: {args.checkpoint}")
    medsam = sam_model_registry["vit_b"](checkpoint=args.checkpoint)
    medsam.to(device)
    medsam.eval()

    all_dice = []
    num_vis_saved = 0

    pbar = tqdm(val_loader, desc="Eval MedSAM (generalist)", leave=False)
    for batch_idx, (images, masks, bboxes, names) in enumerate(pbar):
        # images: (1,3,512,512), masks: (1,1,512,512), bboxes: (1,4)
        images = images.to(device)
        masks = masks.to(device)
        bboxes = bboxes.to(device)

        B = images.size(0)  # should be 1, but keep it general

        # --- Resize images to 1024x1024 as MedSAM expects ---
        images_1024 = F.interpolate(
            images, size=(1024, 1024), mode="bilinear", align_corners=False
        )

        # --- Compute image embeddings ---
        image_embeddings = medsam.image_encoder(images_1024)  # (B,256,64,64)

        # --- Scale bboxes from img_size (e.g., 512) to 1024 ---
        scale = 1024.0 / float(args.img_size)
        bboxes_1024 = bboxes * scale  # (B,4)

        # Prepare boxes for prompt encoder: (B,1,4)
        box_torch = bboxes_1024
        if box_torch.dim() == 2:
            box_torch = box_torch[:, None, :]  # (B,1,4)

        # --- Prompt encoder & mask decoder (same as MedSAM inference flow) ---
        sparse_embeddings, dense_embeddings = medsam.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )
        low_res_logits, _ = medsam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=medsam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )  # (B,1,256,256)

        # Upsample back to Kvasir img_size (e.g. 512x512)
        logits_resized = F.interpolate(
            low_res_logits,
            size=(args.img_size, args.img_size),
            mode="bilinear",
            align_corners=False,
        )  # (B,1,512,512)

        batch_dice = dice_metric_from_logits(logits_resized, masks)  # (B,)
        all_dice.extend(batch_dice.cpu().tolist())
        pbar.set_postfix({"dice": f"{batch_dice.mean().item():.4f}"})

        # --- Save qualitative visualizations ---
        if num_vis_saved < args.num_vis:
            probs = torch.sigmoid(logits_resized)
            preds = (probs > 0.5).float()

            for i in range(B):
                if num_vis_saved >= args.num_vis:
                    break

                img_np = images[i].permute(1, 2, 0).cpu().numpy()
                gt_np = masks[i].squeeze(0).cpu().numpy()
                pred_np = preds[i].squeeze(0).cpu().numpy()

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))

                axs[0].imshow(img_np)
                axs[0].imshow(gt_np, alpha=0.4, cmap="jet")
                axs[0].set_title("GT mask")
                axs[0].axis("off")

                axs[1].imshow(img_np)
                axs[1].imshow(pred_np, alpha=0.4, cmap="jet")
                axs[1].set_title("MedSAM Pred mask")
                axs[1].axis("off")

                axs[2].imshow(gt_np, cmap="gray")
                axs[2].imshow(pred_np, alpha=0.4, cmap="jet")
                axs[2].set_title("GT vs MedSAM Pred")
                axs[2].axis("off")

                plt.tight_layout()
                out_name = f"{batch_idx:03d}_{i:02d}_{names[i]}_medsam.png"
                plt.savefig(join(vis_dir, out_name), dpi=150)
                plt.close()

                num_vis_saved += 1

    mean_dice = sum(all_dice) / max(len(all_dice), 1)
    print(f"MedSAM (generalist) - Validation Dice (mean over {len(all_dice)} images): {mean_dice:.4f}")

    metrics_path = join(args.out_dir, "eval_medsam_results.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Mean validation Dice: {mean_dice:.4f}\n")

    print(f"Saved MedSAM qualitative results to: {vis_dir}")
    print(f"Saved MedSAM metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
