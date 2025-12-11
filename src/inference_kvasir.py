# src/inference_kvasir.py
"""
Inference / evaluation script for MedSam-like model on Kvasir-SEG.

Usage (from project root):
    python -m src.inference_kvasir --run_dir work_dir/medsam_kvasir_YYYYMMDD-HHMMSS
"""

import os
from os.path import join
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.dataset_kvasir import KvasirSegDataset
from src.model_medsam_like import MedSamLikeModel


def dice_metric_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = (2 * intersection + eps) / (union + eps)
    return dice


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="data",
        help="Path to Kvasir-SEG root folder",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to run directory containing checkpoint_best.pth",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=512,
        help="Input size used during training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=8,
        help="Number of qualitative examples to save",
    )

    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt_path = join(args.run_dir, "checkpoint_best.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found at {ckpt_path}")

    # -------- Dataset / loader (val split) --------
    val_dataset = KvasirSegDataset(
        args.data_root, split="val", img_size=args.img_size, bbox_shift=20
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # -------- Model & checkpoint --------
    model = MedSamLikeModel(img_size=args.img_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # -------- Evaluation --------
    all_dice = []
    vis_dir = join(args.run_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    num_vis_saved = 0

    pbar = tqdm(val_loader, desc="Eval", leave=False)
    for batch_idx, (images, masks, bboxes, names) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        bboxes = bboxes.to(device)

        logits = model(images, bboxes)
        batch_dice = dice_metric_from_logits(logits, masks)  # (B,)
        all_dice.extend(batch_dice.cpu().tolist())

        pbar.set_postfix({"dice": f"{batch_dice.mean().item():.4f}"})

        # qualitative visualizations
        if num_vis_saved < args.num_vis:
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            for i in range(images.size(0)):
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
                axs[1].set_title("Pred mask")
                axs[1].axis("off")

                axs[2].imshow(gt_np, cmap="gray")
                axs[2].imshow(pred_np, alpha=0.4, cmap="jet")
                axs[2].set_title("GT vs Pred")
                axs[2].axis("off")

                plt.tight_layout()
                out_name = f"{batch_idx:03d}_{i:02d}_{names[i]}.png"
                plt.savefig(join(vis_dir, out_name), dpi=150)
                plt.close()

                num_vis_saved += 1

    mean_dice = sum(all_dice) / max(len(all_dice), 1)
    print(f"Validation Dice (mean over {len(all_dice)} images): {mean_dice:.4f}")

    with open(join(args.run_dir, "eval_results.txt"), "w") as f:
        f.write(f"Mean validation Dice: {mean_dice:.4f}\n")

    print(f"Saved qualitative results in: {vis_dir}")
    print(f"Saved metrics to: {join(args.run_dir, 'eval_results.txt')}")


if __name__ == "__main__":
    main()
