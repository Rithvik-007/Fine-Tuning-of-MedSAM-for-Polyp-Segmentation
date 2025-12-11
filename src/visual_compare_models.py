# src/visual_compare_models.py

import os
from os.path import join
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from src.dataset_kvasir import KvasirSegDataset
from src.model_medsam_like import MedSamLikeModel
from segment_anything import sam_model_registry


def load_image(path):
    img = np.array(Image.open(path).convert("RGB"))
    return img


def compute_pred_scratch(model, img, bbox, device):
    model.eval()
    # img: HxWx3 uint8
    img_t = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    # bbox: numpy array (4,) -> convert to Tensor (1,4)
    bbox_t = torch.tensor(bbox, dtype=torch.float32, device=device).unsqueeze(0)  # (1,4)
    with torch.no_grad():
        logits = model(img_t, bbox_t)  # model expects tensor bbox
        pred = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
    return pred


def compute_pred_medsam(model, img, bbox, device):
    """
    img: H x W x 3 (uint8)
    bbox: numpy array (4,) in [x_min, y_min, x_max, y_max] on 512x512 space
    """
    model.eval()
    H, W, _ = img.shape

    # Image -> tensor, normalize
    img_t = torch.tensor(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0

    # Resize to 1024x1024 as MedSAM expects
    img_1024 = F.interpolate(
        img_t, size=(1024, 1024), mode="bilinear", align_corners=False
    )

    # Scale bbox from (W,H) to 1024x1024
    bbox = bbox.astype(float)
    x_min, y_min, x_max, y_max = bbox
    scale_x = 1024.0 / W
    scale_y = 1024.0 / H
    x_min *= scale_x
    x_max *= scale_x
    y_min *= scale_y
    y_max *= scale_y
    bbox_scaled = np.array([x_min, y_min, x_max, y_max], dtype=float)

    # Convert to tensor shape (1,1,4) for prompt_encoder
    bbox_t = torch.tensor(bbox_scaled, dtype=torch.float32, device=device).view(1, 1, 4)

    with torch.no_grad():
        img_embed = model.image_encoder(img_1024)
        sparse, dense = model.prompt_encoder(
            points=None,
            boxes=bbox_t,
            masks=None,
        )
        low_res, _ = model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        # Back to original H,W (512x512 in your dataset)
        logits = F.interpolate(
            low_res, size=(H, W), mode="bilinear", align_corners=False
        )
        pred = (torch.sigmoid(logits) > 0.5).float().squeeze().cpu().numpy()
    return pred



def plot_row(img, gt, pred_scratch, pred_general, pred_finetune, out_path):
    fig, axes = plt.subplots(1,5, figsize=(20,5))

    axes[0].imshow(img); axes[0].set_title("Image"); axes[0].axis("off")
    axes[1].imshow(gt, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred_scratch, cmap="gray"); axes[2].set_title("Scratch Model"); axes[2].axis("off")
    axes[3].imshow(pred_general, cmap="gray"); axes[3].set_title("MedSAM General"); axes[3].axis("off")
    axes[4].imshow(pred_finetune, cmap="gray"); axes[4].set_title("MedSAM Finetuned"); axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/Kvasir-SEG")
    parser.add_argument("--scratch_ckpt", required=True)
    parser.add_argument("--medsam_ckpt", required=True)
    parser.add_argument("--finetuned_ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="work_dir/vis_comparison")
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Dataset
    ds = KvasirSegDataset(args.data_root, split="val", img_size=512)

    # Load scratch model
    scratch = MedSamLikeModel().to(device)
    scratch.load_state_dict(torch.load(args.scratch_ckpt, map_location=device)["model"])
    scratch.eval()

    # Load MedSAM general
    medsam_general = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).to(device)
    medsam_general.eval()

    # Load MedSAM fine-tuned
    tmp = torch.load(args.finetuned_ckpt, map_location=device)
    medsam_ft = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt).to(device)
    medsam_ft.load_state_dict(tmp["model"])
    medsam_ft.eval()

    # Process samples
    for i in range(args.num_samples):
        img, gt, bbox, name = ds[i]
        img_np = (img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        gt_np = gt.squeeze().cpu().numpy()
        bbox_np = bbox.cpu().numpy()

        pred_scratch = compute_pred_scratch(scratch, img_np, bbox_np, device)
        pred_general = compute_pred_medsam(medsam_general, img_np, bbox_np, device)
        pred_ft = compute_pred_medsam(medsam_ft, img_np, bbox_np, device)

        out_path = join(args.output_dir, f"compare_{name.replace('.jpg','')}.png")
        plot_row(img_np, gt_np, pred_scratch, pred_general, pred_ft, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
