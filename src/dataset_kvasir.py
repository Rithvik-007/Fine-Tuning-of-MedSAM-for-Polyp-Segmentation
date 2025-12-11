# src/dataset_kvasir.py

import os
from os.path import join
from typing import List, Tuple

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import glob


class KvasirSegDataset(Dataset):
    """
    Kvasir-SEG dataset loader for MedSAM-style training.

    Expects directory structure:

        root_dir/
            images/
                <id>.<ext>
            masks/
                <id>.<ext>
            train.txt
            val.txt

    train.txt / val.txt contain one ID or filename per line, e.g.:
        cju0qkwl35piu0993l0dewei2
    or   cju0qkwl35piu0993l0dewei2.jpg

    We resolve IDs to actual files by searching for matching files in
    images/ and masks/ folders.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 512,
        bbox_shift: int = 20,
    ):
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.root_dir = root_dir
        self.images_dir = join(root_dir, "images")
        self.masks_dir = join(root_dir, "masks")
        self.split = split
        self.img_size = img_size
        self.bbox_shift = bbox_shift

        list_file = join(root_dir, f"{split}.txt")
        if not os.path.isfile(list_file):
            raise FileNotFoundError(f"Split file not found: {list_file}")

        with open(list_file, "r") as f:
            raw_ids = [line.strip() for line in f if line.strip()]

        # Resolve each ID / name to actual image and mask paths
        self.samples: List[Tuple[str, str, str]] = []  # (img_path, mask_path, id_str)

        for id_str in raw_ids:
            img_path = self._resolve_file(self.images_dir, id_str)
            mask_path = self._resolve_file(self.masks_dir, id_str)

            if img_path is not None and mask_path is not None:
                self.samples.append((img_path, mask_path, id_str))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid image/mask pairs found in {root_dir} for split={split}"
            )

        print(f"[KvasirSegDataset] split={split}, samples={len(self.samples)}")

    def _resolve_file(self, folder: str, id_str: str) -> str:
        """
        Resolve an ID or filename to an existing file in `folder`.

        1. If id_str has extension and exists -> use it directly.
        2. Else, search for any file starting with that stem:
           e.g. 'cju0qk...' -> folder/cju0qk...*.jpg/png/etc.
        Returns first match or None.
        """
        # Direct match (if id_str already includes extension)
        direct_path = join(folder, id_str)
        if os.path.isfile(direct_path):
            return direct_path

        # Otherwise remove extension (if any) and search by stem
        stem, _ = os.path.splitext(id_str)
        pattern = join(folder, stem + ".*")
        candidates = glob.glob(pattern)

        if len(candidates) == 0:
            # nothing found
            return None

        # Just take the first one (they should be unique per id)
        return candidates[0]

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: str) -> np.ndarray:
        """Load image as RGB, float32, range [0,1]."""
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to load image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(
            img_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC
        )
        img_resized = img_resized.astype(np.float32)
        img_resized = img_resized - img_resized.min()
        max_val = max(img_resized.max(), 1e-8)
        img_resized = img_resized / max_val
        return img_resized  # (H, W, 3)

    def _load_mask(self, path: str) -> np.ndarray:
        """Load mask as single-channel binary [0,1]."""
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load mask: {path}")
        mask_resized = cv2.resize(
            mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST
        )
        mask_resized = (mask_resized > 0).astype(np.uint8)
        return mask_resized  # (H, W)

    def _mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute bounding box [x_min, y_min, x_max, y_max] from binary mask,
        and add random jitter up to bbox_shift.
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            # No foreground -> fallback to whole image
            x_min, y_min, x_max, y_max = 0, 0, self.img_size - 1, self.img_size - 1
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

        H, W = mask.shape
        if self.bbox_shift > 0:
            x_min = max(0, x_min - np.random.randint(0, self.bbox_shift + 1))
            x_max = min(W - 1, x_max + np.random.randint(0, self.bbox_shift + 1))
            y_min = max(0, y_min - np.random.randint(0, self.bbox_shift + 1))
            y_max = min(H - 1, y_max + np.random.randint(0, self.bbox_shift + 1))

        return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

    def __getitem__(self, idx: int):
        img_path, mask_path, id_str = self.samples[idx]

        img = self._load_image(img_path)   # (H, W, 3), float32 [0,1]
        mask = self._load_mask(mask_path)  # (H, W), uint8 0/1

        bbox = self._mask_to_bbox(mask)    # (4,)

        # HWC -> CHW
        img_chw = np.transpose(img, (2, 0, 1))          # (3, H, W)
        mask_chw = mask[None, :, :].astype(np.float32)  # (1, H, W)

        img_tensor = torch.from_numpy(img_chw).float()
        mask_tensor = torch.from_numpy(mask_chw).float()
        bbox_tensor = torch.from_numpy(bbox).float()

        return img_tensor, mask_tensor, bbox_tensor, id_str


# -------------------------------------------------------------------------
# Sanity check: run this file directly to visualize one batch.
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import random

    root = "data"  # adjust if needed
    dataset = KvasirSegDataset(root, split="train", img_size=512, bbox_shift=20)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    batch = next(iter(loader))
    images, masks, bboxes, names = batch

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(2):
        idx = random.randint(0, images.shape[0] - 1)
        img = images[idx].permute(1, 2, 0).cpu().numpy()
        mask = masks[idx].squeeze(0).cpu().numpy()
        bbox = bboxes[idx].cpu().numpy()

        # Image + mask
        axs[i, 0].imshow(img)
        axs[i, 0].imshow(mask, alpha=0.4, cmap="jet")
        axs[i, 0].axis("off")
        axs[i, 0].set_title(f"{names[idx]} - mask")

        # Image + bbox
        axs[i, 1].imshow(img)
        x_min, y_min, x_max, y_max = bbox
        axs[i, 1].add_patch(
            plt.Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                edgecolor="yellow",
                facecolor="none",
                linewidth=2,
            )
        )
        axs[i, 1].axis("off")
        axs[i, 1].set_title(f"{names[idx]} - bbox")

    plt.tight_layout()
    plt.savefig("kvasir_sanitycheck.png", dpi=200)
    plt.close()
    print("Saved sanity check image to kvasir_sanitycheck.png")
