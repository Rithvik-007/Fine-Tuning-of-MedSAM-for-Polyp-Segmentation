# src/model_medsam_like.py
"""
A simplified MedSAM-like architecture implemented fully in PyTorch.

- ResNet-50 image encoder (pretrained on ImageNet)
- Minimal prompt encoder (bbox -> embeddings)
- Lightweight mask decoder (transformer blocks)
- API:

    logits = model(images, bboxes)

  images: (B,3,img_size,img_size)
  bboxes: (B,4) in pixel coords [x_min,y_min,x_max,y_max]
  logits: (B,1,img_size,img_size)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------
# Positional encoding (sin/cos) for bounding box corner points
# ---------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """
    Converts 2D point coordinates into high-dimensional embeddings.
    Similar spirit to SAM's positional encoding.
    """

    def __init__(self, embed_dim: int = 256, img_size: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.img_size = img_size

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B, 2, 2) -> two points: [x_min,y_min] and [x_max,y_max]
        returns: (B, 2, embed_dim)
        """
        B, N, _ = pts.shape  # N=2

        # normalize to [0,1]
        pts = pts / float(self.img_size)

        # we'll encode x and y separately and concat
        d_half = self.embed_dim // 2  # for x and y
        d_quarter = d_half // 2       # for sin/cos of each

        freqs = torch.arange(d_quarter, device=pts.device).float()
        freqs = 1.0 / (10000 ** (freqs / d_quarter))

        # pts: (B,2,2) -> split x,y
        x = pts[:, :, 0].unsqueeze(-1)  # (B,2,1)
        y = pts[:, :, 1].unsqueeze(-1)  # (B,2,1)

        x_encoded = x * freqs  # (B,2,d_quarter)
        y_encoded = y * freqs  # (B,2,d_quarter)

        sin_x = torch.sin(x_encoded)
        cos_x = torch.cos(x_encoded)
        sin_y = torch.sin(y_encoded)
        cos_y = torch.cos(y_encoded)

        pe_x = torch.cat([sin_x, cos_x], dim=-1)  # (B,2,d_half)
        pe_y = torch.cat([sin_y, cos_y], dim=-1)  # (B,2,d_half)

        pe = torch.cat([pe_x, pe_y], dim=-1)      # (B,2,embed_dim)
        return pe


# ---------------------------------------------------------------
# Prompt Encoder
# ---------------------------------------------------------------
class PromptEncoder(nn.Module):
    """
    Converts bounding box -> sparse & dense embeddings.
    Dense embedding = learned positional grid (like SAM).
    """

    def __init__(self, embed_dim: int = 256, grid_size: int = 64, img_size: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # For bbox points
        self.point_encoder = PositionalEncoding(embed_dim=embed_dim, img_size=img_size)

        # Dense learned positional embedding (1, 256, 64, 64)
        self.pe = nn.Parameter(torch.randn(1, embed_dim, grid_size, grid_size))

    def forward(self, boxes: torch.Tensor):
        """
        boxes: (B, 4) -> [x_min,y_min,x_max,y_max]
        """
        B = boxes.shape[0]

        # Convert to two points
        pts = torch.stack(
            [
                boxes[:, 0:2],      # top-left
                boxes[:, 2:4],      # bottom-right
            ],
            dim=1,
        )  # (B, 2, 2)

        sparse = self.point_encoder(pts)         # (B,2,256)
        dense = self.pe.repeat(B, 1, 1, 1)       # (B,256,64,64)

        return sparse, dense

    def get_dense_pe(self):
        return self.pe


# ---------------------------------------------------------------
# Transformer-based Mask Decoder
# ---------------------------------------------------------------
class MaskDecoder(nn.Module):
    """
    Lightweight transformer + conv decoder.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim, nhead=num_heads, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

        # Final conv layers to produce mask logits
        self.conv1 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_dim, 1, 1)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool = False,
    ):
        """
        image_embeddings:         (B,256,64,64)
        image_pe:                 (1,256,64,64)
        sparse_prompt_embeddings: (B,2,256)
        dense_prompt_embeddings:  (B,256,64,64)  (currently unused directly)
        """
        B, C, H, W = image_embeddings.shape

        # Add positional encoding
        x = image_embeddings + image_pe

        # Flatten for transformer: (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Append sparse prompt tokens
        x = torch.cat([x, sparse_prompt_embeddings], dim=1)  # (B, H*W+2, C)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Remove sparse tokens -> keep image grid only
        x_img = x[:, :-2, :]  # (B, H*W, C)
        x_img = x_img.transpose(1, 2).view(B, C, H, W)

        # Decode to logits
        x = F.relu(self.conv1(x_img))
        low_res_masks = self.conv2(x)  # (B,1,H,W) = (B,1,64,64)

        # Upsample to 256x256 (like SAM)
        masks_256 = F.interpolate(
            low_res_masks, size=(256, 256), mode="bilinear", align_corners=False
        )

        return masks_256, None


# ---------------------------------------------------------------
# Complete MedSAM-like model with ResNet encoder
# ---------------------------------------------------------------
class MedSamLikeModel(nn.Module):
    def __init__(self, img_size: int = 512):
        super().__init__()
        self.img_size = img_size

        # --- ResNet-50 backbone ---
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove avgpool + fc -> feature map
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        # encoder output: (B,2048,H/32,W/32). For 512x512 -> (B,2048,16,16)

        # Project to 256 channels
        self.proj = nn.Conv2d(2048, 256, kernel_size=1)

        # Prompt encoder + mask decoder
        self.prompt_encoder = PromptEncoder(embed_dim=256, grid_size=64, img_size=img_size)
        self.mask_decoder = MaskDecoder(embed_dim=256)

        # Freeze prompt encoder
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor, bboxes: torch.Tensor) -> torch.Tensor:
        """
        images: (B,3,img_size,img_size)
        bboxes: (B,4)  in pixel coords [x_min,y_min,x_max,y_max]
        returns:
            logits: (B,1,img_size,img_size)
        """
        B = images.shape[0]

        # Encoder: ResNet feature map
        feats = self.encoder(images)  # (B,2048,16,16) for img_size=512

        # Project to 256 channels
        feats = self.proj(feats)  # (B,256,16,16)

        # Upsample to 64x64 grid to mimic SAM resolution
        feats = F.interpolate(feats, size=(64, 64), mode="bilinear", align_corners=False)

        # Prompt encoder
        sparse, dense = self.prompt_encoder(bboxes)

        # Mask decoder
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=feats,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )

        # Upsample to final image size
        final_masks = F.interpolate(
            low_res_masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        return final_masks
