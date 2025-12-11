# Domain-Specific Fine-Tuning of MedSAM for Polyp Segmentation

## Overview

This project investigates how well the **MedSAM** medical foundation model generalizes to an under-represented modality (endoscopic polyp segmentation) and whether **domain-specific fine-tuning** improves its performance. We evaluate three models:

- **Original MedSAM (Generalist)** -- Pretrained on 1.57M medical image-mask pairs, evaluated zero-shot
- **Fine-Tuned MedSAM (Specialist)** -- Adapted specifically to the Kvasir-SEG dataset
- **MedSamLike (Scratch Model)** -- A lighter ResNet-based baseline trained only on Kvasir

The primary goal is to test whether a foundation model pretrained on many modalities can outperform or match specialized training, and how much fine-tuning improves its predictions on a narrow domain.

## Results

### Dice Score Comparison

| Model | Mean Dice Score |
|-------|----------------|
| MedSamLike (Scratch) | 0.8640 |
| MedSAM (Generalist) | 0.9081 |
| **MedSAM (Fine-Tuned)** | **0.9448** |

### Key Findings

- The foundation model (MedSAM) performs strongly even zero-shot, achieving 0.9081 Dice score
- Fine-tuning provides a substantial improvement (+0.0367 Dice over generalist)
- Training from scratch on a small dataset performs reasonably but cannot match foundation models
- Fine-tuned MedSAM provides the strongest segmentation quality, supporting the importance of specialization even for powerful foundation models

## Dataset: Kvasir-SEG

The Kvasir-SEG dataset contains: (can be downloaded from kaggle)
- 1,000 RGB endoscopy images
- Pixel-wise binary masks (polyp vs. background)
- Train/validation split: 880 / 120 images

All images were resized to:
- `512×512` for the scratch model
- `1024×1024` for MedSAM (following architectural requirements)

Bounding boxes were computed from ground-truth masks for prompt generation.

## Model Architectures

### MedSAM (Generalist)
- ViT-B image encoder (from Segment Anything Model)
- Prompt encoder (for bounding boxes)
- Mask decoder with upsampling layers
- Evaluated **zero-shot**, without training on Kvasir-SEG

### Fine-Tuned MedSAM
Fine-tuning follows the MedSAM paper methodology:
- Prompt encoder frozen
- Image encoder + mask decoder updated
- Trained with Dice + BCE loss on the Kvasir train set

### MedSamLike (Scratch Model)
A smaller custom segmentation model built with:
- ResNet-based encoder
- Lightweight convolutional decoder
- Bounding-box prompt embedding
- Trained from scratch for 40-50 epochs

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.7 (with CUDA support recommended)
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Project
```

2. Install Segment Anything:
```bash
cd segment-anything
pip install -e .
cd ..
```

3. Install additional dependencies:
```bash
pip install torch torchvision matplotlib opencv-python tqdm
```

4. Download the MedSAM checkpoint:
   - Download `medsam_vit_b.pth` from the [MedSAM repository](https://github.com/bowang-lab/MedSAM)
   - Place it in `work_dir/MedSAM/medsam_vit_b.pth`

5. Prepare the dataset:
   - Download the Kvasir-SEG dataset
   - Extract it to `data/Kvasir-SEG/` with the following structure:
     ```
     data/Kvasir-SEG/
     ├── images/
     ├── masks/
     ├── train.txt
     └── val.txt
     ```

## Usage

### Training Scratch Model (MedSamLike)

Train the custom ResNet-based model from scratch:

```bash
python -m src.train_medsam_kvasir --data_root data/Kvasir-SEG --device cuda:0
```

Optional arguments:
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 2)
- `--lr`: Learning rate (default: 1e-4)

### Evaluating Original MedSAM (Zero-Shot)

Evaluate the pretrained MedSAM model without fine-tuning:

```bash
python -m src.eval_original_medsam_kvasir \
    --data_root data/Kvasir-SEG \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --device cuda:0
```

### Fine-Tuning MedSAM

Fine-tune the MedSAM model on Kvasir-SEG:

```bash
python -m src.train_finetune_medsam \
    --checkpoint work_dir/MedSAM/medsam_vit_b.pth \
    --data_root data/Kvasir-SEG \
    --device cuda:0
```


```

## Training Details

### Loss Functions

- **Dice Loss**: Handles class imbalance effectively
- **Binary Cross-Entropy Loss**: Standard pixel-wise classification loss

Combined loss: `L = Dice_Loss + BCE_Loss`

### Evaluation Metric

All models are evaluated using the **Dice coefficient**:

\[
\text{Dice} = \frac{2|P \cap G|}{|P| + |G|}
\]

where \(P\) is the predicted mask and \(G\) is the ground-truth mask.

### Visual Comparisons

Side-by-side comparison grids are generated showing:
- Original image
- Ground truth mask
- Scratch model prediction
- MedSAM (generalist) prediction
- MedSAM (fine-tuned) prediction

Example outputs are saved in `work_dir/vis_comparison/compare_*.png`

## Conclusion
This project demonstrates that:

1. **MedSAM generalizes well** across unseen modalities, achieving strong zero-shot performance
2. **Fine-tuning on a small dataset** significantly boosts performance, even for powerful foundation models
3. **Scratch-trained models** struggle to match foundation models without large datasets

Fine-tuned MedSAM provides the strongest segmentation quality, supporting the importance of specialization even for powerful foundation models.

## Citation

If you use this code or find it helpful, please consider citing:

```bibtex
@misc{medsam_kvasir_2025,
  title={Domain-Specific Fine-Tuning of MedSAM for Polyp Segmentation},
  author={Ravichandran, Ashwin and Nagaraj, Rithvik Pranao},
  year={2025},
  note={CS 747 Deep Learning Project}
}
```

## License

This project is for educational purposes. Please refer to the original MedSAM and Segment Anything Model licenses for their respective components.

## Acknowledgments

- MedSAM: [Bowang Lab](https://github.com/bowang-lab/MedSAM)
- Segment Anything Model: [Meta AI Research](https://github.com/facebookresearch/segment-anything)
- Kvasir-SEG Dataset: [Kvasir Dataset](https://datasets.simula.no/kvasir-seg/)
