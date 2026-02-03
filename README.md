> ### ⚠️ **Git Large File Storage (LFS) required**
>
> This repository uses **Git LFS** to store large files (e.g., datasets / assets).  
> Make sure Git LFS is installed **before cloning**, otherwise files may be missing or replaced with pointer files.


# Synth2Real Chess Images  
**Synthetic-to-Real Chessboard Image Translation using Conditional GANs**

This repository contains the full codebase for generating realistic chessboard images from synthetic renders using a pix2pix-style conditional GAN. The project focuses on bridging the visual domain gap between clean, rendered chessboard images and real-world photographs, enabling improved downstream perception tasks such as board state recognition.

---

## Project Overview

Chessboard image recognition in real-world settings is challenging due to lighting, reflections, shadows, camera angles, and occlusions (e.g., hands). While synthetic rendering can easily generate perfectly labeled chessboard images, models trained purely on synthetic data often fail to generalize to real images.

This project addresses the problem using a **synthetic-to-real image translation pipeline**, trained on paired synthetic and real chessboard images generated from the same FEN positions.

---

## Method Summary

- **Input**: Synthetic chessboard images rendered from known FEN positions  
- **Output**: Photorealistic chessboard images  
- **Model**: Pix2Pix-style conditional GAN  
  - U-Net generator  
  - PatchGAN discriminator  
- **Losses**:
  - Adversarial loss (Hinge or BCE)
  - L1 reconstruction loss
  - VGG-based perceptual loss
  - Optional R1 gradient penalty

---

## Repository Structure

```
synth_2_real_chess_images/
│
├── scripts/
│   ├── generate/
│   │   ├── generate.py                 # Inference pipeline (Blender → GAN → output)
│   │   ├── chess_position_api_v2_sharp.py
│   │   └── generator.py                # U-Net generator architecture
│   │
│   ├── models_training/
│   │   ├── final_model_training.py     # Main training script
│   │   └── command_line_for_training_final_model.txt
│
├── dataset_root/                        # Expected dataset structure (not included)
│   └── images/
│       └── <fen_folder>/
│           ├── synth.png
│           └── real/
│               ├── *.png / *.jpg
│
├── requirements.txt
└── README.md
```

---

## Environment Setup

### 1️. Clone the repository
```bash
git clone https://github.com/BHK-ImaGen/synth_2_real_chess_images.git
cd synth_2_real_chess_images
```

### 2️. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Combining Dataset Parts

The dataset is split into multiple ZIP files (`dataset_root.zip.part_aa`, `dataset_root.zip.part_ab`, `dataset_root.zip.part_ac`). Combine them into a single `dataset_root/` folder before use:

1. **Create dataset folder**
```bash
mkdir dataset_root
```

2. **Extract all parts**
```bash
unzip dataset_root.zip.part_aa -d dataset_root
unzip dataset_root.zip.part_ab -d dataset_root
unzip dataset_root.zip.part_ac -d dataset_root
```
3. **Verify structure**
```
dataset_root/
└── images/
    └── <fen_folder>/
        ├── synth.png
        └── real/
            ├── real_1.png
            ├── real_2.jpg
            └── ...
```

After this, dataset_root/ can be used in training and inference.






**Notes**
- CUDA is strongly recommended for training.
- Blender is required **only** for synthetic image generation.

---

## Dataset Preparation

### Dataset structure

```
dataset_root/
└── images/
    └── <fen_folder>/
        ├── synth.png
        └── real/
            ├── real_1.png
            ├── real_2.jpg
            └── ...
```

- `<fen_folder>` is a sanitized version of the FEN string
- Each FEN position must contain:
  - Exactly **one** synthetic render (`synth.png`)
  - One or more real images under `real/`

### Filtering
- Positions containing hands are excluded via:
  ```
  fens_with_hands_in_dataset_root.txt
  ```
- Automatic board detection and perspective correction are applied.

---

## Training

### Example training command

```bash
python scripts/models_training/final_model_training.py \
  --dataset_root /path/to/dataset_root \
  --out_root /path/to/output_models \
  --image_size 512 \
  --batch_size 8 \
  --epochs 200 \
  --save_every 10
```

### Key arguments

| Argument | Description |
|--------|------------|
| `--adv_loss` | `hinge` or `bce` |
| `--l1_lambda` | L1 reconstruction weight |
| `--vgg_lambda` | Perceptual loss weight |
| `--r1_gamma` | R1 gradient penalty |
| `--syn_norm` | Synthetic image normalization |
| `--real_norm` | Real image normalization |

- Training runs on the **full dataset** (no train/val split)
- Checkpoints and visualizations are saved automatically

---

## Inference / Image Generation

The inference pipeline:
1. Render a synthetic board from a FEN using Blender
2. Crop and normalize the board
3. Translate the image using the trained generator
4. Save outputs

### Example

```bash
python scripts/generate/generate.py
```

Outputs are saved to:
```
scripts/generate/results/
├── synthetic.png
├── realistic.png
└── side_by_side.png
```

---

## Reproducibility

To reproduce the reported results:

1. Use the same dataset structure
2. Use identical normalization modes
3. Train for the same number of epochs
4. Use the provided generator architecture and loss configuration

### External dependencies (not included)
- **Blender** – synthetic image rendering
- **Fenify-3D** – PGN/FEN alignment
- **MediaPipe Hands** – dataset filtering

Pretrained weights are not bundled unless explicitly stated.

---

## Limitations

- Requires paired synthetic/real data
- GAN training can be unstable without tuning
- Performance depends on accurate board cropping
- Blender rendering is computationally expensive

---

## References

\[1\] Isola et al., ["Image-to-Image Translation with Conditional Adversarial Networks"][1], CVPR 2017. <br>
\[2\] [Fenify repository / tool][2]. <br>
\[3\] [python-chess: a chess library for Python][3] <br>
\[4\] [Google MediaPipe Hands repository][4].


[1]: https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf
[2]: https://github.com/notnil/fenify-3D
[3]: https://python-chess.readthedocs.io/en/latest/

[4]: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md

---

## Authors

#### Yoav Baider Klein, BSc
#### Maor Haak, BSc
#### Guy Kouchly, Bsc
