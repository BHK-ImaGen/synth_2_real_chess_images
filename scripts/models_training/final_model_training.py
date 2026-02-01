#!/usr/bin/env python3
# synth_to_real_chess4_dataset_root_interactive.py
#
# Changes vs original:
# 1) No train/val split: train on the whole dataset.
# 2) Interactive args for hyperparams, normalization modes, discriminator strength, losses, etc.
# 3) After each checkpoint save (every save_every epochs), run inference on:
#      /home/guykou/chess/generate/synth_for_test
#    and save outputs under viz/test_epoch_XXX/.
#
# Keeps dataset_root structure + excludes hands FEN folders:
#   /home/guykou/chess/dataset_root/images/<fen_folder>/synth.png
#   /home/guykou/chess/dataset_root/images/<fen_folder>/real/*.png|jpg|jpeg

import os
import re
import csv
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# 0) Utilities
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image_file(p: Path):
    return p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

# -----------------------------
# 1) DATASET: dataset_root loader
# -----------------------------
class DatasetRootFENDataset(Dataset):
    """
    Loads from <dataset_root>/images

    Structure per FEN folder:
      images/<fen_folder>/
        synth.png
        real/
          *.png / *.jpg / *.jpeg  (one or more)

    Rules:
      - Exclude any FEN listed in fens_with_hands_in_dataset_root.txt (stored as *true* FEN strings)
      - If multiple reals exist: pick random real each __getitem__ call
      - Returns (synth_bgr, real_bgr)
    """
    IMAGE_EXTS = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        dataset_root="/home/guykou/chess/dataset_root",
        images_subdir="images",
        hands_fens_file="fens_with_hands_in_dataset_root.txt",
        require_synth=True,
        verbose=True,
    ):
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / images_subdir
        self.hands_fens_path = self.dataset_root / hands_fens_file

        if not self.images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {self.images_dir}")

        # Load hands FENs and convert to folder names
        self.exclude_fen_folders = set()
        if self.hands_fens_path.exists():
            with open(self.hands_fens_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    fen = line.strip()
                    if not fen:
                        continue
                    self.exclude_fen_folders.add(self.safe_fen_folder_name(fen))

        self.items = []
        fen_dirs = [p for p in self.images_dir.iterdir() if p.is_dir()]
        fen_dirs.sort(key=lambda p: p.name)

        excluded_existing = 0

        for fen_dir in fen_dirs:
            fen_folder = fen_dir.name
            if fen_folder in self.exclude_fen_folders:
                excluded_existing += 1
                continue

            synth_path = fen_dir / "synth.png"
            if require_synth and not synth_path.exists():
                continue

            real_dir = fen_dir / "real"
            if not real_dir.exists():
                continue

            real_images = []
            for rp in real_dir.iterdir():
                if rp.is_file() and rp.suffix.lower() in self.IMAGE_EXTS:
                    real_images.append(rp)

            if len(real_images) == 0:
                continue

            self.items.append({
                "fen_folder": fen_folder,
                "synth_path": synth_path if synth_path.exists() else None,
                "real_images": real_images,
                "source": "dataset_root",
            })

        if verbose:
            print(f"[DatasetRoot] images_dir={self.images_dir}")
            print(f"[DatasetRoot] hands list file: {self.hands_fens_path}")
            print(f"[DatasetRoot] hand-FEN folders in exclude list: {len(self.exclude_fen_folders)}")
            print(f"[DatasetRoot] excluded (actually present) folders: {excluded_existing}")
            print(f"[DatasetRoot] Loaded {len(self.items)} usable FEN folders")

    @staticmethod
    def safe_fen_folder_name(fen: str) -> str:
        """
        Must match naming used in dataset_root creation:
          / -> _
          space -> __
          collapse multiple underscores
        """
        s = fen.strip()
        s = s.replace("/", "_")
        s = s.replace(" ", "__")
        s = s.replace(":", "_").replace("|", "_").replace("\\", "_")
        s = re.sub(r"_+", "_", s)
        return s

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        info = self.items[idx]
        if info["synth_path"] is None:
            raise ValueError(f"Missing synth.png for {info['fen_folder']}")

        real_path = random.choice(info["real_images"])

        synth = cv2.imread(str(info["synth_path"]))
        real = cv2.imread(str(real_path))

        if synth is None:
            raise ValueError(f"Could not read synth image: {info['synth_path']}")
        if real is None:
            raise ValueError(f"Could not read real image: {real_path}")

        return synth, real

# -----------------------------
# 2) PREPROCESSING
# -----------------------------
def preprocess_chess_board(img_bgr, output_size=512, visualize=False):
    """
    Preprocess a board image provided as a BGR numpy array.
    Returns the warped RGB image (HWC uint8) or None if board not found.
    """
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(cnt)
        pts = cv2.boxPoints(rect)
        pts = np.intp(pts)

    rect_ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect_ordered[0] = pts[np.argmin(s)]
    rect_ordered[2] = pts[np.argmax(s)]
    rect_ordered[1] = pts[np.argmin(diff)]
    rect_ordered[3] = pts[np.argmax(diff)]

    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect_ordered, dst)
    warped = cv2.warpPerspective(img_rgb, M, (output_size, output_size))

    if visualize:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Detected Corners")
        plt.imshow(img_rgb)
        for pt in rect_ordered:
            plt.plot(pt[0], pt[1], "ro", markersize=6)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title(f"Warped {output_size}x{output_size}")
        plt.imshow(warped)
        plt.axis("off")

    return warped

def resize_real_image(img_bgr, size=512, visualize=False):
    if isinstance(img_bgr, str):
        img_bgr = cv2.imread(img_bgr)
        if img_bgr is None:
            raise ValueError(f"Could not load image at {img_bgr}")

    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img_rgb); axs[0].set_title("Original"); axs[0].axis("off")
        axs[1].imshow(resized_img); axs[1].set_title(f"Resized {size}"); axs[1].axis("off")
        plt.tight_layout(); plt.show()

    return resized_img

class ProcessedDatasetRoot(Dataset):
    """
    Outputs (syn_t, real_t) tensors in [0,1].
    Applies:
      - preprocess synth board warp (fallback to resize)
      - resize real
    """
    def __init__(self, base_dataset: DatasetRootFENDataset, size=512):
        self.base = base_dataset
        self.size = size
        self.to_tensor = lambda img: torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        synth_bgr, real_bgr = self.base[idx]

        syn_rgb = preprocess_chess_board(synth_bgr, output_size=self.size, visualize=False)
        if syn_rgb is None:
            syn_rgb = cv2.cvtColor(cv2.resize(synth_bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)

        real_rgb = resize_real_image(real_bgr, size=self.size, visualize=False)
        if real_rgb is None:
            raise ValueError(f"Real image preprocess failed at idx={idx}")

        syn_t = self.to_tensor(syn_rgb)   # [0,1]
        real_t = self.to_tensor(real_rgb) # [0,1]
        return syn_t, real_t

# -----------------------------
# 2.5) Normalization modes (interactive)
# -----------------------------
def apply_norm_mode(x01: torch.Tensor, mode: str) -> torch.Tensor:
    """
    x01 is assumed in [0,1].
    Returns tensor in some transformed space depending on mode.
    Supported:
      - "none": [0,1] unchanged
      - "tanh": map to [-1,1] via x*2-1  (DEFAULT behavior you had)
      - "center": identical to tanh (alias)
      - "imagenet": (x-mean)/std in [0,1] space (note: not bounded)
    """
    mode = (mode or "tanh").lower()
    if mode == "none":
        return x01
    if mode in ("tanh", "center"):
        return x01 * 2.0 - 1.0
    if mode == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], device=x01.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x01.device).view(1, 3, 1, 1)
        return (x01 - mean) / std
    raise ValueError(f"Unknown norm mode: {mode}")

# -----------------------------
# 3) MODEL (Pix2Pix)
# -----------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # expects x,y in [-1,1] (if you keep default tanh mode)
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        fx = self.slice(x)
        fy = self.slice(y)
        return (fx - fy).pow(2).mean()

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, norm=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.model(x)
        return torch.cat([x, skip], dim=1)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.d1 = DownBlock(in_channels, 64, norm=False)
        self.d2 = DownBlock(64, 128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.bottleneck = DownBlock(512, 512)
        self.u1 = UpBlock(512, 512, dropout=True)
        self.u2 = UpBlock(1024, 512, dropout=True)
        self.u3 = UpBlock(1024, 256)
        self.u4 = UpBlock(512, 128)
        self.u5 = UpBlock(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        s5 = self.d5(s4)
        b = self.bottleneck(s5)
        x = self.u1(b, s5)
        x = self.u2(x, s4)
        x = self.u3(x, s3)
        x = self.u4(x, s2)
        x = self.u5(x, s1)
        return self.final(x)

def maybe_spectral_norm(conv: nn.Module, use_sn: bool):
    return nn.utils.spectral_norm(conv) if use_sn else conv

class PatchDiscriminator(nn.Module):
    """
    Configurable discriminator strength:
      - base_channels: width multiplier (default 64)
      - n_layers: how many conv blocks (default 4 similar-ish to your current)
      - use_norm: InstanceNorm in blocks
      - use_sn: spectral norm on convs
    """
    def __init__(self, in_channels=3, base_channels=64, n_layers=4, use_norm=True, use_sn=False):
        super().__init__()

        def block(in_c, out_c, stride=2, norm=True):
            conv = nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)
            conv = maybe_spectral_norm(conv, use_sn)
            layers = [conv]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        # first block: no norm typically
        layers += block(in_channels * 2, base_channels, stride=2, norm=False)

        # middle blocks
        ch = base_channels
        for i in range(max(1, n_layers - 2)):
            out_ch = min(ch * 2, 512)
            layers += block(ch, out_ch, stride=2, norm=use_norm)
            ch = out_ch

        # penultimate: stride 1
        layers += block(ch, min(ch * 2, 512), stride=1, norm=use_norm)
        ch = min(ch * 2, 512)

        # final logits
        final_conv = nn.Conv2d(ch, 1, 4, padding=1)
        final_conv = maybe_spectral_norm(final_conv, use_sn)
        layers.append(final_conv)

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# -----------------------------
# 4) ADV LOSS (interactive)
# -----------------------------
def d_loss_hinge(real_pred, fake_pred):
    return F.relu(1.0 - real_pred).mean() + F.relu(1.0 + fake_pred).mean()

def g_loss_hinge(fake_pred):
    return -fake_pred.mean()

def d_loss_bce(real_pred, fake_pred):
    # logits -> targets 1 for real, 0 for fake
    return (
        F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) +
        F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
    )

def g_loss_bce(fake_pred):
    return F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))

# -----------------------------
# 5) TRAIN ONE EPOCH (no validation)
# -----------------------------
def train_one_epoch(
    gen,
    disc,
    dataloader,
    opt_g,
    opt_d,
    vgg_loss_fn,
    scaler,
    device,
    syn_norm_mode: str,
    real_norm_mode: str,
    adv_loss: str,
    l1_lambda: float,
    vgg_lambda: float,
    r1_gamma: float,
    log_every: int,
):
    gen.train()
    disc.train()
    use_amp = (device.type == "cuda")

    g_loss_sum = 0.0
    d_loss_sum = 0.0
    num_batches = 0

    for batch_idx, (syn01, real01) in enumerate(dataloader):
        syn01 = syn01.to(device, non_blocking=True)   # [0,1]
        real01 = real01.to(device, non_blocking=True) # [0,1]

        # Apply chosen normalization modes
        syn_in = apply_norm_mode(syn01, syn_norm_mode)
        real_t = apply_norm_mode(real01, real_norm_mode)

        # ---- D ----
        opt_d.zero_grad(set_to_none=True)

        real_t.requires_grad_(True)  # for R1
        with torch.autocast(device_type=device.type, enabled=use_amp):
            real_pred = disc(syn_in, real_t)

            with torch.no_grad():
                fake_t = gen(syn_in)
            fake_pred = disc(syn_in, fake_t)

            if adv_loss == "bce":
                d_loss = d_loss_bce(real_pred, fake_pred)
            else:
                d_loss = d_loss_hinge(real_pred, fake_pred)

        # R1 penalty on real
        r1_pen = torch.tensor(0.0, device=device)
        if r1_gamma and r1_gamma > 0:
            grad_real = torch.autograd.grad(
                outputs=real_pred.float().sum(),
                inputs=real_t,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            r1_pen = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(dim=1).mean()
            d_loss = d_loss + (0.5 * r1_gamma) * r1_pen

        if use_amp and scaler is not None:
            scaler.scale(d_loss).backward()
            scaler.step(opt_d)
            scaler.update()
        else:
            d_loss.backward()
            opt_d.step()

        real_t.requires_grad_(False)

        # ---- G ----
        opt_g.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            fake_t = gen(syn_in)
            fake_pred = disc(syn_in, fake_t)

            if adv_loss == "bce":
                g_adv = g_loss_bce(fake_pred)
            else:
                g_adv = g_loss_hinge(fake_pred)

            g_l1 = F.l1_loss(fake_t, real_t)
            g_perc = vgg_loss_fn(fake_t, real_t)
            g_loss = g_adv + (l1_lambda * g_l1) + (vgg_lambda * g_perc)

        if use_amp and scaler is not None:
            scaler.scale(g_loss).backward()
            scaler.step(opt_g)
            scaler.update()
        else:
            g_loss.backward()
            opt_g.step()

        g_loss_sum += float(g_loss.detach().item())
        d_loss_sum += float(d_loss.detach().item())
        num_batches += 1

        if log_every and ((batch_idx + 1) % log_every == 0):
            print(
                f"Batch {batch_idx+1}/{len(dataloader)} | "
                f"D: {d_loss.item():.4f} (r1 {r1_pen.item():.4f}) | "
                f"G: {g_loss.item():.4f} "
                f"(adv {g_adv.item():.4f}, l1 {g_l1.item():.4f}, perc {g_perc.item():.4f})"
            )

    return g_loss_sum / max(1, num_batches), d_loss_sum / max(1, num_batches)

# -----------------------------
# 6) VIZ HELPERS
# -----------------------------
def _tensor_to_rgb01(img_t: torch.Tensor) -> np.ndarray:
    """
    img_t: [3,H,W] in arbitrary range; tries to map to [0,1] for visualization.
    If it looks like [-1,1], converts to [0,1].
    """
    img_t = img_t.detach().cpu()
    mn = float(img_t.min().item())
    mx = float(img_t.max().item())

    # heuristic: if range crosses 0 and within [-1.5,1.5], treat as tanh space
    if mn < -0.1 and mx <= 1.5:
        img_t = (img_t + 1.0) / 2.0

    img_t = img_t.clamp(0, 1)
    return img_t.permute(1, 2, 0).numpy()

@torch.no_grad()
def run_test_folder_inference(
    gen: nn.Module,
    test_synth_dir: Path,
    out_dir: Path,
    device,
    image_size: int,
    syn_norm_mode: str,
    save_side_by_side: bool = True,
    max_images: int = 0,   # 0 => no limit
):
    """
    Loads images from test_synth_dir, preprocesses similarly to synth preprocessing,
    runs generator, saves outputs into out_dir.
    """
    gen.eval()
    ensure_dir(out_dir)

    if not test_synth_dir.exists():
        print(f"[TestViz] test_synth_dir does not exist: {test_synth_dir}")
        return

    img_paths = [p for p in sorted(test_synth_dir.iterdir()) if p.is_file() and is_image_file(p)]
    if max_images and max_images > 0:
        img_paths = img_paths[:max_images]

    if len(img_paths) == 0:
        print(f"[TestViz] No images found in: {test_synth_dir}")
        return

    print(f"[TestViz] Running inference on {len(img_paths)} images from: {test_synth_dir}")
    for p in img_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"[TestViz] WARN: could not read: {p}")
            continue

        syn_rgb = preprocess_chess_board(bgr, output_size=image_size, visualize=False)
        if syn_rgb is None:
            syn_rgb = cv2.cvtColor(cv2.resize(bgr, (image_size, image_size)), cv2.COLOR_BGR2RGB)

        syn01 = torch.from_numpy(syn_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        syn01 = syn01.to(device)
        syn_in = apply_norm_mode(syn01, syn_norm_mode)

        fake = gen(syn_in)[0]  # [3,H,W], typically tanh range if default
        fake01 = _tensor_to_rgb01(fake)  # HWC [0,1]
        fake_u8 = (fake01 * 255.0).astype(np.uint8)

        out_fake = out_dir / f"{p.stem}_fake.png"
        cv2.imwrite(str(out_fake), cv2.cvtColor(fake_u8, cv2.COLOR_RGB2BGR))

        if save_side_by_side:
            syn_vis = _tensor_to_rgb01(syn01[0])  # [0,1]
            syn_u8 = (syn_vis * 255.0).astype(np.uint8)

            side = np.concatenate([syn_u8, fake_u8], axis=1)
            out_side = out_dir / f"{p.stem}_in_and_fake.png"
            cv2.imwrite(str(out_side), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

    print(f"[TestViz] Saved test outputs into: {out_dir}")

# -----------------------------
# 7) LOSS SAVING HELPERS (train-only)
# -----------------------------
def save_loss_curves_train_only(logs_dir: Path, run_name: str, train_g_losses, train_d_losses):
    ensure_dir(logs_dir)
    csv_path = logs_dir / f"{run_name}_losses_train_only.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_g", "train_d"])
        for i in range(len(train_g_losses)):
            w.writerow([i + 1, train_g_losses[i], train_d_losses[i]])
    print(f"[Loss] Saved CSV: {csv_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_g_losses) + 1), train_g_losses, label="Train G")
    plt.xlabel("Epoch")
    plt.ylabel("Generator Loss")
    plt.title("Generator Loss (Train)")
    plt.legend()
    plt.tight_layout()
    g_png = logs_dir / f"{run_name}_generator_loss_train.png"
    plt.savefig(g_png, dpi=150)
    plt.close()
    print(f"[Loss] Saved plot: {g_png}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_d_losses) + 1), train_d_losses, label="Train D")
    plt.xlabel("Epoch")
    plt.ylabel("Discriminator Loss")
    plt.title("Discriminator Loss (Train)")
    plt.legend()
    plt.tight_layout()
    d_png = logs_dir / f"{run_name}_discriminator_loss_train.png"
    plt.savefig(d_png, dpi=150)
    plt.close()
    print(f"[Loss] Saved plot: {d_png}")

def save_run_config(logs_dir: Path, config: dict):
    ensure_dir(logs_dir)
    path = logs_dir / "config.json"
    with open(path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[Run] Saved config: {path}")

# -----------------------------
# 8) ARGPARSE
# -----------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Pix2Pix synth->real training on dataset_root with interactive knobs.")

    # data
    p.add_argument("--dataset_root", type=str, default="/home/guykou/chess/dataset_root")
    p.add_argument("--image_size", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=2)

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr_g", type=float, default=2e-4)
    p.add_argument("--lr_d", type=float, default=1e-4)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--l1_lambda", type=float, default=100.0)
    p.add_argument("--vgg_lambda", type=float, default=10.0)
    p.add_argument("--r1_gamma", type=float, default=10.0)
    p.add_argument("--adv_loss", type=str, default="hinge", choices=["hinge", "bce"])
    p.add_argument("--log_every", type=int, default=10)

    # normalization knobs (synthetic & real separately)
    p.add_argument("--syn_norm", type=str, default="tanh", choices=["tanh", "center", "none", "imagenet"])
    p.add_argument("--real_norm", type=str, default="tanh", choices=["tanh", "center", "none", "imagenet"])

    # discriminator strength knobs
    p.add_argument("--disc_base_channels", type=int, default=64)
    p.add_argument("--disc_layers", type=int, default=4)
    p.add_argument("--disc_use_norm", action="store_true", default=True)
    p.add_argument("--disc_no_norm", action="store_true", default=False)
    p.add_argument("--disc_spectral_norm", action="store_true", default=False)

    # checkpointing + test viz
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--test_synth_dir", type=str, default="/home/guykou/chess/generate/synth_for_test")
    p.add_argument("--test_max_images", type=int, default=0)  # 0 => unlimited
    p.add_argument("--test_side_by_side", action="store_true", default=True)
    p.add_argument("--no_test_side_by_side", action="store_true", default=False)

    # run naming
    p.add_argument("--out_root", type=str, default="/home/guykou/chess/final_models")
    p.add_argument("--run_id", type=int, default=0)
    p.add_argument("--tag", type=str, default="default")

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_amp", action="store_true", default=False)

    return p

# -----------------------------
# 9) MAIN
# -----------------------------
def main():
    args = build_argparser().parse_args()

    if args.seed is not None and args.seed >= 0:
        seed_everything(args.seed)

    # resolve toggles
    disc_use_norm = True
    if args.disc_no_norm:
        disc_use_norm = False
    elif args.disc_use_norm:
        disc_use_norm = True

    test_side_by_side = True
    if args.no_test_side_by_side:
        test_side_by_side = False
    elif args.test_side_by_side:
        test_side_by_side = True

    dataset_root = args.dataset_root

    # dataset
    base_ds = DatasetRootFENDataset(
        dataset_root=dataset_root,
        images_subdir="images",
        hands_fens_file="fens_with_hands_in_dataset_root.txt",
        require_synth=True,
        verbose=True,
    )
    full_ds = ProcessedDatasetRoot(base_ds, size=args.image_size)

    loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"[Data] Training on FULL dataset size: {len(full_ds)}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # models
    gen = UNetGenerator().to(device)
    disc = PatchDiscriminator(
        base_channels=args.disc_base_channels,
        n_layers=args.disc_layers,
        use_norm=disc_use_norm,
        use_sn=args.disc_spectral_norm
    ).to(device)
    vgg_loss = VGGPerceptualLoss().to(device)

    # optim
    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(disc.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    use_amp = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"[AMP] enabled={use_amp}")

    # run folder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id_str = f"run{args.run_id:02d}" if args.run_id else "run00"
    run_name = f"{run_id_str}_{args.tag}_{stamp}"

    run_root = Path(args.out_root) / run_name
    ckpt_dir = run_root / "checkpoints"
    viz_dir = run_root / "viz"
    logs_dir = run_root / "logs"
    ensure_dir(ckpt_dir); ensure_dir(viz_dir); ensure_dir(logs_dir)

    print(f"[Run] All outputs will be saved under: {run_root}")

    config = {
        "run_name": run_name,
        "dataset_root": dataset_root,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr_g": args.lr_g,
        "lr_d": args.lr_d,
        "betas": [args.beta1, args.beta2],
        "l1_lambda": args.l1_lambda,
        "vgg_lambda": args.vgg_lambda,
        "r1_gamma": args.r1_gamma,
        "adv_loss": args.adv_loss,
        "syn_norm": args.syn_norm,
        "real_norm": args.real_norm,
        "disc_base_channels": args.disc_base_channels,
        "disc_layers": args.disc_layers,
        "disc_use_norm": disc_use_norm,
        "disc_spectral_norm": args.disc_spectral_norm,
        "save_every": args.save_every,
        "test_synth_dir": args.test_synth_dir,
        "test_max_images": args.test_max_images,
        "seed": args.seed,
        "amp": use_amp,
        "notes": "Train on full dataset_root (no split). Checkpoint & run test-synth inference every save_every epochs.",
    }
    save_run_config(logs_dir, config)

    train_g_losses, train_d_losses = [], []

    print(f"[Train] epochs={args.epochs} | save_every={args.save_every}")
    for epoch in range(args.epochs):
        avg_g, avg_d = train_one_epoch(
            gen=gen,
            disc=disc,
            dataloader=loader,
            opt_g=opt_g,
            opt_d=opt_d,
            vgg_loss_fn=vgg_loss,
            scaler=scaler,
            device=device,
            syn_norm_mode=args.syn_norm,
            real_norm_mode=args.real_norm,
            adv_loss=args.adv_loss,
            l1_lambda=args.l1_lambda,
            vgg_lambda=args.vgg_lambda,
            r1_gamma=args.r1_gamma,
            log_every=args.log_every,
        )

        train_g_losses.append(avg_g)
        train_d_losses.append(avg_d)

        print(f"[Epoch {epoch+1}/{args.epochs}] Train: G {avg_g:.4f} | D {avg_d:.4f}")

        # Save model + test viz every N epochs
        if (epoch + 1) % args.save_every == 0:
            gen_path = ckpt_dir / f"gen_epoch_{epoch+1:03d}.pth"
            disc_path = ckpt_dir / f"disc_epoch_{epoch+1:03d}.pth"
            torch.save(gen.state_dict(), gen_path)
            torch.save(disc.state_dict(), disc_path)
            print("[CKPT] Saved:")
            print(f"  Generator     → {gen_path}")
            print(f"  Discriminator → {disc_path}")

            # Test folder inference
            test_out = viz_dir / f"test_epoch_{epoch+1:03d}"
            run_test_folder_inference(
                gen=gen,
                test_synth_dir=Path(args.test_synth_dir),
                out_dir=test_out,
                device=device,
                image_size=args.image_size,
                syn_norm_mode=args.syn_norm,
                save_side_by_side=test_side_by_side,
                max_images=args.test_max_images,
            )

    # Final save
    gen_path = ckpt_dir / "gen_final.pth"
    disc_path = ckpt_dir / "disc_final.pth"
    torch.save(gen.state_dict(), gen_path)
    torch.save(disc.state_dict(), disc_path)
    print("[CKPT] Final saved:")
    print(f"  Generator     → {gen_path}")
    print(f"  Discriminator → {disc_path}")

    save_loss_curves_train_only(
        logs_dir=logs_dir,
        run_name=run_name,
        train_g_losses=train_g_losses,
        train_d_losses=train_d_losses,
    )

    print(f"[Done] Run folder: {run_root}")

if __name__ == "__main__":
    main()
