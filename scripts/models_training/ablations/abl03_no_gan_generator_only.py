#!/usr/bin/env python3
# abl03_no_gan_generator_only.py
# Significant ablation: remove adversarial training entirely (no discriminator).
# Train generator with ONLY: L1 + VGG perceptual (regression).
# Expect: blurrier / less realistic textures, but geometry often preserved.

import os
import re
import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import argparse

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATASET_ROOT = os.path.join(BASE_DIR, "dataset_root")
DEFAULT_OUT_ROOT = os.path.join(BASE_DIR, "final_models")
DEFAULT_TEST_SYNTH_DIR = os.path.join(BASE_DIR, "generate", "synth_for_test")

ap = argparse.ArgumentParser(description="Ablation 03: No GAN, generator only.")
ap.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
ap.add_argument("--out_root", type=str, default=DEFAULT_OUT_ROOT)
ap.add_argument("--test_synth_dir", type=str, default=DEFAULT_TEST_SYNTH_DIR)
args = ap.parse_args()

dataset_root = args.dataset_root
out_root = args.out_root
test_synth_dir = args.test_synth_dir
os.makedirs(out_root, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

def apply_norm_mode(x01: torch.Tensor, mode: str) -> torch.Tensor:
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

def preprocess_chess_board(img_bgr, output_size=512):
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
    return warped

def resize_real_image(img_bgr, size=512):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA)

def tensor_to_rgb01(img_t: torch.Tensor) -> np.ndarray:
    img_t = img_t.detach().cpu()
    mn = float(img_t.min().item())
    mx = float(img_t.max().item())
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
    max_images: int = 0,
):
    gen.eval()
    ensure_dir(out_dir)

    if not test_synth_dir.exists():
        print(f"[TestViz] Missing test dir: {test_synth_dir}")
        return

    img_paths = [p for p in sorted(test_synth_dir.iterdir()) if p.is_file() and is_image_file(p)]
    if max_images and max_images > 0:
        img_paths = img_paths[:max_images]

    if not img_paths:
        print(f"[TestViz] No images in: {test_synth_dir}")
        return

    print(f"[TestViz] Inference on {len(img_paths)} images from {test_synth_dir}")
    for p in img_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue

        syn_rgb = preprocess_chess_board(bgr, output_size=image_size)
        if syn_rgb is None:
            syn_rgb = cv2.cvtColor(cv2.resize(bgr, (image_size, image_size)), cv2.COLOR_BGR2RGB)

        syn01 = torch.from_numpy(syn_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
        syn01 = syn01.to(device)
        syn_in = apply_norm_mode(syn01, syn_norm_mode)

        fake = gen(syn_in)[0]
        fake01 = tensor_to_rgb01(fake)
        fake_u8 = (fake01 * 255.0).astype(np.uint8)

        out_fake = out_dir / f"{p.stem}_fake.png"
        cv2.imwrite(str(out_fake), cv2.cvtColor(fake_u8, cv2.COLOR_RGB2BGR))

        if save_side_by_side:
            syn_vis = tensor_to_rgb01(syn01[0])
            syn_u8 = (syn_vis * 255.0).astype(np.uint8)
            side = np.concatenate([syn_u8, fake_u8], axis=1)
            out_side = out_dir / f"{p.stem}_in_and_fake.png"
            cv2.imwrite(str(out_side), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))

    print(f"[TestViz] Saved to: {out_dir}")


# -----------------------------
# Dataset
# -----------------------------
class DatasetRootFENDataset(Dataset):
    IMAGE_EXTS = (".png", ".jpg", ".jpeg")

    def __init__(self, dataset_root=dataset_root, images_subdir="images",
                 hands_fens_file="fens_with_hands_in_dataset_root.txt", require_synth=True, verbose=True):
        self.dataset_root = Path(dataset_root)
        self.images_dir = self.dataset_root / images_subdir
        self.hands_fens_path = self.dataset_root / hands_fens_file

        if not self.images_dir.exists():
            raise FileNotFoundError(f"images dir not found: {self.images_dir}")

        self.exclude_fen_folders = set()
        if self.hands_fens_path.exists():
            with open(self.hands_fens_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    fen = line.strip()
                    if fen:
                        self.exclude_fen_folders.add(self.safe_fen_folder_name(fen))

        self.items = []
        fen_dirs = sorted([p for p in self.images_dir.iterdir() if p.is_dir()], key=lambda p: p.name)
        excluded_present = 0

        for fen_dir in fen_dirs:
            fen_folder = fen_dir.name
            if fen_folder in self.exclude_fen_folders:
                excluded_present += 1
                continue

            synth_path = fen_dir / "synth.png"
            if require_synth and not synth_path.exists():
                continue

            real_dir = fen_dir / "real"
            if not real_dir.exists():
                continue

            real_images = [rp for rp in real_dir.iterdir()
                           if rp.is_file() and rp.suffix.lower() in self.IMAGE_EXTS]
            if not real_images:
                continue

            self.items.append({"synth_path": synth_path, "real_images": real_images})

        if verbose:
            print(f"[DatasetRoot] Loaded {len(self.items)} FEN folders (excluded present: {excluded_present})")

    @staticmethod
    def safe_fen_folder_name(fen: str) -> str:
        s = fen.strip()
        s = s.replace("/", "_").replace(" ", "__")
        s = s.replace(":", "_").replace("|", "_").replace("\\", "_")
        s = re.sub(r"_+", "_", s)
        return s

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        info = self.items[idx]
        real_path = random.choice(info["real_images"])
        synth = cv2.imread(str(info["synth_path"]))
        real = cv2.imread(str(real_path))
        if synth is None or real is None:
            raise ValueError("Failed to read synth/real image")
        return synth, real

class ProcessedDatasetRoot(Dataset):
    def __init__(self, base: DatasetRootFENDataset, size=512):
        self.base = base
        self.size = size

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        synth_bgr, real_bgr = self.base[idx]

        syn_rgb = preprocess_chess_board(synth_bgr, output_size=self.size)
        if syn_rgb is None:
            syn_rgb = cv2.cvtColor(cv2.resize(synth_bgr, (self.size, self.size)), cv2.COLOR_BGR2RGB)

        real_rgb = resize_real_image(real_bgr, size=self.size)

        syn_t = torch.from_numpy(syn_rgb.transpose(2, 0, 1)).float() / 255.0
        real_t = torch.from_numpy(real_rgb.transpose(2, 0, 1)).float() / 255.0
        return syn_t, real_t


# -----------------------------
# Model
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
        # expects x,y in [-1,1]
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
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
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
    def __init__(self):
        super().__init__()
        self.d1 = DownBlock(3, 64, norm=False)
        self.d2 = DownBlock(64, 128)
        self.d3 = DownBlock(128, 256)
        self.d4 = DownBlock(256, 512)
        self.d5 = DownBlock(512, 512)
        self.b  = DownBlock(512, 512)
        self.u1 = UpBlock(512, 512, dropout=True)
        self.u2 = UpBlock(1024, 512, dropout=True)
        self.u3 = UpBlock(1024, 256)
        self.u4 = UpBlock(512, 128)
        self.u5 = UpBlock(256, 64)
        self.final = nn.Sequential(nn.ConvTranspose2d(128, 3, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        s1 = self.d1(x)
        s2 = self.d2(s1)
        s3 = self.d3(s2)
        s4 = self.d4(s3)
        s5 = self.d5(s4)
        b = self.b(s5)
        x = self.u1(b, s5)
        x = self.u2(x, s4)
        x = self.u3(x, s3)
        x = self.u4(x, s2)
        x = self.u5(x, s1)
        return self.final(x)


def save_run_config(logs_dir: Path, config: dict):
    ensure_dir(logs_dir)
    with open(logs_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Ablation 03: NO GAN (G-only regression: L1 + VGG).")

    ap.add_argument("--dataset_root", type=str, default=dataset_root)
    ap.add_argument("--out_root", type=str, default=out_root)
    ap.add_argument("--run_id", type=int, default=0)
    ap.add_argument("--tag", type=str, default="abl03_no_gan")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--image_size", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--save_every", type=int, default=10)

    ap.add_argument("--lr_g", type=float, default=2e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)

    ap.add_argument("--l1_lambda", type=float, default=100.0)
    ap.add_argument("--vgg_lambda", type=float, default=10.0)

    ap.add_argument("--syn_norm", type=str, default="tanh", choices=["tanh","center","none","imagenet"])
    ap.add_argument("--real_norm", type=str, default="tanh", choices=["tanh","center","none","imagenet"])

    ap.add_argument("--test_synth_dir", type=str, default=test_synth_dir)
    ap.add_argument("--test_max_images", type=int, default=0)
    ap.add_argument("--no_test_side_by_side", action="store_true", default=False)

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    base_ds = DatasetRootFENDataset(dataset_root=args.dataset_root, verbose=True)
    ds = ProcessedDatasetRoot(base_ds, size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True, drop_last=True)

    gen = UNetGenerator().to(device)
    vgg_loss = VGGPerceptualLoss().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run{args.run_id:02d}_{args.tag}_{stamp}"
    run_root = Path(args.out_root) / run_name
    ckpt_dir = run_root / "checkpoints"
    viz_dir = run_root / "viz"
    logs_dir = run_root / "logs"
    ensure_dir(ckpt_dir); ensure_dir(viz_dir); ensure_dir(logs_dir)

    config = {
        "ablation": "NO_GAN",
        "note": "No discriminator / no adversarial loss. Train generator with L1 + VGG only.",
        "args": vars(args),
    }
    save_run_config(logs_dir, config)
    print(f"[Run] {run_root}")

    for epoch in range(args.epochs):
        gen.train()

        g_sum = 0.0
        nb = 0

        for syn01, real01 in dl:
            syn01 = syn01.to(device, non_blocking=True)
            real01 = real01.to(device, non_blocking=True)

            syn_in = apply_norm_mode(syn01, args.syn_norm)
            real_t = apply_norm_mode(real01, args.real_norm)

            opt_g.zero_grad(set_to_none=True)

            fake_t = gen(syn_in)

            g_l1 = F.l1_loss(fake_t, real_t)
            g_perc = vgg_loss(fake_t, real_t)
            g_loss = (args.l1_lambda * g_l1) + (args.vgg_lambda * g_perc)

            g_loss.backward()
            opt_g.step()

            g_sum += float(g_loss.item())
            nb += 1

        print(f"[Epoch {epoch+1}/{args.epochs}] Train: G {g_sum/max(1,nb):.4f}")

        if (epoch + 1) % args.save_every == 0:
            gen_path = ckpt_dir / f"gen_epoch_{epoch+1:03d}.pth"
            torch.save(gen.state_dict(), gen_path)
            print(f"[CKPT] Saved {gen_path.name}")

            test_out = viz_dir / f"test_epoch_{epoch+1:03d}"
            run_test_folder_inference(
                gen=gen,
                test_synth_dir=Path(args.test_synth_dir),
                out_dir=test_out,
                device=device,
                image_size=args.image_size,
                syn_norm_mode=args.syn_norm,
                save_side_by_side=(not args.no_test_side_by_side),
                max_images=args.test_max_images,
            )

    torch.save(gen.state_dict(), ckpt_dir / "gen_final.pth")
    print("[Done] Final generator checkpoint saved.")


if __name__ == "__main__":
    main()
